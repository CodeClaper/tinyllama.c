/*
 * quants_gpu.cu — CUDA batch dequantization backend.
 *
 * Each thread dequantizes one element directly from the raw GGUF bytes
 * that were uploaded to device memory, mirroring the scalar reference
 * implementation in quants.c (and the SIMD CPU batch kernels in
 * cpu/quants_cpu.c) so GPU output is bit-for-bit comparable.
 *
 * The IQ importance grids / value tables live on the host in quants.c;
 * they are copied into __constant__ memory once at first use.
 */

#include <cuda_runtime.h>
#include <stdlib.h>

#include "../def.h"
#include "quants_gpu.h"

extern "C" {
/* Scalar reference dequant, used only by the CPU fallback paths. */
float gguf_dequant(TensorInfo *ti, const u8 *base, u64 i);

/* IQ grids / value tables defined in quants.c, copied to __constant__ at
 * first use.  (quants.h is not included here because its extern helper
 * prototypes clash with the __device__ versions below.) */
extern const float iq2_xxs_grid[];
extern const float iq2_xs_grid[];
extern const float iq3_xxs_grid[];
extern const float iq1_s_grid[];
extern const float iq4_nl_values[];
extern const float iq3_s_grid[];
}

/* ================================================================
 * Device lookup tables (copied from quants.c into __constant__)
 * ================================================================ */

#define IQ2_XXS_SIZE 256
#define IQ2_XS_SIZE  512
#define IQ3_XXS_SIZE 256
#define IQ1_S_SIZE   256
#define IQ4_NL_SIZE  256
#define IQ3_S_SIZE   512

__device__ __constant__ float d_iq2_xxs_grid[IQ2_XXS_SIZE];
__device__ __constant__ float d_iq2_xs_grid[IQ2_XS_SIZE];
__device__ __constant__ float d_iq3_xxs_grid[IQ3_XXS_SIZE];
__device__ __constant__ float d_iq1_s_grid[IQ1_S_SIZE];
__device__ __constant__ float d_iq4_nl_values[IQ4_NL_SIZE];
__device__ __constant__ float d_iq3_s_grid[IQ3_S_SIZE];

/* ================================================================
 * Device helpers
 * ================================================================ */

/* Identical math to f16_to_f32() in quants.c. */
__device__ static inline float f16_to_f32(u16 bits) {
    u32 sign = (u32)(bits >> 15) << 31;
    u32 exp  = (bits >> 10) & 0x1f;
    u32 mant = bits & 0x3ff;
    u32 f32;
    if (exp == 0) {
        if (mant == 0) {
            f32 = sign;
        } else {
            int e = 1 - 15;
            while ((mant & 0x400) == 0) { mant <<= 1; e--; }
            mant &= 0x3ff;
            f32 = sign | ((u32)(e + 127) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f32 = sign | (0xff << 23) | (mant << 13); /* inf / nan */
    } else {
        f32 = sign | ((u32)(exp - 15 + 127) << 23) | (mant << 13);
    }
    return __int_as_float((int)f32);
}

__device__ static inline float bf16_to_f32(u16 bits) {
    return __int_as_float((int)((u32)bits << 16));
}

/* Alignment-safe multi-byte loads.  GGUF quant blocks start on odd byte
 * strides (e.g. q5_0 is 22 bytes), so a raw `*(const u32*)` cast would
 * fault with a misaligned-address error on some GPUs.  Assembling from
 * bytes is correct for every possible alignment. */
__device__ static inline u16 ld_u16(const u8 *p) {
    return (u16)p[0] | ((u16)p[1] << 8);
}
__device__ static inline u32 ld_u32(const u8 *p) {
    return (u32)p[0] | ((u32)p[1] << 8) | ((u32)p[2] << 16) | ((u32)p[3] << 24);
}
__device__ static inline u64 ld_u64(const u8 *p) {
    return (u64)ld_u32(p) | ((u64)ld_u32(p + 4) << 32);
}
__device__ static inline float ld_f32(const u8 *p) {
    return __uint_as_float(ld_u32(p));
}
__device__ static inline double ld_f64(const u8 *p) {
    return __longlong_as_double((long long)ld_u64(p));
}

__device__ static inline void q4k_scale_min(const u8 *s, u32 sb,
                                            u8 *scale, u8 *min) {
    u8 d  = s[sb & 3];
    u8 m  = s[4 + (sb & 3)];
    u8 md = s[8 + (sb & 3)];
    if (sb < 4) {
        *scale = d & 0x3F;
        *min   = m & 0x3F;
    } else {
        *scale = (md & 0x0F) | ((d >> 2) & 0x30);
        *min   = (md >> 4) | ((m >> 2) & 0x30);
    }
}

/* ================================================================
 * Simple (non-block) types
 * ================================================================ */

__device__ static inline float dequant_f32(const u8 *data, u64 i) {
    return ld_f32(data + i * 4);
}
__device__ static inline float dequant_f16(const u8 *data, u64 i) {
    return f16_to_f32(ld_u16(data + i * 2));
}
__device__ static inline float dequant_bf16(const u8 *data, u64 i) {
    return bf16_to_f32(ld_u16(data + i * 2));
}
__device__ static inline float dequant_f64(const u8 *data, u64 i) {
    return (float)(ld_f64(data + i * 8));
}
__device__ static inline float dequant_i8(const u8 *data, u64 i) {
    return (float)(*(const i8 *)(data + i));
}
__device__ static inline float dequant_i16(const u8 *data, u64 i) {
    return (float)(i16)ld_u16(data + i * 2);
}
__device__ static inline float dequant_i32(const u8 *data, u64 i) {
    return (float)(i32)ld_u32(data + i * 4);
}
__device__ static inline float dequant_i64(const u8 *data, u64 i) {
    return (float)(i64)ld_u64(data + i * 8);
}

/* ================================================================
 * Simple block types (block_size = 32)
 * ================================================================ */

__device__ static inline float dequant_q8_0(const u8 *data, u64 i) {
    u64 bi = i >> 5; u32 o = i & 31;
    const u8 *blk = data + bi * 34;
    float d = f16_to_f32(ld_u16(blk));
    return (float)((const i8 *)(blk + 2))[o] * d;
}

__device__ static inline float dequant_q8_1(const u8 *data, u64 i) {
    u64 bi = i >> 5; u32 o = i & 31;
    const u8 *blk = data + bi * 40;
    float d = ld_f32(blk);
    return (float)((const i8 *)(blk + 8))[o] * d;
}

__device__ static inline float dequant_q4_0(const u8 *data, u64 i) {
    u64 bi = i >> 5; u32 o = i & 31;
    const u8 *blk = data + bi * 18;
    float d = f16_to_f32(ld_u16(blk));
    u32 nib = (blk[2 + (o >> 1)] >> ((o & 1) << 2)) & 0xF;
    return ((float)(i32)nib - 8.0f) * d;
}

__device__ static inline float dequant_q4_1(const u8 *data, u64 i) {
    u64 bi = i >> 5; u32 o = i & 31;
    const u8 *blk = data + bi * 20;
    float d = f16_to_f32(ld_u16(blk));
    float m = f16_to_f32(ld_u16(blk + 2));
    u32 nib = (blk[4 + (o >> 1)] >> ((o & 1) << 2)) & 0xF;
    return (float)nib * d + m;
}

__device__ static inline float dequant_q5_0(const u8 *data, u64 i) {
    u64 bi = i >> 5; u32 o = i & 31;
    const u8 *blk = data + bi * 22;
    float d = f16_to_f32(ld_u16(blk));
    u32 qh = ld_u32(blk + 2);
    u32 hi = (qh >> o) & 1;
    u32 lo = (blk[6 + (o >> 1)] >> ((o & 1) << 2)) & 0xF;
    return ((float)(i32)((hi << 4) | lo) - 16.0f) * d;
}

__device__ static inline float dequant_q5_1(const u8 *data, u64 i) {
    u64 bi = i >> 5; u32 o = i & 31;
    const u8 *blk = data + bi * 24;
    float d = f16_to_f32(ld_u16(blk));
    float m = f16_to_f32(ld_u16(blk + 2));
    u32 qh = ld_u32(blk + 4);
    u32 hi = (qh >> o) & 1;
    u32 lo = (blk[8 + (o >> 1)] >> ((o & 1) << 2)) & 0xF;
    return (float)((hi << 4) | lo) * d + m;
}

/* ================================================================
 * K-quant types (block_size = 256)
 * ================================================================ */

__device__ static inline float dequant_q8_k(const u8 *data, u64 i) {
    u64 bi = i >> 8; u32 o = i & 255;
    const u8 *blk = data + bi * 292;
    float d = ld_f32(blk);
    float bs = f16_to_f32(ld_u16(blk + 4 + 2 * (o >> 4)));
    return (float)((const i8 *)(blk + 36))[o] * bs * d;
}

__device__ static inline float dequant_q6_k(const u8 *data, u64 i) {
    u64 bi = i >> 8; u32 o = i & 255;
    const u8 *blk = data + bi * 210;
    float d = f16_to_f32(ld_u16(blk + 208));
    i32 sc = (i8)blk[192 + (o >> 4)];

    u32 hl      = o & 127;          /* position within 128-element half   */
    u32 which   = hl >> 5;          /* 0..3: which 32-element sub-group   */
    u32 l       = hl & 31;          /* 0..31: position inside sub-group   */
    u32 half    = o >> 7;           /* 0 or 1: which 128-element half     */

    u32 ql_off  = (half << 6) + ((which & 1) << 5) + l;
    u32 lo      = (which >= 2) ? ((blk[ql_off] >> 4) & 0xF)
                               :  (blk[ql_off] & 0xF);

    u32 qh_off  = 128 + (half << 5) + l;
    u32 hi      = (blk[qh_off] >> (which * 2)) & 0x3;

    i32 q  = (i32)(lo | (hi << 4)) - 32;
    return d * (float)sc * (float)q;
}

__device__ static inline float dequant_q5_k(const u8 *data, u64 i) {
    u64 bi = i >> 8; u32 o = i & 255;
    const u8 *blk = data + bi * 176;
    float d    = f16_to_f32(ld_u16(blk));
    float dmin = f16_to_f32(ld_u16(blk + 2));
    u32 sb = o >> 5;
    u8 sc, mn;
    q4k_scale_min(blk + 4, sb, &sc, &mn);
    u32 g  = o >> 6;
    u32 nb = (o >> 5) & 1;
    u32 bc = o & 31;
    u32 lo = (blk[48 + g * 32 + bc] >> (nb * 4)) & 0xF;
    u32 hi = (blk[16 + (o & 31)] >> (o >> 5)) & 1;
    i32 q  = (i32)(lo | (hi << 4));
    return d * (float)sc * (float)q - dmin * (float)mn;
}

__device__ static inline float dequant_q4_k(const u8 *data, u64 i) {
    u64 bi = i >> 8; u32 o = i & 255;
    const u8 *blk = data + bi * 144;
    float d    = f16_to_f32(ld_u16(blk));
    float dmin = f16_to_f32(ld_u16(blk + 2));
    u32 sb = o >> 5;
    u8 sc, mn;
    q4k_scale_min(blk + 4, sb, &sc, &mn);
    u32 g  = o >> 6;
    u32 nb = (o >> 5) & 1;
    u32 bc = o & 31;
    u32 nib = (blk[16 + g * 32 + bc] >> (nb * 4)) & 0xF;
    return d * (float)sc * (float)nib - dmin * (float)mn;
}

__device__ static inline float dequant_q3_k(const u8 *data, u64 i) {
    u64 bi = i >> 8; u32 o = i & 255;
    const u8 *blk = data + bi * 110;
    float d = f16_to_f32(ld_u16(blk + 108));
    u32 s = o >> 4;
    u32 g  = o >> 7;
    u32 pr = (o >> 5) & 3;
    u32 bc = o & 31;
    u32 lo = (blk[32 + g * 32 + bc] >> (pr * 2)) & 3;
    u32 hi = (blk[o & 31] >> (o >> 5)) & 1;
    u32 sc_low  = (blk[96 + (s & 7)]  >> (s >= 8 ? 4 : 0)) & 0xF;
    u32 sc_high = (blk[104 + (s & 3)] >> ((s >> 2) * 2)) & 0x3;
    i32 sc = (i32)(sc_low | (sc_high << 4)) - 32;
    i32 q  = (i32)(lo | (hi << 2)) - 4;
    return d * (float)sc * (float)q;
}

__device__ static inline float dequant_q2_k(const u8 *data, u64 i) {
    u64 bi = i >> 8; u32 o = i & 255;
    const u8 *blk = data + bi * 84;
    float d  = f16_to_f32(ld_u16(blk + 64 + 16));
    float mn = f16_to_f32(ld_u16(blk + 64 + 16 + 2));
    u32 sb  = o >> 4;
    u32 sc  = blk[64 + (sb >> 1)];
    if (sb & 1) sc >>= 4; else sc &= 0xF;
    u32 q2 = (blk[(o >> 2)] >> ((o & 3) << 1)) & 0x3;
    i32 q  = (i32)q2 - (i32)((blk[(o >> 3)] >> (o & 7)) & 1) * 4;
    float sc_f = (float)(i32)sc;
    return d * sc_f * (float)q - mn;
}

/* ================================================================
 * IQ (importance-matrix-aware) types
 * ================================================================ */

__device__ static inline float dequant_iq2_xxs(const u8 *data, u64 i) {
    u64 bi = i >> 8; u32 o = i & 255;
    const u8 *blk = data + bi * 66;
    float d = f16_to_f32(ld_u16(blk));
    u32 q = (blk[2 + (o >> 2)] >> ((o & 3) << 1)) & 0x3;
    u32 sign_bit = (blk[2 + (o >> 3)] >> (o & 7)) & 1;
    float v = d_iq2_xxs_grid[q | (sign_bit << 2)];
    return v * d;
}

__device__ static inline float dequant_iq2_xs(const u8 *data, u64 i) {
    u64 bi = i >> 8; u32 o = i & 255;
    const u8 *blk = data + bi * 74;
    float d = f16_to_f32(ld_u16(blk));
    u32 sh = (blk[2 + (o >> 3)] >> (o & 7)) & 1;
    u32 q  = (blk[2 + 32 + (o >> 2)] >> ((o & 3) << 1)) & 0x3;
    u32 idx = q | (sh << 2);
    i8  sc = (i8)(blk[2 + 32 + 64 + (o >> 4)]);
    return d * (float)sc * d_iq2_xs_grid[idx];
}

__device__ static inline float dequant_iq3_xxs(const u8 *data, u64 i) {
    u64 bi = i >> 8; u32 o = i & 255;
    const u8 *blk = data + bi * 98;
    float d = f16_to_f32(ld_u16(blk));
    u32 qh = (blk[2 + (o >> 5)] >> (o & 31)) & 1;
    u32 qm = (blk[2 + 8 + (o >> 3)] >> (o & 7)) & 1;
    u32 ql = (blk[2 + 8 + 32 + (o >> 1)] >> ((o & 1) << 2)) & 0xF;
    u32 grp = (blk[2 + 8 + 32 + 128 + (o >> 3)] >> ((o & 7) << 1)) & 0x3;
    u32 idx = ql | (qm << 4) | (qh << 5) | (grp << 6);
    i8  sc  = (i8)(blk[2 + 8 + 32 + 128 + 64 + (o >> 4)]);
    return d * (float)sc * d_iq3_xxs_grid[idx];
}

__device__ static inline float dequant_iq1_s(const u8 *data, u64 i) {
    u64 bi = i >> 8; u32 o = i & 255;
    const u8 *blk = data + bi * 110;
    float d = f16_to_f32(ld_u16(blk));
    u32 q  = ((blk[2 + (o >> 3)] >> (o & 7)) & 1) |
             (((blk[2 + 32 + (o >> 3)] >> (o & 7)) & 1) << 1);
    u32 idx = q | (((o >> 4) & 0xF) << 2);
    return d * d_iq1_s_grid[idx];
}

__device__ static inline float dequant_iq4_nl(const u8 *data, u64 i) {
    u64 bi = i >> 8; u32 o = i & 255;
    const u8 *blk = data + bi * 50;
    float d = f16_to_f32(ld_u16(blk));
    u32 q = (blk[2 + (o >> 1)] >> ((o & 1) << 2)) & 0xF;
    return d * d_iq4_nl_values[q | ((o & 0xF) << 4)];
}

__device__ static inline float dequant_iq1_m(const u8 *data, u64 i) {
    u64 bi = i >> 8; u32 o = i & 255;
    const u8 *blk = data + bi * 56;
    float d = f16_to_f32(ld_u16(blk + 54));
    u32 sb    = o >> 4;
    u32 sc    = (blk[sb >> 1] >> ((sb & 1) << 2)) & 0xF;
    float sc_f = (float)(i32)sc + 1.0f;
    u32 val  = (blk[8 + (o >> 3)] >> (o & 7)) & 1;
    u32 sign = (blk[8 + 32 + (o >> 3)] >> (o & 7)) & 1;
    float v = sign ? -(float)val : (float)val;
    return v * sc_f * d;
}

__device__ static inline float dequant_iq3_s(const u8 *data, u64 i) {
    u64 bi = i >> 8; u32 o = i & 255;
    const u8 *blk = data + bi * 110;
    float d = f16_to_f32(ld_u16(blk));
    u32 sign = (blk[2 + (o >> 3)] >> (o & 7)) & 1;
    u32 qh   = (blk[2 + 32 + (o >> 5)] >> (o & 31)) & 1;
    u32 qm   = (blk[2 + 32 + 8 + (o >> 3)] >> (o & 7)) & 1;
    u32 ql   = (blk[2 + 32 + 8 + 32 + (o >> 1)] >> ((o & 1) << 2)) & 0xF;
    u32 idx  = ql | (qm << 4) | (qh << 5) | (sign << 6);
    i8  sc   = (i8)blk[2 + 32 + 8 + 32 + 128 + (o >> 4)];
    return d * (float)sc * d_iq3_s_grid[idx];
}

__device__ static inline float dequant_iq2_s(const u8 *data, u64 i) {
    u64 bi = i >> 8; u32 o = i & 255;
    const u8 *blk = data + bi * 82;
    float d = f16_to_f32(ld_u16(blk));
    u32 sign = (blk[2 + (o >> 3)] >> (o & 7)) & 1;
    u32 qh   = (blk[2 + 32 + (o >> 5)] >> (o & 31)) & 1;
    u32 ql   = (blk[2 + 32 + 8 + (o >> 1)] >> ((o & 1) << 2)) & 0xF;
    u32 idx  = ql | (qh << 4) | (sign << 5);
    i8  sc   = (i8)blk[2 + 32 + 8 + 128 + (o >> 4) + ((o >> 4) & 1) * 16];
    u32 sc_l = (blk[2 + 32 + 8 + 128 + (o >> 5)] >> ((o & 31) >> 2)) & 0x3;
    float sc_f = (float)sc + 0.125f * (float)(i32)sc_l;
    return d * sc_f * d_iq2_xs_grid[idx];
}

__device__ static inline float dequant_iq4_xs(const u8 *data, u64 i) {
    u64 bi = i >> 8; u32 o = i & 255;
    const u8 *blk = data + bi * 136;
    float d = f16_to_f32(ld_u16(blk));
    u32 qh = (blk[2 + (o >> 5)] >> (o & 31)) & 1;
    u32 ql = (blk[2 + 8 + (o >> 1)] >> ((o & 1) << 2)) & 0xF;
    u32 q  = ql | (qh << 4);
    i8  sc_l = (i8)blk[2 + 8 + 128 + (o >> 4)];
    i8  sc_h = (i8)blk[2 + 8 + 128 + 16 + (o >> 4)];
    float sc = (float)(i32)sc_l + ((float)(i32)sc_h / 127.0f);
    return d * sc * d_iq4_nl_values[q];
}

/* ================================================================
 * Per-element dispatcher
 * ================================================================ */

__device__ static inline float dequant_one(u32 type, const u8 *data, u64 i) {
    switch (type) {
        case GGUF_TYPE_F32:     return dequant_f32(data, i);
        case GGUF_TYPE_F16:     return dequant_f16(data, i);
        case GGUF_TYPE_BF16:    return dequant_bf16(data, i);
        case GGUF_TYPE_F64:     return dequant_f64(data, i);
        case GGUF_TYPE_I8:      return dequant_i8(data, i);
        case GGUF_TYPE_I16:     return dequant_i16(data, i);
        case GGUF_TYPE_I32:     return dequant_i32(data, i);
        case GGUF_TYPE_I64:     return dequant_i64(data, i);
        case GGUF_TYPE_Q8_0:    return dequant_q8_0(data, i);
        case GGUF_TYPE_Q8_1:    return dequant_q8_1(data, i);
        case GGUF_TYPE_Q4_0:    return dequant_q4_0(data, i);
        case GGUF_TYPE_Q4_1:    return dequant_q4_1(data, i);
        case GGUF_TYPE_Q5_0:    return dequant_q5_0(data, i);
        case GGUF_TYPE_Q5_1:    return dequant_q5_1(data, i);
        case GGUF_TYPE_Q8_K:    return dequant_q8_k(data, i);
        case GGUF_TYPE_Q6_K:    return dequant_q6_k(data, i);
        case GGUF_TYPE_Q5_K:    return dequant_q5_k(data, i);
        case GGUF_TYPE_Q4_K:    return dequant_q4_k(data, i);
        case GGUF_TYPE_Q3_K:    return dequant_q3_k(data, i);
        case GGUF_TYPE_Q2_K:    return dequant_q2_k(data, i);
        case GGUF_TYPE_IQ2_XXS: return dequant_iq2_xxs(data, i);
        case GGUF_TYPE_IQ2_XS:  return dequant_iq2_xs(data, i);
        case GGUF_TYPE_IQ3_XXS: return dequant_iq3_xxs(data, i);
        case GGUF_TYPE_IQ1_S:   return dequant_iq1_s(data, i);
        case GGUF_TYPE_IQ4_NL:  return dequant_iq4_nl(data, i);
        case GGUF_TYPE_IQ1_M:   return dequant_iq1_m(data, i);
        case GGUF_TYPE_IQ3_S:   return dequant_iq3_s(data, i);
        case GGUF_TYPE_IQ2_S:   return dequant_iq2_s(data, i);
        case GGUF_TYPE_IQ4_XS:  return dequant_iq4_xs(data, i);
        default:                return 0.0f;
    }
}

/* ================================================================
 * Kernels
 * ================================================================ */

/* Dequantize elements [i0, i0+nb) into out.  data points at the start of
 * the block containing element `start_i` (start_i is block-aligned), so a
 * thread dequantizes relative element (i0 + j) - start_i. */
__global__ void dequant_range_kernel(u32 type, const u8 * __restrict__ data,
                                     u64 start_i, u64 i0, u64 nb,
                                     float * __restrict__ out) {
    u64 idx    = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    u64 stride = (u64)gridDim.x * blockDim.x;
    for (u64 j = idx; j < nb; j += stride) {
        out[j] = dequant_one(type, data, (i0 + j) - start_i);
    }
}

/* Fused dequant + dot: sum_{j=0}^{n-1} w[i0+j] * x[j], one partial sum per
 * block, host-side final reduction. */
__global__ void dequant_dot_kernel(u32 type, const u8 * __restrict__ data,
                                   u64 start_i, u64 i0, u64 n,
                                   const float * __restrict__ x,
                                   float * __restrict__ partial) {
    __shared__ float sh[256];
    int tid = threadIdx.x;
    u64 idx    = (u64)blockIdx.x * blockDim.x + tid;
    u64 stride = (u64)gridDim.x * blockDim.x;
    float sum = 0.0f;
    for (u64 j = idx; j < n; j += stride) {
        sum += dequant_one(type, data, (i0 + j) - start_i) * x[j];
    }
    sh[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) sh[tid] += sh[tid + s];
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = sh[0];
}

/* ================================================================
 * Host-side plumbing
 * ================================================================ */

#define GPU_THREADS   256
#define GPU_MAX_BLOCKS 65535u

static int g_tables_loaded = 0;

int gpu_available(void) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) return 0;
    if (count <= 0) return 0;
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return 0;
    return 1;
}

static int gpu_load_tables(void) {
    if (g_tables_loaded) return 0;
    cudaError_t err;
    err = cudaMemcpyToSymbol(d_iq2_xxs_grid, iq2_xxs_grid, IQ2_XXS_SIZE * sizeof(float), 0, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return -1;
    err = cudaMemcpyToSymbol(d_iq2_xs_grid,  iq2_xs_grid,  IQ2_XS_SIZE  * sizeof(float), 0, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return -1;
    err = cudaMemcpyToSymbol(d_iq3_xxs_grid, iq3_xxs_grid, IQ3_XXS_SIZE * sizeof(float), 0, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return -1;
    err = cudaMemcpyToSymbol(d_iq1_s_grid,   iq1_s_grid,   IQ1_S_SIZE   * sizeof(float), 0, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return -1;
    err = cudaMemcpyToSymbol(d_iq4_nl_values, iq4_nl_values, IQ4_NL_SIZE * sizeof(float), 0, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return -1;
    err = cudaMemcpyToSymbol(d_iq3_s_grid,   iq3_s_grid,   IQ3_S_SIZE   * sizeof(float), 0, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return -1;
    g_tables_loaded = 1;
    return 0;
}

/* Block geometry per GGUF type — must match the gguf_types[] table in core.c. */
static int gpu_geom(u32 type, u64 *block_elems, u64 *block_bytes) {
    switch (type) {
        case GGUF_TYPE_F32:     *block_elems = 1;   *block_bytes = 4;   return 0;
        case GGUF_TYPE_F16:     *block_elems = 1;   *block_bytes = 2;   return 0;
        case GGUF_TYPE_BF16:    *block_elems = 1;   *block_bytes = 2;   return 0;
        case GGUF_TYPE_F64:     *block_elems = 1;   *block_bytes = 8;   return 0;
        case GGUF_TYPE_I8:      *block_elems = 1;   *block_bytes = 1;   return 0;
        case GGUF_TYPE_I16:     *block_elems = 1;   *block_bytes = 2;   return 0;
        case GGUF_TYPE_I32:     *block_elems = 1;   *block_bytes = 4;   return 0;
        case GGUF_TYPE_I64:     *block_elems = 1;   *block_bytes = 8;   return 0;
        case GGUF_TYPE_Q4_0:    *block_elems = 32;  *block_bytes = 18;  return 0;
        case GGUF_TYPE_Q4_1:    *block_elems = 32;  *block_bytes = 20;  return 0;
        case GGUF_TYPE_Q5_0:    *block_elems = 32;  *block_bytes = 22;  return 0;
        case GGUF_TYPE_Q5_1:    *block_elems = 32;  *block_bytes = 24;  return 0;
        case GGUF_TYPE_Q8_0:    *block_elems = 32;  *block_bytes = 34;  return 0;
        case GGUF_TYPE_Q8_1:    *block_elems = 32;  *block_bytes = 40;  return 0;
        case GGUF_TYPE_Q2_K:    *block_elems = 256; *block_bytes = 84;  return 0;
        case GGUF_TYPE_Q3_K:    *block_elems = 256; *block_bytes = 110; return 0;
        case GGUF_TYPE_Q4_K:    *block_elems = 256; *block_bytes = 144; return 0;
        case GGUF_TYPE_Q5_K:    *block_elems = 256; *block_bytes = 176; return 0;
        case GGUF_TYPE_Q6_K:    *block_elems = 256; *block_bytes = 210; return 0;
        case GGUF_TYPE_Q8_K:    *block_elems = 256; *block_bytes = 292; return 0;
        case GGUF_TYPE_IQ2_XXS: *block_elems = 256; *block_bytes = 66;  return 0;
        case GGUF_TYPE_IQ2_XS:  *block_elems = 256; *block_bytes = 74;  return 0;
        case GGUF_TYPE_IQ3_XXS: *block_elems = 256; *block_bytes = 98;  return 0;
        case GGUF_TYPE_IQ1_S:   *block_elems = 256; *block_bytes = 110; return 0;
        case GGUF_TYPE_IQ4_NL:  *block_elems = 256; *block_bytes = 50;  return 0;
        case GGUF_TYPE_IQ3_S:   *block_elems = 256; *block_bytes = 110; return 0;
        case GGUF_TYPE_IQ2_S:   *block_elems = 256; *block_bytes = 82;  return 0;
        case GGUF_TYPE_IQ4_XS:  *block_elems = 256; *block_bytes = 136; return 0;
        case GGUF_TYPE_IQ1_M:   *block_elems = 256; *block_bytes = 56;  return 0;
        default: return -1;
    }
}

static inline unsigned gpu_block_count(u64 n) {
    u64 want = (n + GPU_THREADS - 1) / GPU_THREADS;
    return (unsigned)(want < GPU_MAX_BLOCKS ? want : GPU_MAX_BLOCKS);
}

/* Scalar CPU fallbacks — used when CUDA is unavailable or a type is
 * unsupported, so callers never need a separate error path. */
static void scalar_dequant_range(TensorInfo *ti, const u8 *base,
                                 u64 i0, u64 nb, float *out) {
    for (u64 j = 0; j < nb; j++)
        out[j] = gguf_dequant(ti, base, i0 + j);
}

static float scalar_dot(TensorInfo *ti, const u8 *base,
                        u64 i, u64 n, const float *x) {
    double sum = 0.0;
    for (u64 j = 0; j < n; j++)
        sum += (double)gguf_dequant(ti, base, i + j) * (double)x[j];
    return (float)sum;
}

int gpu_dequant_batch(TensorInfo *ti, const u8 *base, u64 i0, u64 nb, float *out) {
    if (!ti || !base || !out || nb == 0) return 0;
    if (i0 + nb > ti->n_element) return -1;

    if (!gpu_available() || gpu_load_tables() != 0) {
        scalar_dequant_range(ti, base, i0, nb, out);
        return 0;
    }

    u64 be, bb;
    if (gpu_geom(ti->type, &be, &bb) != 0) {
        scalar_dequant_range(ti, base, i0, nb, out);
        return 0;
    }

    u64 start_i   = (i0 / be) * be;
    u64 end_i     = i0 + nb;
    u64 n_blocks  = (end_i - start_i + be - 1) / be;
    u64 bytes     = n_blocks * bb;
    const u8 *src = base + ti->offset + (start_i / be) * bb;

    cudaError_t err;
    u8  *d_data = NULL;
    float *d_out = NULL;
    err = cudaMalloc(&d_data, bytes);
    if (err != cudaSuccess) return -1;
    err = cudaMalloc(&d_out, nb * sizeof(float));
    if (err != cudaSuccess) { cudaFree(d_data); return -1; }
    err = cudaMemcpy(d_data, src, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { cudaFree(d_out); cudaFree(d_data); return -1; }

    unsigned blocks = gpu_block_count(nb);
    dequant_range_kernel<<<blocks, GPU_THREADS>>>(ti->type, d_data, start_i, i0, nb, d_out);
    err = cudaGetLastError();
    if (err != cudaSuccess) { cudaFree(d_out); cudaFree(d_data); return -1; }

    err = cudaMemcpy(out, d_out, nb * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_out);
    cudaFree(d_data);
    if (err != cudaSuccess) return -1;
    return 0;
}

int gpu_dequant_tensor(TensorInfo *ti, const u8 *base, float *out) {
    return gpu_dequant_batch(ti, base, 0, ti->n_element, out);
}

float gpu_dot_batch(TensorInfo *ti, const u8 *base, u64 i, u64 n, const float *x) {
    if (!ti || !base || !x || n == 0) return 0.0f;
    if (i + n > ti->n_element) return 0.0f;

    if (!gpu_available() || gpu_load_tables() != 0)
        return scalar_dot(ti, base, i, n, x);

    u64 be, bb;
    if (gpu_geom(ti->type, &be, &bb) != 0)
        return scalar_dot(ti, base, i, n, x);

    u64 start_i   = (i / be) * be;
    u64 end_i     = i + n;
    u64 n_blocks  = (end_i - start_i + be - 1) / be;
    u64 bytes     = n_blocks * bb;
    const u8 *src = base + ti->offset + (start_i / be) * bb;

    unsigned blocks = gpu_block_count(n);
    cudaError_t err;
    u8   *d_data    = NULL;
    float *d_x      = NULL;
    float *d_partial = NULL;
    err = cudaMalloc(&d_data, bytes);
    if (err != cudaSuccess) return 0.0f;
    err = cudaMalloc(&d_x, n * sizeof(float));
    if (err != cudaSuccess) { cudaFree(d_data); return 0.0f; }
    err = cudaMalloc(&d_partial, blocks * sizeof(float));
    if (err != cudaSuccess) { cudaFree(d_x); cudaFree(d_data); return 0.0f; }
    err = cudaMemcpy(d_data, src, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { cudaFree(d_partial); cudaFree(d_x); cudaFree(d_data); return 0.0f; }
    err = cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { cudaFree(d_partial); cudaFree(d_x); cudaFree(d_data); return 0.0f; }

    dequant_dot_kernel<<<blocks, GPU_THREADS>>>(ti->type, d_data, start_i, i, n, d_x, d_partial);
    err = cudaGetLastError();
    if (err != cudaSuccess) { cudaFree(d_partial); cudaFree(d_x); cudaFree(d_data); return 0.0f; }

    float *partial = (float *)malloc(blocks * sizeof(float));
    if (!partial) { cudaFree(d_partial); cudaFree(d_x); cudaFree(d_data); return 0.0f; }
    err = cudaMemcpy(partial, d_partial, blocks * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_partial);
    cudaFree(d_x);
    cudaFree(d_data);
    if (err != cudaSuccess) { free(partial); return 0.0f; }

    double sum = 0.0;
    for (unsigned b = 0; b < blocks; b++)
        sum += (double)partial[b];
    free(partial);
    return (float)sum;
}
