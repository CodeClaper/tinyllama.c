/*
 * gpu.cu — CUDA batch dequantization backend.
 *
 * Each thread dequantizes one element directly from the raw GGUF bytes
 * that were uploaded to device memory, mirroring the scalar reference
 * implementation in quants.c (and the SIMD CPU batch kernels in
 * backend/cpu/cpu.c) so GPU output is bit-for-bit comparable.
 *
 * The IQ importance grids / value tables live on the host in quants.c;
 * they are copied into __constant__ memory once at first use.
 */

#include <cuda_runtime.h>
#include <stdlib.h>

#include "../../def.h"
#include "gpu.h"

extern "C" {
/* Scalar reference dequant, used only by the CPU fallback paths. */
float gguf_dequant(TensorInfo *ti, u64 i);

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

__device__ static inline void q4k_scale_min(const u8 *s, u32 sb, u8 *scale, u8 *min) {
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
 * Block-cooperative GEMV
 *
 * The per-element helpers above take a flat element index, so they must
 * re-derive (and re-convert) their quant block's scale/min header on
 * every call: two software f16_to_f32() calls, plus the header loads,
 * per ELEMENT.  On K-quants that decode work is the whole kernel —
 * matmul_kernel_nontrans() plateaued near 40 GB/s on a card capable of
 * ~170 GB/s, because it spent its time converting the same two f16
 * headers 256 times per block.
 *
 * The helpers below split decode in two, so the header is converted
 * once per lane per block instead of once per element:
 *
 *   blk_scale<TYPE>()  block scale/min for this lane's element range
 *   blk_elem <TYPE>()  one element, given the already-decoded scale
 *
 * Lane `lane` owns the EPT consecutive elements
 *   [lane*EPT, lane*EPT + EPT)
 * of a block of QK elements, so a 32-lane warp covers a whole block
 * (32 * EPT == QK) and every byte of the block is read by exactly the
 * lanes that need it.  Each quant type's sub-block scale is chosen so
 * that it is constant across a lane's EPT elements — verified for
 * every type below, since blk_scale() indexes it with the lane's first
 * element only.
 *
 * Requires cols % QK == 0 so no block straddles a row; matmul_launch()
 * checks that and falls back to the per-element kernels otherwise.
 * ================================================================ */

/* Decoded per-lane block header.  Slot meanings are type-specific; most
 * types use v[0] as the block scale and v[1] as the block minimum. */
struct BlkS { float v[4]; };

/* Per-type block geometry.  QK is the element count of one quant block;
 * EPT is the number of elements one lane decodes from that block.
 *
 * Only quantized types are listed.  The unquantized ones (F32/F16/BF16/
 * I8/I16/I32/I64/F64) have no block header, so there is nothing for
 * blk_scale() to hoist: their per-element decode is already one load +
 * one convert + one FMA, and measured against a 220 GB/s read ceiling
 * on the reference card the plain warp-per-row kernel already reaches
 * ~205 GB/s on BF16 (94%).  Leaving QK = 0 for them keeps them on that
 * kernel, which measured 7% faster than this one (the returned BlkS is
 * dead weight the compiler cannot always sink). */
template<int TYPE> struct QT { enum { QK = 0, BS = 0, EPT = 0 }; };

#define QT_DEF(T, qk, bs)                                 \
    template<> struct QT<GGUF_TYPE_##T> {                 \
        enum { QK = (qk), BS = (bs), EPT = (qk) / 32 };   \
    };

QT_DEF(Q4_0,  32,  18)
QT_DEF(Q4_1,  32,  20)
QT_DEF(Q5_0,  32,  22)
QT_DEF(Q5_1,  32,  24)
QT_DEF(Q8_0,  32,  34)
QT_DEF(Q8_1,  32,  40)
QT_DEF(Q2_K, 256,  84)
QT_DEF(Q3_K, 256, 110)
QT_DEF(Q4_K, 256, 144)
QT_DEF(Q5_K, 256, 176)
QT_DEF(Q6_K, 256, 210)
QT_DEF(Q8_K, 256, 292)
#undef QT_DEF

/* Block header for the element range this lane owns. */
template<int TYPE>
__device__ __forceinline__ BlkS blk_scale(const u8 *blk, u32 lane) {
    constexpr int EPT = QT<TYPE>::EPT;
    const u32 o0 = lane * (u32)EPT;   /* first element owned by this lane */
    BlkS s = {};   /* unquantized types have no header; leave it zeroed */

    if constexpr (TYPE == GGUF_TYPE_Q4_K || TYPE == GGUF_TYPE_Q5_K) {
        /* sb = o >> 5 is lane >> 2 across the lane's EPT = 8 elements. */
        s.v[0] = f16_to_f32(ld_u16(blk));
        s.v[1] = f16_to_f32(ld_u16(blk + 2));
        u8 sc, mn;
        q4k_scale_min(blk + 4, o0 >> 5, &sc, &mn);
        s.v[2] = (float)sc;
        s.v[3] = (float)mn;
    } else if constexpr (TYPE == GGUF_TYPE_Q6_K) {
        s.v[0] = f16_to_f32(ld_u16(blk + 208));
        s.v[1] = (float)(i8)blk[192 + (o0 >> 4)];
    } else if constexpr (TYPE == GGUF_TYPE_Q8_K) {
        /* Sub-block scale covers 16 elements -> constant over EPT = 8. */
        s.v[0] = ld_f32(blk);
        s.v[1] = f16_to_f32(ld_u16(blk + 4 + 2 * (o0 >> 4)));
    } else if constexpr (TYPE == GGUF_TYPE_Q2_K) {
        /* Sub-block scale covers 16 elements -> constant over EPT = 8. */
        u32 sb = o0 >> 4;
        u32 sc = blk[64 + (sb >> 1)];
        sc = (sb & 1) ? (sc >> 4) : (sc & 0xF);
        s.v[0] = f16_to_f32(ld_u16(blk + 80));
        s.v[1] = f16_to_f32(ld_u16(blk + 82));
        s.v[2] = (float)(i32)sc;
    } else if constexpr (TYPE == GGUF_TYPE_Q3_K) {
        /* 6-bit scale, sub-block covers 16 elements -> constant over 8. */
        u32 t = o0 >> 4;
        u32 sc_low  = (blk[96 + (t & 7)]  >> (t >= 8 ? 4 : 0)) & 0xF;
        u32 sc_high = (blk[104 + (t & 3)] >> ((t >> 2) * 2))    & 0x3;
        s.v[0] = f16_to_f32(ld_u16(blk + 108));
        s.v[1] = (float)((i32)(sc_low | (sc_high << 4)) - 32);
    } else if constexpr (TYPE == GGUF_TYPE_Q8_0 || TYPE == GGUF_TYPE_Q4_0 ||
                         TYPE == GGUF_TYPE_Q5_0) {
        s.v[0] = f16_to_f32(ld_u16(blk));
    } else if constexpr (TYPE == GGUF_TYPE_Q4_1 || TYPE == GGUF_TYPE_Q5_1) {
        s.v[0] = f16_to_f32(ld_u16(blk));
        s.v[1] = f16_to_f32(ld_u16(blk + 2));
    } else if constexpr (TYPE == GGUF_TYPE_Q8_1) {
        s.v[0] = ld_f32(blk);
    }
    return s;
}

/* Element `lane*EPT + j` of the block, given the header from
 * blk_scale().  Every byte read here is one this lane owns, so a warp's
 * reads for fixed j are contiguous within the block. */
template<int TYPE>
__device__ __forceinline__ float blk_elem(const u8 *blk, u32 lane, u32 j, const BlkS &s) {
    constexpr int EPT = QT<TYPE>::EPT;
    const u32 o = lane * (u32)EPT + j;

    if constexpr (TYPE == GGUF_TYPE_Q4_K || TYPE == GGUF_TYPE_Q5_K) {
        /* g = o >> 6 == lane >> 3, nb = (o >> 5) & 1 == (lane >> 2) & 1,
         * bc = o & 31 == 8*(lane & 3) + j  -> 8 contiguous payload bytes. */
        const u32 g  = lane >> 3;
        const u32 bc = 8u * (lane & 3u) + j;
        const u32 nb = (lane >> 2) & 1u;
        u32 q;
        if constexpr (TYPE == GGUF_TYPE_Q4_K) {
            q = (blk[16 + g * 32 + bc] >> (nb * 4)) & 0xF;
        } else {
            const u32 lo = (blk[48 + g * 32 + bc] >> (nb * 4)) & 0xF;
            const u32 hi = (blk[16 + bc] >> (lane >> 2)) & 1u;
            q = lo | (hi << 4);
        }
        return s.v[0] * s.v[2] * (float)q - s.v[1] * s.v[3];
    } else if constexpr (TYPE == GGUF_TYPE_Q6_K) {
        /* half = o >> 7 == lane >> 4, m = lane & 15, which = m >> 2,
         * l_idx = 8*(m & 3) + j. */
        const u32 m     = lane & 15u;
        const u32 which = m >> 2;
        const u32 l     = 8u * (m & 3u) + j;
        const u32 ql    = 64u * (lane >> 4) + ((which & 1u) << 5) + l;
        const u32 lo    = (which >= 2) ? ((blk[ql] >> 4) & 0xF) : (blk[ql] & 0xF);
        const u32 hi    = (blk[128 + (lane >> 4) * 32 + l] >> (which * 2)) & 0x3;
        return s.v[0] * s.v[1] * (float)((i32)(lo | (hi << 4)) - 32);
    } else if constexpr (TYPE == GGUF_TYPE_Q8_K) {
        return (float)(i8)blk[36 + o] * s.v[1] * s.v[0];
    } else if constexpr (TYPE == GGUF_TYPE_Q2_K) {
        /* o = 8*lane + j: q2 lives in blk[2*lane + j/2], the sign bit in
         * blk[lane] -> three bytes cover all 8 elements. */
        const u32 q2 = (blk[2u * lane + (j >> 2)] >> ((j & 3u) << 1)) & 0x3;
        const i32 q  = (i32)q2 - (i32)((blk[lane] >> j) & 1u) * 4;
        return s.v[0] * s.v[2] * (float)q - s.v[1];
    } else if constexpr (TYPE == GGUF_TYPE_Q3_K) {
        /* g = o >> 7 == lane >> 4, pr = (o >> 5) & 3 == (lane >> 2) & 3,
         * bc = o & 31 == 8*(lane & 3) + j. */
        const u32 bc = 8u * (lane & 3u) + j;
        const u32 lo = (blk[32 + (lane >> 4) * 32 + bc] >> (((lane >> 2) & 3u) * 2)) & 0x3;
        const u32 hi = (blk[bc] >> (lane >> 2)) & 1u;
        return s.v[0] * s.v[1] * (float)((i32)(lo | (hi << 2)) - 4);
    } else if constexpr (TYPE == GGUF_TYPE_Q8_0) {
        return (float)(i8)blk[2 + o] * s.v[0];
    } else if constexpr (TYPE == GGUF_TYPE_Q8_1) {
        return (float)(i8)blk[8 + o] * s.v[0];
    } else if constexpr (TYPE == GGUF_TYPE_Q4_0) {
        const u32 nib = (blk[2 + (o >> 1)] >> ((o & 1u) << 2)) & 0xF;
        return ((float)(i32)nib - 8.0f) * s.v[0];
    } else if constexpr (TYPE == GGUF_TYPE_Q4_1) {
        const u32 nib = (blk[4 + (o >> 1)] >> ((o & 1u) << 2)) & 0xF;
        return (float)nib * s.v[0] + s.v[1];
    } else if constexpr (TYPE == GGUF_TYPE_Q5_0) {
        const u32 qh = ld_u32(blk + 2);
        const u32 hi = (qh >> o) & 1u;
        const u32 lo = (blk[6 + (o >> 1)] >> ((o & 1u) << 2)) & 0xF;
        return ((float)(i32)((hi << 4) | lo) - 16.0f) * s.v[0];
    } else if constexpr (TYPE == GGUF_TYPE_Q5_1) {
        const u32 qh = ld_u32(blk + 4);
        const u32 hi = (qh >> o) & 1u;
        const u32 lo = (blk[8 + (o >> 1)] >> ((o & 1u) << 2)) & 0xF;
        return (float)((hi << 4) | lo) * s.v[0] + s.v[1];
    }
    /* No branch for the unquantized types (F32/F16/BF16/I8/I16/I32/I64/
     * F64): QT<TYPE>::QK == 0 keeps them off this kernel entirely. */
    return 0.0f;
}

/* One warp per output row: y[b*rows + r] = sum_c w[r*cols + c] * x[b*cols + c].
 *
 * Rows are covered grid-stride so the launch works for any `rows`
 * (lm_head is ~152k rows).  Within a row the warp walks the row's
 * quant blocks, decoding each block header once per lane. */
template<int TYPE>
__global__ void gemv_block_kernel(const u8 *__restrict__ w,
                                  const float *__restrict__ x,
                                  float *__restrict__ y,
                                  u64 rows, u64 cols) {
    constexpr int QK  = QT<TYPE>::QK;
    constexpr int BS  = QT<TYPE>::BS;
    constexpr int EPT = QT<TYPE>::EPT;

    const u32 lane = threadIdx.x & 31u;
    const u64 bpr  = cols / (u64)QK;                  /* blocks per row  */
    const float *xr = x + (u64)blockIdx.y * cols;
    float *yr       = y + (u64)blockIdx.y * rows;

    for (u64 r = (u64)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
         r < rows;
         r += (u64)gridDim.x * (blockDim.x >> 5)) {
        u64 blk0 = (r * cols) / (u64)QK;              /* first block of row */
        float sum = 0.0f;
        for (u64 b = 0; b < bpr; b++) {
            const u8 *bp = w + (blk0 + b) * (u64)BS;
            const BlkS s = blk_scale<TYPE>(bp, lane);
            #pragma unroll
            for (int j = 0; j < EPT; j++)
                sum += blk_elem<TYPE>(bp, lane, j, s) * xr[b * (u64)QK + lane * (u64)EPT + j];
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            sum += __shfl_down_sync(0xffffffffu, sum, o);
        if (lane == 0) yr[r] = sum;
    }
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

/* Compile-time twin of dequant_one(): instantiated once per quant
 * type, so each matmul kernel instantiation gets a straight-line
 * per-element decode (no runtime switch in the inner loop). */
template<int TYPE>
__device__ static inline float dequant_one_t(const u8 *data, u64 i) {
    if constexpr (TYPE == GGUF_TYPE_F32)          return dequant_f32(data, i);
    else if constexpr (TYPE == GGUF_TYPE_F16)     return dequant_f16(data, i);
    else if constexpr (TYPE == GGUF_TYPE_BF16)    return dequant_bf16(data, i);
    else if constexpr (TYPE == GGUF_TYPE_F64)     return dequant_f64(data, i);
    else if constexpr (TYPE == GGUF_TYPE_I8)      return dequant_i8(data, i);
    else if constexpr (TYPE == GGUF_TYPE_I16)     return dequant_i16(data, i);
    else if constexpr (TYPE == GGUF_TYPE_I32)     return dequant_i32(data, i);
    else if constexpr (TYPE == GGUF_TYPE_I64)     return dequant_i64(data, i);
    else if constexpr (TYPE == GGUF_TYPE_Q8_0)    return dequant_q8_0(data, i);
    else if constexpr (TYPE == GGUF_TYPE_Q8_1)    return dequant_q8_1(data, i);
    else if constexpr (TYPE == GGUF_TYPE_Q4_0)    return dequant_q4_0(data, i);
    else if constexpr (TYPE == GGUF_TYPE_Q4_1)    return dequant_q4_1(data, i);
    else if constexpr (TYPE == GGUF_TYPE_Q5_0)    return dequant_q5_0(data, i);
    else if constexpr (TYPE == GGUF_TYPE_Q5_1)    return dequant_q5_1(data, i);
    else if constexpr (TYPE == GGUF_TYPE_Q8_K)    return dequant_q8_k(data, i);
    else if constexpr (TYPE == GGUF_TYPE_Q6_K)    return dequant_q6_k(data, i);
    else if constexpr (TYPE == GGUF_TYPE_Q5_K)    return dequant_q5_k(data, i);
    else if constexpr (TYPE == GGUF_TYPE_Q4_K)    return dequant_q4_k(data, i);
    else if constexpr (TYPE == GGUF_TYPE_Q3_K)    return dequant_q3_k(data, i);
    else if constexpr (TYPE == GGUF_TYPE_Q2_K)    return dequant_q2_k(data, i);
    else if constexpr (TYPE == GGUF_TYPE_IQ2_XXS) return dequant_iq2_xxs(data, i);
    else if constexpr (TYPE == GGUF_TYPE_IQ2_XS)  return dequant_iq2_xs(data, i);
    else if constexpr (TYPE == GGUF_TYPE_IQ3_XXS) return dequant_iq3_xxs(data, i);
    else if constexpr (TYPE == GGUF_TYPE_IQ1_S)   return dequant_iq1_s(data, i);
    else if constexpr (TYPE == GGUF_TYPE_IQ4_NL)  return dequant_iq4_nl(data, i);
    else if constexpr (TYPE == GGUF_TYPE_IQ1_M)   return dequant_iq1_m(data, i);
    else if constexpr (TYPE == GGUF_TYPE_IQ3_S)   return dequant_iq3_s(data, i);
    else if constexpr (TYPE == GGUF_TYPE_IQ2_S)   return dequant_iq2_s(data, i);
    else if constexpr (TYPE == GGUF_TYPE_IQ4_XS)  return dequant_iq4_xs(data, i);
    else return 0.0f;
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

/* Embedding-style gather dequant: one output element per thread,
 * out[idx] = dequant(w, ids[p]*base_mul + j*stride) with p = idx/count,
 * j = idx%count.  ids live in device memory (OP_INPUT leaves), so token
 * lookups never round-trip through the host. */
__global__ void dequant_gather_kernel(u32 type, const u8 * __restrict__ w,
                                      const u32 * __restrict__ ids, u32 n_ids,
                                      u32 count, u64 base_mul, u64 stride,
                                      float * __restrict__ out) {
    u64 idx     = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    u64 gstride = (u64)gridDim.x * blockDim.x;
    for (; idx < (u64)n_ids * count; idx += gstride) {
        u32 p = (u32)(idx / count);
        u32 j = (u32)(idx % count);
        out[idx] = dequant_one(type, w, (u64)ids[p] * base_mul + (u64)j * stride);
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
 * Persistent-weight matmul kernels
 *
 * w points at the device copy of the whole tensor; all indices are
 * FLAT element indices (i = r*cols + c / c*rows + r), identical to
 * the scalar reference gguf_dequant().  GGUF quantized tensors have
 * no per-row padding — block boundaries may fall mid-row — so the
 * dequant helpers must derive the block from the flat index, never
 * from a precomputed row pointer.
 * ================================================================ */

/* Non-transposed: y[b*rows + r] = sum_c w[r*cols + c] * x[b*cols + c].
 * One warp per output row; lanes stride c by 32 so each iteration
 * covers exactly one quant block (QK=32 types) — coalesced loads.
 * Grid = (ceil(rows/8), batch); r is warp-uniform so the bounds
 * check exits whole warps (shuffle mask stays full). */
template<int TYPE>
__global__ void matmul_kernel_nontrans(const u8 * __restrict__ w,
                                       const float * __restrict__ x,
                                       float * __restrict__ y,
                                       u64 rows, u64 cols, u64 batch) {
    u64 r = (u64)blockIdx.x * 8 + (threadIdx.x >> 5);
    if (r >= rows) return;
    const float *xr = x + (u64)blockIdx.y * cols;
    float sum = 0.0f;
    for (u64 c = threadIdx.x & 31; c < cols; c += 32)
        sum += dequant_one_t<TYPE>(w, r * cols + c) * xr[c];
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, o);
    if ((threadIdx.x & 31) == 0) y[(u64)blockIdx.y * rows + r] = sum;
}

/* Transposed: y[b*rows + idx] = sum_c w[c*rows + idx] * x[b*cols + c].
 * One thread per output element; 256 threads cover 256 outputs, so per
 * c the 32 lanes of a warp read the same quant block (QK=32 types:
 * block headers broadcast, elements gathered within the block). */
template<int TYPE>
__global__ void matmul_kernel_trans(const u8 * __restrict__ w,
                                    const float * __restrict__ x,
                                    float * __restrict__ y,
                                    u64 rows, u64 cols, u64 batch) {
    u64 idx = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows) return;
    const float *xr = x + (u64)blockIdx.y * cols;
    float sum = 0.0f;
    for (u64 c = 0; c < cols; c++)
        sum += dequant_one_t<TYPE>(w, c * rows + idx) * xr[c];
    y[(u64)blockIdx.y * rows + idx] = sum;
}

/* ================================================================
 * Host-side plumbing
 * ================================================================ */

#define GPU_THREADS   256
#define GPU_MAX_BLOCKS 65535u

static int g_tables_loaded = 0;

int gpu_available(void) {
    static int cached = -1; /* memoized: checked on every matmul call */
    if (cached >= 0) return cached;
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) { cached = 0; return 0; }
    if (count <= 0) { cached = 0; return 0; }
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) { cached = 0; return 0; }
    cached = 1;
    return 1;
}

static void gpu_load_tables(void) {
    if (g_tables_loaded) return;
    CHECK(cudaMemcpyToSymbol(d_iq2_xxs_grid, iq2_xxs_grid, IQ2_XXS_SIZE * sizeof(float), 0, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpyToSymbol(d_iq2_xs_grid,  iq2_xs_grid,  IQ2_XS_SIZE  * sizeof(float), 0, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpyToSymbol(d_iq3_xxs_grid, iq3_xxs_grid, IQ3_XXS_SIZE * sizeof(float), 0, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpyToSymbol(d_iq1_s_grid,   iq1_s_grid,   IQ1_S_SIZE   * sizeof(float), 0, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpyToSymbol(d_iq4_nl_values, iq4_nl_values, IQ4_NL_SIZE * sizeof(float), 0, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpyToSymbol(d_iq3_s_grid,   iq3_s_grid,   IQ3_S_SIZE   * sizeof(float), 0, cudaMemcpyHostToDevice));
    g_tables_loaded = 1;
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
static void scalar_dequant_range(TensorInfo *ti,
                                 u64 i0, u64 nb, float *out) {
    for (u64 j = 0; j < nb; j++)
        out[j] = gguf_dequant(ti, i0 + j);
}

static float scalar_dot(TensorInfo *ti,
                        u64 i, u64 n, const float *x) {
    double sum = 0.0;
    for (u64 j = 0; j < n; j++)
        sum += (double)gguf_dequant(ti, i + j) * (double)x[j];
    return (float)sum;
}

int gpu_dequant_batch(TensorInfo *ti, u64 i0, u64 nb, float *out) {
    if (!ti || !ti->data || !out || nb == 0) return 0;
    if (i0 + nb > ti->n_element) return -1;

    if (!gpu_available()) {
        scalar_dequant_range(ti, i0, nb, out);
        return 0;
    }
    gpu_load_tables();

    u64 be, bb;
    if (gpu_geom(ti->type, &be, &bb) != 0) {
        scalar_dequant_range(ti, i0, nb, out);
        return 0;
    }

    u64 start_i   = (i0 / be) * be;
    u64 end_i     = i0 + nb;
    u64 n_blocks  = (end_i - start_i + be - 1) / be;
    u64 bytes     = n_blocks * bb;
    const u8 *src = (const u8 *)ti->data + (start_i / be) * bb;
    /* Copy through the tensor's declared end, not just the decoded
     * blocks: some IQ decoders read a few bytes past the last block
     * (block-relative offsets spilling beyond the block end), so the
     * device buffer must include the tensor's tail to keep those reads
     * deterministic.  The unit-test tensors carry a padding tail inside
     * ti->bytes; production tensors have ti->bytes == n_blocks*bb, so
     * this is a no-op there. */
    bytes = ti->bytes - (start_i / be) * bb;

    u8  *d_data = NULL;
    float *d_out = NULL;
    CHECK(cudaMalloc(&d_data, bytes));
    CHECK(cudaMalloc(&d_out, nb * sizeof(float)));
    CHECK(cudaMemcpy(d_data, src, bytes, cudaMemcpyHostToDevice));

    unsigned blocks = gpu_block_count(nb);
    dequant_range_kernel<<<blocks, GPU_THREADS>>>(ti->type, d_data, start_i, i0, nb, d_out);
    CHECK(cudaGetLastError());

    CHECK(cudaMemcpy(out, d_out, nb * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_out);
    cudaFree(d_data);
    return 0;
}

int gpu_dequant_tensor(TensorInfo *ti, float *out) {
    return gpu_dequant_batch(ti, 0, ti->n_element, out);
}

float gpu_dot_batch(TensorInfo *ti, u64 i, u64 n, const float *x) {
    if (!ti || !ti->data || !x || n == 0) return 0.0f;
    if (i + n > ti->n_element) return 0.0f;

    if (!gpu_available())
        return scalar_dot(ti, i, n, x);
    gpu_load_tables();

    u64 be, bb;
    if (gpu_geom(ti->type, &be, &bb) != 0)
        return scalar_dot(ti, i, n, x);

    u64 start_i   = (i / be) * be;
    u64 end_i     = i + n;
    u64 n_blocks  = (end_i - start_i + be - 1) / be;
    u64 bytes     = n_blocks * bb;
    const u8 *src = (const u8 *)ti->data + (start_i / be) * bb;

    unsigned blocks = gpu_block_count(n);
    u8   *d_data    = NULL;
    float *d_x      = NULL;
    float *d_partial = NULL;
    CHECK(cudaMalloc(&d_data, bytes));
    CHECK(cudaMalloc(&d_x, n * sizeof(float)));
    CHECK(cudaMalloc(&d_partial, blocks * sizeof(float)));
    CHECK(cudaMemcpy(d_data, src, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice));

    dequant_dot_kernel<<<blocks, GPU_THREADS>>>(ti->type, d_data, start_i, i, n, d_x, d_partial);
    CHECK(cudaGetLastError());

    float *partial = (float *)malloc(blocks * sizeof(float));
    if (!partial) { cudaFree(d_partial); cudaFree(d_x); cudaFree(d_data); return 0.0f; }
    CHECK(cudaMemcpy(partial, d_partial, blocks * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_partial);
    cudaFree(d_x);
    cudaFree(d_data);

    double sum = 0.0;
    for (unsigned b = 0; b < blocks; b++)
        sum += (double)partial[b];
    free(partial);
    return (float)sum;
}

/* ================================================================
 * Persistent-weight matmul backend
 * ================================================================ */

/* Host-side dispatch: one kernel instantiation per quant type, so the
 * device-side switch is compiled away.  Table indexed by GGUF type
 * value; NULL slots (type 4/5 don't exist) mean "fall back to CPU". */
typedef void (*matmul_launch_fn)(const u8 *w, const float *x, float *y,
                                 u64 rows, u64 cols, u64 batch, bool trans);

template<int TYPE>
static void matmul_launch(const u8 *w, const float *x, float *y,
                          u64 rows, u64 cols, u64 batch, bool trans) {
    /* Fast path — one warp per output row, with each quant block's
     * header decoded once per lane instead of once per element.  Only
     * the non-transposed orientation has that layout, and it needs
     * cols % QK == 0 so no block straddles a row.  Everything else
     * falls through to the per-element kernels below. */
    if constexpr (QT<TYPE>::QK > 0) {
        if (!trans && cols % (u64)QT<TYPE>::QK == 0) {
            u64 want = (rows + 7) / 8;   /* 8 warps per block */
            dim3 grid((unsigned)(want < GPU_MAX_BLOCKS ? want : GPU_MAX_BLOCKS),
                      (unsigned)batch);
            gemv_block_kernel<TYPE><<<grid, 256>>>(w, x, y, rows, cols);
            CHECK(cudaGetLastError());
            return;
        }
    }
    dim3 block(256);
    dim3 grid(trans ? (unsigned)((rows + 255) / 256) : (unsigned)((rows + 7) / 8), (unsigned)batch);
    if (trans) matmul_kernel_trans<TYPE><<<grid, block>>>(w, x, y, rows, cols, batch);
    else matmul_kernel_nontrans<TYPE><<<grid, block>>>(w, x, y, rows, cols, batch);
    /* CHECK also clears the sticky error flag, so a failed launch
     * cannot poison a later cudaGetLastError(). */
    CHECK(cudaGetLastError());
}

#define GPU_TYPE_LIST(X) \
    X(F32) X(F16) X(Q4_0) X(Q4_1) \
    X(Q5_0) X(Q5_1) X(Q8_0) X(Q8_1) \
    X(Q2_K) X(Q3_K) X(Q4_K) X(Q5_K) \
    X(Q6_K) X(Q8_K) X(IQ2_XXS) X(IQ2_XS) \
    X(IQ3_XXS) X(IQ1_S) X(IQ4_NL) X(IQ3_S) \
    X(IQ2_S) X(IQ4_XS) X(I8) X(I16) \
    X(I32) X(I64) X(F64) X(IQ1_M) \
    X(BF16)

/* Built at first use — GCC's C++ frontend rejects designated
 * initializers carrying template addresses, so a runtime fill. */
static matmul_launch_fn gpu_matmul_dispatch[31];

static void gpu_dispatch_init(void) {
    static int done = 0;
    if (done) return;
#define GPU_DISPATCH_X(T) gpu_matmul_dispatch[GGUF_TYPE_##T] = matmul_launch<GGUF_TYPE_##T>;
    GPU_TYPE_LIST(GPU_DISPATCH_X)
#undef GPU_DISPATCH_X
    done = 1;
}

/* Device weight cache: (ti->data, ti) identifies a tensor
 * unambiguously (ti->data is the resolved host buffer; ti the flat
 * tensor array element).  Uploaded lazily on first use; never
 * evicted — weights are read-only and the model is loaded once.  A
 * failed upload is fatal (CHECK) rather than a silent fallback to CPU.
 *
 * A content snapshot guards the pointer key: if a caller reuses a
 * freed (ti->data, ti) identity for a new tensor (the unit tests hit
 * this via malloc address reuse), the stale device copy is replaced
 * rather than served.  The engine never aliases identities, so the
 * snapshot always matches there and the replace path is never taken. */
#define GPU_CACHE_MAX 4096
static const u8   *g_cache_base[GPU_CACHE_MAX];
static TensorInfo *g_cache_ti[GPU_CACHE_MAX];
static u8         *g_cache_dev[GPU_CACHE_MAX];
static u64         g_cache_elems[GPU_CACHE_MAX];
static u64         g_cache_bytes[GPU_CACHE_MAX];
static u32         g_cache_type[GPU_CACHE_MAX];
static u64         g_cache_off[GPU_CACHE_MAX];
static int         g_cache_n = 0;

static u8 *gpu_cache_upload(TensorInfo *ti) {
    u8 *d = NULL;
    CHECK(cudaMalloc(&d, ti->bytes));
    CHECK(cudaMemcpy(d, ti->data, ti->bytes, cudaMemcpyHostToDevice));
    return d;
}

static u8 *gpu_cache_get(TensorInfo *ti) {
    for (int i = 0; i < g_cache_n; i++) {
        if (g_cache_base[i] != (const u8 *)ti->data || g_cache_ti[i] != ti) continue;
        if (g_cache_elems[i] == ti->n_element && g_cache_bytes[i] == ti->bytes &&
            g_cache_type[i] == ti->type && g_cache_off[i] == ti->offset)
            return g_cache_dev[i]; /* same identity, same content */
        /* Same pointers, different content: replace the dead copy. */
        cudaFree(g_cache_dev[i]);
        u8 *d = gpu_cache_upload(ti);
        if (!d) return NULL;
        g_cache_dev[i]   = d;
        g_cache_elems[i] = ti->n_element;
        g_cache_bytes[i] = ti->bytes;
        g_cache_type[i]  = ti->type;
        g_cache_off[i]   = ti->offset;
        return d;
    }
    if (g_cache_n >= GPU_CACHE_MAX || ti->bytes == 0) return NULL;
    u8 *d = gpu_cache_upload(ti);
    if (!d) return NULL;
    g_cache_base[g_cache_n] = (const u8 *)ti->data;
    g_cache_ti[g_cache_n]   = ti;
    g_cache_dev[g_cache_n]  = d;
    g_cache_elems[g_cache_n] = ti->n_element;
    g_cache_bytes[g_cache_n] = ti->bytes;
    g_cache_type[g_cache_n]  = ti->type;
    g_cache_off[g_cache_n]   = ti->offset;
    g_cache_n++;
    return d;
}

static int gpu_matmul_prep(TensorInfo *ti, u8 **d_w_out) {
    if (!gpu_available()) return -1;
    gpu_load_tables();
    gpu_dispatch_init();
    if (ti->type >= 31 || !gpu_matmul_dispatch[ti->type]) return -1;
    u8 *d_w = gpu_cache_get(ti);
    if (!d_w) return -1;
    *d_w_out = d_w;
    return 0;
}

/* Persistent scratch for the decode hot path.
 *
 * gpu_matvec runs ~150-200 times per token with only a handful of
 * distinct shapes, and each cudaMalloc/cudaFree is a driver round trip:
 * measured 48 µs + 20 µs on the reference card, i.e. ~11 ms of every
 * token on a 248k-vocab model.  Growing two buffers once removes all of
 * it.  Buffers past GPU_SCRATCH_MAX_FLOATS are still allocated per call
 * so an unusually wide vocab cannot pin unbounded memory. */
#define GPU_SCRATCH_MAX_FLOATS (8u << 20)   /* 32 MB */

static float *g_sx = NULL; static u64 g_sx_n = 0;
static float *g_sy = NULL; static u64 g_sy_n = 0;

static float *gpu_scratch(float **buf, u64 *cap, u64 n) {
    if (n > GPU_SCRATCH_MAX_FLOATS) return NULL;
    if (n > *cap) {
        if (*buf) CHECK(cudaFree(*buf));
        CHECK(cudaMalloc(buf, n * sizeof(float)));
        *cap = n;
    }
    return *buf;
}

int gpu_matvec(TensorInfo *ti, const float *x, float *y,
               u64 rows, u64 cols, bool trans) {
    if (!ti || !ti->data || !x || !y || rows == 0 || cols == 0) return -1;
    if (rows * cols > ti->n_element) return -1; /* dim sanity (CPU checks dims) */

    u8 *d_w;
    if (gpu_matmul_prep(ti, &d_w) != 0) return -1;

    float *d_x = gpu_scratch(&g_sx, &g_sx_n, cols);
    float *d_y = gpu_scratch(&g_sy, &g_sy_n, rows);
    bool own = false;
    if (!d_x) { CHECK(cudaMalloc(&d_x, cols * sizeof(float))); own = true; }
    if (!d_y) { CHECK(cudaMalloc(&d_y, rows * sizeof(float))); own = true; }
    CHECK(cudaMemcpy(d_x, x, cols * sizeof(float), cudaMemcpyHostToDevice));
    gpu_matmul_dispatch[ti->type](d_w, d_x, d_y, rows, cols, 1, trans);
    CHECK(cudaMemcpy(y, d_y, rows * sizeof(float), cudaMemcpyDeviceToHost));
    if (own) { if (d_x != g_sx) cudaFree(d_x); if (d_y != g_sy) cudaFree(d_y); }
    return 0;
}

int gpu_matmat(TensorInfo *ti, const float *X, float *Y,
               u64 batch, u64 rows, u64 cols, bool trans) {
    if (!ti || !ti->data || !X || !Y || batch == 0 || rows == 0 || cols == 0) return -1;
    if (rows * cols > ti->n_element) return -1;

    u8 *d_w;
    if (gpu_matmul_prep(ti, &d_w) != 0) return -1;

    float *d_x = NULL, *d_y = NULL;
    CHECK(cudaMalloc(&d_x, batch * cols * sizeof(float)));
    CHECK(cudaMalloc(&d_y, batch * rows * sizeof(float)));
    CHECK(cudaMemcpy(d_x, X, batch * cols * sizeof(float), cudaMemcpyHostToDevice));
    gpu_matmul_dispatch[ti->type](d_w, d_x, d_y, rows, cols, batch, trans);
    CHECK(cudaMemcpy(Y, d_y, batch * rows * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_x);
    cudaFree(d_y);
    return 0;
}

/* ================================================================
 * Device-resident primitives (graph-op backend, see gpu.h)
 * ================================================================ */

u8 *gpu_weight_dev(TensorInfo *ti) {
    if (!ti || !ti->data || !gpu_available()) return NULL;
    return gpu_cache_get(ti);   /* NULL when the cache is full */
}

int gpu_dequant_dev(u32 type, const u8 *d_w, u64 i0, u64 nb, float *d_out) {
    if (!d_w || !d_out || nb == 0) return -1;
    if (!gpu_available()) return -1;
    gpu_load_tables();
    u64 be, bb;
    if (gpu_geom(type, &be, &bb) != 0) return -1;
    u64 start_i   = (i0 / be) * be;
    const u8 *src = d_w + (start_i / be) * bb;
    dequant_range_kernel<<<gpu_block_count(nb), GPU_THREADS>>>(type, src, start_i, i0, nb, d_out);
    CHECK(cudaGetLastError());
    return 0;
}

int gpu_dequant_gather_dev(u32 type, const u8 *d_w, const u32 *d_ids, u32 n_ids,
                           u32 count, u64 base_mul, u64 stride, float *d_out) {
    if (!d_w || !d_ids || !d_out || n_ids == 0 || count == 0) return -1;
    if (!gpu_available()) return -1;
    gpu_load_tables();
    dequant_gather_kernel<<<gpu_block_count((u64)n_ids * count), GPU_THREADS>>>(
        type, d_w, d_ids, n_ids, count, base_mul, stride, d_out);
    CHECK(cudaGetLastError());
    return 0;
}

int gpu_matmul_dev(TensorInfo *ti, const float *d_x, float *d_y,
                   u64 batch, u64 rows, u64 cols, bool trans) {
    if (!ti || !ti->data || !d_x || !d_y || batch == 0 || rows == 0 || cols == 0) return -1;
    if (rows * cols > ti->n_element) return -1;

    u8 *d_w;
    if (gpu_matmul_prep(ti, &d_w) != 0) return -1;
    gpu_matmul_dispatch[ti->type](d_w, d_x, d_y, rows, cols, batch, trans);
    return 0;
}

void gpu_shutdown(void) {
    for (int i = 0; i < g_cache_n; i++)
        if (g_cache_dev[i]) cudaFree(g_cache_dev[i]);
    g_cache_n = 0;
    if (g_sx) { cudaFree(g_sx); g_sx = NULL; g_sx_n = 0; }
    if (g_sy) { cudaFree(g_sy); g_sy = NULL; g_sy_n = 0; }
}
