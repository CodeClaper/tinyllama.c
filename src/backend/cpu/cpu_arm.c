/*
 * cpu_arm.c — NEON SIMD batch dequantization
 *
 * One block-level dequant function per GGUF quant type.
 * The dispatcher gguf_dequant_batch() processes full blocks with
 * these kernels and handles edge elements via the scalar gguf_dequant().
 */
#include <arm_neon.h>
#include <string.h>
#include <math.h>

#include "../../def.h"
#include "../../quants.h"
#include "cpu.h"

/* ================================================================
 * Block sizes
 * ================================================================ */

#define BLOCK_SIMPLE  32
#define BLOCK_KQUANT 256

/* ================================================================
 * NEON helpers
 * ================================================================ */

/* Widen 16 × uint8 to 16 × float32 (as 4 × float32x4).
 *   out = (float)(i32)(vu8[n] - bias) * scale + add
 */
static inline void neon_u8_to_f32_x16(uint8x16_t vu8,
                                       float32x4_t scale, float32x4_t add, i32 bias,
                                       float32x4_t *r0, float32x4_t *r1,
                                       float32x4_t *r2, float32x4_t *r3) {
    int16x8_t v16_0 = (int16x8_t)vmovl_u8(vget_low_u8(vu8));
    int16x8_t v16_1 = (int16x8_t)vmovl_u8(vget_high_u8(vu8));

    int32x4_t v32_0 = vmovl_s16(vget_low_s16(v16_0));
    int32x4_t v32_1 = vmovl_s16(vget_high_s16(v16_0));
    int32x4_t v32_2 = vmovl_s16(vget_low_s16(v16_1));
    int32x4_t v32_3 = vmovl_s16(vget_high_s16(v16_1));

    if (bias) {
        int32x4_t vb = vdupq_n_s32(bias);
        v32_0 = vsubq_s32(v32_0, vb);
        v32_1 = vsubq_s32(v32_1, vb);
        v32_2 = vsubq_s32(v32_2, vb);
        v32_3 = vsubq_s32(v32_3, vb);
    }

    *r0 = vmlaq_f32(add, vcvtq_f32_s32(v32_0), scale);
    *r1 = vmlaq_f32(add, vcvtq_f32_s32(v32_1), scale);
    *r2 = vmlaq_f32(add, vcvtq_f32_s32(v32_2), scale);
    *r3 = vmlaq_f32(add, vcvtq_f32_s32(v32_3), scale);
}

/* Widen 16 × int8 to 16 × float32 (as 4 × float32x4).
 *   out = (float)(i32)(vs8[n]) * scale + add
 * vadd is broadcast addend, may be vdupq_n_f32(0.0f).
 */
static inline void neon_s8_to_f32_x16(int8x16_t vs8,
                                       float32x4_t scale, float32x4_t add,
                                       float32x4_t *r0, float32x4_t *r1,
                                       float32x4_t *r2, float32x4_t *r3) {
    int16x8_t v16_0 = vmovl_s8(vget_low_s8(vs8));
    int16x8_t v16_1 = vmovl_s8(vget_high_s8(vs8));

    int32x4_t v32_0 = vmovl_s16(vget_low_s16(v16_0));
    int32x4_t v32_1 = vmovl_s16(vget_high_s16(v16_0));
    int32x4_t v32_2 = vmovl_s16(vget_low_s16(v16_1));
    int32x4_t v32_3 = vmovl_s16(vget_high_s16(v16_1));

    *r0 = vmlaq_f32(add, vcvtq_f32_s32(v32_0), scale);
    *r1 = vmlaq_f32(add, vcvtq_f32_s32(v32_1), scale);
    *r2 = vmlaq_f32(add, vcvtq_f32_s32(v32_2), scale);
    *r3 = vmlaq_f32(add, vcvtq_f32_s32(v32_3), scale);
}

/* Store 4 × float32x4 to out[0..15] */
static inline void neon_store_f32_x16(float *out,
                                       float32x4_t r0, float32x4_t r1,
                                       float32x4_t r2, float32x4_t r3) {
    vst1q_f32(out +  0, r0);
    vst1q_f32(out +  4, r1);
    vst1q_f32(out +  8, r2);
    vst1q_f32(out + 12, r3);
}

/* ================================================================
 * Non-block types — direct NEON copy / convert
 * ================================================================ */

/* F32: straight memcpy is fine; use NEON copy for consistency. */
static void batch_f32(const u8 *data, float *out, u64 n) {
    u64 i = 0;
    for (; i + 16 <= n; i += 16) {
        vst1q_f32(out + i +  0, vld1q_f32((const float *)data + i +  0));
        vst1q_f32(out + i +  4, vld1q_f32((const float *)data + i +  4));
        vst1q_f32(out + i +  8, vld1q_f32((const float *)data + i +  8));
        vst1q_f32(out + i + 12, vld1q_f32((const float *)data + i + 12));
    }
    for (; i < n; i++) out[i] = ((const float *)data)[i];
}

/* F16 → F32 (scalar fallback — portable, fast enough for bulk copy). */
static void batch_f16(const u8 *data, float *out, u64 n) {
    const u16 *h = (const u16 *)data;
    for (u64 i = 0; i < n; i++) out[i] = f16_to_f32(h[i]);
}

/* BF16 → F32: left-shift 16 bits. */
static void batch_bf16(const u8 *data, float *out, u64 n) {
    const u16 *b = (const u16 *)data;
    u64 i = 0;
    for (; i + 8 <= n; i += 8) {
        uint16x8_t vh = vld1q_u16(b + i);
        uint32x4_t vl = vshll_n_u16(vget_low_u16(vh), 16);
        uint32x4_t vh2 = vshll_n_u16(vget_high_u16(vh), 16);
        vst1q_f32(out + i + 0, vreinterpretq_f32_u32(vl));
        vst1q_f32(out + i + 4, vreinterpretq_f32_u32(vh2));
    }
    for (; i < n; i++) {
        u32 f32 = (u32)b[i] << 16;
        memcpy(out + i, &f32, sizeof(float));
    }
}

/* F64 → F32 narrow. */
static void batch_f64(const u8 *data, float *out, u64 n) {
    const double *d = (const double *)data;
    for (u64 i = 0; i < n; i++) out[i] = (float)d[i];
}

/* I8 → F32. */
static void batch_i8(const u8 *data, float *out, u64 n) {
    const i8 *s = (const i8 *)data;
    u64 i = 0;
    for (; i + 16 <= n; i += 16) {
        int8x16_t v = vld1q_s8(s + i);
        float32x4_t r0, r1, r2, r3;
        neon_s8_to_f32_x16(v, vdupq_n_f32(1.0f), vdupq_n_f32(0.0f), &r0, &r1, &r2, &r3);
        neon_store_f32_x16(out + i, r0, r1, r2, r3);
    }
    for (; i < n; i++) out[i] = (float)s[i];
}

/* I16 → F32. */
static void batch_i16(const u8 *data, float *out, u64 n) {
    const i16 *s = (const i16 *)data;
    u64 i = 0;
    for (; i + 8 <= n; i += 8) {
        int16x8_t v = vld1q_s16(s + i);
        int32x4_t v0 = vmovl_s16(vget_low_s16(v));
        int32x4_t v1 = vmovl_s16(vget_high_s16(v));
        vst1q_f32(out + i + 0, vcvtq_f32_s32(v0));
        vst1q_f32(out + i + 4, vcvtq_f32_s32(v1));
    }
    for (; i < n; i++) out[i] = (float)s[i];
}

/* I32 → F32. */
static void batch_i32(const u8 *data, float *out, u64 n) {
    const i32 *s = (const i32 *)data;
    u64 i = 0;
    for (; i + 4 <= n; i += 4) {
        vst1q_f32(out + i, vcvtq_f32_s32(vld1q_s32(s + i)));
    }
    for (; i < n; i++) out[i] = (float)s[i];
}

/* I64 → F32. */
static void batch_i64(const u8 *data, float *out, u64 n) {
    const i64 *s = (const i64 *)data;
    /* No great NEON path for i64→f32 — just scalar. */
    for (u64 i = 0; i < n; i++) out[i] = (float)s[i];
}

/* ================================================================
 * Simple block types  (block_size = 32)
 * ================================================================ */

/*
 * Q8_0 block (34 bytes): [d:f16][qs:32×i8]
 *   out[j] = (float)(i8)qs[j] * d
 */
static void neon_block_q8_0(const u8 *data, float *out) {
    float d = f16_to_f32(*(const u16 *)data);
    float32x4_t vd = vdupq_n_f32(d);
    const i8 *qs = (const i8 *)(data + 2);

    int8x16_t v0 = vld1q_s8(qs +  0);
    int8x16_t v1 = vld1q_s8(qs + 16);

    float32x4_t r0, r1, r2, r3, r4, r5, r6, r7;
    neon_s8_to_f32_x16(v0, vd, vdupq_n_f32(0.0f), &r0, &r1, &r2, &r3);
    neon_s8_to_f32_x16(v1, vd, vdupq_n_f32(0.0f), &r4, &r5, &r6, &r7);

    neon_store_f32_x16(out +  0, r0, r1, r2, r3);
    neon_store_f32_x16(out + 16, r4, r5, r6, r7);
}

/*
 * Q8_1 block (40 bytes): [d:f32][pad:4][qs:32×i8]
 *   out[j] = (float)(i8)qs[j] * d
 */
static void neon_block_q8_1(const u8 *data, float *out) {
    float d;
    memcpy(&d, data, 4);
    float32x4_t vd = vdupq_n_f32(d);
    const i8 *qs = (const i8 *)(data + 8);

    int8x16_t v0 = vld1q_s8(qs +  0);
    int8x16_t v1 = vld1q_s8(qs + 16);

    float32x4_t r0, r1, r2, r3, r4, r5, r6, r7;
    neon_s8_to_f32_x16(v0, vd, vdupq_n_f32(0.0f), &r0, &r1, &r2, &r3);
    neon_s8_to_f32_x16(v1, vd, vdupq_n_f32(0.0f), &r4, &r5, &r6, &r7);

    neon_store_f32_x16(out +  0, r0, r1, r2, r3);
    neon_store_f32_x16(out + 16, r4, r5, r6, r7);
}

/*
 * Q4_0 block (18 bytes): [d:f16][qs:16 bytes → 32 nibbles]
 *   nib = qs[o>>1] >> ((o&1)<<2) & 0xF
 *   out[o] = (float)((i32)nib - 8) * d
 */
static void neon_block_q4_0(const u8 *data, float *out) {
    float d = f16_to_f32(*(const u16 *)data);
    float32x4_t vd = vdupq_n_f32(d);
    const u8 *qs = data + 2;

    /* Load 16 bytes; extract low & high nibbles; interleave with vzip
     * so each half holds 16 consecutive elements in natural order. */
    uint8x16_t vq   = vld1q_u8(qs);
    uint8x16_t vlo  = vandq_u8(vq, vdupq_n_u8(0x0F));
    uint8x16_t vhi  = vshrq_n_u8(vq, 4);
    uint8x16x2_t vz = vzipq_u8(vlo, vhi);

    float32x4_t r0, r1, r2, r3, r4, r5, r6, r7;

    neon_u8_to_f32_x16(vz.val[0], vd, vdupq_n_f32(0.0f), 8,
                        &r0, &r1, &r2, &r3);
    neon_u8_to_f32_x16(vz.val[1], vd, vdupq_n_f32(0.0f), 8,
                        &r4, &r5, &r6, &r7);

    neon_store_f32_x16(out +  0, r0, r1, r2, r3);
    neon_store_f32_x16(out + 16, r4, r5, r6, r7);
}

/*
 * Q4_1 block (20 bytes): [d:f16][m:f16][qs:16 bytes → 32 nibbles]
 *   out[o] = (float)nib * d + m
 */
static void neon_block_q4_1(const u8 *data, float *out) {
    float d = f16_to_f32(*(const u16 *)data);
    float m = f16_to_f32(*(const u16 *)(data + 2));
    float32x4_t vd = vdupq_n_f32(d);
    float32x4_t vm = vdupq_n_f32(m);
    const u8 *qs = data + 4;

    uint8x16_t vq   = vld1q_u8(qs);
    uint8x16_t vlo  = vandq_u8(vq, vdupq_n_u8(0x0F));
    uint8x16_t vhi  = vshrq_n_u8(vq, 4);
    uint8x16x2_t vz = vzipq_u8(vlo, vhi);

    float32x4_t r0, r1, r2, r3, r4, r5, r6, r7;

    neon_u8_to_f32_x16(vz.val[0], vd, vm, 0, &r0, &r1, &r2, &r3);
    neon_u8_to_f32_x16(vz.val[1], vd, vm, 0, &r4, &r5, &r6, &r7);

    neon_store_f32_x16(out +  0, r0, r1, r2, r3);
    neon_store_f32_x16(out + 16, r4, r5, r6, r7);
}

/*
 * Q5_0 block (22 bytes): [d:f16][qh:u32][ql:16 bytes → 32 nibbles]
 *   lo  = ql[o>>1] >> ((o&1)<<2) & 0xF
 *   hi  = (qh >> o) & 1
 *   out[o] = (float)((i32)((hi << 4) | lo) - 16) * d
 *
 * We unpack hi bits to a byte array first (only 32 bytes, once per block),
 * then combine with NEON.
 */
static void neon_block_q5_0(const u8 *data, float *out) {
    float d = f16_to_f32(*(const u16 *)data);
    float32x4_t vd = vdupq_n_f32(d);
    u32 qh;
    memcpy(&qh, data + 2, 4);
    const u8 *ql = data + 6;

    /* Unpack 32 hi bits into 16 pairs of bytes (hi in bit 4 position). */
    u8 hi_bytes[16];
    for (int j = 0; j < 16; j++) {
        u8 b = 0;
        if ((qh >> (j * 2))     & 1) b |= 0x10;   /* element 2*j     */
        if ((qh >> (j * 2 + 1)) & 1) b |= 0x10;   /* element 2*j+1   — wait, these go to different lanes */
        hi_bytes[j] = b;
    }

    /* Actually each element needs its own hi bit at position 4.
     * Let's unpack to a uint8x16 of hi bits (each 0 or 1→shifted to 0x10). */
    /* Better: unpack into 32-byte scratch, then load with NEON. */
    u8 hi[32];
    for (int o = 0; o < 32; o++)
        hi[o] = ((qh >> o) & 1) << 4;

    /* Load lo nibbles, interleave for natural ordering. */
    uint8x16_t vql  = vld1q_u8(ql);
    uint8x16_t vlo  = vandq_u8(vql, vdupq_n_u8(0x0F));
    uint8x16_t vhi_lo = vshrq_n_u8(vql, 4);
    uint8x16x2_t vz = vzipq_u8(vlo, vhi_lo);

    /* Load corresponding hi bits. */
    uint8x16_t vhi0 = vld1q_u8(hi +  0);
    uint8x16_t vhi1 = vld1q_u8(hi + 16);

    /* Combine: val = lo | hi → subtract 16 → multiply by d. */
    for (int h = 0; h < 2; h++) {
        uint8x16_t vnib = vorrq_u8(
            (h == 0) ? vz.val[0] : vz.val[1],
            (h == 0) ? vhi0 : vhi1);

        float32x4_t r0, r1, r2, r3;
        neon_u8_to_f32_x16(vnib, vd, vdupq_n_f32(0.0f), 16,
                            &r0, &r1, &r2, &r3);
        neon_store_f32_x16(out + h * 16, r0, r1, r2, r3);
    }
}

/*
 * Q5_1 block (24 bytes): [d:f16][m:f16][qh:u32][ql:16 bytes → 32 nibbles]
 *   out[o] = (float)((hi << 4) | lo) * d + m
 */
static void neon_block_q5_1(const u8 *data, float *out) {
    float d = f16_to_f32(*(const u16 *)data);
    float m = f16_to_f32(*(const u16 *)(data + 2));
    float32x4_t vd = vdupq_n_f32(d);
    float32x4_t vm = vdupq_n_f32(m);
    u32 qh;
    memcpy(&qh, data + 4, 4);
    const u8 *ql = data + 8;

    u8 hi[32];
    for (int o = 0; o < 32; o++)
        hi[o] = ((qh >> o) & 1) << 4;

    uint8x16_t vql     = vld1q_u8(ql);
    uint8x16_t vlo     = vandq_u8(vql, vdupq_n_u8(0x0F));
    uint8x16_t vhi_lo  = vshrq_n_u8(vql, 4);
    uint8x16x2_t vz    = vzipq_u8(vlo, vhi_lo);

    uint8x16_t vhi0 = vld1q_u8(hi +  0);
    uint8x16_t vhi1 = vld1q_u8(hi + 16);

    for (int h = 0; h < 2; h++) {
        uint8x16_t vnib = vorrq_u8(
            (h == 0) ? vz.val[0] : vz.val[1],
            (h == 0) ? vhi0 : vhi1);

        float32x4_t r0, r1, r2, r3;
        neon_u8_to_f32_x16(vnib, vd, vm, 0, &r0, &r1, &r2, &r3);
        neon_store_f32_x16(out + h * 16, r0, r1, r2, r3);
    }
}

/* ================================================================
 * K-quant types  (block_size = 256)
 * ================================================================ */

/*
 * Q8_K block (292 bytes): [d:f32][bs:16×f16][qs:256×i8]
 *   out[o] = (float)qs[o] * bs[o>>4] * d
 *
 * Process 16 sub-blocks of 16 elements each.
 */
static void neon_block_q8_k(const u8 *data, float *out) {
    float d;
    memcpy(&d, data, 4);
    float32x4_t vd = vdupq_n_f32(d);
    const u16 *bs = (const u16 *)(data + 4);
    const i8   *qs = (const i8 *)(data + 36);

    for (int sb = 0; sb < 16; sb++) {
        float32x4_t vscale = vmulq_n_f32(vd, f16_to_f32(bs[sb]));
        int off = sb * 16;

        int8x16_t vq = vld1q_s8(qs + off);
        float32x4_t r0, r1, r2, r3;
        neon_s8_to_f32_x16(vq, vscale, vdupq_n_f32(0.0f), &r0, &r1, &r2, &r3);
        neon_store_f32_x16(out + off, r0, r1, r2, r3);
    }
}

/*
 * Q6_K block (210 bytes):
 *   [ql:128][qh:64][scales:16×i8][d:f16]
 *   ql holds low 4 bits, qh holds high 2 bits per element.
 *   Sub-block scale at scales[o>>4]; d at byte 208-209.
 *
 *   For each 128-element half:
 *     - 4 groups of 32 elements, taking low or high nibble from ql
 *       and 2 hi bits from qh.
 *   out[o] = d * (float)(i8)scales[o>>4] * (float)(q - 32)
 */
static void neon_block_q6_k(const u8 *data, float *out) {
    float d = f16_to_f32(*(const u16 *)(data + 208));
    const u8 *scales  = data + 192;   /* 16 × i8 */
    const u8 *ql      = data;          /* 128 bytes */
    const u8 *qh      = data + 128;    /* 64 bytes  */

    for (int sb = 0; sb < 16; sb++) {
        i32 sc       = (i8)scales[sb];
        u32 start    = (u32)sb * 16;

        /* Build 16 combined values (lo + hi<<4) in a uint8 buffer,
         * then widen with NEON. */
        u8 vals[16];
        for (int j = 0; j < 16; j++) {
            u32 o     = start + (u32)j;
            u32 half  = o >> 7;              /* 0 or 1 */
            u32 hl    = o & 127;             /* position within half */
            u32 which = hl >> 5;             /* 0..3 group of 32 */
            u32 l     = hl & 31;             /* 0..31 within group */
            u32 ql_off = (half << 6) + ((which & 1) << 5) + l;
            u32 lo = (which >= 2) ? ((ql[ql_off] >> 4) & 0xF)
                                  :  (ql[ql_off] & 0xF);
            u32 qh_off = (half << 5) + l;
            u32 hi = (qh[qh_off] >> (which * 2)) & 0x3;
            vals[j] = (u8)(lo | (hi << 4));
        }

        float32x4_t vscale = vdupq_n_f32(d * (float)sc);
        int32x4_t   vbias  = vdupq_n_s32(32);

        /* Load as uint8, widen, subtract 32, convert, multiply. */
        uint8x16_t vu8 = vld1q_u8(vals);
        int16x8_t v16_0 = (int16x8_t)vmovl_u8(vget_low_u8(vu8));
        int16x8_t v16_1 = (int16x8_t)vmovl_u8(vget_high_u8(vu8));

        int32x4_t v32_0 = vsubq_s32(vmovl_s16(vget_low_s16(v16_0)), vbias);
        int32x4_t v32_1 = vsubq_s32(vmovl_s16(vget_high_s16(v16_0)), vbias);
        int32x4_t v32_2 = vsubq_s32(vmovl_s16(vget_low_s16(v16_1)), vbias);
        int32x4_t v32_3 = vsubq_s32(vmovl_s16(vget_high_s16(v16_1)), vbias);

        float32x4_t r0 = vmulq_f32(vcvtq_f32_s32(v32_0), vscale);
        float32x4_t r1 = vmulq_f32(vcvtq_f32_s32(v32_1), vscale);
        float32x4_t r2 = vmulq_f32(vcvtq_f32_s32(v32_2), vscale);
        float32x4_t r3 = vmulq_f32(vcvtq_f32_s32(v32_3), vscale);

        neon_store_f32_x16(out + start, r0, r1, r2, r3);
    }
}

/*
 * Q4_K block (144 bytes):
 *   [d:f16][dmin:f16][scales+mins:12 bytes (8×6-bit)][qs:128 bytes → 256 nibbles]
 *
 *   8 sub-blocks of 32 elements. Each sub-block has its own scale/min.
 *   Within a super-block, quant data is organised in 4 × 32-byte groups.
 *   Sub-blocks 2g (nb=0) use low nibbles, sub-blocks 2g+1 (nb=1) use high nibbles.
 *
 *   out[o] = d * (float)sc * (float)nib - dmin * (float)mn
 */
static void neon_block_q4_k(const u8 *data, float *out) {
    float d    = f16_to_f32(*(const u16 *)data);
    float dmin = f16_to_f32(*(const u16 *)(data + 2));
    const u8 *sm  = data + 4;     /* 12 bytes scales+mins  */
    const u8 *qs  = data + 16;    /* 128 bytes quant data   */

    for (int sb = 0; sb < 8; sb++) {
        u8 sc, mn;
        q4k_scale_min(sm, sb, &sc, &mn);

        float32x4_t vscale = vdupq_n_f32(d * (float)sc);
        float32x4_t vmin   = vdupq_n_f32(dmin * (float)mn);

        u32 g  = (u32)sb >> 1;          /* 0..3: which 32-byte qs group */
        u32 nb = (u32)sb & 1;           /* 0 → low nibble, 1 → high nibble */
        const u8 *src = qs + g * 32;    /* 32 bytes for this group-pair */

        u8 buf[32];  /* 32 nibbles for this sub-block */
        for (int j = 0; j < 32; j++)
            buf[j] = (nb ? (src[j] >> 4) : (src[j] & 0xF));

        /* Process 32 values in two 16-element halves via NEON. */
        for (int half = 0; half < 2; half++) {
            uint8x16_t vu8 = vld1q_u8(buf + half * 16);
            float32x4_t r0, r1, r2, r3;
            neon_u8_to_f32_x16(vu8, vscale, vnegq_f32(vmin), 0,
                                &r0, &r1, &r2, &r3);
            neon_store_f32_x16(out + (u32)sb * 32 + half * 16, r0, r1, r2, r3);
        }
    }
}

/*
 * Q5_K block (176 bytes):
 *   [d:f16][dmin:f16][scales+mins:12][qh:32 bytes hi bits][qs:128 bytes lo nibbles]
 *   out[o] = d * (float)sc * (float)q - dmin * (float)mn
 *
 * Similar structure to Q4_K but with 1 extra hi bit per element (packed in qh).
 */
static void neon_block_q5_k(const u8 *data, float *out) {
    float d    = f16_to_f32(*(const u16 *)data);
    float dmin = f16_to_f32(*(const u16 *)(data + 2));
    const u8 *sm  = data + 4;     /* 12 bytes scales+mins */
    const u8 *qh  = data + 16;    /* 32 bytes hi bits     */
    const u8 *qs  = data + 48;    /* 128 bytes lo nibbles */

    for (int sb = 0; sb < 8; sb++) {
        u8 sc, mn;
        q4k_scale_min(sm, sb, &sc, &mn);

        float32x4_t vscale = vdupq_n_f32(d * (float)sc);
        float32x4_t vmin   = vdupq_n_f32(dmin * (float)mn);

        u32 g  = (u32)sb >> 1;
        u32 nb = (u32)sb & 1;
        const u8 *src_lo = qs + g * 32;

        u8 vals[32];
        for (int j = 0; j < 32; j++) {
            u32 o   = (u32)sb * 32 + (u32)j;
            u32 hi  = (qh[o & 31] >> (o >> 5)) & 1;
            u32 lo  = nb ? (src_lo[j] >> 4) : (src_lo[j] & 0xF);
            vals[j] = (u8)(lo | (hi << 4));
        }

        for (int half = 0; half < 2; half++) {
            uint8x16_t vu8 = vld1q_u8(vals + half * 16);
            float32x4_t r0, r1, r2, r3;
            neon_u8_to_f32_x16(vu8, vscale, vnegq_f32(vmin), 0,
                                &r0, &r1, &r2, &r3);
            neon_store_f32_x16(out + (u32)sb * 32 + half * 16, r0, r1, r2, r3);
        }
    }
}

/*
 * Q3_K block (110 bytes):
 *   [...][d:f16 at byte 108]
 *   out[o] = d * (float)sc * (float)(q - 4)
 *
 * 3-bit quants: 2 lo bits (qs) + 1 hi bit (qh).
 * 16 sub-blocks of 16 elements. Scale: 4 lower + 2 upper bits packed.
 */
static void neon_block_q3_k(const u8 *data, float *out) {
    float d = f16_to_f32(*(const u16 *)(data + 108));
    const u8 *qh      = data;         /* 32 bytes: hi bits (1 per element) */
    const u8 *qs      = data + 32;    /* 64 bytes: lo 2 bits (2 per byte, 4 groups × 32 bytes) */
    const u8 *scales  = data + 96;    /* 8+4 bytes: packed 4+2 bit scales */

    for (int sb = 0; sb < 16; sb++) {
        u32 s = (u32)sb;
        u32 sc_low  = (scales[(s & 7)] >> ((s >= 8) ? 4 : 0)) & 0xF;
        u32 sc_high = (scales[8 + (s & 3)] >> (((s >> 2) & 3) * 2)) & 0x3;
        i32 sc = (i32)(sc_low | (sc_high << 4)) - 32;

        float32x4_t vscale = vdupq_n_f32(d * (float)sc);
        float32x4_t vzero  = vdupq_n_f32(0.0f);

        u32 start = (u32)sb * 16;
        u8 vals[16];
        for (int j = 0; j < 16; j++) {
            u32 o   = start + (u32)j;
            u32 g   = o >> 7;               /* 0 or 1 */
            u32 pr  = (o >> 5) & 3;         /* which pair within group */
            u32 bc  = o & 31;
            u32 lo  = (qs[g * 32 + bc] >> (pr * 2)) & 0x3;
            u32 hi  = (qh[o & 31] >> (o >> 5)) & 1;
            vals[j]  = (u8)(lo | (hi << 2));
        }

        uint8x16_t vu8  = vld1q_u8(vals);
        int16x8_t v16_0 = (int16x8_t)vmovl_u8(vget_low_u8(vu8));
        int16x8_t v16_1 = (int16x8_t)vmovl_u8(vget_high_u8(vu8));

        int32x4_t vb4 = vdupq_n_s32(4);
        int32x4_t v32_0 = vsubq_s32(vmovl_s16(vget_low_s16(v16_0)), vb4);
        int32x4_t v32_1 = vsubq_s32(vmovl_s16(vget_high_s16(v16_0)), vb4);
        int32x4_t v32_2 = vsubq_s32(vmovl_s16(vget_low_s16(v16_1)), vb4);
        int32x4_t v32_3 = vsubq_s32(vmovl_s16(vget_high_s16(v16_1)), vb4);

        float32x4_t r0 = vmlaq_f32(vzero, vcvtq_f32_s32(v32_0), vscale);
        float32x4_t r1 = vmlaq_f32(vzero, vcvtq_f32_s32(v32_1), vscale);
        float32x4_t r2 = vmlaq_f32(vzero, vcvtq_f32_s32(v32_2), vscale);
        float32x4_t r3 = vmlaq_f32(vzero, vcvtq_f32_s32(v32_3), vscale);

        neon_store_f32_x16(out + start, r0, r1, r2, r3);
    }
}

/*
 * Q2_K block (84 bytes):
 *   [...][d:f16 at 80][mn:f16 at 82]
 *   out[o] = d * (float)sc * (float)q - mn
 *
 * 2-bit quants: qs gives 2 bits, sign bit from another byte.
 * Scales: 4-bit each (2 per byte, 16 blocks of 16).
 */
static void neon_block_q2_k(const u8 *data, float *out) {
    float d  = f16_to_f32(*(const u16 *)(data + 80));
    float mn = f16_to_f32(*(const u16 *)(data + 82));
    const u8 *scales = data + 64;   /* 16 bytes: packed 4-bit scales */

    for (int sb = 0; sb < 16; sb++) {
        u32 sc = scales[sb >> 1];
        if (sb & 1) sc >>= 4; else sc &= 0xF;

        float32x4_t vscale = vdupq_n_f32(d * (float)(i32)sc);
        float32x4_t vmn    = vdupq_n_f32(mn);

        u32 start = (u32)sb * 16;
        u8 vals[16];
        for (int j = 0; j < 16; j++) {
            u32 o   = start + (u32)j;
            u32 q2  = (data[(o >> 2)] >> ((o & 3) << 1)) & 0x3;
            u32 sign = (data[(o >> 3)] >> (o & 7)) & 1;
            u32 q   = q2 - sign * 4;
            vals[j]  = (u8)(i32)q;  /* small signed value in u8 */
        }

        /* vals[j] = q - sign*4, a signed value in [-4, 3].
         * Widen as uint8 (will be 0..255), but the signed range is small enough
         * that the u8→i16→i32 chain works if we adjust.  Let's cast through i8. */
        int8x16_t vs8 = vld1q_s8((const i8 *)vals);
        float32x4_t r0, r1, r2, r3;
        neon_s8_to_f32_x16(vs8, vscale, vnegq_f32(vmn), &r0, &r1, &r2, &r3);
        neon_store_f32_x16(out + start, r0, r1, r2, r3);
    }
}

/* ================================================================
 * IQ block types — scalar block dequant
 *
 * These types rely on lookup tables.  We dequant one block at a
 * time with a tight scalar loop; the table lookups dominate, so
 * NEON wouldn't help much.
 * ================================================================ */

static void scalar_block_iq2_xxs(const u8 *data, float *out) {
    float d = f16_to_f32(*(const u16 *)data);
    for (u32 o = 0; o < 256; o++) {
        u32 q  = (data[2 + (o >> 2)] >> ((o & 3) << 1)) & 0x3;
        u32 sb = (data[2 + (o >> 3)] >> (o & 7)) & 1;
        out[o] = iq2_xxs_grid[q | (sb << 2)] * d;
    }
}

static void scalar_block_iq2_xs(const u8 *data, float *out) {
    float d = f16_to_f32(*(const u16 *)data);
    for (u32 o = 0; o < 256; o++) {
        u32 sh = (data[2 + (o >> 3)] >> (o & 7)) & 1;
        u32 q  = (data[2 + 32 + (o >> 2)] >> ((o & 3) << 1)) & 0x3;
        i8  sc = (i8)data[2 + 32 + 64 + (o >> 4)];
        out[o] = d * (float)sc * iq2_xs_grid[q | (sh << 2)];
    }
}

static void scalar_block_iq3_xxs(const u8 *data, float *out) {
    float d = f16_to_f32(*(const u16 *)data);
    for (u32 o = 0; o < 256; o++) {
        u32 qh  = (data[2 + (o >> 5)] >> (o & 31)) & 1;
        u32 qm  = (data[10 + (o >> 3)] >> (o & 7)) & 1;
        u32 ql  = (data[42 + (o >> 1)] >> ((o & 1) << 2)) & 0xF;
        u32 grp = (data[42 + 128 + (o >> 3)] >> ((o & 7) << 1)) & 0x3;
        i8  sc  = (i8)data[42 + 128 + 64 + (o >> 4)];
        out[o] = d * (float)sc * iq3_xxs_grid[ql | (qm << 4) | (qh << 5) | (grp << 6)];
    }
}

static void scalar_block_iq1_s(const u8 *data, float *out) {
    float d = f16_to_f32(*(const u16 *)data);
    for (u32 o = 0; o < 256; o++) {
        u32 q  = ((data[2 + (o >> 3)] >> (o & 7)) & 1) |
                 (((data[34 + (o >> 3)] >> (o & 7)) & 1) << 1);
        u32 idx = q | (((o >> 4) & 0xF) << 2);
        out[o] = d * iq1_s_grid[idx];
    }
}

static void scalar_block_iq4_nl(const u8 *data, float *out) {
    float d = f16_to_f32(*(const u16 *)data);
    for (u32 o = 0; o < 256; o++) {
        u32 q = (data[2 + (o >> 1)] >> ((o & 1) << 2)) & 0xF;
        out[o] = d * iq4_nl_values[q | ((o & 0xF) << 4)];
    }
}

static void scalar_block_iq1_m(const u8 *data, float *out) {
    float d = f16_to_f32(*(const u16 *)(data + 54));
    for (u32 o = 0; o < 256; o++) {
        u32 sb    = o >> 4;
        u32 sc    = (data[sb >> 1] >> ((sb & 1) << 2)) & 0xF;
        u32 val   = (data[8 + (o >> 3)] >> (o & 7)) & 1;
        u32 sign  = (data[40 + (o >> 3)] >> (o & 7)) & 1;
        out[o] = (sign ? -(float)val : (float)val) * ((float)(i32)sc + 1.0f) * d;
    }
}

static void scalar_block_iq3_s(const u8 *data, float *out) {
    float d = f16_to_f32(*(const u16 *)data);
    for (u32 o = 0; o < 256; o++) {
        u32 sign = (data[2 + (o >> 3)] >> (o & 7)) & 1;
        u32 qh   = (data[34 + (o >> 5)] >> (o & 31)) & 1;
        u32 qm   = (data[42 + (o >> 3)] >> (o & 7)) & 1;
        u32 ql   = (data[74 + (o >> 1)] >> ((o & 1) << 2)) & 0xF;
        i8  sc   = (i8)data[74 + 128 + (o >> 4)];
        out[o] = d * (float)sc * iq3_s_grid[ql | (qm << 4) | (qh << 5) | (sign << 6)];
    }
}

static void scalar_block_iq2_s(const u8 *data, float *out) {
    float d = f16_to_f32(*(const u16 *)data);
    for (u32 o = 0; o < 256; o++) {
        u32 sign = (data[2 + (o >> 3)] >> (o & 7)) & 1;
        u32 qh   = (data[34 + (o >> 5)] >> (o & 31)) & 1;
        u32 ql   = (data[42 + (o >> 1)] >> ((o & 1) << 2)) & 0xF;
        i8  sc   = (i8)data[42 + 128 + (o >> 4) + ((o >> 4) & 1) * 16];
        u32 sc_l = (data[42 + 128 + (o >> 5)] >> ((o & 31) >> 2)) & 0x3;
        out[o] = d * ((float)sc + 0.125f * (float)(i32)sc_l) * iq2_xs_grid[ql | (qh << 4) | (sign << 5)];
    }
}

static void scalar_block_iq4_xs(const u8 *data, float *out) {
    float d = f16_to_f32(*(const u16 *)data);
    for (u32 o = 0; o < 256; o++) {
        u32 qh = (data[2 + (o >> 5)] >> (o & 31)) & 1;
        u32 ql = (data[10 + (o >> 1)] >> ((o & 1) << 2)) & 0xF;
        i8  sc_l = (i8)data[10 + 128 + (o >> 4)];
        i8  sc_h = (i8)data[10 + 128 + 16 + (o >> 4)];
        out[o] = d * ((float)(i32)sc_l + ((float)(i32)sc_h / 127.0f)) * iq4_nl_values[ql | (qh << 4)];
    }
}

/* ================================================================
 * Type-dispatch table
 * ================================================================ */

typedef void (*block_dequant_fn)(const u8 *data, float *out);

static block_dequant_fn block_table[] = {
    [GGUF_TYPE_F32]      = NULL,   /* handled inline in batch dispatcher */
    [GGUF_TYPE_F16]      = NULL,
    [GGUF_TYPE_Q4_0]     = neon_block_q4_0,
    [GGUF_TYPE_Q4_1]     = neon_block_q4_1,
    [GGUF_TYPE_Q5_0]     = neon_block_q5_0,
    [GGUF_TYPE_Q5_1]     = neon_block_q5_1,
    [GGUF_TYPE_Q8_0]     = neon_block_q8_0,
    [GGUF_TYPE_Q8_1]     = neon_block_q8_1,
    [GGUF_TYPE_Q2_K]     = neon_block_q2_k,
    [GGUF_TYPE_Q3_K]     = neon_block_q3_k,
    [GGUF_TYPE_Q4_K]     = neon_block_q4_k,
    [GGUF_TYPE_Q5_K]     = neon_block_q5_k,
    [GGUF_TYPE_Q6_K]     = neon_block_q6_k,
    [GGUF_TYPE_Q8_K]     = neon_block_q8_k,
    [GGUF_TYPE_IQ2_XXS]  = scalar_block_iq2_xxs,
    [GGUF_TYPE_IQ2_XS]   = scalar_block_iq2_xs,
    [GGUF_TYPE_IQ3_XXS]  = scalar_block_iq3_xxs,
    [GGUF_TYPE_IQ1_S]    = scalar_block_iq1_s,
    [GGUF_TYPE_IQ4_NL]   = scalar_block_iq4_nl,
    [GGUF_TYPE_IQ3_S]    = scalar_block_iq3_s,
    [GGUF_TYPE_IQ2_S]    = scalar_block_iq2_s,
    [GGUF_TYPE_IQ4_XS]   = scalar_block_iq4_xs,
    [GGUF_TYPE_I8]       = NULL,   /* non-block */
    [GGUF_TYPE_I16]      = NULL,
    [GGUF_TYPE_I32]      = NULL,
    [GGUF_TYPE_I64]      = NULL,
    [GGUF_TYPE_F64]      = NULL,
    [GGUF_TYPE_IQ1_M]    = scalar_block_iq1_m,
    [GGUF_TYPE_BF16]     = NULL,
};

/* ================================================================
 * Public API
 * ================================================================ */

void gguf_dequant_batch(TensorInfo *ti, u64 i0, u64 nb, float *out) {
    const u8 *data = ti->data;
    u32 type = ti->type;

    /* ---- Non-block / simple conversion types: direct NEON path ---- */
    switch (type) {
    case GGUF_TYPE_F32:  batch_f32(data + i0 * 4, out, nb); return;
    case GGUF_TYPE_F16:  batch_f16(data + i0 * 2, out, nb); return;
    case GGUF_TYPE_BF16: batch_bf16(data + i0 * 2, out, nb); return;
    case GGUF_TYPE_F64:  batch_f64(data + i0 * 8, out, nb); return;
    case GGUF_TYPE_I8:   batch_i8(data + i0, out, nb); return;
    case GGUF_TYPE_I16:  batch_i16(data + i0 * 2, out, nb); return;
    case GGUF_TYPE_I32:  batch_i32(data + i0 * 4, out, nb); return;
    case GGUF_TYPE_I64:  batch_i64(data + i0 * 8, out, nb); return;
    default: break;
    }

    /* ---- Block-quantised types ---- */
    u32 block_elems;
    u32 block_bytes;

    /* Determine block parameters from the GGUF type. */
    if (type <= GGUF_TYPE_Q8_1) {
        block_elems = BLOCK_SIMPLE;
        switch (type) {
            case GGUF_TYPE_Q4_0: block_bytes = 18; break;
            case GGUF_TYPE_Q4_1: block_bytes = 20; break;
            case GGUF_TYPE_Q5_0: block_bytes = 22; break;
            case GGUF_TYPE_Q5_1: block_bytes = 24; break;
            case GGUF_TYPE_Q8_0: block_bytes = 34; break;
            case GGUF_TYPE_Q8_1: block_bytes = 40; break;
            default: goto fallback;
        }
    } else {
        block_elems = BLOCK_KQUANT;
        switch (type) {
            case GGUF_TYPE_Q2_K:     block_bytes =  84; break;
            case GGUF_TYPE_Q3_K:     block_bytes = 110; break;
            case GGUF_TYPE_Q4_K:     block_bytes = 144; break;
            case GGUF_TYPE_Q5_K:     block_bytes = 176; break;
            case GGUF_TYPE_Q6_K:     block_bytes = 210; break;
            case GGUF_TYPE_Q8_K:     block_bytes = 292; break;
            case GGUF_TYPE_IQ2_XXS:  block_bytes =  66; break;
            case GGUF_TYPE_IQ2_XS:   block_bytes =  74; break;
            case GGUF_TYPE_IQ3_XXS:  block_bytes =  98; break;
            case GGUF_TYPE_IQ1_S:    block_bytes = 110; break;
            case GGUF_TYPE_IQ4_NL:   block_bytes =  50; break;
            case GGUF_TYPE_IQ3_S:    block_bytes = 110; break;
            case GGUF_TYPE_IQ2_S:    block_bytes =  82; break;
            case GGUF_TYPE_IQ4_XS:   block_bytes = 136; break;
            case GGUF_TYPE_IQ1_M:    block_bytes =  56; break;
            default: goto fallback;
        }
    }

    block_dequant_fn fn = block_table[type];

    /* Walk blocks intersecting [i0, i0+nb).  Full blocks → NEON;
     * partial edge blocks → scalar per-element fallback. */
    u64 end       = i0 + nb;
    u64 bi0       = i0 / block_elems;
    u64 bi1       = (end + block_elems - 1) / block_elems;

    for (u64 bi = bi0; fn && bi < bi1; bi++) {
        u64 bs = bi * block_elems;               /* block start (element)    */
        u64 be = bs + block_elems;               /* block end (exclusive)    */
        u64 cs = i0 > bs ? i0 : bs;              /* chunk start              */
        u64 ce = end < be ? end : be;            /* chunk end                */

        if (cs == bs && ce == be) {
            /* Fully contained — use NEON / scalar block fn. */
            fn(data + bi * block_bytes, out + (cs - i0));
        } else {
            /* Partial — fall back to scalar. */
            for (u64 j = cs; j < ce; j++)
                out[j - i0] = gguf_dequant(ti, j);
        }
    }

    /* If a block fn was available the loop above handled everything. */
    if (fn) return;

fallback:
    /* No block function available — fall back to per-element scalar. */
    for (u64 j = i0; j < i0 + nb; j++)
        out[j - i0] = gguf_dequant(ti, j);
}

/* ================================================================
 * Fused dequant + dot product  (NEON SIMD)
 *
 * Each function computes  sum(w[j] * x[j])  for one complete
 * block of quantised weights, dequantising and multiplying in
 * NEON registers — no intermediate float buffer.
 * ================================================================ */

/* Horizontal sum of a float32x4 into a scalar. */
static inline float neon_hsum_f32x4(float32x4_t v) {
    float32x2_t s2 = vadd_f32(vget_low_f32(v), vget_high_f32(v));
    s2 = vpadd_f32(s2, s2);
    return vget_lane_f32(s2, 0);
}

/* ---- Non-block types ------------------------------------------- */

static float dot_f32(const u8 *data, const float *x, u64 n) {
    const float *w = (const float *)data;
    float32x4_t s0 = vdupq_n_f32(0), s1 = vdupq_n_f32(0);
    float32x4_t s2 = vdupq_n_f32(0), s3 = vdupq_n_f32(0);
    u64 i = 0;
    for (; i + 16 <= n; i += 16) {
        s0 = vmlaq_f32(s0, vld1q_f32(w + i +  0), vld1q_f32(x + i +  0));
        s1 = vmlaq_f32(s1, vld1q_f32(w + i +  4), vld1q_f32(x + i +  4));
        s2 = vmlaq_f32(s2, vld1q_f32(w + i +  8), vld1q_f32(x + i +  8));
        s3 = vmlaq_f32(s3, vld1q_f32(w + i + 12), vld1q_f32(x + i + 12));
    }
    float sum = neon_hsum_f32x4(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
    for (; i < n; i++) sum += w[i] * x[i];
    return sum;
}

static float dot_f16(const u8 *data, const float *x, u64 n) {
    float sum = 0.0f;
    const u16 *h = (const u16 *)data;
    for (u64 i = 0; i < n; i++)
        sum += f16_to_f32(h[i]) * x[i];
    return sum;
}

/* ---- Simple block types (block_size = 32) ---------------------- */

/* Q8_0: sum = d * Σ(qs[j] * x[j])  */
static float neon_dot_q8_0_block(const u8 *data, const float *x) {
    float d = f16_to_f32(*(const u16 *)data);
    const i8 *qs = (const i8 *)(data + 2);

    float32x4_t s0 = vdupq_n_f32(0), s1 = vdupq_n_f32(0);
    float32x4_t s2 = vdupq_n_f32(0), s3 = vdupq_n_f32(0);

    for (int j = 0; j < 32; j += 16) {
        int8x16_t vq = vld1q_s8(qs + j);
        int16x8_t v16_0 = vmovl_s8(vget_low_s8(vq));
        int16x8_t v16_1 = vmovl_s8(vget_high_s8(vq));
        float32x4_t qf0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(v16_0)));
        float32x4_t qf1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(v16_0)));
        float32x4_t qf2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(v16_1)));
        float32x4_t qf3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(v16_1)));

        s0 = vmlaq_f32(s0, qf0, vld1q_f32(x + j +  0));
        s1 = vmlaq_f32(s1, qf1, vld1q_f32(x + j +  4));
        s2 = vmlaq_f32(s2, qf2, vld1q_f32(x + j +  8));
        s3 = vmlaq_f32(s3, qf3, vld1q_f32(x + j + 12));
    }
    return d * neon_hsum_f32x4(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
}

/* Q4_0: sum = d * Σ((nib - 8) * x) = d * (Σ(nib*x) - 8*Σ(x)) */
static float neon_dot_q4_0_block(const u8 *data, const float *x) {
    float d = f16_to_f32(*(const u16 *)data);
    const u8 *qs = data + 2;

    float32x4_t dot0 = vdupq_n_f32(0), dot1 = vdupq_n_f32(0);
    float32x4_t dot2 = vdupq_n_f32(0), dot3 = vdupq_n_f32(0);
    float32x4_t sum0 = vdupq_n_f32(0), sum1 = vdupq_n_f32(0);

    for (int j = 0; j < 32; j += 16) {
        /* For Q4_0, nibbles within a block alternate low/high per byte.
         * Extract low nibbles (even elements 0,2,4,...) and high nibbles
         * (odd elements 1,3,5,...), then interleave for natural order. */
        uint8x16_t vq  = vld1q_u8(qs + j / 2);
        uint8x16_t vlo = vandq_u8(vq, vdupq_n_u8(0x0F));
        uint8x16_t vhi = vshrq_n_u8(vq, 4);
        uint8x16x2_t vz = vzipq_u8(vlo, vhi);

        /* Process first 8 elements of each half (vz.val[0]) then last 8 (vz.val[1]).
         * Actually vzip gives: val[0] = [l0,h0,l1,h1,...,l7,h7], val[1]=[l8,h8,...,l15,h15].
         * So each val[h] has 16 consecutively ordered elements. */
        for (int h = 0; h < 2; h++) {
            uint8x16_t vn = (h == 0) ? vz.val[0] : vz.val[1];
            int off = j + h * 8;
            (void)off; /* j/2 bytes → j elements; use separate offsets */

            int16x8_t v16_0 = (int16x8_t)vmovl_u8(vget_low_u8(vn));
            int16x8_t v16_1 = (int16x8_t)vmovl_u8(vget_high_u8(vn));
            float32x4_t nf0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(v16_0)));
            float32x4_t nf1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(v16_0)));
            float32x4_t nf2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(v16_1)));
            float32x4_t nf3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(v16_1)));

            int jj = j + h * 16;
            float32x4_t xv0 = vld1q_f32(x + jj +  0);
            float32x4_t xv1 = vld1q_f32(x + jj +  4);
            float32x4_t xv2 = vld1q_f32(x + jj +  8);
            float32x4_t xv3 = vld1q_f32(x + jj + 12);

            dot0 = vmlaq_f32(dot0, nf0, xv0); dot1 = vmlaq_f32(dot1, nf1, xv1);
            dot2 = vmlaq_f32(dot2, nf2, xv2); dot3 = vmlaq_f32(dot3, nf3, xv3);
            sum0 = vaddq_f32(sum0, xv0);  sum1 = vaddq_f32(sum1, xv1);
            /* sum2, sum3 — reuse dot accumulators; just add x to separate regs */
            /* Actually let me just accumulate all x values into sum0/sum1 */
        }
    }

    /* Recompute: we need sum_x for all 32 elements. Let me re-accumulate. */
    float32x4_t sx0 = vdupq_n_f32(0), sx1 = vdupq_n_f32(0);
    float32x4_t sx2 = vdupq_n_f32(0), sx3 = vdupq_n_f32(0);
    for (int j = 0; j < 32; j += 16) {
        sx0 = vaddq_f32(sx0, vld1q_f32(x + j +  0));
        sx1 = vaddq_f32(sx1, vld1q_f32(x + j +  4));
        sx2 = vaddq_f32(sx2, vld1q_f32(x + j +  8));
        sx3 = vaddq_f32(sx3, vld1q_f32(x + j + 12));
    }
    float sum_x = neon_hsum_f32x4(vaddq_f32(vaddq_f32(sx0, sx1), vaddq_f32(sx2, sx3)));

    float dot_all = neon_hsum_f32x4(vaddq_f32(vaddq_f32(dot0, dot1), vaddq_f32(dot2, dot3)));
    return d * (dot_all - 8.0f * sum_x);
}

/* Q4_1: sum = Σ((nib * d + m) * x) = d * Σ(nib*x) + m * Σ(x) */
static float neon_dot_q4_1_block(const u8 *data, const float *x) {
    float d = f16_to_f32(*(const u16 *)data);
    float m = f16_to_f32(*(const u16 *)(data + 2));
    const u8 *qs = data + 4;

    float32x4_t dot0 = vdupq_n_f32(0), dot1 = vdupq_n_f32(0);
    float32x4_t dot2 = vdupq_n_f32(0), dot3 = vdupq_n_f32(0);
    float32x4_t sx0 = vdupq_n_f32(0), sx1 = vdupq_n_f32(0);
    float32x4_t sx2 = vdupq_n_f32(0), sx3 = vdupq_n_f32(0);

    for (int j = 0; j < 32; j += 16) {
        uint8x16_t vq  = vld1q_u8(qs + j / 2);
        uint8x16_t vlo = vandq_u8(vq, vdupq_n_u8(0x0F));
        uint8x16_t vhi = vshrq_n_u8(vq, 4);
        uint8x16x2_t vz = vzipq_u8(vlo, vhi);

        for (int h = 0; h < 2; h++) {
            uint8x16_t vn = (h == 0) ? vz.val[0] : vz.val[1];
            int jj = j + h * 16;

            int16x8_t v16_0 = (int16x8_t)vmovl_u8(vget_low_u8(vn));
            int16x8_t v16_1 = (int16x8_t)vmovl_u8(vget_high_u8(vn));
            float32x4_t nf0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(v16_0)));
            float32x4_t nf1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(v16_0)));
            float32x4_t nf2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(v16_1)));
            float32x4_t nf3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(v16_1)));

            float32x4_t xv0 = vld1q_f32(x + jj +  0);
            float32x4_t xv1 = vld1q_f32(x + jj +  4);
            float32x4_t xv2 = vld1q_f32(x + jj +  8);
            float32x4_t xv3 = vld1q_f32(x + jj + 12);

            dot0 = vmlaq_f32(dot0, nf0, xv0); dot1 = vmlaq_f32(dot1, nf1, xv1);
            dot2 = vmlaq_f32(dot2, nf2, xv2); dot3 = vmlaq_f32(dot3, nf3, xv3);
            sx0 = vaddq_f32(sx0, xv0); sx1 = vaddq_f32(sx1, xv1);
            sx2 = vaddq_f32(sx2, xv2); sx3 = vaddq_f32(sx3, xv3);
        }
    }
    float dot_nib = neon_hsum_f32x4(vaddq_f32(vaddq_f32(dot0, dot1), vaddq_f32(dot2, dot3)));
    float sum_x   = neon_hsum_f32x4(vaddq_f32(vaddq_f32(sx0, sx1), vaddq_f32(sx2, sx3)));
    return d * dot_nib + m * sum_x;
}

/* ---- K-quant types  (block_size = 256) ------------------------- */

/*
 * Q8_K: sum = d * Σ(bs[sb] * Σ(qs[j] * x[j]))
 *
 * Process 16 sub-blocks × 16 elements.  Each sub-block has its own
 * f16 scale bs[sb]; the overall scale d is f32.
 */
static float neon_dot_q8_k_block(const u8 *data, const float *x) {
    float d; memcpy(&d, data, 4);
    const u16 *bs = (const u16 *)(data + 4);
    const i8  *qs = (const i8 *)(data + 36);

    float acc = 0.0f;
    for (int sb = 0; sb < 16; sb++) {
        float32x4_t s0 = vdupq_n_f32(0), s1 = vdupq_n_f32(0);
        float32x4_t s2 = vdupq_n_f32(0), s3 = vdupq_n_f32(0);
        int off = sb * 16;

        int8x16_t vq = vld1q_s8(qs + off);
        int16x8_t v16_0 = vmovl_s8(vget_low_s8(vq));
        int16x8_t v16_1 = vmovl_s8(vget_high_s8(vq));
        float32x4_t qf0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(v16_0)));
        float32x4_t qf1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(v16_0)));
        float32x4_t qf2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(v16_1)));
        float32x4_t qf3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(v16_1)));

        s0 = vmlaq_f32(s0, qf0, vld1q_f32(x + off +  0));
        s1 = vmlaq_f32(s1, qf1, vld1q_f32(x + off +  4));
        s2 = vmlaq_f32(s2, qf2, vld1q_f32(x + off +  8));
        s3 = vmlaq_f32(s3, qf3, vld1q_f32(x + off + 12));

        float dot_qs = neon_hsum_f32x4(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
        acc += dot_qs * f16_to_f32(bs[sb]);
    }
    return d * acc;
}

/*
 * Q6_K: sum = d * Σ(sc[sb] * Σ((q - 32) * x))
 *          = d * Σ(sc[sb] * (Σ(q * x) - 32 * Σ(x)))
 *
 * 16 sub-blocks of 16 elements each.  Scale is i8 per sub-block.
 * q = lo | (hi<<4), in range 0..63 → q - 32 in [-32, 31].
 */
static float neon_dot_q6_k_block(const u8 *data, const float *x) {
    float d = f16_to_f32(*(const u16 *)(data + 208));
    const u8 *scales = data + 192;
    const u8 *ql     = data;          /* 128 bytes */
    const u8 *qh     = data + 128;    /* 64 bytes  */

    float acc = 0.0f;
    for (int sb = 0; sb < 16; sb++) {
        i32 sc = (i8)scales[sb];
        u32 start = (u32)sb * 16;

        /* Vectorised unpack, bit-identical to the scalar walk:
         *   lo = 4-bit nibbles, 2 elems per byte; 16 contiguous bytes
         *        per sub-block, low or high nibbles depending on which
         *        quarter of the 64-element half this sub-block is in.
         *   hi = 2 extra bits, 4 elems per byte; the 16 bytes at
         *        (half<<5) are shared by 4 quarters via a 2-bit shift. */
        u32 sub7   = (u32)sb & 7;
        u32 half   = (u32)sb >> 3;
        u32 which  = sub7 >> 1;
        u32 sbpar  = sub7 & 1;

        const u8 *qsrc = ql + (half << 6) + ((which & 1) << 5) + (sbpar << 4);
        uint8x16_t qb  = vld1q_u8(qsrc);
        uint8x16_t lo  = (which >= 2) ? vshrq_n_u8(qb, 4)
                                      : vandq_u8(qb, vdupq_n_u8(0x0F));

        const u8 *hsrc = qh + (half << 5) + (sbpar << 4);
        uint8x16_t hb  = vld1q_u8(hsrc);
        uint8x16_t hi;
        switch (which) {   /* vshrq_n requires a compile-time shift */
            case 0:  hi = hb;                    break;
            case 1:  hi = vshrq_n_u8(hb, 2);     break;
            case 2:  hi = vshrq_n_u8(hb, 4);     break;
            default: hi = vshrq_n_u8(hb, 6);     break;
        }
        hi = vandq_u8(hi, vdupq_n_u8(0x3));

        uint8x16_t vals = vorrq_u8(lo, vshlq_n_u8(hi, 4));

        /* NEON dot: Σ(vals[j] * x[j]) and Σ(x[j]). */
        float32x4_t dot0 = vdupq_n_f32(0), dot1 = vdupq_n_f32(0);
        float32x4_t dot2 = vdupq_n_f32(0), dot3 = vdupq_n_f32(0);
        float32x4_t sx0  = vdupq_n_f32(0), sx1  = vdupq_n_f32(0);

        int16x8_t v16_0 = (int16x8_t)vmovl_u8(vget_low_u8(vals));
        int16x8_t v16_1 = (int16x8_t)vmovl_u8(vget_high_u8(vals));
        float32x4_t vf0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(v16_0)));
        float32x4_t vf1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(v16_0)));
        float32x4_t vf2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(v16_1)));
        float32x4_t vf3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(v16_1)));

        float32x4_t xv0 = vld1q_f32(x + start +  0);
        float32x4_t xv1 = vld1q_f32(x + start +  4);
        float32x4_t xv2 = vld1q_f32(x + start +  8);
        float32x4_t xv3 = vld1q_f32(x + start + 12);

        dot0 = vmlaq_f32(dot0, vf0, xv0); dot1 = vmlaq_f32(dot1, vf1, xv1);
        dot2 = vmlaq_f32(dot2, vf2, xv2); dot3 = vmlaq_f32(dot3, vf3, xv3);
        sx0 = vaddq_f32(sx0, xv0); sx1 = vaddq_f32(sx1, xv1);
        /* accumulate remaining x sums */
        float32x4_t sx_all = vaddq_f32(vaddq_f32(vaddq_f32(sx0, sx1), xv2), xv3);

        float dot_q = neon_hsum_f32x4(vaddq_f32(vaddq_f32(dot0, dot1), vaddq_f32(dot2, dot3)));
        float sum_x = neon_hsum_f32x4(sx_all);

        acc += (float)sc * (dot_q - 32.0f * sum_x);
    }
    return d * acc;
}

/*
 * Q4_K: result = d * Σ(sc[sb] * Σ(nib * x)) - dmin * Σ(mn[sb] * Σ(x))
 *
 * 8 sub-blocks of 32 elements.  Scales/mins are 6-bit packed.
 * Nibble extraction: sub-blocks within a 32-byte group share bytes;
 * even sb use low nibbles, odd sb use high nibbles.
 */
static float neon_dot_q4_k_block(const u8 *data, const float *x) {
    float d    = f16_to_f32(*(const u16 *)data);
    float dmin = f16_to_f32(*(const u16 *)(data + 2));
    const u8 *sm  = data + 4;
    const u8 *qs  = data + 16;

    float dot_acc = 0.0f, sum_acc = 0.0f;

    for (int sb = 0; sb < 8; sb++) {
        u8 sc, mn;
        q4k_scale_min(sm, (u32)sb, &sc, &mn);

        u32 g  = (u32)sb >> 1;
        u32 nb = (u32)sb & 1;
        const u8 *src = qs + g * 32;   /* 32 bytes for this group-pair */
        u32 off = (u32)sb * 32;

        /* Extract 32 nibbles (all low or all high), compute dot + sum_x. */
        float32x4_t dot0 = vdupq_n_f32(0), dot1 = vdupq_n_f32(0);
        float32x4_t dot2 = vdupq_n_f32(0), dot3 = vdupq_n_f32(0);
        float32x4_t sx0  = vdupq_n_f32(0), sx1  = vdupq_n_f32(0);
        float32x4_t sx2  = vdupq_n_f32(0), sx3  = vdupq_n_f32(0);

        for (int j = 0; j < 32; j += 16) {
            uint8x16_t vbytes = vld1q_u8(src + j);
            uint8x16_t vnibs;
            if (nb == 0) vnibs = vandq_u8(vbytes, vdupq_n_u8(0x0F));
            else         vnibs = vshrq_n_u8(vbytes, 4);

            int16x8_t v16_0 = (int16x8_t)vmovl_u8(vget_low_u8(vnibs));
            int16x8_t v16_1 = (int16x8_t)vmovl_u8(vget_high_u8(vnibs));
            float32x4_t nf0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(v16_0)));
            float32x4_t nf1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(v16_0)));
            float32x4_t nf2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(v16_1)));
            float32x4_t nf3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(v16_1)));

            float32x4_t xv0 = vld1q_f32(x + off + j +  0);
            float32x4_t xv1 = vld1q_f32(x + off + j +  4);
            float32x4_t xv2 = vld1q_f32(x + off + j +  8);
            float32x4_t xv3 = vld1q_f32(x + off + j + 12);

            dot0 = vmlaq_f32(dot0, nf0, xv0); dot1 = vmlaq_f32(dot1, nf1, xv1);
            dot2 = vmlaq_f32(dot2, nf2, xv2); dot3 = vmlaq_f32(dot3, nf3, xv3);
            sx0 = vaddq_f32(sx0, xv0); sx1 = vaddq_f32(sx1, xv1);
            sx2 = vaddq_f32(sx2, xv2); sx3 = vaddq_f32(sx3, xv3);
        }

        float dot_nib = neon_hsum_f32x4(vaddq_f32(vaddq_f32(dot0, dot1),
                                                    vaddq_f32(dot2, dot3)));
        float sum_x   = neon_hsum_f32x4(vaddq_f32(vaddq_f32(sx0, sx1),
                                                    vaddq_f32(sx2, sx3)));

        dot_acc += (float)sc * dot_nib;
        sum_acc += (float)mn * sum_x;
    }
    return d * dot_acc - dmin * sum_acc;
}

/* Per-byte logical right shift (vshrq_n needs a compile-time shift). */
static uint8x16_t neon_u8_shr(uint8x16_t v, u32 k) {
    switch (k) {
        case 0:  return v;
        case 1:  return vshrq_n_u8(v, 1);
        case 2:  return vshrq_n_u8(v, 2);
        case 3:  return vshrq_n_u8(v, 3);
        case 4:  return vshrq_n_u8(v, 4);
        case 5:  return vshrq_n_u8(v, 5);
        case 6:  return vshrq_n_u8(v, 6);
        default: return vshrq_n_u8(v, 7);
    }
}

/*
 * Q5_K: result = d * Σ(sc[sb] * Σ(q * x)) - dmin * Σ(mn[sb] * Σ(x))
 *
 * Same structure as Q4_K but with 1 extra hi bit per element.
 */
static float neon_dot_q5_k_block(const u8 *data, const float *x) {
    float d    = f16_to_f32(*(const u16 *)data);
    float dmin = f16_to_f32(*(const u16 *)(data + 2));
    const u8 *sm  = data + 4;
    const u8 *qh  = data + 16;
    const u8 *qs  = data + 48;

    float dot_acc = 0.0f, sum_acc = 0.0f;

    for (int sb = 0; sb < 8; sb++) {
        u8 sc, mn;
        q4k_scale_min(sm, (u32)sb, &sc, &mn);

        u32 g  = (u32)sb >> 1;
        u32 nb = (u32)sb & 1;
        const u8 *src_lo = qs + g * 32;
        u32 start = (u32)sb * 32;

        /* Vectorised unpack, bit-identical to the scalar walk:
         *  lo = 4-bit nibbles of 16 contiguous qs bytes (low or high
         *       nibbles by sub-block parity);
         *  hi = bit sb of qh[0..31], one byte per element. */
        uint8x16_t n0 = vld1q_u8(src_lo);
        uint8x16_t n1 = vld1q_u8(src_lo + 16);
        uint8x16_t lo0 = nb ? vshrq_n_u8(n0, 4) : vandq_u8(n0, vdupq_n_u8(0x0F));
        uint8x16_t lo1 = nb ? vshrq_n_u8(n1, 4) : vandq_u8(n1, vdupq_n_u8(0x0F));
        uint8x16_t hb0 = vandq_u8(neon_u8_shr(vld1q_u8(qh), (u32)sb),
                                  vdupq_n_u8(0x1));
        uint8x16_t hb1 = vandq_u8(neon_u8_shr(vld1q_u8(qh + 16), (u32)sb),
                                  vdupq_n_u8(0x1));
        uint8x16_t val[2] = {
            vorrq_u8(lo0, vshlq_n_u8(hb0, 4)),
            vorrq_u8(lo1, vshlq_n_u8(hb1, 4))
        };

        float32x4_t dot0 = vdupq_n_f32(0), dot1 = vdupq_n_f32(0);
        float32x4_t dot2 = vdupq_n_f32(0), dot3 = vdupq_n_f32(0);
        float32x4_t sx0  = vdupq_n_f32(0), sx1  = vdupq_n_f32(0);
        float32x4_t sx2  = vdupq_n_f32(0), sx3  = vdupq_n_f32(0);

        for (int j = 0; j < 32; j += 16) {
            uint8x16_t vu8 = val[j >> 4];
            int16x8_t v16_0 = (int16x8_t)vmovl_u8(vget_low_u8(vu8));
            int16x8_t v16_1 = (int16x8_t)vmovl_u8(vget_high_u8(vu8));
            float32x4_t vf0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(v16_0)));
            float32x4_t vf1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(v16_0)));
            float32x4_t vf2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(v16_1)));
            float32x4_t vf3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(v16_1)));

            float32x4_t xv0 = vld1q_f32(x + start + j +  0);
            float32x4_t xv1 = vld1q_f32(x + start + j +  4);
            float32x4_t xv2 = vld1q_f32(x + start + j +  8);
            float32x4_t xv3 = vld1q_f32(x + start + j + 12);

            dot0 = vmlaq_f32(dot0, vf0, xv0); dot1 = vmlaq_f32(dot1, vf1, xv1);
            dot2 = vmlaq_f32(dot2, vf2, xv2); dot3 = vmlaq_f32(dot3, vf3, xv3);
            sx0 = vaddq_f32(sx0, xv0); sx1 = vaddq_f32(sx1, xv1);
            sx2 = vaddq_f32(sx2, xv2); sx3 = vaddq_f32(sx3, xv3);
        }

        float dot_q = neon_hsum_f32x4(vaddq_f32(vaddq_f32(dot0, dot1),
                                                  vaddq_f32(dot2, dot3)));
        float sum_x = neon_hsum_f32x4(vaddq_f32(vaddq_f32(sx0, sx1),
                                                  vaddq_f32(sx2, sx3)));

        dot_acc += (float)sc * dot_q;
        sum_acc += (float)mn * sum_x;
    }
    return d * dot_acc - dmin * sum_acc;
}

/*
 * Q3_K: sum = d * Σ(sc[sb] * Σ((q - 4) * x))
 *          = d * Σ(sc[sb] * (Σ(q*x) - 4*Σ(x)))
 */
static float neon_dot_q3_k_block(const u8 *data, const float *x) {
    float d = f16_to_f32(*(const u16 *)(data + 108));
    const u8 *qh     = data;
    const u8 *qs     = data + 32;
    const u8 *scales = data + 96;

    float acc = 0.0f;
    for (int sb = 0; sb < 16; sb++) {
        u32 s = (u32)sb;
        u32 sc_low  = (scales[(s & 7)] >> ((s >= 8) ? 4 : 0)) & 0xF;
        u32 sc_high = (scales[8 + (s & 3)] >> (((s >> 2) & 3) * 2)) & 0x3;
        i32 sc = (i32)(sc_low | (sc_high << 4)) - 32;

        u32 start = (u32)sb * 16;

        /* Vectorised unpack, bit-identical to the scalar walk:
         *  lo = 2-bit pairs, 4 per byte; a sub-block reads 16 contiguous
         *       qs bytes (window g = sb>>3 half, (sb&1)*16 offset) at the
         *       2-bit lane ((sb>>1)&3).
         *  hi = 1 extra bit per element from the 32 qh bytes, bit lane
         *       (sb>>1), byte offset (sb&1)*16. */
        const u8 *lsrc = qs + (((u32)sb >> 3) << 5) + (((u32)sb & 1) << 4);
        uint8x16_t lb  = vld1q_u8(lsrc);
        uint8x16_t lo;
        switch (((u32)sb >> 1) & 3) {   /* vshrq_n needs a constant */
            case 0:  lo = lb;                break;
            case 1:  lo = vshrq_n_u8(lb, 2); break;
            case 2:  lo = vshrq_n_u8(lb, 4); break;
            default: lo = vshrq_n_u8(lb, 6); break;
        }
        lo = vandq_u8(lo, vdupq_n_u8(0x3));

        const u8 *hsrc = qh + (((u32)sb & 1) << 4);
        uint8x16_t hb  = vld1q_u8(hsrc);
        uint8x16_t hi;
        switch ((u32)sb >> 1) {         /* bit lane 0..7 */
            case 0:  hi = hb;                break;
            case 1:  hi = vshrq_n_u8(hb, 1); break;
            case 2:  hi = vshrq_n_u8(hb, 2); break;
            case 3:  hi = vshrq_n_u8(hb, 3); break;
            case 4:  hi = vshrq_n_u8(hb, 4); break;
            case 5:  hi = vshrq_n_u8(hb, 5); break;
            case 6:  hi = vshrq_n_u8(hb, 6); break;
            default: hi = vshrq_n_u8(hb, 7); break;
        }
        hi = vandq_u8(hi, vdupq_n_u8(0x1));

        uint8x16_t vu8 = vorrq_u8(lo, vshlq_n_u8(hi, 2));
        int16x8_t v16_0 = (int16x8_t)vmovl_u8(vget_low_u8(vu8));
        int16x8_t v16_1 = (int16x8_t)vmovl_u8(vget_high_u8(vu8));
        float32x4_t vf0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(v16_0)));
        float32x4_t vf1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(v16_0)));
        float32x4_t vf2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(v16_1)));
        float32x4_t vf3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(v16_1)));

        float32x4_t xv0 = vld1q_f32(x + start +  0);
        float32x4_t xv1 = vld1q_f32(x + start +  4);
        float32x4_t xv2 = vld1q_f32(x + start +  8);
        float32x4_t xv3 = vld1q_f32(x + start + 12);

        float32x4_t dot0 = vmulq_f32(vf0, xv0), dot1 = vmulq_f32(vf1, xv1);
        float32x4_t dot2 = vmulq_f32(vf2, xv2), dot3 = vmulq_f32(vf3, xv3);
        float dot_q = neon_hsum_f32x4(vaddq_f32(vaddq_f32(dot0, dot1),
                                                  vaddq_f32(dot2, dot3)));
        float sum_x = neon_hsum_f32x4(vaddq_f32(vaddq_f32(xv0, xv1),
                                                  vaddq_f32(xv2, xv3)));

        acc += (float)sc * (dot_q - 4.0f * sum_x);
    }
    return d * acc;
}

/*
 * Q2_K: sum = d * Σ(sc[sb] * Σ(q * x)) - mn * Σ(x)
 *        = d * Σ(sc[sb] * dot_q[sb]) - mn * total_sum_x
 */
static float neon_dot_q2_k_block(const u8 *data, const float *x) {
    float d  = f16_to_f32(*(const u16 *)(data + 80));
    float mn = f16_to_f32(*(const u16 *)(data + 82));
    const u8 *scales = data + 64;

    float dot_acc = 0.0f;
    float32x4_t total_sx0 = vdupq_n_f32(0), total_sx1 = vdupq_n_f32(0);
    float32x4_t total_sx2 = vdupq_n_f32(0), total_sx3 = vdupq_n_f32(0);

    for (int sb = 0; sb < 16; sb++) {
        u32 sc = scales[sb >> 1];
        if (sb & 1) sc >>= 4; else sc &= 0xF;

        u32 start = (u32)sb * 16;

        /* Vectorised unpack, bit-identical to the scalar walk:
         *  mag  = 2-bit value of element j from byte (j>>2) of the 4-byte
         *         sub-block window at lane (j&3) — splat each of the 4
         *         bytes to 4 lanes with a table lookup, then shift every
         *         lane right by (j&3)*2 with a per-lane shift vector;
         *  sign = bit (j&7) of byte (j>>3) of the 2-byte window.
         *  val  = mag - 4*sign, mod 2^8 == the scalar i8 value. */
        const uint8x16_t mag_idx = {
            0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3
        };
        const int8x16_t mag_sh = {
            0, -2, -4, -6, 0, -2, -4, -6, 0, -2, -4, -6, 0, -2, -4, -6
        };
        const uint8x16_t sg_idx = {
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1
        };
        const int8x16_t sg_sh = {
            0, -1, -2, -3, -4, -5, -6, -7, 0, -1, -2, -3, -4, -5, -6, -7
        };
        uint8x16_t mags = vqtbl1q_u8(vld1q_u8(data + (sb << 2)), mag_idx);
        mags = vandq_u8(vshlq_u8(mags, mag_sh), vdupq_n_u8(0x3));
        uint8x16_t sg = vqtbl1q_u8(vld1q_u8(data + (sb << 1)), sg_idx);
        sg = vandq_u8(vshlq_u8(sg, sg_sh), vdupq_n_u8(0x1));

        int8x16_t vs8 = (int8x16_t)vsubq_u8(mags, vshlq_n_u8(sg, 2));
        int16x8_t v16_0 = vmovl_s8(vget_low_s8(vs8));
        int16x8_t v16_1 = vmovl_s8(vget_high_s8(vs8));
        float32x4_t vf0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(v16_0)));
        float32x4_t vf1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(v16_0)));
        float32x4_t vf2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(v16_1)));
        float32x4_t vf3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(v16_1)));

        float32x4_t xv0 = vld1q_f32(x + start +  0);
        float32x4_t xv1 = vld1q_f32(x + start +  4);
        float32x4_t xv2 = vld1q_f32(x + start +  8);
        float32x4_t xv3 = vld1q_f32(x + start + 12);

        float32x4_t d0 = vmulq_f32(vf0, xv0), d1 = vmulq_f32(vf1, xv1);
        float32x4_t d2 = vmulq_f32(vf2, xv2), d3 = vmulq_f32(vf3, xv3);

        dot_acc += (float)(i32)sc * neon_hsum_f32x4(
            vaddq_f32(vaddq_f32(d0, d1), vaddq_f32(d2, d3)));

        total_sx0 = vaddq_f32(total_sx0, xv0); total_sx1 = vaddq_f32(total_sx1, xv1);
        total_sx2 = vaddq_f32(total_sx2, xv2); total_sx3 = vaddq_f32(total_sx3, xv3);
    }

    float total_sum_x = neon_hsum_f32x4(
        vaddq_f32(vaddq_f32(total_sx0, total_sx1), vaddq_f32(total_sx2, total_sx3)));
    return d * dot_acc - mn * total_sum_x;
}

/* ================================================================
 * Public API
 * ================================================================ */

typedef float (*block_dot_fn)(const u8 *data, const float *x);

float gguf_dot_batch(TensorInfo *ti, u64 i, u64 n, const float *x) {
    const u8 *data = ti->data;
    u32 type = ti->type;

    /* ---- Non-block types: direct NEON dot ---- */
    switch (type) {
    case GGUF_TYPE_F32: return dot_f32(data + i * 4, x, n);
    case GGUF_TYPE_F16: return dot_f16(data + i * 2, x, n);
    /* For BF16 / integer types, fall through to scalar. */
    default: break;
    }

    /* ---- Block-quantised types ---- */
    u32 block_elems;
    u32 block_bytes;
    block_dot_fn dot_fn = NULL;

    if (type <= GGUF_TYPE_Q8_1) {
        block_elems = BLOCK_SIMPLE;
        switch (type) {
            case GGUF_TYPE_Q4_0: block_bytes = 18; dot_fn = neon_dot_q4_0_block; break;
            case GGUF_TYPE_Q4_1: block_bytes = 20; dot_fn = neon_dot_q4_1_block; break;
            case GGUF_TYPE_Q5_0: block_bytes = 22; break;
            case GGUF_TYPE_Q5_1: block_bytes = 24; break;
            case GGUF_TYPE_Q8_0: block_bytes = 34; dot_fn = neon_dot_q8_0_block; break;
            case GGUF_TYPE_Q8_1: block_bytes = 40; break;
            default: goto fallback_dot;
        }
    } else {
        block_elems = BLOCK_KQUANT;
        switch (type) {
            case GGUF_TYPE_Q2_K:     block_bytes =  84; dot_fn = neon_dot_q2_k_block; break;
            case GGUF_TYPE_Q3_K:     block_bytes = 110; dot_fn = neon_dot_q3_k_block; break;
            case GGUF_TYPE_Q4_K:     block_bytes = 144; dot_fn = neon_dot_q4_k_block; break;
            case GGUF_TYPE_Q5_K:     block_bytes = 176; dot_fn = neon_dot_q5_k_block; break;
            case GGUF_TYPE_Q6_K:     block_bytes = 210; dot_fn = neon_dot_q6_k_block; break;
            case GGUF_TYPE_Q8_K:     block_bytes = 292; dot_fn = neon_dot_q8_k_block; break;
            case GGUF_TYPE_IQ2_XXS:  block_bytes =  66; break;
            case GGUF_TYPE_IQ2_XS:   block_bytes =  74; break;
            case GGUF_TYPE_IQ3_XXS:  block_bytes =  98; break;
            case GGUF_TYPE_IQ1_S:    block_bytes = 110; break;
            case GGUF_TYPE_IQ4_NL:   block_bytes =  50; break;
            case GGUF_TYPE_IQ3_S:    block_bytes = 110; break;
            case GGUF_TYPE_IQ2_S:    block_bytes =  82; break;
            case GGUF_TYPE_IQ4_XS:   block_bytes = 136; break;
            case GGUF_TYPE_IQ1_M:    block_bytes =  56; break;
            default: goto fallback_dot;
        }
    }

    /* Walk blocks intersecting [i, i+n).  Full blocks → fused NEON dot;
     * partial edges → scalar per-element. */
    u64 end = i + n;
    u64 bi0 = i / block_elems;
    u64 bi1 = (end + block_elems - 1) / block_elems;
    float result = 0.0f;

    for (u64 bi = bi0; dot_fn && bi < bi1; bi++) {
        u64 bs = bi * block_elems;
        u64 be = bs + block_elems;
        u64 cs = i > bs ? i : bs;
        u64 ce = end < be ? end : be;

        if (cs == bs && ce == be) {
            /* Fully contained block — fused NEON dot. */
            result += dot_fn(data + bi * block_bytes, x + (cs - i));
        } else {
            /* Partial block — scalar per-element. */
            for (u64 j = cs; j < ce; j++)
                result += gguf_dequant(ti, j) * x[j - i];
        }
    }

    if (dot_fn) return result;

fallback_dot:
    /* No fused dot function — scalar per-element. */
    for (u64 j = i; j < i + n; j++)
        result += gguf_dequant(ti, j) * x[j - i];
    return result;
}

/* ================================================================
 * vdotq_s32  i8 dot-product path  (requires ARMv8.2+ dotprod)
 *
 * Key insight: instead of widening i8→i32→f32 for each weight
 * and then doing f32 multiply-add (5 instructions / 4 elements),
 * we quantise the x vector to i8 once and use vdotq_s32 which
 * does 4 i8×i8→i32 multiplies + accumulate in ONE instruction.
 *
 *   vdotq_s32(acc, q_i8, x_i8):  1 instruction → 4 MACs
 *   widening chain:              5+ instructions → 4 MACs
 * ================================================================ */

#if defined(__ARM_FEATURE_DOTPROD)

/* Quantise a float vector to int8.
 *   scale = 127 / max(|x|)
 *   out[i] = round(clamp(x[i] * scale, -128, 127))
 * Returns scale. */
float quantize_f32_to_i8(const float *x, i8 *out, u64 n) {
    /* Find max absolute value. */
    float32x4_t vmax0 = vdupq_n_f32(0), vmax1 = vdupq_n_f32(0);
    u64 i = 0;
    for (; i + 8 <= n; i += 8) {
        vmax0 = vmaxq_f32(vmax0, vabsq_f32(vld1q_f32(x + i + 0)));
        vmax1 = vmaxq_f32(vmax1, vabsq_f32(vld1q_f32(x + i + 4)));
    }
    float32x4_t vmax = vmaxq_f32(vmax0, vmax1);
    float32x2_t s2 = vpmax_f32(vget_low_f32(vmax), vget_high_f32(vmax));
    float max_abs = vget_lane_f32(vpmax_f32(s2, s2), 0);
    for (; i < n; i++) {
        float v = fabsf(x[i]);
        if (v > max_abs) max_abs = v;
    }
    if (max_abs < 1e-8f) max_abs = 1e-8f;

    float scale = 127.0f / max_abs;
    float32x4_t vscale = vdupq_n_f32(scale);

    /* NEON quantise: f32 → round → clamp → s32 → saturating narrow → s8. */
    i = 0;
    for (; i + 16 <= n; i += 16) {
        float32x4_t f0 = vld1q_f32(x + i +  0), f1 = vld1q_f32(x + i +  4);
        float32x4_t f2 = vld1q_f32(x + i +  8), f3 = vld1q_f32(x + i + 12);

        int32x4_t i0 = vcvtnq_s32_f32(vmulq_f32(f0, vscale));
        int32x4_t i1 = vcvtnq_s32_f32(vmulq_f32(f1, vscale));
        int32x4_t i2 = vcvtnq_s32_f32(vmulq_f32(f2, vscale));
        int32x4_t i3 = vcvtnq_s32_f32(vmulq_f32(f3, vscale));

        int16x4_t s16_0 = vqmovn_s32(i0), s16_1 = vqmovn_s32(i1);
        int16x4_t s16_2 = vqmovn_s32(i2), s16_3 = vqmovn_s32(i3);

        int16x8_t s16_01 = vcombine_s16(s16_0, s16_1);
        int16x8_t s16_23 = vcombine_s16(s16_2, s16_3);

        int8x8_t s8_0 = vqmovn_s16(s16_01);
        int8x8_t s8_1 = vqmovn_s16(s16_23);

        vst1q_s8(out + i, vcombine_s8(s8_0, s8_1));
    }
    /* Tail. */
    for (; i < n; i++) {
        float v = x[i] * scale;
        if (v > 127.0f) v = 127.0f;
        if (v < -128.0f) v = -128.0f;
        out[i] = (i8)(i32)roundf(v);
    }
    return scale;
}

/* ---- vdotq i8 dot kernels ------------------------------------- */

/*
 * Q8_0 i8-dot: dot_i32 = Σ(qs[j] * x_i8[j])  (32 elements)
 *   result = (d / x_scale) * dot_i32
 */
static float vdot_dot_q8_0_block(const u8 *data, const i8 *x_i8,
                                  float d, float x_scale) {
    const i8 *qs = (const i8 *)(data + 2);

    int32x4_t acc0 = vdupq_n_s32(0), acc1 = vdupq_n_s32(0);

    acc0 = vdotq_s32(acc0, vld1q_s8(qs +  0), vld1q_s8(x_i8 +  0));
    acc1 = vdotq_s32(acc1, vld1q_s8(qs + 16), vld1q_s8(x_i8 + 16));

    int32x4_t acc  = vaddq_s32(acc0, acc1);
    int32x2_t s2   = vadd_s32(vget_low_s32(acc), vget_high_s32(acc));
    s2 = vpadd_s32(s2, s2);
    i32 dot = vget_lane_s32(s2, 0);

    return (d / x_scale) * (float)dot;
}

/*
 * Q8_K i8-dot: 16 sub-blocks × 16 elements; each has f16 scale bs[sb].
 *   dot_i32 = Σ(qs[j] * x_i8[j]) per sub-block
 *   result = (d / x_scale) * Σ(bs[sb] * dot_i32[sb])
 */
static float vdot_dot_q8_k_block(const u8 *data, const i8 *x_i8,
                                  float d, float x_scale) {
    const u16 *bs = (const u16 *)(data + 4);
    const i8  *qs = (const i8 *)(data + 36);

    float acc = 0.0f;
    for (int sb = 0; sb < 16; sb++) {
        int off = sb * 16;

        int32x4_t a0 = vdupq_n_s32(0);
        a0 = vdotq_s32(a0, vld1q_s8(qs + off), vld1q_s8(x_i8 + off));

        int32x2_t s2 = vadd_s32(vget_low_s32(a0), vget_high_s32(a0));
        s2 = vpadd_s32(s2, s2);
        i32 dot = vget_lane_s32(s2, 0);

        acc += (float)dot * f16_to_f32(bs[sb]);
    }
    return (d / x_scale) * acc;
}

/* ---- i8 dot dispatcher ---------------------------------------- */

typedef float (*i8_block_dot_fn)(const u8 *data, const i8 *x_i8,
                                  float d, float x_scale);

float gguf_dot_i8_batch(TensorInfo *ti, u64 i, u64 n,
                         const i8 *x_i8, float x_scale) {
    const u8 *data = ti->data;
    u32 type = ti->type;
    u32 block_elems, block_bytes;
    i8_block_dot_fn fn = NULL;
    float d_scale = 1.0f;

    /* Determine block params and scale factor. */
    if (type == GGUF_TYPE_Q8_0) {
        block_elems = BLOCK_SIMPLE; block_bytes = 34;
        d_scale = f16_to_f32(*(const u16 *)data); /* same d for all blocks of this tensor */
        fn = (i8_block_dot_fn)vdot_dot_q8_0_block;
    } else if (type == GGUF_TYPE_Q8_K) {
        block_elems = BLOCK_KQUANT; block_bytes = 292;
        float d; memcpy(&d, data, 4);  /* f32 d */
        d_scale = d;
        fn = (i8_block_dot_fn)vdot_dot_q8_k_block;
    } else {
        goto fallback_i8;
    }

    /* Walk blocks. */
    u64 end = i + n;
    u64 bi0 = i / block_elems;
    u64 bi1 = (end + block_elems - 1) / block_elems;
    float result = 0.0f;

    for (u64 bi = bi0; fn && bi < bi1; bi++) {
        u64 bs = bi * block_elems;
        u64 be = bs + block_elems;
        u64 cs = i > bs ? i : bs;
        u64 ce = end < be ? end : be;
        u64 off = (cs - i);

        if (cs == bs && ce == be) {
            result += fn(data + bi * block_bytes, x_i8 + off, d_scale, x_scale);
        } else {
            for (u64 j = cs; j < ce; j++)
                result += gguf_dequant(ti, j) * ((float)(i8)x_i8[j - i] / x_scale);
        }
    }

    if (fn) return result;

fallback_i8:
    /* Fall back: dequant with scalar, multiply with original x scaled back. */
    for (u64 j = i; j < i + n; j++)
        result += gguf_dequant(ti, j) * ((float)(i8)x_i8[j - i] / x_scale);
    return result;
}

#else  /* !__ARM_FEATURE_DOTPROD — stubs, callers fall back to gguf_dot_batch */

float quantize_f32_to_i8(const float *x, i8 *out, u64 n) {
    (void)x; (void)out; (void)n; return 0.0f;
}
float gguf_dot_i8_batch(TensorInfo *ti, u64 i, u64 n,
                         const i8 *x_i8, float x_scale) {
    (void)ti; (void)i; (void)n; (void)x_i8; (void)x_scale;
    return 0.0f;
}

#endif /* __ARM_FEATURE_DOTPROD */
