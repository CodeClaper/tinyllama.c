/*
 * quants_x86.c — SSE/SSE4.1 SIMD batch dequantization
 *
 * One block-level dequant function per GGUF quant type.
 * The dispatcher gguf_dequant_batch() processes full blocks with
 * these kernels and handles edge elements via the scalar gguf_dequant().
 */
#include <smmintrin.h>
#include <string.h>
#include <math.h>

#include "../def.h"
#include "../quants.h"
#include "quants_cpu.h"

/* ================================================================
 * Block sizes
 * ================================================================ */

#define BLOCK_SIMPLE  32
#define BLOCK_KQUANT 256

/* ================================================================
 * SSE helpers
 * ================================================================ */

/* Widen 16 × uint8 to 16 × float32 (as 4 × __m128).
 *   out = (float)(i32)(vu8[n] - bias) * scale + add
 */
static inline void sse_u8_to_f32_x16(__m128i vu8,
                                      __m128 scale, __m128 add, i32 bias,
                                      __m128 *r0, __m128 *r1,
                                      __m128 *r2, __m128 *r3) {
    __m128i v16_0 = _mm_cvtepu8_epi16(vu8);
    __m128i v16_1 = _mm_cvtepu8_epi16(_mm_srli_si128(vu8, 8));

    __m128i v32_0 = _mm_cvtepi16_epi32(v16_0);
    __m128i v32_1 = _mm_cvtepi16_epi32(_mm_srli_si128(v16_0, 8));
    __m128i v32_2 = _mm_cvtepi16_epi32(v16_1);
    __m128i v32_3 = _mm_cvtepi16_epi32(_mm_srli_si128(v16_1, 8));

    if (bias) {
        __m128i vb = _mm_set1_epi32(bias);
        v32_0 = _mm_sub_epi32(v32_0, vb);
        v32_1 = _mm_sub_epi32(v32_1, vb);
        v32_2 = _mm_sub_epi32(v32_2, vb);
        v32_3 = _mm_sub_epi32(v32_3, vb);
    }

    *r0 = _mm_add_ps(add, _mm_mul_ps(_mm_cvtepi32_ps(v32_0), scale));
    *r1 = _mm_add_ps(add, _mm_mul_ps(_mm_cvtepi32_ps(v32_1), scale));
    *r2 = _mm_add_ps(add, _mm_mul_ps(_mm_cvtepi32_ps(v32_2), scale));
    *r3 = _mm_add_ps(add, _mm_mul_ps(_mm_cvtepi32_ps(v32_3), scale));
}

/* Widen 16 × int8 to 16 × float32 (as 4 × __m128).
 *   out = (float)(i32)(vs8[n]) * scale + add
 */
static inline void sse_s8_to_f32_x16(__m128i vs8,
                                      __m128 scale, __m128 add,
                                      __m128 *r0, __m128 *r1,
                                      __m128 *r2, __m128 *r3) {
    __m128i v16_0 = _mm_cvtepi8_epi16(vs8);
    __m128i v16_1 = _mm_cvtepi8_epi16(_mm_srli_si128(vs8, 8));

    __m128i v32_0 = _mm_cvtepi16_epi32(v16_0);
    __m128i v32_1 = _mm_cvtepi16_epi32(_mm_srli_si128(v16_0, 8));
    __m128i v32_2 = _mm_cvtepi16_epi32(v16_1);
    __m128i v32_3 = _mm_cvtepi16_epi32(_mm_srli_si128(v16_1, 8));

    *r0 = _mm_add_ps(add, _mm_mul_ps(_mm_cvtepi32_ps(v32_0), scale));
    *r1 = _mm_add_ps(add, _mm_mul_ps(_mm_cvtepi32_ps(v32_1), scale));
    *r2 = _mm_add_ps(add, _mm_mul_ps(_mm_cvtepi32_ps(v32_2), scale));
    *r3 = _mm_add_ps(add, _mm_mul_ps(_mm_cvtepi32_ps(v32_3), scale));
}

/* Store 4 × __m128 to out[0..15] */
static inline void sse_store_f32_x16(float *out,
                                      __m128 r0, __m128 r1,
                                      __m128 r2, __m128 r3) {
    _mm_storeu_ps(out +  0, r0);
    _mm_storeu_ps(out +  4, r1);
    _mm_storeu_ps(out +  8, r2);
    _mm_storeu_ps(out + 12, r3);
}

/* Horizontal sum of a __m128 into a scalar. */
static inline float sse_hsum_f32x4(__m128 v) {
    v = _mm_hadd_ps(v, v);
    v = _mm_hadd_ps(v, v);
    return _mm_cvtss_f32(v);
}

/* Extract high nibbles from each byte (>> 4) into bytes 0..0x0F.
 * SSE2 trick: mask out low nibble, shift as 16-bit, mask result.
 */
static inline __m128i sse_hi_nibbles(__m128i v) {
    __m128i mask = _mm_set1_epi8(0xF0);
    __m128i hi = _mm_and_si128(v, mask);
    hi = _mm_srli_epi16(hi, 4);
    hi = _mm_and_si128(hi, _mm_set1_epi8(0x0F));
    return hi;
}

/* Extract low nibbles from each byte. */
static inline __m128i sse_lo_nibbles(__m128i v) {
    return _mm_and_si128(v, _mm_set1_epi8(0x0F));
}

/* ================================================================
 * Non-block types — direct SSE copy / convert
 * ================================================================ */

/* F32: straight memcpy is fine; use SSE copy for consistency. */
static void batch_f32(const u8 *data, float *out, u64 n) {
    u64 i = 0;
    for (; i + 16 <= n; i += 16) {
        _mm_storeu_ps(out + i +  0, _mm_loadu_ps((const float *)data + i +  0));
        _mm_storeu_ps(out + i +  4, _mm_loadu_ps((const float *)data + i +  4));
        _mm_storeu_ps(out + i +  8, _mm_loadu_ps((const float *)data + i +  8));
        _mm_storeu_ps(out + i + 12, _mm_loadu_ps((const float *)data + i + 12));
    }
    for (; i < n; i++) out[i] = ((const float *)data)[i];
}

/* F16 → F32 (scalar fallback). */
static void batch_f16(const u8 *data, float *out, u64 n) {
    const u16 *h = (const u16 *)data;
    for (u64 i = 0; i < n; i++) out[i] = f16_to_f32(h[i]);
}

/* BF16 → F32: left-shift 16 bits. */
static void batch_bf16(const u8 *data, float *out, u64 n) {
    const u16 *b = (const u16 *)data;
    u64 i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i vh = _mm_loadu_si128((const __m128i *)(b + i));
        __m128i lo = _mm_cvtepu16_epi32(vh);
        __m128i hi = _mm_cvtepu16_epi32(_mm_srli_si128(vh, 8));
        lo = _mm_slli_epi32(lo, 16);
        hi = _mm_slli_epi32(hi, 16);
        _mm_storeu_ps(out + i + 0, _mm_castsi128_ps(lo));
        _mm_storeu_ps(out + i + 4, _mm_castsi128_ps(hi));
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
        __m128i v = _mm_loadu_si128((const __m128i *)(s + i));
        __m128 r0, r1, r2, r3;
        sse_s8_to_f32_x16(v, _mm_set1_ps(1.0f), _mm_setzero_ps(), &r0, &r1, &r2, &r3);
        sse_store_f32_x16(out + i, r0, r1, r2, r3);
    }
    for (; i < n; i++) out[i] = (float)s[i];
}

/* I16 → F32. */
static void batch_i16(const u8 *data, float *out, u64 n) {
    const i16 *s = (const i16 *)data;
    u64 i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i v = _mm_loadu_si128((const __m128i *)(s + i));
        __m128i v0 = _mm_cvtepi16_epi32(v);
        __m128i v1 = _mm_cvtepi16_epi32(_mm_srli_si128(v, 8));
        _mm_storeu_ps(out + i + 0, _mm_cvtepi32_ps(v0));
        _mm_storeu_ps(out + i + 4, _mm_cvtepi32_ps(v1));
    }
    for (; i < n; i++) out[i] = (float)s[i];
}

/* I32 → F32. */
static void batch_i32(const u8 *data, float *out, u64 n) {
    const i32 *s = (const i32 *)data;
    u64 i = 0;
    for (; i + 4 <= n; i += 4) {
        _mm_storeu_ps(out + i, _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i *)(s + i))));
    }
    for (; i < n; i++) out[i] = (float)s[i];
}

/* I64 → F32. */
static void batch_i64(const u8 *data, float *out, u64 n) {
    const i64 *s = (const i64 *)data;
    for (u64 i = 0; i < n; i++) out[i] = (float)s[i];
}

/* ================================================================
 * Simple block types  (block_size = 32)
 * ================================================================ */

/*
 * Q8_0 block (34 bytes): [d:f16][qs:32×i8]
 *   out[j] = (float)(i8)qs[j] * d
 */
static void sse_block_q8_0(const u8 *data, float *out) {
    float d = f16_to_f32(*(const u16 *)data);
    __m128 vd = _mm_set1_ps(d);
    const i8 *qs = (const i8 *)(data + 2);

    __m128i v0 = _mm_loadu_si128((const __m128i *)(qs +  0));
    __m128i v1 = _mm_loadu_si128((const __m128i *)(qs + 16));

    __m128 r0, r1, r2, r3, r4, r5, r6, r7;
    sse_s8_to_f32_x16(v0, vd, _mm_setzero_ps(), &r0, &r1, &r2, &r3);
    sse_s8_to_f32_x16(v1, vd, _mm_setzero_ps(), &r4, &r5, &r6, &r7);

    sse_store_f32_x16(out +  0, r0, r1, r2, r3);
    sse_store_f32_x16(out + 16, r4, r5, r6, r7);
}

/*
 * Q8_1 block (40 bytes): [d:f32][pad:4][qs:32×i8]
 *   out[j] = (float)(i8)qs[j] * d
 */
static void sse_block_q8_1(const u8 *data, float *out) {
    float d;
    memcpy(&d, data, 4);
    __m128 vd = _mm_set1_ps(d);
    const i8 *qs = (const i8 *)(data + 8);

    __m128i v0 = _mm_loadu_si128((const __m128i *)(qs +  0));
    __m128i v1 = _mm_loadu_si128((const __m128i *)(qs + 16));

    __m128 r0, r1, r2, r3, r4, r5, r6, r7;
    sse_s8_to_f32_x16(v0, vd, _mm_setzero_ps(), &r0, &r1, &r2, &r3);
    sse_s8_to_f32_x16(v1, vd, _mm_setzero_ps(), &r4, &r5, &r6, &r7);

    sse_store_f32_x16(out +  0, r0, r1, r2, r3);
    sse_store_f32_x16(out + 16, r4, r5, r6, r7);
}

/*
 * Q4_0 block (18 bytes): [d:f16][qs:16 bytes → 32 nibbles]
 *   nib = qs[o>>1] >> ((o&1)<<2) & 0xF
 *   out[o] = (float)((i32)nib - 8) * d
 */
static void sse_block_q4_0(const u8 *data, float *out) {
    float d = f16_to_f32(*(const u16 *)data);
    __m128 vd = _mm_set1_ps(d);
    const u8 *qs = data + 2;

    __m128i vq   = _mm_loadu_si128((const __m128i *)qs);
    __m128i vlo  = sse_lo_nibbles(vq);
    __m128i vhi  = sse_hi_nibbles(vq);
    __m128i z0 = _mm_unpacklo_epi8(vlo, vhi);
    __m128i z1 = _mm_unpackhi_epi8(vlo, vhi);

    __m128 r0, r1, r2, r3, r4, r5, r6, r7;
    sse_u8_to_f32_x16(z0, vd, _mm_setzero_ps(), 8, &r0, &r1, &r2, &r3);
    sse_u8_to_f32_x16(z1, vd, _mm_setzero_ps(), 8, &r4, &r5, &r6, &r7);

    sse_store_f32_x16(out +  0, r0, r1, r2, r3);
    sse_store_f32_x16(out + 16, r4, r5, r6, r7);
}

/*
 * Q4_1 block (20 bytes): [d:f16][m:f16][qs:16 bytes → 32 nibbles]
 *   out[o] = (float)nib * d + m
 */
static void sse_block_q4_1(const u8 *data, float *out) {
    float d = f16_to_f32(*(const u16 *)data);
    float m = f16_to_f32(*(const u16 *)(data + 2));
    __m128 vd = _mm_set1_ps(d);
    __m128 vm = _mm_set1_ps(m);
    const u8 *qs = data + 4;

    __m128i vq   = _mm_loadu_si128((const __m128i *)qs);
    __m128i vlo  = sse_lo_nibbles(vq);
    __m128i vhi  = sse_hi_nibbles(vq);
    __m128i z0 = _mm_unpacklo_epi8(vlo, vhi);
    __m128i z1 = _mm_unpackhi_epi8(vlo, vhi);

    __m128 r0, r1, r2, r3, r4, r5, r6, r7;
    sse_u8_to_f32_x16(z0, vd, vm, 0, &r0, &r1, &r2, &r3);
    sse_u8_to_f32_x16(z1, vd, vm, 0, &r4, &r5, &r6, &r7);

    sse_store_f32_x16(out +  0, r0, r1, r2, r3);
    sse_store_f32_x16(out + 16, r4, r5, r6, r7);
}

/*
 * Q5_0 block (22 bytes): [d:f16][qh:u32][ql:16 bytes → 32 nibbles]
 *   lo  = ql[o>>1] >> ((o&1)<<2) & 0xF
 *   hi  = (qh >> o) & 1
 *   out[o] = (float)((i32)((hi << 4) | lo) - 16) * d
 */
static void sse_block_q5_0(const u8 *data, float *out) {
    float d = f16_to_f32(*(const u16 *)data);
    __m128 vd = _mm_set1_ps(d);
    u32 qh;
    memcpy(&qh, data + 2, 4);
    const u8 *ql = data + 6;

    u8 hi[32];
    for (int o = 0; o < 32; o++)
        hi[o] = ((qh >> o) & 1) << 4;

    __m128i vql  = _mm_loadu_si128((const __m128i *)ql);
    __m128i vlo  = sse_lo_nibbles(vql);
    __m128i vhi_lo = sse_hi_nibbles(vql);
    __m128i z0 = _mm_unpacklo_epi8(vlo, vhi_lo);
    __m128i z1 = _mm_unpackhi_epi8(vlo, vhi_lo);

    __m128i vhi0 = _mm_loadu_si128((const __m128i *)(hi +  0));
    __m128i vhi1 = _mm_loadu_si128((const __m128i *)(hi + 16));

    for (int h = 0; h < 2; h++) {
        __m128i vnib = _mm_or_si128(h == 0 ? z0 : z1, h == 0 ? vhi0 : vhi1);
        __m128 r0, r1, r2, r3;
        sse_u8_to_f32_x16(vnib, vd, _mm_setzero_ps(), 16, &r0, &r1, &r2, &r3);
        sse_store_f32_x16(out + h * 16, r0, r1, r2, r3);
    }
}

/*
 * Q5_1 block (24 bytes): [d:f16][m:f16][qh:u32][ql:16 bytes → 32 nibbles]
 *   out[o] = (float)((hi << 4) | lo) * d + m
 */
static void sse_block_q5_1(const u8 *data, float *out) {
    float d = f16_to_f32(*(const u16 *)data);
    float m = f16_to_f32(*(const u16 *)(data + 2));
    __m128 vd = _mm_set1_ps(d);
    __m128 vm = _mm_set1_ps(m);
    u32 qh;
    memcpy(&qh, data + 4, 4);
    const u8 *ql = data + 8;

    u8 hi[32];
    for (int o = 0; o < 32; o++)
        hi[o] = ((qh >> o) & 1) << 4;

    __m128i vql     = _mm_loadu_si128((const __m128i *)ql);
    __m128i vlo     = sse_lo_nibbles(vql);
    __m128i vhi_lo  = sse_hi_nibbles(vql);
    __m128i z0 = _mm_unpacklo_epi8(vlo, vhi_lo);
    __m128i z1 = _mm_unpackhi_epi8(vlo, vhi_lo);

    __m128i vhi0 = _mm_loadu_si128((const __m128i *)(hi +  0));
    __m128i vhi1 = _mm_loadu_si128((const __m128i *)(hi + 16));

    for (int h = 0; h < 2; h++) {
        __m128i vnib = _mm_or_si128(h == 0 ? z0 : z1, h == 0 ? vhi0 : vhi1);
        __m128 r0, r1, r2, r3;
        sse_u8_to_f32_x16(vnib, vd, vm, 0, &r0, &r1, &r2, &r3);
        sse_store_f32_x16(out + h * 16, r0, r1, r2, r3);
    }
}

/* ================================================================
 * K-quant types  (block_size = 256)
 * ================================================================ */

/*
 * Q8_K block (292 bytes): [d:f32][bs:16×f16][qs:256×i8]
 *   out[o] = (float)qs[o] * bs[o>>4] * d
 * Process 16 sub-blocks of 16 elements each.
 */
static void sse_block_q8_k(const u8 *data, float *out) {
    float d;
    memcpy(&d, data, 4);
    __m128 vd = _mm_set1_ps(d);
    const u16 *bs = (const u16 *)(data + 4);
    const i8   *qs = (const i8 *)(data + 36);

    for (int sb = 0; sb < 16; sb++) {
        __m128 vscale = _mm_mul_ps(vd, _mm_set1_ps(f16_to_f32(bs[sb])));
        int off = sb * 16;

        __m128i vq = _mm_loadu_si128((const __m128i *)(qs + off));
        __m128 r0, r1, r2, r3;
        sse_s8_to_f32_x16(vq, vscale, _mm_setzero_ps(), &r0, &r1, &r2, &r3);
        sse_store_f32_x16(out + off, r0, r1, r2, r3);
    }
}

/*
 * Q6_K block (210 bytes):
 *   [ql:128][qh:64][scales:16×i8][d:f16]
 *   ql holds low 4 bits, qh holds high 2 bits per element.
 *   out[o] = d * (float)(i8)scales[o>>4] * (float)(q - 32)
 */
static void sse_block_q6_k(const u8 *data, float *out) {
    float d = f16_to_f32(*(const u16 *)(data + 208));
    const u8 *scales = data + 192;
    const u8 *ql     = data;
    const u8 *qh     = data + 128;

    for (int sb = 0; sb < 16; sb++) {
        i32 sc       = (i8)scales[sb];
        u32 start    = (u32)sb * 16;

        u8 vals[16];
        for (int j = 0; j < 16; j++) {
            u32 o     = start + (u32)j;
            u32 half  = o >> 7;
            u32 hl    = o & 127;
            u32 which = hl >> 5;
            u32 l     = hl & 31;
            u32 ql_off = (half << 6) + ((which & 1) << 5) + l;
            u32 lo = (which >= 2) ? ((ql[ql_off] >> 4) & 0xF) : (ql[ql_off] & 0xF);
            u32 qh_off = (half << 5) + l;
            u32 hi = (qh[qh_off] >> (which * 2)) & 0x3;
            vals[j] = (u8)(lo | (hi << 4));
        }

        __m128 vscale = _mm_set1_ps(d * (float)sc);

        __m128i vu8 = _mm_loadu_si128((const __m128i *)vals);
        __m128i v16_0 = _mm_cvtepu8_epi16(vu8);
        __m128i v16_1 = _mm_cvtepu8_epi16(_mm_srli_si128(vu8, 8));

        __m128i vbias = _mm_set1_epi32(32);
        __m128i v32_0 = _mm_sub_epi32(_mm_cvtepi16_epi32(v16_0), vbias);
        __m128i v32_1 = _mm_sub_epi32(_mm_cvtepi16_epi32(_mm_srli_si128(v16_0, 8)), vbias);
        __m128i v32_2 = _mm_sub_epi32(_mm_cvtepi16_epi32(v16_1), vbias);
        __m128i v32_3 = _mm_sub_epi32(_mm_cvtepi16_epi32(_mm_srli_si128(v16_1, 8)), vbias);

        __m128 r0 = _mm_mul_ps(_mm_cvtepi32_ps(v32_0), vscale);
        __m128 r1 = _mm_mul_ps(_mm_cvtepi32_ps(v32_1), vscale);
        __m128 r2 = _mm_mul_ps(_mm_cvtepi32_ps(v32_2), vscale);
        __m128 r3 = _mm_mul_ps(_mm_cvtepi32_ps(v32_3), vscale);

        sse_store_f32_x16(out + start, r0, r1, r2, r3);
    }
}

/*
 * Q4_K block (144 bytes):
 *   [d:f16][dmin:f16][scales+mins:12 bytes (8×6-bit)][qs:128 bytes → 256 nibbles]
 *   8 sub-blocks of 32 elements.
 *   out[o] = d * (float)sc * (float)nib - dmin * (float)mn
 */
static void sse_block_q4_k(const u8 *data, float *out) {
    float d    = f16_to_f32(*(const u16 *)data);
    float dmin = f16_to_f32(*(const u16 *)(data + 2));
    const u8 *sm = data + 4;
    const u8 *qs = data + 16;

    for (int sb = 0; sb < 8; sb++) {
        u8 sc, mn;
        q4k_scale_min(sm, sb, &sc, &mn);

        __m128 vscale = _mm_set1_ps(d * (float)sc);
        __m128 vmin   = _mm_set1_ps(dmin * (float)mn);

        u32 g  = (u32)sb >> 1;
        u32 nb = (u32)sb & 1;
        const u8 *src = qs + g * 32;

        u8 buf[32];
        for (int j = 0; j < 32; j++)
            buf[j] = (nb ? (src[j] >> 4) : (src[j] & 0xF));

        for (int half = 0; half < 2; half++) {
            __m128i vu8 = _mm_loadu_si128((const __m128i *)(buf + half * 16));
            __m128 r0, r1, r2, r3;
            __m128 neg_vmin = _mm_sub_ps(_mm_setzero_ps(), vmin);
            sse_u8_to_f32_x16(vu8, vscale, neg_vmin, 0, &r0, &r1, &r2, &r3);
            sse_store_f32_x16(out + (u32)sb * 32 + half * 16, r0, r1, r2, r3);
        }
    }
}

/*
 * Q5_K block (176 bytes):
 *   [d:f16][dmin:f16][scales+mins:12][qh:32 bytes hi bits][qs:128 bytes lo nibbles]
 *   out[o] = d * (float)sc * (float)q - dmin * (float)mn
 */
static void sse_block_q5_k(const u8 *data, float *out) {
    float d    = f16_to_f32(*(const u16 *)data);
    float dmin = f16_to_f32(*(const u16 *)(data + 2));
    const u8 *sm = data + 4;
    const u8 *qh = data + 16;
    const u8 *qs = data + 48;

    for (int sb = 0; sb < 8; sb++) {
        u8 sc, mn;
        q4k_scale_min(sm, sb, &sc, &mn);

        __m128 vscale = _mm_set1_ps(d * (float)sc);
        __m128 vmin   = _mm_set1_ps(dmin * (float)mn);

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
            __m128i vu8 = _mm_loadu_si128((const __m128i *)(vals + half * 16));
            __m128 r0, r1, r2, r3;
            __m128 neg_vmin = _mm_sub_ps(_mm_setzero_ps(), vmin);
            sse_u8_to_f32_x16(vu8, vscale, neg_vmin, 0, &r0, &r1, &r2, &r3);
            sse_store_f32_x16(out + (u32)sb * 32 + half * 16, r0, r1, r2, r3);
        }
    }
}

/*
 * Q3_K block (110 bytes):
 *   [...][d:f16 at byte 108]
 *   out[o] = d * (float)sc * (float)(q - 4)
 * 3-bit quants: 2 lo bits (qs) + 1 hi bit (qh).
 * 16 sub-blocks of 16 elements.
 */
static void sse_block_q3_k(const u8 *data, float *out) {
    float d = f16_to_f32(*(const u16 *)(data + 108));
    const u8 *qh      = data;
    const u8 *qs      = data + 32;
    const u8 *scales  = data + 96;

    for (int sb = 0; sb < 16; sb++) {
        u32 s = (u32)sb;
        u32 sc_low  = (scales[(s & 7)] >> ((s >= 8) ? 4 : 0)) & 0xF;
        u32 sc_high = (scales[8 + (s & 3)] >> (((s >> 2) & 3) * 2)) & 0x3;
        i32 sc = (i32)(sc_low | (sc_high << 4)) - 32;

        __m128 vscale = _mm_set1_ps(d * (float)sc);

        u32 start = (u32)sb * 16;
        u8 vals[16];
        for (int j = 0; j < 16; j++) {
            u32 o   = start + (u32)j;
            u32 g   = o >> 7;
            u32 pr  = (o >> 5) & 3;
            u32 bc  = o & 31;
            u32 lo  = (qs[g * 32 + bc] >> (pr * 2)) & 0x3;
            u32 hi  = (qh[o & 31] >> (o >> 5)) & 1;
            vals[j]  = (u8)(lo | (hi << 2));
        }

        __m128i vu8  = _mm_loadu_si128((const __m128i *)vals);
        __m128i v16_0 = _mm_cvtepu8_epi16(vu8);
        __m128i v16_1 = _mm_cvtepu8_epi16(_mm_srli_si128(vu8, 8));

        __m128i vb4 = _mm_set1_epi32(4);
        __m128i v32_0 = _mm_sub_epi32(_mm_cvtepi16_epi32(v16_0), vb4);
        __m128i v32_1 = _mm_sub_epi32(_mm_cvtepi16_epi32(_mm_srli_si128(v16_0, 8)), vb4);
        __m128i v32_2 = _mm_sub_epi32(_mm_cvtepi16_epi32(v16_1), vb4);
        __m128i v32_3 = _mm_sub_epi32(_mm_cvtepi16_epi32(_mm_srli_si128(v16_1, 8)), vb4);

        __m128 r0 = _mm_add_ps(_mm_setzero_ps(), _mm_mul_ps(_mm_cvtepi32_ps(v32_0), vscale));
        __m128 r1 = _mm_add_ps(_mm_setzero_ps(), _mm_mul_ps(_mm_cvtepi32_ps(v32_1), vscale));
        __m128 r2 = _mm_add_ps(_mm_setzero_ps(), _mm_mul_ps(_mm_cvtepi32_ps(v32_2), vscale));
        __m128 r3 = _mm_add_ps(_mm_setzero_ps(), _mm_mul_ps(_mm_cvtepi32_ps(v32_3), vscale));

        sse_store_f32_x16(out + start, r0, r1, r2, r3);
    }
}

/*
 * Q2_K block (84 bytes):
 *   [...][d:f16 at 80][mn:f16 at 82]
 *   out[o] = d * (float)sc * (float)q - mn
 * 2-bit quants: qs gives 2 bits, sign bit from another byte.
 */
static void sse_block_q2_k(const u8 *data, float *out) {
    float d  = f16_to_f32(*(const u16 *)(data + 80));
    float mn = f16_to_f32(*(const u16 *)(data + 82));
    const u8 *scales = data + 64;

    for (int sb = 0; sb < 16; sb++) {
        u32 sc = scales[sb >> 1];
        if (sb & 1) sc >>= 4; else sc &= 0xF;

        __m128 vscale = _mm_set1_ps(d * (float)(i32)sc);
        __m128 vmn    = _mm_set1_ps(mn);

        u32 start = (u32)sb * 16;
        u8 vals[16];
        for (int j = 0; j < 16; j++) {
            u32 o   = start + (u32)j;
            u32 q2  = (data[(o >> 2)] >> ((o & 3) << 1)) & 0x3;
            u32 sign = (data[(o >> 3)] >> (o & 7)) & 1;
            u32 q   = q2 - sign * 4;
            vals[j]  = (u8)(i32)q;
        }

        __m128i vs8 = _mm_loadu_si128((const __m128i *)vals);
        __m128 r0, r1, r2, r3;
        __m128 neg_vmn = _mm_sub_ps(_mm_setzero_ps(), vmn);
        sse_s8_to_f32_x16(vs8, vscale, neg_vmn, &r0, &r1, &r2, &r3);
        sse_store_f32_x16(out + start, r0, r1, r2, r3);
    }
}

/* ================================================================
 * IQ block types — scalar block dequant
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
    [GGUF_TYPE_F32]      = NULL,
    [GGUF_TYPE_F16]      = NULL,
    [GGUF_TYPE_Q4_0]     = sse_block_q4_0,
    [GGUF_TYPE_Q4_1]     = sse_block_q4_1,
    [GGUF_TYPE_Q5_0]     = sse_block_q5_0,
    [GGUF_TYPE_Q5_1]     = sse_block_q5_1,
    [GGUF_TYPE_Q8_0]     = sse_block_q8_0,
    [GGUF_TYPE_Q8_1]     = sse_block_q8_1,
    [GGUF_TYPE_Q2_K]     = sse_block_q2_k,
    [GGUF_TYPE_Q3_K]     = sse_block_q3_k,
    [GGUF_TYPE_Q4_K]     = sse_block_q4_k,
    [GGUF_TYPE_Q5_K]     = sse_block_q5_k,
    [GGUF_TYPE_Q6_K]     = sse_block_q6_k,
    [GGUF_TYPE_Q8_K]     = sse_block_q8_k,
    [GGUF_TYPE_IQ2_XXS]  = scalar_block_iq2_xxs,
    [GGUF_TYPE_IQ2_XS]   = scalar_block_iq2_xs,
    [GGUF_TYPE_IQ3_XXS]  = scalar_block_iq3_xxs,
    [GGUF_TYPE_IQ1_S]    = scalar_block_iq1_s,
    [GGUF_TYPE_IQ4_NL]   = scalar_block_iq4_nl,
    [GGUF_TYPE_IQ3_S]    = scalar_block_iq3_s,
    [GGUF_TYPE_IQ2_S]    = scalar_block_iq2_s,
    [GGUF_TYPE_IQ4_XS]   = scalar_block_iq4_xs,
    [GGUF_TYPE_I8]       = NULL,
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

void gguf_dequant_batch(TensorInfo *ti, const u8 *base, u64 i0, u64 nb, float *out) {
    const u8 *data = base + ti->offset;
    u32 type = ti->type;

    /* ---- Non-block / simple conversion types: direct SSE path ---- */
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

    u64 end       = i0 + nb;
    u64 bi0       = i0 / block_elems;
    u64 bi1       = (end + block_elems - 1) / block_elems;

    for (u64 bi = bi0; fn && bi < bi1; bi++) {
        u64 bs = bi * block_elems;
        u64 be = bs + block_elems;
        u64 cs = i0 > bs ? i0 : bs;
        u64 ce = end < be ? end : be;

        if (cs == bs && ce == be) {
            fn(data + bi * block_bytes, out + (cs - i0));
        } else {
            for (u64 j = cs; j < ce; j++)
                out[j - i0] = gguf_dequant(ti, base, j);
        }
    }

    if (fn) return;

fallback:
    for (u64 j = i0; j < i0 + nb; j++)
        out[j - i0] = gguf_dequant(ti, base, j);
}

/* ================================================================
 * Fused dequant + dot product  (SSE SIMD)
 * ================================================================ */

/* ---- Non-block types ------------------------------------------- */

static float dot_f32(const u8 *data, const float *x, u64 n) {
    const float *w = (const float *)data;
    __m128 s0 = _mm_setzero_ps(), s1 = _mm_setzero_ps();
    __m128 s2 = _mm_setzero_ps(), s3 = _mm_setzero_ps();
    u64 i = 0;
    for (; i + 16 <= n; i += 16) {
        s0 = _mm_add_ps(s0, _mm_mul_ps(_mm_loadu_ps(w + i +  0), _mm_loadu_ps(x + i +  0)));
        s1 = _mm_add_ps(s1, _mm_mul_ps(_mm_loadu_ps(w + i +  4), _mm_loadu_ps(x + i +  4)));
        s2 = _mm_add_ps(s2, _mm_mul_ps(_mm_loadu_ps(w + i +  8), _mm_loadu_ps(x + i +  8)));
        s3 = _mm_add_ps(s3, _mm_mul_ps(_mm_loadu_ps(w + i + 12), _mm_loadu_ps(x + i + 12)));
    }
    float sum = sse_hsum_f32x4(_mm_add_ps(_mm_add_ps(s0, s1), _mm_add_ps(s2, s3)));
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
static float sse_dot_q8_0_block(const u8 *data, const float *x) {
    float d = f16_to_f32(*(const u16 *)data);
    const i8 *qs = (const i8 *)(data + 2);

    __m128 s0 = _mm_setzero_ps(), s1 = _mm_setzero_ps();
    __m128 s2 = _mm_setzero_ps(), s3 = _mm_setzero_ps();

    for (int j = 0; j < 32; j += 16) {
        __m128i vq = _mm_loadu_si128((const __m128i *)(qs + j));
        __m128i v16_0 = _mm_cvtepi8_epi16(vq);
        __m128i v16_1 = _mm_cvtepi8_epi16(_mm_srli_si128(vq, 8));
        __m128 qf0 = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(v16_0));
        __m128 qf1 = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_srli_si128(v16_0, 8)));
        __m128 qf2 = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(v16_1));
        __m128 qf3 = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_srli_si128(v16_1, 8)));

        s0 = _mm_add_ps(s0, _mm_mul_ps(qf0, _mm_loadu_ps(x + j +  0)));
        s1 = _mm_add_ps(s1, _mm_mul_ps(qf1, _mm_loadu_ps(x + j +  4)));
        s2 = _mm_add_ps(s2, _mm_mul_ps(qf2, _mm_loadu_ps(x + j +  8)));
        s3 = _mm_add_ps(s3, _mm_mul_ps(qf3, _mm_loadu_ps(x + j + 12)));
    }
    return d * sse_hsum_f32x4(_mm_add_ps(_mm_add_ps(s0, s1), _mm_add_ps(s2, s3)));
}

/* Q4_0: sum = d * Σ((nib - 8) * x) = d * (Σ(nib*x) - 8*Σ(x)) */
static float sse_dot_q4_0_block(const u8 *data, const float *x) {
    float d = f16_to_f32(*(const u16 *)data);
    const u8 *qs = data + 2;

    __m128 dot0 = _mm_setzero_ps(), dot1 = _mm_setzero_ps();
    __m128 dot2 = _mm_setzero_ps(), dot3 = _mm_setzero_ps();

    for (int j = 0; j < 32; j += 16) {
        __m128i vq  = _mm_loadu_si128((const __m128i *)(qs + j / 2));
        __m128i vlo = sse_lo_nibbles(vq);
        __m128i vhi = sse_hi_nibbles(vq);
        __m128i z0 = _mm_unpacklo_epi8(vlo, vhi);
        __m128i z1 = _mm_unpackhi_epi8(vlo, vhi);

        for (int h = 0; h < 2; h++) {
            __m128i vn = (h == 0) ? z0 : z1;
            int jj = j + h * 16;

            __m128i v16_0 = _mm_cvtepu8_epi16(vn);
            __m128i v16_1 = _mm_cvtepu8_epi16(_mm_srli_si128(vn, 8));
            __m128 nf0 = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(v16_0));
            __m128 nf1 = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_srli_si128(v16_0, 8)));
            __m128 nf2 = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(v16_1));
            __m128 nf3 = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_srli_si128(v16_1, 8)));

            __m128 xv0 = _mm_loadu_ps(x + jj +  0);
            __m128 xv1 = _mm_loadu_ps(x + jj +  4);
            __m128 xv2 = _mm_loadu_ps(x + jj +  8);
            __m128 xv3 = _mm_loadu_ps(x + jj + 12);

            dot0 = _mm_add_ps(dot0, _mm_mul_ps(nf0, xv0));
            dot1 = _mm_add_ps(dot1, _mm_mul_ps(nf1, xv1));
            dot2 = _mm_add_ps(dot2, _mm_mul_ps(nf2, xv2));
            dot3 = _mm_add_ps(dot3, _mm_mul_ps(nf3, xv3));
        }
    }

    __m128 sx0 = _mm_setzero_ps(), sx1 = _mm_setzero_ps();
    __m128 sx2 = _mm_setzero_ps(), sx3 = _mm_setzero_ps();
    for (int j = 0; j < 32; j += 16) {
        sx0 = _mm_add_ps(sx0, _mm_loadu_ps(x + j +  0));
        sx1 = _mm_add_ps(sx1, _mm_loadu_ps(x + j +  4));
        sx2 = _mm_add_ps(sx2, _mm_loadu_ps(x + j +  8));
        sx3 = _mm_add_ps(sx3, _mm_loadu_ps(x + j + 12));
    }
    float sum_x = sse_hsum_f32x4(_mm_add_ps(_mm_add_ps(sx0, sx1), _mm_add_ps(sx2, sx3)));

    float dot_all = sse_hsum_f32x4(_mm_add_ps(_mm_add_ps(dot0, dot1), _mm_add_ps(dot2, dot3)));
    return d * (dot_all - 8.0f * sum_x);
}

/* Q4_1: sum = Σ((nib * d + m) * x) = d * Σ(nib*x) + m * Σ(x) */
static float sse_dot_q4_1_block(const u8 *data, const float *x) {
    float d = f16_to_f32(*(const u16 *)data);
    float m = f16_to_f32(*(const u16 *)(data + 2));
    const u8 *qs = data + 4;

    __m128 dot0 = _mm_setzero_ps(), dot1 = _mm_setzero_ps();
    __m128 dot2 = _mm_setzero_ps(), dot3 = _mm_setzero_ps();
    __m128 sx0 = _mm_setzero_ps(), sx1 = _mm_setzero_ps();
    __m128 sx2 = _mm_setzero_ps(), sx3 = _mm_setzero_ps();

    for (int j = 0; j < 32; j += 16) {
        __m128i vq  = _mm_loadu_si128((const __m128i *)(qs + j / 2));
        __m128i vlo = sse_lo_nibbles(vq);
        __m128i vhi = sse_hi_nibbles(vq);
        __m128i z0 = _mm_unpacklo_epi8(vlo, vhi);
        __m128i z1 = _mm_unpackhi_epi8(vlo, vhi);

        for (int h = 0; h < 2; h++) {
            __m128i vn = (h == 0) ? z0 : z1;
            int jj = j + h * 16;

            __m128i v16_0 = _mm_cvtepu8_epi16(vn);
            __m128i v16_1 = _mm_cvtepu8_epi16(_mm_srli_si128(vn, 8));
            __m128 nf0 = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(v16_0));
            __m128 nf1 = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_srli_si128(v16_0, 8)));
            __m128 nf2 = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(v16_1));
            __m128 nf3 = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_srli_si128(v16_1, 8)));

            __m128 xv0 = _mm_loadu_ps(x + jj +  0);
            __m128 xv1 = _mm_loadu_ps(x + jj +  4);
            __m128 xv2 = _mm_loadu_ps(x + jj +  8);
            __m128 xv3 = _mm_loadu_ps(x + jj + 12);

            dot0 = _mm_add_ps(dot0, _mm_mul_ps(nf0, xv0));
            dot1 = _mm_add_ps(dot1, _mm_mul_ps(nf1, xv1));
            dot2 = _mm_add_ps(dot2, _mm_mul_ps(nf2, xv2));
            dot3 = _mm_add_ps(dot3, _mm_mul_ps(nf3, xv3));
            sx0 = _mm_add_ps(sx0, xv0); sx1 = _mm_add_ps(sx1, xv1);
            sx2 = _mm_add_ps(sx2, xv2); sx3 = _mm_add_ps(sx3, xv3);
        }
    }
    float dot_nib = sse_hsum_f32x4(_mm_add_ps(_mm_add_ps(dot0, dot1), _mm_add_ps(dot2, dot3)));
    float sum_x   = sse_hsum_f32x4(_mm_add_ps(_mm_add_ps(sx0, sx1), _mm_add_ps(sx2, sx3)));
    return d * dot_nib + m * sum_x;
}

/* ---- K-quant types  (block_size = 256) ------------------------- */

/* Q8_K: sum = d * Σ(bs[sb] * Σ(qs[j] * x[j])) */
static float sse_dot_q8_k_block(const u8 *data, const float *x) {
    float d; memcpy(&d, data, 4);
    const u16 *bs = (const u16 *)(data + 4);
    const i8  *qs = (const i8 *)(data + 36);

    float acc = 0.0f;
    for (int sb = 0; sb < 16; sb++) {
        __m128 s0 = _mm_setzero_ps(), s1 = _mm_setzero_ps();
        __m128 s2 = _mm_setzero_ps(), s3 = _mm_setzero_ps();
        int off = sb * 16;

        __m128i vq = _mm_loadu_si128((const __m128i *)(qs + off));
        __m128i v16_0 = _mm_cvtepi8_epi16(vq);
        __m128i v16_1 = _mm_cvtepi8_epi16(_mm_srli_si128(vq, 8));
        __m128 qf0 = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(v16_0));
        __m128 qf1 = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_srli_si128(v16_0, 8)));
        __m128 qf2 = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(v16_1));
        __m128 qf3 = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_srli_si128(v16_1, 8)));

        s0 = _mm_add_ps(s0, _mm_mul_ps(qf0, _mm_loadu_ps(x + off +  0)));
        s1 = _mm_add_ps(s1, _mm_mul_ps(qf1, _mm_loadu_ps(x + off +  4)));
        s2 = _mm_add_ps(s2, _mm_mul_ps(qf2, _mm_loadu_ps(x + off +  8)));
        s3 = _mm_add_ps(s3, _mm_mul_ps(qf3, _mm_loadu_ps(x + off + 12)));

        float dot_qs = sse_hsum_f32x4(_mm_add_ps(_mm_add_ps(s0, s1), _mm_add_ps(s2, s3)));
        acc += dot_qs * f16_to_f32(bs[sb]);
    }
    return d * acc;
}

/* Q6_K: sum = d * Σ(sc[sb] * Σ((q - 32) * x)) */
static float sse_dot_q6_k_block(const u8 *data, const float *x) {
    float d = f16_to_f32(*(const u16 *)(data + 208));
    const u8 *scales = data + 192;
    const u8 *ql     = data;
    const u8 *qh     = data + 128;

    float acc = 0.0f;
    for (int sb = 0; sb < 16; sb++) {
        i32 sc = (i8)scales[sb];
        u32 start = (u32)sb * 16;

        u8 vals[16];
        for (int j = 0; j < 16; j++) {
            u32 o     = start + (u32)j;
            u32 half  = o >> 7;
            u32 hl    = o & 127;
            u32 which = hl >> 5;
            u32 l     = hl & 31;
            u32 ql_off = (half << 6) + ((which & 1) << 5) + l;
            u32 lo = (which >= 2) ? ((ql[ql_off] >> 4) & 0xF) : (ql[ql_off] & 0xF);
            u32 qh_off = (half << 5) + l;
            u32 hi = (qh[qh_off] >> (which * 2)) & 0x3;
            vals[j] = (u8)(lo | (hi << 4));
        }

        __m128 dot0 = _mm_setzero_ps(), dot1 = _mm_setzero_ps();
        __m128 dot2 = _mm_setzero_ps(), dot3 = _mm_setzero_ps();
        __m128 sx0  = _mm_setzero_ps(), sx1  = _mm_setzero_ps();

        __m128i vu8 = _mm_loadu_si128((const __m128i *)vals);
        __m128i v16_0 = _mm_cvtepu8_epi16(vu8);
        __m128i v16_1 = _mm_cvtepu8_epi16(_mm_srli_si128(vu8, 8));
        __m128 vf0 = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(v16_0));
        __m128 vf1 = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_srli_si128(v16_0, 8)));
        __m128 vf2 = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(v16_1));
        __m128 vf3 = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_srli_si128(v16_1, 8)));

        __m128 xv0 = _mm_loadu_ps(x + start +  0);
        __m128 xv1 = _mm_loadu_ps(x + start +  4);
        __m128 xv2 = _mm_loadu_ps(x + start +  8);
        __m128 xv3 = _mm_loadu_ps(x + start + 12);

        dot0 = _mm_add_ps(dot0, _mm_mul_ps(vf0, xv0));
        dot1 = _mm_add_ps(dot1, _mm_mul_ps(vf1, xv1));
        dot2 = _mm_add_ps(dot2, _mm_mul_ps(vf2, xv2));
        dot3 = _mm_add_ps(dot3, _mm_mul_ps(vf3, xv3));
        sx0 = _mm_add_ps(sx0, xv0); sx1 = _mm_add_ps(sx1, xv1);

        __m128 sx_all = _mm_add_ps(_mm_add_ps(_mm_add_ps(sx0, sx1), xv2), xv3);

        float dot_q = sse_hsum_f32x4(_mm_add_ps(_mm_add_ps(dot0, dot1), _mm_add_ps(dot2, dot3)));
        float sum_x = sse_hsum_f32x4(sx_all);

        acc += (float)sc * (dot_q - 32.0f * sum_x);
    }
    return d * acc;
}

/* Q4_K: result = d * Σ(sc[sb] * Σ(nib * x)) - dmin * Σ(mn[sb] * Σ(x)) */
static float sse_dot_q4_k_block(const u8 *data, const float *x) {
    float d    = f16_to_f32(*(const u16 *)data);
    float dmin = f16_to_f32(*(const u16 *)(data + 2));
    const u8 *sm = data + 4;
    const u8 *qs = data + 16;

    float dot_acc = 0.0f, sum_acc = 0.0f;

    for (int sb = 0; sb < 8; sb++) {
        u8 sc, mn;
        q4k_scale_min(sm, (u32)sb, &sc, &mn);

        u32 g  = (u32)sb >> 1;
        u32 nb = (u32)sb & 1;
        const u8 *src = qs + g * 32;
        u32 off = (u32)sb * 32;

        __m128 dot0 = _mm_setzero_ps(), dot1 = _mm_setzero_ps();
        __m128 dot2 = _mm_setzero_ps(), dot3 = _mm_setzero_ps();
        __m128 sx0  = _mm_setzero_ps(), sx1  = _mm_setzero_ps();
        __m128 sx2  = _mm_setzero_ps(), sx3  = _mm_setzero_ps();

        for (int j = 0; j < 32; j += 16) {
            __m128i vbytes = _mm_loadu_si128((const __m128i *)(src + j));
            __m128i vnibs;
            if (nb == 0) vnibs = sse_lo_nibbles(vbytes);
            else         vnibs = sse_hi_nibbles(vbytes);

            __m128i v16_0 = _mm_cvtepu8_epi16(vnibs);
            __m128i v16_1 = _mm_cvtepu8_epi16(_mm_srli_si128(vnibs, 8));
            __m128 nf0 = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(v16_0));
            __m128 nf1 = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_srli_si128(v16_0, 8)));
            __m128 nf2 = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(v16_1));
            __m128 nf3 = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_srli_si128(v16_1, 8)));

            __m128 xv0 = _mm_loadu_ps(x + off + j +  0);
            __m128 xv1 = _mm_loadu_ps(x + off + j +  4);
            __m128 xv2 = _mm_loadu_ps(x + off + j +  8);
            __m128 xv3 = _mm_loadu_ps(x + off + j + 12);

            dot0 = _mm_add_ps(dot0, _mm_mul_ps(nf0, xv0));
            dot1 = _mm_add_ps(dot1, _mm_mul_ps(nf1, xv1));
            dot2 = _mm_add_ps(dot2, _mm_mul_ps(nf2, xv2));
            dot3 = _mm_add_ps(dot3, _mm_mul_ps(nf3, xv3));
            sx0 = _mm_add_ps(sx0, xv0); sx1 = _mm_add_ps(sx1, xv1);
            sx2 = _mm_add_ps(sx2, xv2); sx3 = _mm_add_ps(sx3, xv3);
        }

        float dot_nib = sse_hsum_f32x4(_mm_add_ps(_mm_add_ps(dot0, dot1), _mm_add_ps(dot2, dot3)));
        float sum_x   = sse_hsum_f32x4(_mm_add_ps(_mm_add_ps(sx0, sx1), _mm_add_ps(sx2, sx3)));

        dot_acc += (float)sc * dot_nib;
        sum_acc += (float)mn * sum_x;
    }
    return d * dot_acc - dmin * sum_acc;
}

/* Q5_K: result = d * Σ(sc[sb] * Σ(q * x)) - dmin * Σ(mn[sb] * Σ(x)) */
static float sse_dot_q5_k_block(const u8 *data, const float *x) {
    float d    = f16_to_f32(*(const u16 *)data);
    float dmin = f16_to_f32(*(const u16 *)(data + 2));
    const u8 *sm = data + 4;
    const u8 *qh = data + 16;
    const u8 *qs = data + 48;

    float dot_acc = 0.0f, sum_acc = 0.0f;

    for (int sb = 0; sb < 8; sb++) {
        u8 sc, mn;
        q4k_scale_min(sm, (u32)sb, &sc, &mn);

        u32 g  = (u32)sb >> 1;
        u32 nb = (u32)sb & 1;
        const u8 *src_lo = qs + g * 32;
        u32 start = (u32)sb * 32;

        u8 vals[32];
        for (int j = 0; j < 32; j++) {
            u32 o  = start + (u32)j;
            u32 hi = (qh[o & 31] >> (o >> 5)) & 1;
            u32 lo = nb ? (src_lo[j] >> 4) : (src_lo[j] & 0xF);
            vals[j] = (u8)(lo | (hi << 4));
        }

        __m128 dot0 = _mm_setzero_ps(), dot1 = _mm_setzero_ps();
        __m128 dot2 = _mm_setzero_ps(), dot3 = _mm_setzero_ps();
        __m128 sx0  = _mm_setzero_ps(), sx1  = _mm_setzero_ps();
        __m128 sx2  = _mm_setzero_ps(), sx3  = _mm_setzero_ps();

        for (int j = 0; j < 32; j += 16) {
            __m128i vu8 = _mm_loadu_si128((const __m128i *)(vals + j));
            __m128i v16_0 = _mm_cvtepu8_epi16(vu8);
            __m128i v16_1 = _mm_cvtepu8_epi16(_mm_srli_si128(vu8, 8));
            __m128 vf0 = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(v16_0));
            __m128 vf1 = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_srli_si128(v16_0, 8)));
            __m128 vf2 = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(v16_1));
            __m128 vf3 = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_srli_si128(v16_1, 8)));

            __m128 xv0 = _mm_loadu_ps(x + start + j +  0);
            __m128 xv1 = _mm_loadu_ps(x + start + j +  4);
            __m128 xv2 = _mm_loadu_ps(x + start + j +  8);
            __m128 xv3 = _mm_loadu_ps(x + start + j + 12);

            dot0 = _mm_add_ps(dot0, _mm_mul_ps(vf0, xv0));
            dot1 = _mm_add_ps(dot1, _mm_mul_ps(vf1, xv1));
            dot2 = _mm_add_ps(dot2, _mm_mul_ps(vf2, xv2));
            dot3 = _mm_add_ps(dot3, _mm_mul_ps(vf3, xv3));
            sx0 = _mm_add_ps(sx0, xv0); sx1 = _mm_add_ps(sx1, xv1);
            sx2 = _mm_add_ps(sx2, xv2); sx3 = _mm_add_ps(sx3, xv3);
        }

        float dot_q = sse_hsum_f32x4(_mm_add_ps(_mm_add_ps(dot0, dot1), _mm_add_ps(dot2, dot3)));
        float sum_x = sse_hsum_f32x4(_mm_add_ps(_mm_add_ps(sx0, sx1), _mm_add_ps(sx2, sx3)));

        dot_acc += (float)sc * dot_q;
        sum_acc += (float)mn * sum_x;
    }
    return d * dot_acc - dmin * sum_acc;
}

/* Q3_K: sum = d * Σ(sc[sb] * Σ((q - 4) * x)) */
static float sse_dot_q3_k_block(const u8 *data, const float *x) {
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
        u8 vals[16];
        for (int j = 0; j < 16; j++) {
            u32 o   = start + (u32)j;
            u32 g   = o >> 7;
            u32 pr  = (o >> 5) & 3;
            u32 bc  = o & 31;
            u32 lo  = (qs[g * 32 + bc] >> (pr * 2)) & 0x3;
            u32 hi  = (qh[o & 31] >> (o >> 5)) & 1;
            vals[j]  = (u8)(lo | (hi << 2));
        }

        __m128i vu8 = _mm_loadu_si128((const __m128i *)vals);
        __m128i v16_0 = _mm_cvtepu8_epi16(vu8);
        __m128i v16_1 = _mm_cvtepu8_epi16(_mm_srli_si128(vu8, 8));
        __m128 vf0 = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(v16_0));
        __m128 vf1 = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_srli_si128(v16_0, 8)));
        __m128 vf2 = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(v16_1));
        __m128 vf3 = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_srli_si128(v16_1, 8)));

        __m128 xv0 = _mm_loadu_ps(x + start +  0);
        __m128 xv1 = _mm_loadu_ps(x + start +  4);
        __m128 xv2 = _mm_loadu_ps(x + start +  8);
        __m128 xv3 = _mm_loadu_ps(x + start + 12);

        __m128 dot0 = _mm_mul_ps(vf0, xv0), dot1 = _mm_mul_ps(vf1, xv1);
        __m128 dot2 = _mm_mul_ps(vf2, xv2), dot3 = _mm_mul_ps(vf3, xv3);
        float dot_q = sse_hsum_f32x4(_mm_add_ps(_mm_add_ps(dot0, dot1), _mm_add_ps(dot2, dot3)));
        float sum_x = sse_hsum_f32x4(_mm_add_ps(_mm_add_ps(xv0, xv1), _mm_add_ps(xv2, xv3)));

        acc += (float)sc * (dot_q - 4.0f * sum_x);
    }
    return d * acc;
}

/* Q2_K: sum = d * Σ(sc[sb] * Σ(q * x)) - mn * Σ(x) */
static float sse_dot_q2_k_block(const u8 *data, const float *x) {
    float d  = f16_to_f32(*(const u16 *)(data + 80));
    float mn = f16_to_f32(*(const u16 *)(data + 82));
    const u8 *scales = data + 64;

    float dot_acc = 0.0f;
    __m128 total_sx0 = _mm_setzero_ps(), total_sx1 = _mm_setzero_ps();
    __m128 total_sx2 = _mm_setzero_ps(), total_sx3 = _mm_setzero_ps();

    for (int sb = 0; sb < 16; sb++) {
        u32 sc = scales[sb >> 1];
        if (sb & 1) sc >>= 4; else sc &= 0xF;

        u32 start = (u32)sb * 16;
        i8 vals[16];
        for (int j = 0; j < 16; j++) {
            u32 o   = start + (u32)j;
            u32 q2  = (data[(o >> 2)] >> ((o & 3) << 1)) & 0x3;
            u32 sign = (data[(o >> 3)] >> (o & 7)) & 1;
            vals[j] = (i8)((i32)q2 - (i32)sign * 4);
        }

        __m128i vs8  = _mm_loadu_si128((const __m128i *)vals);
        __m128i v16_0 = _mm_cvtepi8_epi16(vs8);
        __m128i v16_1 = _mm_cvtepi8_epi16(_mm_srli_si128(vs8, 8));
        __m128 vf0 = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(v16_0));
        __m128 vf1 = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_srli_si128(v16_0, 8)));
        __m128 vf2 = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(v16_1));
        __m128 vf3 = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_srli_si128(v16_1, 8)));

        __m128 xv0 = _mm_loadu_ps(x + start +  0);
        __m128 xv1 = _mm_loadu_ps(x + start +  4);
        __m128 xv2 = _mm_loadu_ps(x + start +  8);
        __m128 xv3 = _mm_loadu_ps(x + start + 12);

        __m128 d0 = _mm_mul_ps(vf0, xv0), d1 = _mm_mul_ps(vf1, xv1);
        __m128 d2 = _mm_mul_ps(vf2, xv2), d3 = _mm_mul_ps(vf3, xv3);

        dot_acc += (float)(i32)sc * sse_hsum_f32x4(
            _mm_add_ps(_mm_add_ps(d0, d1), _mm_add_ps(d2, d3)));

        total_sx0 = _mm_add_ps(total_sx0, xv0); total_sx1 = _mm_add_ps(total_sx1, xv1);
        total_sx2 = _mm_add_ps(total_sx2, xv2); total_sx3 = _mm_add_ps(total_sx3, xv3);
    }

    float total_sum_x = sse_hsum_f32x4(
        _mm_add_ps(_mm_add_ps(total_sx0, total_sx1), _mm_add_ps(total_sx2, total_sx3)));
    return d * dot_acc - mn * total_sum_x;
}

/* ================================================================
 * Public API
 * ================================================================ */

typedef float (*block_dot_fn)(const u8 *data, const float *x);

float gguf_dot_batch(TensorInfo *ti, const u8 *base, u64 i, u64 n, const float *x) {
    const u8 *data = base + ti->offset;
    u32 type = ti->type;

    /* ---- Non-block types: direct SSE dot ---- */
    switch (type) {
    case GGUF_TYPE_F32: return dot_f32(data + i * 4, x, n);
    case GGUF_TYPE_F16: return dot_f16(data + i * 2, x, n);
    default: break;
    }

    /* ---- Block-quantised types ---- */
    u32 block_elems;
    u32 block_bytes;
    block_dot_fn dot_fn = NULL;

    if (type <= GGUF_TYPE_Q8_1) {
        block_elems = BLOCK_SIMPLE;
        switch (type) {
            case GGUF_TYPE_Q4_0: block_bytes = 18; dot_fn = sse_dot_q4_0_block; break;
            case GGUF_TYPE_Q4_1: block_bytes = 20; dot_fn = sse_dot_q4_1_block; break;
            case GGUF_TYPE_Q5_0: block_bytes = 22; break;
            case GGUF_TYPE_Q5_1: block_bytes = 24; break;
            case GGUF_TYPE_Q8_0: block_bytes = 34; dot_fn = sse_dot_q8_0_block; break;
            case GGUF_TYPE_Q8_1: block_bytes = 40; break;
            default: goto fallback_dot;
        }
    } else {
        block_elems = BLOCK_KQUANT;
        switch (type) {
            case GGUF_TYPE_Q2_K:     block_bytes =  84; dot_fn = sse_dot_q2_k_block; break;
            case GGUF_TYPE_Q3_K:     block_bytes = 110; dot_fn = sse_dot_q3_k_block; break;
            case GGUF_TYPE_Q4_K:     block_bytes = 144; dot_fn = sse_dot_q4_k_block; break;
            case GGUF_TYPE_Q5_K:     block_bytes = 176; dot_fn = sse_dot_q5_k_block; break;
            case GGUF_TYPE_Q6_K:     block_bytes = 210; dot_fn = sse_dot_q6_k_block; break;
            case GGUF_TYPE_Q8_K:     block_bytes = 292; dot_fn = sse_dot_q8_k_block; break;
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
            result += dot_fn(data + bi * block_bytes, x + (cs - i));
        } else {
            for (u64 j = cs; j < ce; j++)
                result += gguf_dequant(ti, base, j) * x[j - i];
        }
    }

    if (dot_fn) return result;

fallback_dot:
    for (u64 j = i; j < i + n; j++)
        result += gguf_dequant(ti, base, j) * x[j - i];
    return result;
}

/* ================================================================
 * i8 dot-product path — not available on x86 (no vdotq_s32).
 * Stubs ensure callers (guarded by __ARM_FEATURE_DOTPROD) fall
 * through to gguf_dot_batch on x86.
 * ================================================================ */

float quantize_f32_to_i8(const float *x, i8 *out, u64 n) {
    (void)x; (void)out; (void)n;
    return 0.0f;
}

float gguf_dot_i8_batch(TensorInfo *ti, const u8 *base, u64 i, u64 n,
                         const i8 *x_i8, float x_scale) {
    (void)ti; (void)base; (void)i; (void)n; (void)x_i8; (void)x_scale;
    return 0.0f;
}
