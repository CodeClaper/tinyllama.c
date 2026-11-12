#ifndef __QUANTS_CPU_H__
#define __QUANTS_CPU_H__

#include "../../def.h"

/*
 * Batch dequantize nb elements starting at index i0 from tensor ti.
 * out must have room for nb floats.
 *
 * Full blocks are processed with NEON SIMD; partial edges fall back
 * to the scalar gguf_dequant().
 */
void gguf_dequant_batch(TensorInfo *ti, u64 i0, u64 nb, float *out);

/*
 * Fused dequant + dot product: returns sum_{j=0}^{n-1} w[i+j] * x[j].
 * Full blocks use NEON SIMD that dequantises and multiplies in
 * registers without writing to an intermediate buffer.
 * Edge elements fall back to the scalar gguf_dequant().
 */
float gguf_dot_batch(TensorInfo *ti, u64 i, u64 n, const float *x);

/*
 * i8 dot-product path (ARMv8.2+ dotprod, or x86 AVX2/VNNI).
 *
 * Quantise x to i8 once, then reuse across all rows.  Each dot
 * product does 4 i8×i8→i32 MACs per instruction instead of the
 * widening f32 chain.
 */
#if defined(__ARM_FEATURE_DOTPROD) || \
    ((defined(__x86_64__) || defined(__i386__)) && defined(__AVX2__) && defined(__FMA__))
#  define CPU_HAVE_I8_DOT 1
#else
#  define CPU_HAVE_I8_DOT 0
#endif

float quantize_f32_to_i8(const float *x, i8 *out, u64 n);
float gguf_dot_i8_batch(TensorInfo *ti, u64 i, u64 n, const i8 *x_i8, float x_scale);
bool cpu_i8_dot_supported(u32 type);

/*
 * Plain f32 matrix-matrix multiply: C = A @ B.
 * A is [M x K], B is [K x N], C is [M x N], all row-major.
 */
void mul_mat_mat(float *C, const float *A, const float *B, u64 M, u64 K, u64 N);

/*
 * Plain f32 matrix-vector multiply: y = A @ x.
 * A is [M x K] row-major, x is [K]; the result y is [M].
 */
void mul_mat_vec(float *y, const float *A, const float *x, u64 M, u64 K);

#endif
