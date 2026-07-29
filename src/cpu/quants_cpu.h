#ifndef __QUANTS_CPU_H__
#define __QUANTS_CPU_H__

#include "../def.h"

/*
 * Batch dequantize nb elements starting at index i0 from tensor ti.
 * out must have room for nb floats.
 *
 * Full blocks are processed with NEON SIMD; partial edges fall back
 * to the scalar gguf_dequant().
 */
void gguf_dequant_batch(TensorInfo *ti, const u8 *base, u64 i0, u64 nb, float *out);

/*
 * Fused dequant + dot product: returns sum_{j=0}^{n-1} w[i+j] * x[j].
 * Full blocks use NEON SIMD that dequantises and multiplies in
 * registers without writing to an intermediate buffer.
 * Edge elements fall back to the scalar gguf_dequant().
 */
float gguf_dot_batch(TensorInfo *ti, const u8 *base, u64 i, u64 n, const float *x);

/*
 * vdotq_s32 i8 dot-product path (ARMv8.2+ dotprod).
 *
 * Quantise x to i8 once, then reuse across all rows.  Each dot
 * product uses vdotq_s32: one instruction for 4 i8×i8→i32 MACs
 * instead of the 5-instruction widening chain.
 */
float quantize_f32_to_i8(const float *x, i8 *out, u64 n);
float gguf_dot_i8_batch(TensorInfo *ti, const u8 *base, u64 i, u64 n, const i8 *x_i8, float x_scale);

#endif
