#ifndef __QUANTS_GPU_H__
#define __QUANTS_GPU_H__

#include "../def.h"

/*
 * GPU (CUDA) batch dequantization backend.
 *
 * Mirrors the CPU API in cpu/quants_cpu.h so the two backends are
 * drop-in interchangeable:
 *   gpu_dequant_batch  — dequantize nb elements starting at i0 into out
 *   gpu_dequant_tensor — dequantize an entire tensor into out
 *   gpu_dot_batch      — fused dequant + dot product
 *
 * When no usable CUDA device is present every wrapper falls back to the
 * scalar gguf_dequant() CPU path, so callers may use the same API
 * unconditionally.
 */

#ifdef __cplusplus
extern "C" {
#endif

/* Returns nonzero if a usable CUDA device is present. */
int gpu_available(void);

/* Dequantize nb elements starting at i0 from tensor ti into out[0..nb-1].
 * out must have room for nb floats. Returns 0 on success, -1 on error. */
int gpu_dequant_batch(TensorInfo *ti, const u8 *base, u64 i0, u64 nb, float *out);

/* Dequantize the whole tensor. out must hold ti->n_element floats. */
int gpu_dequant_tensor(TensorInfo *ti, const u8 *base, float *out);

/* Fused dequant + dot product: sum_{j=0}^{n-1} w[i+j] * x[j]. */
float gpu_dot_batch(TensorInfo *ti, const u8 *base, u64 i, u64 n, const float *x);

#ifdef __cplusplus
}
#endif

#endif
