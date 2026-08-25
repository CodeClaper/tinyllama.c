#ifndef __QUANTS_METAL_H__
#define __QUANTS_METAL_H__

#include <stdbool.h>
#include "../def.h"

/*
 * Metal batch dequantization backend (macOS only).
 *
 * Mirrors the CPU API in cpu/quants_cpu.h and the CUDA backend in
 * gpu/quants_gpu.h so the backends are drop-in interchangeable:
 *   metal_dequant_batch  — dequantize nb elements starting at i0 into out
 *   metal_dequant_tensor — dequantize an entire tensor into out
 *   metal_dot_batch      — fused dequant + dot product
 *
 * When no usable Metal device is present every wrapper falls back to the
 * scalar gguf_dequant() CPU path, so callers may use the same API
 * unconditionally.
 */

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Persistent-weight matmul backend (decode hot path) ----
 *
 * Weight tensors are uploaded to device memory once on first use and
 * cached for the lifetime of the process; every call is one kernel
 * launch + a small x/y round trip.  All index math matches the CPU
 * scalar reference (flat element indices, no per-row padding), so
 * results are bit-comparable to gguf_dequant().
 *
 * A Metal error at any point is fatal: the error is printed and the
 * process exits (matching CHECK() in the CUDA backend).  Only
 * non-error conditions are non-fatal — no usable device, unsupported
 * quant type, or a full cache return nonzero and leave y/Y untouched
 * so the caller falls back to the CPU path. */
#define METAL_MAT_MIN_OPS 65536u /* below this many MACs, stay on CPU */

/* Returns nonzero if a usable Metal device is present. */
int metal_available(void);

/* Dequantize nb elements starting at i0 from tensor ti into out[0..nb-1].
 * out must have room for nb floats. Returns 0 on success, -1 on error. */
int metal_dequant_batch(TensorInfo *ti, const u8 *base, u64 i0, u64 nb, float *out);

/* Dequantize the whole tensor. out must hold ti->n_element floats. */
int metal_dequant_tensor(TensorInfo *ti, const u8 *base, float *out);

/* Fused dequant + dot product: sum_{j=0}^{n-1} w[i+j] * x[j]. */
float metal_dot_batch(TensorInfo *ti, const u8 *base, u64 i, u64 n, const float *x);

/* y = W @ x (trans=false) or y = W^T @ x (trans=true), W = tensor ti.
 * rows/cols must match ti's dims (mat_vec_mul's validation). */
int metal_matvec(TensorInfo *ti, const u8 *base, const float *x, float *y, u64 rows, u64 cols, bool trans);

/* Y[b] = W @ X[b] for b in [0,batch).  X is [batch x cols], Y is
 * [batch x rows].  A single launch covers all batches. */
int metal_matmat(TensorInfo *ti, const u8 *base, const float *X, float *Y, u64 batch, u64 rows, u64 cols, bool trans);

/* Free all cached device weight buffers.  Call once at teardown. */
void metal_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif
