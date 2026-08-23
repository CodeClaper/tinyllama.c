#ifndef __QUANTS_GPU_H__
#define __QUANTS_GPU_H__

#include <stdbool.h>
#include <stdio.h>  /* CHECK uses printf() */
#include <stdlib.h> /* CHECK uses exit() */
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

/* ---- Persistent-weight matmul backend (decode hot path) ----
 *
 * Weight tensors are uploaded to device memory once on first use and
 * cached for the lifetime of the process; every call is one kernel
 * launch + a small x/y round trip.  All index math matches the CPU
 * scalar reference (flat element indices, no per-row padding), so
 * results are bit-comparable to gguf_dequant().
 *
 * A CUDA error at any point is fatal: CHECK() prints the failing call
 * site and exits(1).  Only non-error conditions are non-fatal — no
 * usable device, unsupported quant type, or a full cache return
 * nonzero and leave y/Y untouched so the caller falls back to the
 * CPU path. */
#define GPU_MAT_MIN_OPS 65536u /* below this many MACs, stay on CPU */
#define CHECK(call)                                                         \
{                                                                           \
    const cudaError_t error = call;                                         \
    if (error != cudaSuccess)                                               \
    {                                                                       \
        printf("Error: %s: %d", __FILE__, __LINE__);                        \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1);                                                            \
    }                                                                       \
}                                                                           \

/* Returns nonzero if a usable CUDA device is present. */
int gpu_available(void);

/* Dequantize nb elements starting at i0 from tensor ti into out[0..nb-1].
 * out must have room for nb floats. Returns 0 on success, -1 on error. */
int gpu_dequant_batch(TensorInfo *ti, const u8 *base, u64 i0, u64 nb, float *out);

/* Dequantize the whole tensor. out must hold ti->n_element floats. */
int gpu_dequant_tensor(TensorInfo *ti, const u8 *base, float *out);

/* Fused dequant + dot product: sum_{j=0}^{n-1} w[i+j] * x[j]. */
float gpu_dot_batch(TensorInfo *ti, const u8 *base, u64 i, u64 n, const float *x);

/* y = W @ x (trans=false) or y = W^T @ x (trans=true), W = tensor ti.
 * rows/cols must match ti's dims (mat_vec_mul's validation). */
int gpu_matvec(TensorInfo *ti, const u8 *base, const float *x, float *y, u64 rows, u64 cols, bool trans);

/* Y[b] = W @ X[b] for b in [0,batch).  X is [batch x cols], Y is
 * [batch x rows].  A single launch covers all batches. */
int gpu_matmat(TensorInfo *ti, const u8 *base, const float *X, float *Y, u64 batch, u64 rows, u64 cols, bool trans);

/* Free all cached device weight buffers.  Call once at teardown. */
void gpu_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif
