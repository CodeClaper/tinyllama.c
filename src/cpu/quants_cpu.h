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

#endif
