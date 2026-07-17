/*
 * quants_cpu.c — Architecture dispatcher for SIMD batch dequantisation
 * and fused dot-product.
 *
 * ARM NEON (aarch64):  includes quants_arm.c
 * x86 (x86_64/i386):   includes quants_x86.c (SSE4.1 SIMD)
 */
#include "../def.h"
#include "../quants.h"
#include "quants_cpu.h"

#if defined(__aarch64__)

#  include "quants_arm.c"

#elif defined(__x86_64__) || defined(__i386__)

#  include "quants_x86.c"

#else
#  error "Unsupported architecture: only aarch64 and x86_64/i386 are supported"
#endif
