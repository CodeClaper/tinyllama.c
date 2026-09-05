/*
 * cpu.c — Architecture dispatcher for SIMD batch dequantisation
 * and fused dot-product.
 *
 * ARM NEON (aarch64):  includes cpu_arm.c
 * x86 (x86_64/i386):   includes cpu_x86.c (SSE4.1 SIMD)
 */
#include "../../def.h"
#include "../../quants.h"
#include "cpu.h"

#if defined(__aarch64__)

#  include "cpu_arm.c"

#elif defined(__x86_64__) || defined(__i386__)

#  include "cpu_x86.c"

#else
#  error "Unsupported architecture: only aarch64 and x86_64/i386 are supported"
#endif
