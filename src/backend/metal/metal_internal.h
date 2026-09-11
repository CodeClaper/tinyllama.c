#ifndef __METAL_INTERNAL_H__
#define __METAL_INTERNAL_H__

/*
 * metal_internal.h — ObjC-only bridge between metal.m (device, weight
 * cache, embedded kernel library) and metal_op.m (device-resident graph
 * operators).  Not installed and not included by C translation units.
 *
 * metal.m owns the MTLDevice/queue/library, the concatenated IQ table
 * buffer, and the persistent raw-weight cache; metal_op.m owns the
 * graph arena registry and its per-op shadow buffers.  This header is
 * the narrow seam between the two.
 */

#import <Metal/Metal.h>
#include "../../def.h"

/* Ensure device/queue/library/tables/weight-cache are usable.  0 = ready. */
int metal_i_ready(void);

id<MTLDevice>               metal_i_dev(void);
id<MTLCommandQueue>         metal_i_queue(void);
id<MTLBuffer>               metal_i_tables(void);
id<MTLComputePipelineState> metal_i_pipe(NSString *name);

/* Raw quantized bytes of ti as a cached device buffer (NULL on failure). */
id<MTLBuffer> metal_i_weight(TensorInfo *ti);

/* Arena registry: metal_arena_alloc registers the backing MTLBuffer so
 * metal_op can resolve any interior pointer back to (buffer, offset).
 * lookup returns 0 and fills buf/off when ptr lies inside one. */
int  metal_i_lookup(const void *ptr, id<MTLBuffer> __strong *buf, NSUInteger *off);
void metal_i_register(void *base, NSUInteger len, id<MTLBuffer> buf);
void metal_i_unregister(void *base);

#endif /* __METAL_INTERNAL_H__ */
