#ifndef __METAL_OP_H__
#define __METAL_OP_H__

#include <stdbool.h>
#include "../../def.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Metal graph operators — device twins of the op_table in graph.c.
 *
 * The executor runs with the arena (one shared MTLBuffer) and the
 * token ids CPU/GPU-visible, so activations flow through the same
 * storage the kernels read and write; only weights originate on the
 * host and are served from metal.m's persistent raw-weight cache.
 * Per-op scalars are staged in an auxiliary shared buffer and every
 * dispatch of one graph_compute() is encoded into a single command
 * buffer, committed by metal_graph_flush().
 *
 * Mirrors backend/gpu/gpu_op.h.  Returns false for unsupported ops
 * (OP_INPUT / OP_H2D / OP_D2H stay with the executor) or structural
 * failures.
 */
bool metal_graph_op(OpCtx *c);

/* Commit + wait for the command buffer accumulated by metal_graph_op()
 * calls.  Must be called once after each graph_compute() node walk. */
void metal_graph_flush(void);

/* Free the op-level caches (dequantized f32 weights, KV/state shadows). */
void metal_op_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif
