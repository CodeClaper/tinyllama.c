#ifndef __GPU_OP_H__
#define __GPU_OP_H__

#include <stdbool.h>
#include "../../def.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * CUDA graph operators — device twins of the op_table in graph.c.
 *
 * The executor calls gpu_graph_op() once per node with every pointer
 * already resident in VRAM (graph arena, KV cache, SSM state, token
 * ids), so activations flow device-to-device without H2D/D2H.  Weight
 * tensors are reached through the persistent device cache in gpu.cu
 * (gpu_weight_dev / gpu_dequant_dev / gpu_dequant_gather_dev /
 * gpu_matmul_dev), uploaded once on first use.
 *
 * The context type is the shared backend-agnostic OpCtx from def.h;
 * on this path its pointers address device memory.
 */

/* Execute c->node->op on the device.  Returns false for unsupported
 * ops (OP_INPUT / OP_H2D / OP_D2H stay with the executor) or
 * structural failures; CUDA errors are fatal via CHECK(). */
bool gpu_graph_op(OpCtx *c);

/* Free the op-level device caches (dequantized f32 weights). */
void gpu_op_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif
