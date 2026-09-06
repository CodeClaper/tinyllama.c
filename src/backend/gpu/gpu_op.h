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
 */

/* Execution context for one node — mirrors the CPU OpCtx in graph.c,
 * but every data pointer addresses device memory. */
typedef struct {
    Graph      *g;
    Session    *s;
    ArchConfig *cfg;
    GraphNode  *node;  /* node being executed                        */
    float      *dst;   /* node->data: output slot (device)           */
    u32         od;    /* output row width (floats)                  */
    u32         r;     /* rows to produce (sinks: 1)                 */
    u32         base;  /* batch row of local row 0                   */
    u32         pos;   /* absolute position of batch row 0           */
    u32         n;     /* batch rows                                 */
    float      *scr;   /* [pos + n] score scratch (device; GPU attn  */
                       /* is self-contained and does not need it)    */
    float       scale; /* 1 / sqrt(kv_head_dim)                      */
    u32         q_dim; /* n_head    * head_dim                       */
    u32         kv_dim;/* n_kv_head * kv_head_dim                    */
} GpuOpCtx;

/* Execute c->node->op on the device.  Returns false for unsupported
 * ops (OP_INPUT / OP_H2D / OP_D2H stay with the executor) or
 * structural failures; CUDA errors are fatal via CHECK(). */
bool gpu_graph_op(GpuOpCtx *c);

/* Free the op-level device caches (dequantized f32 weights). */
void gpu_op_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif
