#ifndef __GRAPH_H__
#define __GRAPH_H__

#include "def.h"


Graph *graph_new(void);
void graph_free(Graph *g);
/* Op builders: append a node, return its index (a tensor handle). */
u32 graph_input(Graph *g, u32 n_tokens);
u32 graph_embed(Graph *g, u32 token_id, TensorInfo *weight, u32 n_tokens);
u32 graph_rms_norm(Graph *g, u32 src, TensorInfo *weight);
u32 graph_mul_mat(Graph *g, u32 src, TensorInfo *weight, bool trans);
u32 graph_binary(Graph *g, GraphOp op, u32 a, u32 b);
/* RoPE: theta / head layout baked into the node at build time, so the
 * executor does not need to know the arch. */
u32 graph_rope(Graph *g, u32 src, float theta, u32 n_heads, u32 head_dim);
u32 graph_silu(Graph *g, u32 src);
u32 graph_softmax(Graph *g, u32 src);
u32 graph_bias(Graph *g, u32 src, TensorInfo *bias);
/* graph_attn bakes the layer index into the node so the executor can
 * reach the per-layer session KV cache. */
u32 graph_attn(Graph *g, u32 q, u32 k, u32 v, u32 layer);
GraphPlan graph_plan(Graph *g);
/* Execute the graph over one batch of n rows at absolute position
 * b->pos (incremental, KV-cached): rows are RoPE'd at pos+i, and each
 * OP_ATTN writes the batch's K/V into the session cache, then attends
 * causally over the cached keys.  The sink node (e.g. the LM head) is
 * computed for the batch's last row only; its output is copied to
 * s->logits.  The graph itself is built once at capacity and reused
 * for batches of any n <= capacity. */
bool graph_compute(Graph *g, GraphPlan *plan, const GraphBatch *b, Session *s);

#endif
