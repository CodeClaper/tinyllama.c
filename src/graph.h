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
u32 graph_attn(Graph *g, u32 q, u32 k, u32 v);
GraphPlan graph_plan(Graph *g);
/* Execute the whole graph over the first s->n_tokens rows of `tokens`
 * (row i = absolute position i).  The sink node (e.g. the LM head) is
 * computed for the last row only; its output is copied to s->logits. */
bool graph_compute(Graph *g, GraphPlan *plan,const u32 *tokens, Session *s);

#endif
