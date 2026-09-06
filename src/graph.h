#ifndef __GRAPH_H__
#define __GRAPH_H__

#include "def.h"

/* Builders append a node and return its index (a tensor handle).
 * Sources always point backwards: build order == execution order. */

Graph *graph_new(void);
void graph_free(Graph *g);
u32 graph_input(Graph *g, u32 n_tokens);
u32 graph_embed(Graph *g, u32 token_id, TensorInfo *weight, u32 n_tokens);
u32 graph_rms_norm(Graph *g, u32 src, TensorInfo *weight);
/* Per-head rms(x)*w over n_heads slices of head_dim, read at in_stride,
 * written at out_stride. */
u32 graph_rms_norm_heads(Graph *g, u32 src, TensorInfo *weight,
                         u32 n_heads, u32 head_dim, u32 in_stride, u32 out_stride);
u32 graph_mul_mat(Graph *g, u32 src, TensorInfo *weight, bool trans);
u32 graph_binary(Graph *g, GraphOp op, u32 a, u32 b);
/* rope_dim <= head_dim rotates only the first rope_dim dims (partial RoPE). */
u32 graph_rope(Graph *g, u32 src, float theta, u32 n_heads, u32 head_dim, u32 rope_dim);
u32 graph_silu(Graph *g, u32 src);
u32 graph_softmax(Graph *g, u32 src);
u32 graph_bias(Graph *g, u32 src, TensorInfo *bias);
/* out = a * sigmoid(gate); gate is the 2nd half of each 2*head_dim block. */
u32 graph_sigmoid_gate(Graph *g, u32 a, u32 gate, u32 n_heads, u32 head_dim);
/* Gated DeltaNet; state is a graph_state() handle. */
u32 graph_ssm_conv(Graph *g, u32 src, TensorInfo *weight, u32 state, u32 kernel);
u32 graph_ssm_delta(Graph *g, u32 fused, u32 alpha, u32 beta,
                    TensorInfo *ssm_a, TensorInfo *dt_bias, TensorInfo *norm,
                    u32 state, u32 n_v_heads, u32 n_k_heads, u32 head_dim);
/* Borrows a caller-owned buffer; graph_free() does not free it. */
u32 graph_state(Graph *g, void *ptr);
u32 graph_attn(Graph *g, u32 q, u32 k, u32 v, u32 layer);
/* Runs one batch at b->pos: OP_ATTN writes K/V into the session cache, then
 * attends causally.  The sink node runs on the last row only and its output
 * is copied to s->logits.  The graph is built once at ctx_size capacity and
 * reused for any n <= capacity. */
/* Auto-selects the compute backend (recorded into plan->backend); a
 * plan shell must exist.  graph_plan() calls this after the arena
 * sweep, so callers normally do not need it directly. */
bool backend_plan(Graph *g);
/* Generates the execution plan (arena slot layout) once: node->data
 * pointers are baked into the graph and any n <= cfg-sized batch
 * reuses the slots.  graph_compute() calls this lazily on first run. */
bool graph_plan(Graph *g, Session *s);
bool graph_compute(Graph *g, const GraphBatch *b, Session *s);
/* Session-level entry point (ArchOps.graph_execute): builds the graph on
 * first use, appends the batch at s->n_tokens and runs it. */
bool graph_execute(Session *s, const u32 *tokens, u32 n_tokens, float *logits);

#endif
