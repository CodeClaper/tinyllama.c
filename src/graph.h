#ifndef __GRAPH_H__
#define __GRAPH_H__

#include "def.h"


Graph *graph_new(void);
void graph_free(Graph *g);
/* Op builders: append a node, return its index (a tensor handle). */
u32 graph_input(Graph *g, u32 n_tokens);
u32 graph_embed(Graph *g, u32 token_id, TensorInfo *weight, u32 n_tokens);
u32 graph_rms_norm(Graph *g, u32 src, TensorInfo *weight);
/* Per-head RMS norm: applies rms(x)*w independently to `n_heads` slices of
 * `head_dim` elements, reading every slice at `in_stride` and writing it at
 * `out_stride`.  Covers both K-norm (in == out == head_dim, whole row) and
 * Q-norm over a fused [q, gate] projection (in == 2*head_dim, out ==
 * head_dim, which also drops the gate half). */
u32 graph_rms_norm_heads(Graph *g, u32 src, TensorInfo *weight,
                         u32 n_heads, u32 head_dim, u32 in_stride, u32 out_stride);
u32 graph_mul_mat(Graph *g, u32 src, TensorInfo *weight, bool trans);
u32 graph_binary(Graph *g, GraphOp op, u32 a, u32 b);
/* RoPE: theta / head layout baked into the node at build time, so the
 * executor does not need to know the arch.  `rope_dim` <= `head_dim`
 * rotates only the first rope_dim elements of each head (partial RoPE);
 * passing rope_dim == head_dim gives the classic full-width neox RoPE. */
u32 graph_rope(Graph *g, u32 src, float theta, u32 n_heads, u32 head_dim, u32 rope_dim);
u32 graph_silu(Graph *g, u32 src);
u32 graph_softmax(Graph *g, u32 src);
u32 graph_bias(Graph *g, u32 src, TensorInfo *bias);
/* out = a * sigmoid(gate), where `gate` is a fused [n_heads][2*head_dim]
 * projection whose second half of each head holds the raw gate logits
 * (Qwen3.5 fused Q+gate attention). */
u32 graph_sigmoid_gate(Graph *g, u32 a, u32 gate, u32 n_heads, u32 head_dim);
/* Gated DeltaNet: depthwise causal conv1d over the fused QKV projection.
 * `state` is a graph_state() handle for the per-layer ring buffer
 * [(kernel-1) * channels]; `weight` is ssm_conv1d [channels, kernel]. */
u32 graph_ssm_conv(Graph *g, u32 src, TensorInfo *weight, u32 state, u32 kernel);
/* Gated DeltaNet recurrence: reads the SiLU'd fused [q|k|v] block plus the
 * alpha/beta projections, updates the recurrent state in place and yields
 * the normed delta output [n_groups * d_state].  `state` is a graph_state()
 * handle for [n_groups * d_state * d_state]. */
u32 graph_ssm_delta(Graph *g, u32 fused, u32 alpha, u32 beta,
                    TensorInfo *ssm_a, TensorInfo *dt_bias, TensorInfo *norm,
                    u32 state, u32 n_groups, u32 d_state);
/* Register a buffer owned by the caller (typically the arch workspace) for
 * use by stateful ops; returns a handle to bake into op params.  The graph
 * borrows the pointer: graph_free() does not free it. */
u32 graph_state(Graph *g, void *ptr);
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
