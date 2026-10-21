#ifndef __GRAPH_H__
#define __GRAPH_H__

#include "def.h"

/* Builders append a node to the graph's linked list and return it (a
 * tensor handle).  Sources always point backwards: build order ==
 * execution order. */

/* Inputs for graph_ssm_delta() (Gated DeltaNet recurrence). */
typedef struct {
    GraphNode  *fused;      /* conv + SiLU output                         */
    GraphNode  *alpha;      /* decay projection   [n_v_heads]             */
    GraphNode  *beta;       /* update projection  [n_v_heads]             */
    TensorInfo *ssm_a;      /* per-v-head decay parameter                 */
    TensorInfo *dt_bias;    /* per-v-head dt bias                         */
    TensorInfo *norm;       /* per-v-head RMS norm weight (may be NULL)   */
    u32         state;      /* graph_state() handle for the recurrent state */
    u32         n_v_heads;
    u32         n_k_heads;
    u32         head_dim;
} GraphSsmDelta;

Graph *graph_new(void);
void graph_free(Graph *g);

/* Print an ASCII rendering of the built graph workflow: every node in
 * build (= execution) order with its op, layer, backend, inputs and
 * attached weights, followed by a summary of the sink nodes. */
void graph_drawer(const Graph *g);

/* Tag every node appended after this call with `layer`.  Builders call
 * it once per transformer layer; layer == cfg.n_layer marks the final
 * norm / LM head so it stays on the CPU side of an offload split. */
void graph_set_layer(Graph *g, u32 layer);

/* Backend a given transformer layer should use under the session's
 * -ngl setting.  Also used by arch code to place recurrent state. */
BackendType graph_layer_backend(const Session *s, u32 layer);

/* Insert a device->host copy after a layer when it is the last GPU
 * layer of a split plan; returns `src` unchanged otherwise. */
GraphNode *graph_layer_boundary(Graph *g, GraphNode *src, Session *s, u32 layer);

/* Backend-aware recurrent-state allocation (CPU host memory or CUDA
 * device memory; Metal keeps a host mirror). */
void *graph_state_alloc(size_t bytes, BackendType backend);
void  graph_state_free(void *ptr, BackendType backend);
void  graph_state_zero(void *ptr, size_t bytes, BackendType backend);

GraphNode *graph_input(Graph *g, u32 n_tokens);
GraphNode *graph_embed(Graph *g, GraphNode *token_id, TensorInfo *weight, u32 n_tokens);
GraphNode *graph_rms_norm(Graph *g, GraphNode *src, TensorInfo *weight);
GraphNode *graph_rms_norm_heads(Graph *g, GraphNode *src, TensorInfo *weight, u32 n_heads, u32 head_dim, u32 in_stride, u32 out_stride);
GraphNode *graph_mul_mat(Graph *g, GraphNode *src, TensorInfo *weight, bool trans);
GraphNode *graph_binary(Graph *g, GraphOp op, GraphNode *a, GraphNode *b);
GraphNode *graph_rope(Graph *g, GraphNode *src, float theta, u32 n_heads, u32 head_dim, u32 rope_dim);
GraphNode *graph_rope_norm(Graph *g, GraphNode *src, float theta, u32 n_heads, u32 head_dim, u32 rope_dim);
GraphNode *graph_silu(Graph *g, GraphNode *src);
GraphNode *graph_gelu(Graph *g, GraphNode *src);
GraphNode *graph_softmax(Graph *g, GraphNode *src);
GraphNode *graph_scale(Graph *g, GraphNode *src, float scale);
GraphNode *graph_softcap(Graph *g, GraphNode *src, float cap);
GraphNode *graph_bias(Graph *g, GraphNode *src, TensorInfo *bias);
GraphNode *graph_sigmoid_gate(Graph *g, GraphNode *a, GraphNode *gate, u32 n_heads, u32 head_dim);
GraphNode *graph_ssm_conv(Graph *g, GraphNode *src, TensorInfo *weight, u32 state, u32 kernel);
GraphNode *graph_ssm_delta(Graph *g, const GraphSsmDelta *args);
GraphNode *graph_mamba2_conv(Graph *g, GraphNode *src, TensorInfo *weight,
                             TensorInfo *bias, u32 state, u32 kernel,
                             u32 d_inner, u32 n_group, u32 d_state,
                             u32 n_head);
GraphNode *graph_mamba2_scan(Graph *g, GraphNode *conv, GraphNode *proj,
                             TensorInfo *ssm_a, TensorInfo *ssm_d,
                             TensorInfo *dt_bias, TensorInfo *norm,
                             u32 state, u32 n_head, u32 head_dim,
                             u32 d_state, u32 n_group, u32 d_inner);
GraphNode *graph_attn(Graph *g, GraphNode *q, GraphNode *k, GraphNode *v, u32 layer, u32 window);
u32 graph_state(Graph *g, void *ptr);

/* Runs tokens through the graph, splitting batches larger than
 * Session.n_ubatch into micro-batches.  Generates the execution plan
 * (arena slot layout) once: node->data pointers are baked into the graph
 * and any n <= n_ubatch reuses the slots. */
bool graph_execute(Session *s, const u32 *tokens, u32 n_tokens, float *logits);

/* Build the graph (if needed), generate the execution plan and preload
 * every weight onto the device backend.  Call once at startup so the
 * first request does not pay the graph-build / planning / H2D cost
 * (warmup).  Safe to call repeatedly. */
bool graph_prepare(Session *s);

#endif
