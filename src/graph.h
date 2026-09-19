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

GraphNode *graph_input(Graph *g, u32 n_tokens);
GraphNode *graph_embed(Graph *g, GraphNode *token_id, TensorInfo *weight, u32 n_tokens);
GraphNode *graph_rms_norm(Graph *g, GraphNode *src, TensorInfo *weight);
GraphNode *graph_rms_norm_heads(Graph *g, GraphNode *src, TensorInfo *weight, u32 n_heads, u32 head_dim, u32 in_stride, u32 out_stride);
GraphNode *graph_mul_mat(Graph *g, GraphNode *src, TensorInfo *weight, bool trans);
GraphNode *graph_binary(Graph *g, GraphOp op, GraphNode *a, GraphNode *b);
GraphNode *graph_rope(Graph *g, GraphNode *src, float theta, u32 n_heads, u32 head_dim, u32 rope_dim);
GraphNode *graph_silu(Graph *g, GraphNode *src);
GraphNode *graph_softmax(Graph *g, GraphNode *src);
GraphNode *graph_scale(Graph *g, GraphNode *src, float scale);
GraphNode *graph_bias(Graph *g, GraphNode *src, TensorInfo *bias);
GraphNode *graph_sigmoid_gate(Graph *g, GraphNode *a, GraphNode *gate, u32 n_heads, u32 head_dim);
GraphNode *graph_ssm_conv(Graph *g, GraphNode *src, TensorInfo *weight, u32 state, u32 kernel);
GraphNode *graph_ssm_delta(Graph *g, const GraphSsmDelta *args);
GraphNode *graph_attn(Graph *g, GraphNode *q, GraphNode *k, GraphNode *v, u32 layer);
u32 graph_state(Graph *g, void *ptr);

/* Generates the execution plan (arena slot layout) once: node->data
 * pointers are baked into the graph and any n <= cfg-sized batch
 * reuses the slots.  graph_compute() calls this lazily on first run. */
bool graph_execute(Session *s, const u32 *tokens, u32 n_tokens, float *logits);

#endif
