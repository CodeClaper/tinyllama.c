#ifndef __GRAPH_H__
#define __GRAPH_H__

#include "def.h"

#define GRAPH_NODE_NONE ((u32)-1)  /* invalid node handle (builder return) */

typedef enum {
    OP_INPUT,      /* leaf: caller-seeded value, produced externally */
    OP_ADD,        /* out = a + b                 (element-wise)     */
    OP_MUL,        /* out = a * b                 (element-wise)     */
    OP_MATMUL,     /* out = W @ x                 (mat-vec)          */
    OP_MATMULARRY, /* out[b] = W @ x[b]           (mat-mat, batch)   */
    OP_MATMUL_T,   /* out = W^T @ x               (mat-vec)          */
    OP_RMS_NORM,   /* out = rms(x) * w                               */
    OP_ROPE_NEOX,  /* out = rope(in)                                 */
    OP_SOFTMAX,    /* out = softmax(x)  (per-row over last dim)      */
    OP_SILU,       /* out = silu(x)                                  */
    OP_EMBED,      /* out = token_embd[token]                        */
} GraphOp;


typedef struct {
    GraphOp     op;
    int         src[4];     /* source node indices; -1 = unused */
    TensorInfo *weight;

    /* OP_ATTENTION only: this layer's KV cache. */
    AttnKvCache *cache;

    /* Output shape, computed at build time. */
    u64  out_elems;
    u32  out_ndim;
    u64  out_dim[MAX_DIMS];
} GraphNode;

/* Static DAG.  Nodes are appended in build order; because every edge
 * points strictly backward, build order is also a valid topo order. */
typedef struct {
    GraphNode *node;
    u32        n_node;
    u32        cap;
} Graph;

typedef struct {

} GraphPlan;

Graph *graph_new(void);
void graph_free(Graph *g);
/* Op builders: append a node, return its index (a tensor handle). */
u32 graph_embed(Graph *g, u32 token_id, TensorInfo *weight, u32 n_tokens);
u32 graph_rms_norm(Graph *g, u32 src, TensorInfo *weight);
u32 graph_mul_mat(Graph *g, u32 src, TensorInfo *weight, bool trans);
u32 graph_mul_mat2(Graph *g, u32 src, TensorInfo *weight);
u32 graph_binary(Graph *g, GraphOp op, u32 a, u32 b);
u32 graph_rope(Graph *g, u32 src);
u32 graph_silu(Graph *g, u32 src);
u32 graph_softmax(Graph *g, u32 src);
GraphPlan graph_plan(Graph *g);
bool graph_compute(Graph *g, GraphPlan *plan,const u32 *tokens, Session *s);

#endif
