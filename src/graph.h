#ifndef __GRAPH_H__
#define __GRAPH_H__

#include "def.h"
#include "pthreads.h"

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
    OP_ATTENTION,  /* fused per-head scored attention + KV update    */
} GraphOp;

typedef struct {
    GGUFType type;
    u64     dim[MAX_DIMS];
    void    *data;
} GraphTensor;


typedef struct {
    GraphOp     op;
    int         src[4];     /* source node indices; -1 = unused */
    TensorInfo *weight;
    void       *param;      /* op-specific data (GraphAttnParam) */

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

/* Per-run context: params not carried by the node itself. */
typedef struct {
    const u8   *base;       /* mmap base for weight dequant */
    pthreads_t *pool;
    u32         n_head;

    /* RoPE / norm config (used by rope + attention nodes). */
    u32         head_dim;
    u32         rope_dim;
    float       theta_base;

    float       eps;
} GraphRunCtx;

/* Build modes for graph_build(). */
typedef enum {
    GRAPH_PREFILL,  /* process n_tokens new tokens into cache */
    GRAPH_GENERATE, /* process 1 token and read full cache    */
} GraphMode;

Graph  *graph_new(void);
void    graph_free(Graph *g);

/* Leaf: an externally-provided value (token ids, hand-seeded). */
u32     graph_input(Graph *g, u32 n_element);

/* Op builders: append a node, return its index (a tensor handle). */
u32 graph_embed(Graph *g, u32 token_id, TensorInfo *weight, u32 n_tokens);
u32 graph_rms_norm(Graph *g, u32 src, TensorInfo *weight);
u32 graph_mul_mat(Graph *g, u32 src, TensorInfo *weight, bool trans);
u32 graph_mul_mat2(Graph *g, u32 src, TensorInfo *weight);
u32 graph_binary(Graph *g, GraphOp op, u32 a, u32 b);
u32 graph_rope(Graph *g, u32 src);
u32 graph_silu(Graph *g, u32 src);
u32 graph_softmax(Graph *g, u32 src);
u32 graph_attention(Graph *g, u32 q, u32 k, u32 v, GraphAttnParam *param);

/* Build a whole forward pass (embed -> layers -> logits).  Returns a
 * graph whose single OP_INPUT leaf is the token-id buffer (n_tokens
 * u32 values), seeded from `tokens` by graph_compute().  On success
 * *logits_out is the logits node. */
Graph *graph_build(Session *s, GraphMode mode, const u32 *tokens, u32 n_tokens, u32 *logits_out);

/* Execute the graph: allocate one output buffer per node, seed OP_INPUT
 * leaves from `tokens`, run every node in build order, then release
 * the buffers. */
bool graph_compute(Graph *g, const u32 *tokens, const GraphRunCtx *ctx);

/* Fused per-head causal attention + KV-cache update (used by an
 * OP_ATTENTION node; also callable directly). */
bool graph_attention_run(GraphAttnParam *p, float *q, const float *k, const float *v, float *out, const GraphRunCtx *ctx);

#endif
