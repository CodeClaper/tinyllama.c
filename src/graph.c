#include <string.h>
#include <math.h>
#include "graph.h"
#include "core.h"
#include "mm.h"
#include "slog.h"

/* RoPE frequency base (models without a metadata override). */
#define GRAPH_ROPE_THETA 10000.0f

/* ---------------------------------------------------------------- *
 * Operators
 * ---------------------------------------------------------------- */

/* Grow a dynamic array, doubling capacity when full. */
static bool grow(void **arr, u32 *cap, u32 need, size_t rec) {
    if (*cap >= need) return true;
    u32 nc = *cap ? *cap * 2 : 16;
    if (nc < need) nc = need;
    void *p = srealloc(*arr, (size_t)nc * rec);
    if (!p) return false;
    *arr = p;
    *cap = nc;
    return true;
}

/* Append a node; output shapes are derived at execution time.
 * ne[]  = source node indices, src[] = per-source weight tensors,
 * params[] = op-specific parameters (unused ops may pass NULL). */
static u32 node_add(Graph *g, GraphOp op, const int *ne, TensorInfo *const *src, const u32 *params) {
    if (!g) return GRAPH_NODE_NONE;
    if (!grow((void **)&g->node, &g->cap, g->n_node + 1, sizeof(GraphNode))) return GRAPH_NODE_NONE;
    GraphNode *n = &g->node[g->n_node];
    n->op = op;
    for (int i = 0; i < 4; i++) {
        n->ne[i]  = ne ? ne[i] : -1;
        n->src[i] = src ? src[i] : NULL;
    }
    if (params) memcpy(n->params, params, sizeof(n->params));
    return g->n_node++;
}

/* Leaf input: carries a run of token ids; width stored in ne[0]. */
static u32 graph_input(Graph *g, u32 n_element) {
    if (!g || n_element == 0) return GRAPH_NODE_NONE;
    return node_add(g, OP_INPUT, (int[]){ (int)n_element, -1, -1, -1 }, NULL, NULL);
}

/* mat-vec: rows from weight dims. */
static bool shape_matvec(u64 a_elems, TensorInfo *w, bool trans, u64 *rows, u64 *cols) {
    if (!w || w->ndim < 2) return false;
    if (trans) { *rows = w->dim[1]; *cols = w->dim[0]; }
    else       { *rows = w->dim[0]; *cols = w->dim[1]; }
    if (a_elems != *cols) return false;
    return true;
}

/* mat-mat: batch is derived from src elems / cols. */
static bool shape_matmat(u64 a_elems, TensorInfo *w, bool trans, u64 *batch, u64 *rows, u64 *cols) {
    if (!w || w->ndim < 2) return false;
    if (trans) { *rows = w->dim[1]; *cols = w->dim[0]; }
    else       { *rows = w->dim[0]; *cols = w->dim[1]; }
    if (*cols == 0 || a_elems % *cols != 0) return false;
    *batch = a_elems / *cols;
    return true;
}

u32 graph_rms_norm(Graph *g, u32 src, TensorInfo *weight) {
    if (!g || src >= g->n_node || !weight) return GRAPH_NODE_NONE;
    return node_add(g, OP_RMS_NORM, (int[]){ (int)src, -1, -1, -1 },
                    (TensorInfo *[]){ weight, NULL, NULL, NULL }, NULL);
}

u32 graph_mul_mat(Graph *g, u32 src, TensorInfo *weight, bool trans) {
    if (!g || src >= g->n_node || !weight || weight->ndim < 2) return GRAPH_NODE_NONE;
    return node_add(g, trans ? OP_MATMUL_T : OP_MATMUL, (int[]){ (int)src, -1, -1, -1 },
                    (TensorInfo *[]){ weight, NULL, NULL, NULL }, NULL);
}

u32 graph_mul_mat2(Graph *g, u32 src, TensorInfo *weight) {
    if (!g || src >= g->n_node || !weight || weight->ndim < 2) return GRAPH_NODE_NONE;
    return node_add(g, OP_MATMULARRY, (int[]){ (int)src, -1, -1, -1 },
                    (TensorInfo *[]){ weight, NULL, NULL, NULL }, NULL);
}

u32 graph_binary(Graph *g, GraphOp op, u32 a_id, u32 b_id) {
    if (op != OP_ADD && op != OP_MUL) return GRAPH_NODE_NONE;
    if (!g || a_id >= g->n_node || b_id >= g->n_node) return GRAPH_NODE_NONE;
    return node_add(g, op, (int[]){ (int)a_id, (int)b_id, -1, -1 }, NULL, NULL);
}

u32 graph_silu(Graph *g, u32 src) {
    if (!g || src >= g->n_node) return GRAPH_NODE_NONE;
    return node_add(g, OP_SILU, (int[]){ (int)src, -1, -1, -1 }, NULL, NULL);
}

u32 graph_softmax(Graph *g, u32 src) {
    if (!g || src >= g->n_node) return GRAPH_NODE_NONE;
    return node_add(g, OP_SOFTMAX, (int[]){ (int)src, -1, -1, -1 }, NULL, NULL);
}

u32 graph_bias(Graph *g, u32 src, TensorInfo *bias) {
    if (!g || src >= g->n_node || !bias || bias->ndim < 1) return GRAPH_NODE_NONE;
    return node_add(g, OP_BIAS, (int[]){ (int)src, -1, -1, -1 },
                    (TensorInfo *[]){ bias, NULL, NULL, NULL }, NULL);
}

u32 graph_attn(Graph *g, u32 q, u32 k, u32 v) {
    if (!g || q >= g->n_node || k >= g->n_node || v >= g->n_node) return GRAPH_NODE_NONE;
    return node_add(g, OP_ATTN, (int[]){ (int)q, (int)k, (int)v, -1 }, NULL, NULL);
}

u32 graph_rope(Graph *g, u32 src) {
    if (!g || src >= g->n_node) return GRAPH_NODE_NONE;
    return node_add(g, OP_ROPE_NEOX, (int[]){ (int)src, -1, -1, -1 }, NULL, NULL);
}

u32 graph_embed(Graph *g, u32 token_id, TensorInfo *weight, u32 n_tokens) {
    if (!g || !weight || weight->ndim < 2 || n_tokens == 0) return GRAPH_NODE_NONE;
    if (token_id == GRAPH_NODE_NONE) token_id = graph_input(g, n_tokens);
    if (token_id == GRAPH_NODE_NONE || token_id >= g->n_node) return GRAPH_NODE_NONE;
    const GraphNode *tok = &g->node[token_id];
    if (tok->op != OP_INPUT || tok->ne[0] != (int)n_tokens) return GRAPH_NODE_NONE;
    return node_add(g, OP_EMBED, (int[]){ (int)token_id, -1, -1, -1 },
                    (TensorInfo *[]){ weight, NULL, NULL, NULL }, NULL);
}


/* ---------------------------------------------------------------- *
 * Graph creation
 * ---------------------------------------------------------------- */

Graph *graph_new(void) {
    return scalloc(1, sizeof(Graph));
}

void graph_free(Graph *g) {
    if (!g) return;
    sfree(g->node);
    sfree(g);
}

GraphPlan graph_plan(Graph *g) {
    (void)g;
    GraphPlan p;
    return p;
}


/* ---------------------------------------------------------------- *
 * Graph execution
 * ---------------------------------------------------------------- */
bool graph_compute(Graph *g, GraphPlan *plan, const u32 *tokens, Session *s) {
    if (!g || !s) {
        slog(WARN, "graph_compute: missing graph / session");
        return false;
    }
    return true;
}
