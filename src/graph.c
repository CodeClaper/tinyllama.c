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

/* Append a node with a precomputed output shape. */
static u32 node_add(Graph *g, GraphOp op, const int *src, TensorInfo *weight, u64 out_elems, u32 out_ndim, const u64 *out_dim) {
    if (!g || out_elems == 0) return GRAPH_NODE_NONE;
    if (!grow((void **)&g->node, &g->cap, g->n_node + 1, sizeof(GraphNode))) return GRAPH_NODE_NONE;

    GraphNode *n = &g->node[g->n_node];
    n->op        = op;
    n->weight    = weight;
    n->cache     = NULL;
    n->out_elems = out_elems;
    n->out_ndim  = out_ndim;
    for (u32 i = 0; i < MAX_DIMS; i++) n->out_dim[i] = out_dim ? out_dim[i] : 0;
    for (int i = 0; i < 4; i++) n->src[i] = src ? src[i] : -1;
    return g->n_node++;
}

/* mat-vec: rows from weight dims as in mat_vec_mul(). */
static bool shape_matvec(const GraphNode *a, TensorInfo *w, bool trans,
                         u64 *rows, u64 *cols) {
    if (!w || w->ndim < 2) return false;
    if (trans) { *rows = w->dim[1]; *cols = w->dim[0]; }
    else       { *rows = w->dim[0]; *cols = w->dim[1]; }
    if (a->out_elems != *cols) return false;
    return true;
}

/* mat-mat: batch is derived from src elems / cols. */
static bool shape_matmat(const GraphNode *a, TensorInfo *w, bool trans,
                         u64 *batch, u64 *rows, u64 *cols) {
    if (!w || w->ndim < 2) return false;
    if (trans) { *rows = w->dim[1]; *cols = w->dim[0]; }
    else       { *rows = w->dim[0]; *cols = w->dim[1]; }
    if (*cols == 0 || a->out_elems % *cols != 0) return false;
    *batch = a->out_elems / *cols;
    return true;
}

u32 graph_rms_norm(Graph *g, u32 src, TensorInfo *weight) {
    const GraphNode *a = src < g->n_node ? &g->node[src] : NULL;
    if (!a || !weight) return GRAPH_NODE_NONE;
    return node_add(g, OP_RMS_NORM, (int[]){src,-1,-1,-1}, weight,
                    a->out_elems, a->out_ndim, a->out_dim);
}

u32 graph_mul_mat(Graph *g, u32 src, TensorInfo *weight, bool trans) {
    const GraphNode *a = src < g->n_node ? &g->node[src] : NULL;
    u64 rows = 0, cols = 0;
    if (!a || !shape_matvec(a, weight, trans, &rows, &cols)) return GRAPH_NODE_NONE;
    u64 dim[1] = { rows };
    return node_add(g, trans ? OP_MATMUL_T : OP_MATMUL, (int[]){src,-1,-1,-1},
                    weight, rows, 1, dim);
}

u32 graph_mul_mat2(Graph *g, u32 src, TensorInfo *weight) {
    const GraphNode *a = src < g->n_node ? &g->node[src] : NULL;
    u64 batch = 0, rows = 0, cols = 0;
    if (!a || !shape_matmat(a, weight, false, &batch, &rows, &cols)) return GRAPH_NODE_NONE;
    u64 dim[2] = { batch, rows };
    return node_add(g, OP_MATMULARRY, (int[]){src,-1,-1,-1}, weight,
                    batch * rows, 2, dim);
}

u32 graph_binary(Graph *g, GraphOp op, u32 a_id, u32 b_id) {
    if (op != OP_ADD && op != OP_MUL) return GRAPH_NODE_NONE;
    const GraphNode *a = a_id < g->n_node ? &g->node[a_id] : NULL;
    const GraphNode *b = b_id < g->n_node ? &g->node[b_id] : NULL;
    if (!a || !b || a->out_elems != b->out_elems) return GRAPH_NODE_NONE;
    return node_add(g, op, (int[]){a_id,b_id,-1,-1}, NULL,
                    a->out_elems, a->out_ndim, a->out_dim);
}

u32 graph_silu(Graph *g, u32 src) {
    const GraphNode *a = src < g->n_node ? &g->node[src] : NULL;
    if (!a) return GRAPH_NODE_NONE;
    return node_add(g, OP_SILU, (int[]){src,-1,-1,-1}, NULL,
                    a->out_elems, a->out_ndim, a->out_dim);
}

u32 graph_softmax(Graph *g, u32 src) {
    const GraphNode *a = src < g->n_node ? &g->node[src] : NULL;
    if (!a) return GRAPH_NODE_NONE;
    return node_add(g, OP_SOFTMAX, (int[]){src,-1,-1,-1}, NULL,
                    a->out_elems, a->out_ndim, a->out_dim);
}

u32 graph_rope(Graph *g, u32 src) {
    const GraphNode *a = src < g->n_node ? &g->node[src] : NULL;
    if (!a) return GRAPH_NODE_NONE;
    return node_add(g, OP_ROPE_NEOX, (int[]){src,-1,-1,-1}, NULL,
                    a->out_elems, a->out_ndim, a->out_dim);
}

u32 graph_embed(Graph *g, u32 token_id, TensorInfo *weight, u32 n_tokens) {
    const GraphNode *tok = token_id < g->n_node ? &g->node[token_id] : NULL;
    if (!tok || !weight || weight->ndim < 2) return GRAPH_NODE_NONE;
    u64 n_embd = weight->dim[weight->ndim - 1];
    u64 dim[2] = { n_tokens, n_embd };
    return node_add(g, OP_EMBED, (int[]){token_id,-1,-1,-1}, weight,
                    n_tokens * n_embd, 2, dim);
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


/* ---------------------------------------------------------------- *
 * Graph execution
 * ---------------------------------------------------------------- */

static void per_row_softmax(float *x, u64 rows, u64 cols) {
    for (u64 r = 0; r < rows; r++)
        softmax(x + r * cols, (u32)cols);
}

bool graph_compute(Graph *g, GraphPlan *plan, const u32 *tokens, Session *s) {
    if (!g || !s) {
        slog(WARN, "graph_compute: missing graph / session");
        return false;
    }
    u32 cnt = g->n_node;

    bool ok = true;

    /* One output buffer per node, all released at the end. */
    float **out = scalloc(cnt, sizeof(float *));
    for (u32 i = 0; i < cnt; i++)
        out[i] = smalloc(g->node[i].out_elems * sizeof(float));

    /* Seed OP_INPUT leaves with token ids (stored as float). */
    for (u32 i = 0; i < cnt; i++) {
        if (g->node[i].op != OP_INPUT) continue;
        if (!tokens) {
            slog(WARN, "graph_compute: OP_INPUT node requires tokens");
            ok = false;
            goto done;
        }
        float *dst = out[i];
        for (u64 j = 0; j < g->node[i].out_elems; j++)
            dst[j] = (float)tokens[j];
    }

    for (u32 i = 0; i < cnt; i++) {
        GraphNode *nd = &g->node[i];
        float *o = out[i];

        if (nd->op == OP_INPUT) continue; /* seeded above */

        float *xa = nd->src[0] >= 0 ? out[nd->src[0]] : NULL;
        float *xb = nd->src[1] >= 0 ? out[nd->src[1]] : NULL;

        switch (nd->op) {
            case OP_ADD:
            case OP_MUL:
                for (u64 j = 0; j < nd->out_elems; j++)
                    o[j] = (nd->op == OP_ADD) ? xa[j] + xb[j] : xa[j] * xb[j];
                break;
            case OP_SILU:
                memcpy(o, xa, nd->out_elems * sizeof(float));
                silu(o, (int)nd->out_elems);
                break;
            case OP_SOFTMAX: {
                memcpy(o, xa, nd->out_elems * sizeof(float));
                u64 cols = nd->out_ndim > 0 ? nd->out_dim[nd->out_ndim - 1] : 1;
                u64 rows = cols ? nd->out_elems / cols : 1;
                per_row_softmax(o, rows, cols);
                break;
            }
            case OP_RMS_NORM: {
                u64 ncols = nd->weight ? nd->weight->dim[0] : 1;
                if (ncols == 0) ncols = 1;
                u64 rows = nd->out_elems / ncols;
                for (u64 r = 0; r < rows; r++)
                    rms_norm(o + r * ncols, xa + r * ncols,
                             nd->weight, (int)ncols, DEFAULT_EPS);
                break;
            }
            case OP_MATMUL:
            case OP_MATMUL_T: {
                u64 rows, cols;
                bool trans = nd->op == OP_MATMUL_T;
                const GraphNode *an = &g->node[nd->src[0]];
                if (!shape_matvec(an, nd->weight, trans, &rows, &cols)) {
                    slog(WARN, "graph_compute: matvec shape mismatch at node %u", i);
                    ok = false;
                    goto done;
                }
                mat_vec_mul(o, nd->weight, xa, rows, cols, trans, s->pthreads);
                break;
            }
            case OP_MATMULARRY: {
                u64 batch, rows, cols;
                const GraphNode *an = &g->node[nd->src[0]];
                if (!shape_matmat(an, nd->weight, false, &batch, &rows, &cols)) {
                    slog(WARN, "graph_compute: matmat shape mismatch at node %u", i);
                    ok = false;
                    goto done;
                }
                mat_mat_mul(o, nd->weight, xa, batch, rows, cols, false, s->pthreads);
                break;
            }
            case OP_EMBED: {
                u64 n_embd = nd->weight->dim[nd->weight->ndim - 1];
                u64 n_tok = nd->out_elems / n_embd;
                for (u64 t = 0; t < n_tok; t++) {
                    i64 id = (i64)xa[t];
                    float *dst = o + t * n_embd;
                    if (id < 0) { memset(dst, 0, n_embd * sizeof(float)); continue; }
                    for (u64 j = 0; j < n_embd; j++)
                        dst[j] = tensor_get_f32(nd->weight, (u64)id * n_embd + j);
                }
                break;
            }
            case OP_ROPE_NEOX: {
                memcpy(o, xa, nd->out_elems * sizeof(float));
                u64 n_tok = nd->out_dim[0];
                u32 per = (u32)(nd->out_elems / n_tok);
                u32 hd = s->cfg.head_dim;
                u32 nh = hd ? per / hd : 1;
                if (!hd) hd = per;
                for (u64 t = 0; t < n_tok; t++)
                    rope_partial(o + t * per, nh, hd, s->cfg.rope_dim,
                                 (u32)t, GRAPH_ROPE_THETA);
                break;
            }
            default:
                slog(WARN, "graph_compute: node %u unknown op %d", i, nd->op);
                ok = false;
                goto done;
        }
    }

done:
    for (u32 i = 0; i < cnt; i++) sfree(out[i]);
    sfree(out);
    return ok;
}

