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

/* Per-node output shape, recomputed from the static DAG at run time. */
typedef struct {
    u64 elems;   /* total output elements        */
    u64 dim0;    /* first dimension              */
    u64 last;    /* last dimension               */
} Shape;

static void per_row_softmax(float *x, u64 rows, u64 cols) {
    for (u64 r = 0; r < rows; r++)
        softmax(x + r * cols, (u32)cols);
}

bool graph_compute(Graph *g, GraphPlan *plan, const u32 *tokens, Session *s) {
    if (!g || !s) {
        slog(WARN, "graph_compute: missing graph / session");
        return false;
    }
    (void)plan;

    u32 cnt = g->n_node;
    if (cnt == 0) return true;

    Shape *shape = scalloc(cnt, sizeof(Shape));

    bool ok = true;

    /* Pass 1: derive every node's output shape (build order == topo order). */
    for (u32 i = 0; i < cnt && ok; i++) {
        const GraphNode *nd = &g->node[i];
        Shape sh = { 0, 0, 0 };

        switch (nd->op) {
            case OP_INPUT:
                sh.elems = nd->ne[0] > 0 ? (u64)nd->ne[0] : 0;
                sh.dim0  = sh.elems;
                sh.last  = sh.elems;
                break;
            case OP_SILU:
            case OP_SOFTMAX:
            case OP_ROPE_NEOX:
            case OP_RMS_NORM:
            case OP_BIAS: {
                u32 a = nd->ne[0];
                if (a >= cnt) { ok = false; break; }
                sh = shape[a];
                break;
            }
            case OP_ATTN: {
                u32 a = nd->ne[0], b = nd->ne[1], c = nd->ne[2];
                if (a >= cnt || b >= cnt || c >= cnt ||
                    shape[a].dim0 != shape[b].dim0 ||
                    shape[a].dim0 != shape[c].dim0) {
                    slog(WARN, "graph_compute: attention shape mismatch at node %u", i);
                    ok = false;
                    break;
                }
                sh = shape[a];
                break;
            }
            case OP_ADD:
            case OP_MUL: {
                u32 a = nd->ne[0], b = nd->ne[1];
                if (a >= cnt || b >= cnt || shape[a].elems != shape[b].elems) {
                    slog(WARN, "graph_compute: binary op shape mismatch at node %u", i);
                    ok = false;
                    break;
                }
                sh = shape[a];
                break;
            }
            case OP_MATMUL:
            case OP_MATMUL_T: {
                u32 a = nd->ne[0];
                u64 rows, cols;
                if (a >= cnt ||
                    !shape_matvec(shape[a].elems, nd->src[0], nd->op == OP_MATMUL_T, &rows, &cols)
                ) {
                    slog(WARN, "graph_compute: matvec shape mismatch at node %u", i);
                    ok = false;
                    break;
                }
                sh.elems = rows;
                sh.dim0  = rows;
                sh.last  = rows;
                break;
            }
            case OP_MATMULARRY: {
                u32 a = nd->ne[0];
                u64 batch, rows, cols;
                if (a >= cnt ||
                    !shape_matmat(shape[a].elems, nd->src[0], false, &batch, &rows, &cols)
                ) {
                    slog(WARN, "graph_compute: matmat shape mismatch at node %u", i);
                    ok = false;
                    break;
                }
                sh.elems = batch * rows;
                sh.dim0  = batch;
                sh.last  = rows;
                break;
            }
            case OP_EMBED: {
                u32 a = nd->ne[0];
                if (a >= cnt || !nd->src[0] || nd->src[0]->ndim < 2) {
                    slog(WARN, "graph_compute: embed shape mismatch at node %u", i);
                    ok = false;
                    break;
                }
                u64 n_embd = nd->src[0]->dim[nd->src[0]->ndim - 1];
                u64 n_tok  = shape[a].elems;
                sh.elems = n_tok * n_embd;
                sh.dim0  = n_tok;
                sh.last  = n_embd;
                break;
            }
            default:
                slog(WARN, "graph_compute: node %u unknown op %d", i, nd->op);
                ok = false;
                break;
        }

        shape[i] = sh;
    }

    if (ok) {
        /* One output buffer per node, all released at the end. */
        float **out = scalloc(cnt, sizeof(float *));
        for (u32 i = 0; i < cnt; i++)
            out[i] = smalloc(shape[i].elems * sizeof(float));

        /* Seed OP_INPUT leaves with token ids (stored as float). */
        for (u32 i = 0; i < cnt; i++) {
            if (g->node[i].op != OP_INPUT) continue;
            if (!tokens) {
                slog(WARN, "graph_compute: OP_INPUT node requires tokens");
                ok = false;
                goto done;
            }
            float *dst = out[i];
            for (u64 j = 0; j < shape[i].elems; j++)
                dst[j] = (float)tokens[j];
        }

        for (u32 i = 0; i < cnt; i++) {
            GraphNode *nd = &g->node[i];
            float *o = out[i];
            Shape *sh = &shape[i];

            if (nd->op == OP_INPUT) continue; /* seeded above */

            float *xa = nd->ne[0] >= 0 ? out[nd->ne[0]] : NULL;
            float *xb = nd->ne[1] >= 0 ? out[nd->ne[1]] : NULL;
            float *xc = nd->ne[2] >= 0 ? out[nd->ne[2]] : NULL;

            switch (nd->op) {
                case OP_ADD:
                case OP_MUL:
                    for (u64 j = 0; j < sh->elems; j++)
                        o[j] = (nd->op == OP_ADD) ? xa[j] + xb[j] : xa[j] * xb[j];
                    break;
                case OP_SILU:
                    memcpy(o, xa, sh->elems * sizeof(float));
                    silu(o, (int)sh->elems);
                    break;
                case OP_SOFTMAX: {
                    memcpy(o, xa, sh->elems * sizeof(float));
                    u64 cols = sh->last ? sh->last : 1;
                    u64 rows = cols ? sh->elems / cols : 1;
                    per_row_softmax(o, rows, cols);
                    break;
                }
                case OP_RMS_NORM: {
                    u64 ncols = nd->src[0] ? nd->src[0]->dim[0] : 1;
                    if (ncols == 0) ncols = 1;
                    u64 rows = sh->elems / ncols;
                    for (u64 r = 0; r < rows; r++)
                        rms_norm(o + r * ncols, xa + r * ncols, nd->src[0], (int)ncols, DEFAULT_EPS);
                    break;
                }
                case OP_BIAS: {
                    u64 ncols = nd->src[0] ? nd->src[0]->n_element : 0;
                    if (ncols == 0 || sh->elems % ncols != 0) {
                        slog(WARN, "graph_compute: bias shape mismatch at node %u", i);
                        ok = false;
                        goto done;
                    }
                    float *bias = smalloc(ncols * sizeof(float));
                    if (!bias) { ok = false; goto done; }
                    tensor_get_f32_batch(nd->src[0], 0, ncols, bias);
                    u64 rows = sh->elems / ncols;
                    for (u64 r = 0; r < rows; r++) {
                        float *dst = o + r * ncols;
                        for (u64 j = 0; j < ncols; j++)
                            dst[j] = xa[r * ncols + j] + bias[j];
                    }
                    sfree(bias);
                    break;
                }
                case OP_MATMUL:
                case OP_MATMUL_T: {
                    u64 rows, cols;
                    bool trans = nd->op == OP_MATMUL_T;
                    if (!shape_matvec(shape[nd->ne[0]].elems, nd->src[0],
                                      trans, &rows, &cols)) {
                        slog(WARN, "graph_compute: matvec shape mismatch at node %u", i);
                        ok = false;
                        goto done;
                    }
                    mat_vec_mul(o, nd->src[0], xa, rows, cols, trans, s->pthreads);
                    break;
                }
                case OP_MATMULARRY: {
                    u64 batch, rows, cols;
                    if (!shape_matmat(shape[nd->ne[0]].elems, nd->src[0], false,
                                      &batch, &rows, &cols)) {
                        slog(WARN, "graph_compute: matmat shape mismatch at node %u", i);
                        ok = false;
                        goto done;
                    }
                    mat_mat_mul(o, nd->src[0], xa, batch, rows, cols, false, s->pthreads);
                    break;
                }
                case OP_EMBED: {
                    u64 n_embd = nd->src[0]->dim[nd->src[0]->ndim - 1];
                    u64 n_tok = sh->elems / n_embd;
                    for (u64 t = 0; t < n_tok; t++) {
                        i64 id = (i64)xa[t];
                        float *dst = o + t * n_embd;
                        if (id < 0) { memset(dst, 0, n_embd * sizeof(float)); continue; }
                        for (u64 j = 0; j < n_embd; j++)
                            dst[j] = tensor_get_f32(nd->src[0], (u64)id * n_embd + j);
                    }
                    break;
                }
                case OP_ROPE_NEOX: {
                    memcpy(o, xa, sh->elems * sizeof(float));
                    u64 n_tok = sh->dim0 ? sh->dim0 : 1;
                    u32 per = (u32)(sh->elems / n_tok);
                    u32 hd = s->cfg.head_dim;
                    u32 nh = hd ? per / hd : 1;
                    if (!hd) hd = per;
                    for (u64 t = 0; t < n_tok; t++)
                        rope_partial(o + t * per, nh, hd, s->cfg.rope_dim, (u32)t, GRAPH_ROPE_THETA);
                    break;
                }
                case OP_ATTN: {
                    u32 n_head  = s->cfg.n_head;
                    u32 n_kv_h  = s->cfg.n_kv_head;
                    u32 q_hd    = s->cfg.head_dim;
                    u32 kv_hd   = s->cfg.kv_head_dim;
                    if (!n_head || !n_kv_h || !q_hd || !kv_hd || n_head % n_kv_h != 0) {
                        slog(WARN, "graph_compute: attention config invalid at node %u", i);
                        ok = false;
                        goto done;
                    }
                    u32 gqa   = n_head / n_kv_h;
                    float scale = 1.0f / sqrtf((float)kv_hd);
                    u64 n_tok = sh->dim0 ? sh->dim0 : 1;
                    u32 q_dim = (u32)(sh->elems / n_tok);
                    u32 kv_dim = (u32)(shape[nd->ne[1]].elems / n_tok);
                    float *scores = smalloc((u64)n_tok * sizeof(float));
                    if (!scores) { ok = false; goto done; }
                    memset(o, 0, sh->elems * sizeof(float));
                    for (u32 h = 0; h < n_head; h++) {
                        u32 kv_h = h / gqa;
                        for (u64 qi = 0; qi < n_tok; qi++) {
                            const float *qh = xa + qi * q_dim + (u64)h * q_hd;
                            u32 n_keys = (u32)qi + 1;
                            for (u32 kj = 0; kj < n_keys; kj++) {
                                const float *kk = xb + (u64)kj * kv_dim + (u64)kv_h * kv_hd;
                                float s = 0.0f;
                                for (u32 d = 0; d < kv_hd; d++)
                                    s += qh[d] * kk[d];
                                scores[kj] = s * scale;
                            }
                            softmax(scores, n_keys);
                            float *oh = o + qi * q_dim + (u64)h * q_hd;
                            for (u32 kj = 0; kj < n_keys; kj++) {
                                const float *vv = xc + (u64)kj * kv_dim + (u64)kv_h * kv_hd;
                                float st = scores[kj];
                                for (u32 d = 0; d < kv_hd; d++)
                                    oh[d] += st * vv[d];
                            }
                        }
                    }
                    sfree(scores);
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
    }

    sfree(shape);
    return ok;
}
