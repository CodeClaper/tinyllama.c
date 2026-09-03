#include <string.h>
#include <math.h>
#include "graph.h"
#include "core.h"
#include "mm.h"
#include "slog.h"

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
static u32 node_add(Graph *g, GraphOp op, const int *src, TensorInfo *weight,
                    void *param, u64 out_elems, u32 out_ndim, const u64 *out_dim) {
    if (!g || out_elems == 0) return GRAPH_NODE_NONE;
    if (!grow((void **)&g->node, &g->cap, g->n_node + 1, sizeof(GraphNode))) return GRAPH_NODE_NONE;

    GraphNode *n = &g->node[g->n_node];
    n->op        = op;
    n->weight    = weight;
    n->param     = param;
    n->out_elems = out_elems;
    n->out_ndim  = out_ndim;
    for (u32 i = 0; i < MAX_DIMS; i++) n->out_dim[i] = out_dim ? out_dim[i] : 0;
    for (int i = 0; i < 4; i++) n->src[i] = src ? src[i] : -1;
    return g->n_node++;
}

u32 graph_input(Graph *g, u32 n_element) {
    u64 dim[1] = { n_element };
    return node_add(g, OP_INPUT, NULL, NULL, NULL, n_element, 1, dim);
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
    return node_add(g, OP_RMS_NORM, (int[]){src,-1,-1,-1}, weight, NULL,
                    a->out_elems, a->out_ndim, a->out_dim);
}

u32 graph_mul_mat(Graph *g, u32 src, TensorInfo *weight, bool trans) {
    const GraphNode *a = src < g->n_node ? &g->node[src] : NULL;
    u64 rows = 0, cols = 0;
    if (!a || !shape_matvec(a, weight, trans, &rows, &cols)) return GRAPH_NODE_NONE;
    u64 dim[1] = { rows };
    return node_add(g, trans ? OP_MATMUL_T : OP_MATMUL, (int[]){src,-1,-1,-1},
                    weight, NULL, rows, 1, dim);
}

u32 graph_mul_mat2(Graph *g, u32 src, TensorInfo *weight) {
    const GraphNode *a = src < g->n_node ? &g->node[src] : NULL;
    u64 batch = 0, rows = 0, cols = 0;
    if (!a || !shape_matmat(a, weight, false, &batch, &rows, &cols)) return GRAPH_NODE_NONE;
    u64 dim[2] = { batch, rows };
    return node_add(g, OP_MATMULARRY, (int[]){src,-1,-1,-1}, weight, NULL,
                    batch * rows, 2, dim);
}

u32 graph_binary(Graph *g, GraphOp op, u32 a_id, u32 b_id) {
    if (op != OP_ADD && op != OP_MUL) return GRAPH_NODE_NONE;
    const GraphNode *a = a_id < g->n_node ? &g->node[a_id] : NULL;
    const GraphNode *b = b_id < g->n_node ? &g->node[b_id] : NULL;
    if (!a || !b || a->out_elems != b->out_elems) return GRAPH_NODE_NONE;
    return node_add(g, op, (int[]){a_id,b_id,-1,-1}, NULL, NULL,
                    a->out_elems, a->out_ndim, a->out_dim);
}

u32 graph_silu(Graph *g, u32 src) {
    const GraphNode *a = src < g->n_node ? &g->node[src] : NULL;
    if (!a) return GRAPH_NODE_NONE;
    return node_add(g, OP_SILU, (int[]){src,-1,-1,-1}, NULL, NULL,
                    a->out_elems, a->out_ndim, a->out_dim);
}

u32 graph_softmax(Graph *g, u32 src) {
    const GraphNode *a = src < g->n_node ? &g->node[src] : NULL;
    if (!a) return GRAPH_NODE_NONE;
    return node_add(g, OP_SOFTMAX, (int[]){src,-1,-1,-1}, NULL, NULL,
                    a->out_elems, a->out_ndim, a->out_dim);
}

u32 graph_rope(Graph *g, u32 src) {
    const GraphNode *a = src < g->n_node ? &g->node[src] : NULL;
    if (!a) return GRAPH_NODE_NONE;
    return node_add(g, OP_ROPE_NEOX, (int[]){src,-1,-1,-1}, NULL, NULL,
                    a->out_elems, a->out_ndim, a->out_dim);
}

u32 graph_embed(Graph *g, u32 token_id, TensorInfo *weight, u32 n_tokens) {
    const GraphNode *tok = token_id < g->n_node ? &g->node[token_id] : NULL;
    if (!tok || !weight || weight->ndim < 2) return GRAPH_NODE_NONE;
    u64 n_embd = weight->dim[weight->ndim - 1];
    u64 dim[2] = { n_tokens, n_embd };
    return node_add(g, OP_EMBED, (int[]){token_id,-1,-1,-1}, weight, NULL,
                    n_tokens * n_embd, 2, dim);
}

u32 graph_attention(Graph *g, u32 q, u32 k, u32 v, GraphAttnParam *param) {
    if (!param) return GRAPH_NODE_NONE;
    const GraphNode *nq = q < g->n_node ? &g->node[q] : NULL;
    const GraphNode *nk = k < g->n_node ? &g->node[k] : NULL;
    const GraphNode *nv = v < g->n_node ? &g->node[v] : NULL;
    if (!nq || !nk || !nv) return GRAPH_NODE_NONE;

    /* q output: [batch, n_head * head_dim]; attn matches this shape. */
    u64 batch = param->n_tokens;
    u64 elems = batch * (u64)param->n_heads * param->head_dim;
    u64 dim[2] = { batch, (u64)param->n_heads * param->head_dim };
    return node_add(g, OP_ATTENTION, (int[]){q,k,v,-1}, NULL, param, elems, 2, dim);
}

/* ---------------------------------------------------------------- *
 * Graph creation
 * ---------------------------------------------------------------- */

Graph *graph_new(void) {
    return scalloc(1, sizeof(Graph));
}

void graph_free(Graph *g) {
    if (!g) return;
    for (u32 i = 0; i < g->n_node; i++)
        sfree(g->node[i].param);
    sfree(g->node);
    sfree(g);
}

Graph *graph_build(Session *s, GraphMode mode, const u32 *tokens,
                   u32 n_tokens, u32 *logits_out) {
    if (!s || !s->en || !s->en->weights || n_tokens == 0) return NULL;
    Weights  *w = s->en->weights;
    ArchConfig *c = &s->cfg;

    if (s->en->model->arch == ARCH_DEEPSEEK) {
        slog(WARN, "graph_build: MLA (DeepSeek) forward not yet built");
        return NULL;
    }

    Graph *g = graph_new();

    u32 input = graph_input(g, n_tokens);

    u32 hid = graph_embed(g, input, w->tensors[TENSOR_TOKEN_EMBD], n_tokens);

    /* start = first new token's position in the cache. */
    u32 start = s->n_tokens;

    for (u32 li = 0; li < c->n_layer; li++) {
        LayerWeights *lw = &w->layers[li];

        /* ---- Attention ---- */
        TensorInfo *pre_norm = lw->tensors[TENSOR_ATTN_NORM];
        u32 hn = graph_rms_norm(g, hid, pre_norm);

        TensorInfo *wq = lw->tensors[TENSOR_ATTN_Q];
        TensorInfo *wk = lw->tensors[TENSOR_ATTN_K];
        TensorInfo *wv = lw->tensors[TENSOR_ATTN_V];
        TensorInfo *wo = lw->tensors[TENSOR_ATTN_OUT];
        if (!wq || !wk || !wv || !wo) {
            slog(WARN, "graph_build: layer %u missing attention weights", li);
            graph_free(g);
            return NULL;
        }

        u32 q = graph_mul_mat2(g, hn, wq);
        u32 k = graph_mul_mat2(g, hn, wk);
        u32 v = graph_mul_mat2(g, hn, wv);

        GraphAttnParam *ap = smalloc(sizeof(*ap));
        memset(ap, 0, sizeof(*ap));
        ap->cache = &s->cache.std[li];
        ap->n_heads = c->n_head ? c->n_head : c->n_embd / (c->head_dim ? c->head_dim : 1);
        ap->n_kv_head = c->n_kv_head;
        ap->head_dim = c->head_dim;
        ap->kv_head_dim = c->kv_head_dim;
        ap->ctx_cap = s->ctx_size;
        ap->start = start;
        ap->n_tokens = n_tokens;
        ap->rope_theta = 10000.0f;

        u32 a = graph_attention(g, q, k, v, ap);
        if (a == GRAPH_NODE_NONE) { graph_free(g); return NULL; }

        u32 ao = graph_mul_mat2(g, a, wo);
        hid = graph_binary(g, OP_ADD, hid, ao);

        /* ---- FFN ---- */
        TensorInfo *post_norm = lw->tensors[TENSOR_POST_ATTN_NORM];
        if (!post_norm) post_norm = lw->tensors[TENSOR_ATTN_NORM];
        TensorInfo *wg = lw->tensors[TENSOR_FFN_GATE];
        TensorInfo *wu = lw->tensors[TENSOR_FFN_UP];
        TensorInfo *wd = lw->tensors[TENSOR_FFN_DOWN];
        if (!wg || !wu || !wd) {
            slog(WARN, "graph_build: layer %u missing FFN weights", li);
            graph_free(g);
            return NULL;
        }

        u32 hn2 = graph_rms_norm(g, hid, post_norm);
        u32 gate = graph_mul_mat2(g, hn2, wg);
        u32 up   = graph_mul_mat2(g, hn2, wu);
        u32 gs   = graph_silu(g, gate);
        u32 gated = graph_binary(g, OP_MUL, gs, up);
        u32 down = graph_mul_mat2(g, gated, wd);
        hid = graph_binary(g, OP_ADD, hid, down);
    }

    TensorInfo *out_norm = w->tensors[TENSOR_OUTPUT_NORM];
    TensorInfo *out_w = w->tensors[TENSOR_OUTPUT];
    if (!out_norm || !out_w) {
        slog(WARN, "graph_build: missing output norm / weight");
        graph_free(g);
        return NULL;
    }

    u32 hn = graph_rms_norm(g, hid, out_norm);
    u32 logits = graph_mul_mat2(g, hn, out_w);
    if (logits_out) *logits_out = logits;

    (void)mode;
    (void)tokens;
    return g;
}

/* ---------------------------------------------------------------- *
 * Graph execution
 * ---------------------------------------------------------------- */

static void per_row_softmax(float *x, u64 rows, u64 cols) {
    for (u64 r = 0; r < rows; r++)
        softmax(x + r * cols, (u32)cols);
}

bool graph_compute(Graph *g, const u32 *tokens, const GraphRunCtx *ctx) {
    if (!g || !ctx) {
        slog(WARN, "graph_compute: missing graph / ctx");
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
                             nd->weight, (int)ncols, ctx->eps);
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
                mat_vec_mul(o, nd->weight, xa, rows, cols, trans, ctx->pool);
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
                mat_mat_mul(o, nd->weight, xa, batch, rows, cols, false, ctx->pool);
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
                u32 nh = (ctx->head_dim && (u64)ctx->head_dim) ? per / ctx->head_dim : 1;
                u32 hd = ctx->head_dim ? ctx->head_dim : per;
                for (u64 t = 0; t < n_tok; t++)
                    rope_partial(o + t * per, nh, hd, ctx->rope_dim,
                                 (u32)t, ctx->theta_base);
                break;
            }
            case OP_ATTENTION: {
                GraphAttnParam *ap = nd->param;
                float *q = out[nd->src[0]];
                float *k = out[nd->src[1]];
                float *v = out[nd->src[2]];
                if (!graph_attention_run(ap, q, k, v, o, ctx)) {
                    slog(WARN, "graph_compute: attention failed at node %u", i);
                    ok = false;
                    goto done;
                }
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

/* Fused per-head causal attention + KV-cache update (OP_ATTENTION). */
bool graph_attention_run(GraphAttnParam *p, float *q, const float *k,
                         const float *v, float *out, const GraphRunCtx *ctx) {
    u32 n_head = p->n_heads, n_kv = p->n_kv_head;
    u32 hd = p->head_dim, kvhd = p->kv_head_dim;
    u32 n_tok = p->n_tokens, start = p->start, cap = p->ctx_cap;
    if (!p->cache || hd == 0 || kvhd == 0) return false;
    if (n_kv == 0) n_kv = n_head;
    u32 group = n_head / n_kv;
    float theta = ctx->theta_base ? ctx->theta_base : p->rope_theta;

    AttnKvCache *kc = p->cache;
    u32 pos_stride = n_kv * kvhd; /* per-position K/V block */

    /* 1) write rope-applied K and raw V into the cache. */
    for (u32 t = 0; t < n_tok; t++) {
        u32 pos = start + t;
        if (pos >= cap) return false;
        for (u32 h = 0; h < n_kv; h++) {
            const float *kv = k + t * pos_stride + h * kvhd;
            float *dstk = kc->k + (u64)pos * pos_stride + h * kvhd;
            memcpy(dstk, kv, kvhd * sizeof(float));
            if (ctx->rope_dim)
                rope_partial(dstk, 1, kvhd, ctx->rope_dim, pos, theta);
            else
                rope_neox(dstk, 1, kvhd, pos, theta);
            const float *kv2 = v + t * pos_stride + h * kvhd;
            float *dstv = kc->v + (u64)pos * pos_stride + h * kvhd;
            memcpy(dstv, kv2, kvhd * sizeof(float));
        }
    }

    /* scratch for scores */
    float *scores = smalloc((cap <= (u32)-1 ? (u64)cap : 0) * sizeof(float));
    if (!scores) return false;

    /* 2) per token, per q-head: causal attention over [0, pos]. */
    for (u32 t = 0; t < n_tok; t++) {
        u32 pos = start + t;
        for (u32 h = 0; h < n_head; h++) {
            float *qh = (float *)(q + (u64)t * n_head * hd + h * hd);
            if (ctx->rope_dim)
                rope_partial(qh, 1, hd, ctx->rope_dim, pos, theta);
            else
                rope_neox(qh, 1, hd, pos, theta);

            u32 kvh = h / group;
            u32 nkey = pos + 1; /* causal window */

            float mx = -INFINITY;
            for (u32 pk = 0; pk < nkey; pk++) {
                const float *kk = kc->k + (u64)pk * pos_stride + kvh * kvhd;
                float d = 0.0f;
                for (u32 j = 0; j < kvhd; j++) d += qh[j] * kk[j];
                scores[pk] = d / sqrtf((float)kvhd);
                if (scores[pk] > mx) mx = scores[pk];
            }
            float sum = 0.0f;
            for (u32 pk = 0; pk < nkey; pk++) {
                scores[pk] = expf(scores[pk] - mx);
                sum += scores[pk];
            }
            for (u32 pk = 0; pk < nkey; pk++) scores[pk] /= sum;

            float *oh = out + (u64)t * n_head * hd + h * hd;
            for (u32 j = 0; j < hd; j++) oh[j] = 0.0f;
            for (u32 pk = 0; pk < nkey; pk++) {
                float w = scores[pk];
                const float *vv = kc->v + (u64)pk * pos_stride + kvh * kvhd;
                for (u32 j = 0; j < hd && j < kvhd; j++) oh[j] += w * vv[j];
            }
        }
    }

    sfree(scores);
    return true;
}
