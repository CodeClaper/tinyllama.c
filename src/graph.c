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
 * src[]   = source node indices, weights[] = per-source weight tensors,
 * params[] = op-specific parameters (unused ops may pass NULL). */
static u32 node_add(Graph *g, GraphOp op, const int *src, TensorInfo *const *weights, const u32 *params) {
    if (!g) return GRAPH_NODE_NONE;
    if (!grow((void **)&g->node, &g->cap, g->n_node + 1, sizeof(GraphNode))) return GRAPH_NODE_NONE;
    GraphNode *n = &g->node[g->n_node];
    n->op = op;
    for (int i = 0; i < 4; i++) {
        n->src[i]     = src ? src[i] : -1;
        n->weights[i] = weights ? weights[i] : NULL;
    }
    if (params) memcpy(n->params, params, sizeof(n->params));
    return g->n_node++;
}

/* Leaf input: carries a run of token ids.  Shape is recorded in the node
 * itself (ne[0] = n_tokens, nb[0] = element size); the data buffer is
 * allocated empty at build time and filled by graph_compute() at
 * execution time with the concrete tokens. */
u32 graph_input(Graph *g, u32 n_tokens) {
    if (!g || n_tokens == 0) return GRAPH_NODE_NONE;

    u32 n = node_add(g, OP_INPUT, (int[]){ -1, -1, -1, -1 }, NULL, NULL);
    if (n == GRAPH_NODE_NONE) return GRAPH_NODE_NONE;

    void *data = scalloc(n_tokens, sizeof(u32));   /* filled at compute time */
    if (!data) return GRAPH_NODE_NONE;

    GraphNode *nd = &g->node[n];
    nd->ne[0]    = n_tokens;       /* capacity: [n_tokens]        */
    nd->nb[0]    = sizeof(u32);    /* element stride              */
    nd->data     = data;
    nd->data_cap = (size_t)n_tokens * sizeof(u32);
    return n;
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

u32 graph_attn(Graph *g, u32 q, u32 k, u32 v, u32 layer) {
    if (!g || q >= g->n_node || k >= g->n_node || v >= g->n_node) return GRAPH_NODE_NONE;
    u32 params[] = { layer };
    return node_add(g, OP_ATTN, (int[]){ (int)q, (int)k, (int)v, -1 }, NULL, params);
}

u32 graph_rope(Graph *g, u32 src, float theta, u32 n_heads, u32 head_dim) {
    if (!g || src >= g->n_node) return GRAPH_NODE_NONE;
    u32 bits;
    memcpy(&bits, &theta, sizeof(bits));   /* float bits in params[0] */
    u32 params[] = { bits, n_heads, head_dim };
    return node_add(g, OP_ROPE_NEOX, (int[]){ (int)src, -1, -1, -1 }, NULL, params);
}

u32 graph_embed(Graph *g, u32 src, TensorInfo *weight, u32 n_tokens) {
    if (!g || !weight || weight->ndim < 2 || n_tokens == 0) return GRAPH_NODE_NONE;
    if (src == GRAPH_NODE_NONE || src >= g->n_node) return GRAPH_NODE_NONE;
    const GraphNode *tok = &g->node[src];
    if (tok->op != OP_INPUT || tok->ne[0] != (int)n_tokens) return GRAPH_NODE_NONE;
    return node_add(g, OP_EMBED, (int[]){ (int)src, -1, -1, -1 },
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
    /* The OP_INPUT buffer (u32 token ids) is allocated at build time;
     * every other node's data buffer is owned by the executor. */
    for (u32 i = 0; i < g->n_node; i++)
        sfree(g->node[i].data);
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
 * ---------------------------------------------------------------- *
 * Incremental (Model B): one call runs one batch of n rows starting at
 * absolute position b->pos.  Each row is RoPE'd at its absolute
 * position; OP_ATTN first writes the batch's K/V into the session KV
 * cache and then attends causally over the cached keys.  Consecutive
 * calls with (n = 1) at growing positions are incremental decoding —
 * the history is never recomputed, so per-token cost stays flat.
 *
 * Sink nodes (no consumers — the LM head in practice) are computed for
 * the batch's last row only, mirroring qwen25_prefill, which runs its
 * final norm + LM head on the last position only.
 *
 * The graph is built once at capacity (ctx_size rows); batches of any
 * n <= capacity reuse it.                                              */

/* Grow a node output buffer to `need` bytes (grow-only; sessions only
 * ever grow their maximum batch size within one graph lifetime). */
static bool ensure_data(GraphNode *nd, size_t need) {
    if (nd->data_cap >= need) return true;
    void *p = srealloc(nd->data, need);
    if (!p) return false;
    nd->data     = p;
    nd->data_cap = need;
    return true;
}

bool graph_compute(Graph *g, GraphPlan *plan, const GraphBatch *b, Session *s) {
    (void)plan;
    if (!g || !s || !b || g->n_node == 0) {
        slog(WARN, "graph_compute: missing graph / session / batch");
        return false;
    }

    u32 pos = b->pos;
    u32 n   = b->n;
    if (n == 0) return true;
    if ((u64)pos + n > s->ctx_size) {
        slog(WARN, "graph_compute: batch pos=%u n=%u exceeds ctx_size=%u",
             pos, n, s->ctx_size);
        return false;
    }

    /* Bind the batch's token ids into each OP_INPUT leaf. */
    for (u32 i = 0; i < g->n_node; i++) {
        GraphNode *nd = &g->node[i];
        if (nd->op != OP_INPUT) continue;
        if (n > (u32)nd->ne[0]) {
            slog(WARN, "graph_compute: batch of %u exceeds graph capacity", n);
            return false;
        }
        if (b->tokens) memcpy(nd->data, b->tokens, (size_t)n * sizeof(u32));
    }

    ArchConfig *c = &s->cfg;
    u32 n_node = g->n_node;

    /* Sink detection: a node is a sink iff no other node reads it.
     * (Edges point strictly backward, so scanning all nodes is enough.) */
    u32 sink[n_node];
    for (u32 i = 0; i < n_node; i++) sink[i] = 1;
    for (u32 j = 0; j < n_node; j++)
        for (int k = 0; k < 4; k++)
            if (g->node[j].src[k] >= 0) sink[(u32)g->node[j].src[k]] = 0;

    /* Elements per row of each node's output (float count). */
    u32 dims[n_node];
    for (u32 i = 0; i < n_node; i++) {
        GraphNode *nd = &g->node[i];
        switch (nd->op) {
            case OP_INPUT:
                dims[i] = 0;
                break;
            case OP_MATMUL_T:
            case OP_MATMUL:
            case OP_MATMULARRY: {
                TensorInfo *w = nd->weights[0];
                dims[i] = (u32)(nd->op == OP_MATMUL_T ? w->dim[1] : w->dim[0]);
                break;
            }
            case OP_EMBED: {
                TensorInfo *w = nd->weights[0];
                dims[i] = (u32)(w->dim[0] == (i64)c->n_vocab ? w->dim[1] : w->dim[0]);
                break;
            }
            case OP_RMS_NORM:
                dims[i] = (u32)nd->weights[0]->n_element;
                break;
            case OP_ATTN:
                dims[i] = c->n_head * c->head_dim;
                break;
            default:
                dims[i] = (u32)nd->src[0] >= 0 ? dims[(u32)nd->src[0]] : 0;
        }
    }

    /* Attention score scratch: the batch's last row attends every key
     * in the cache, positions 0..pos+n-1. */
    float *scr = smalloc((size_t)(pos + n) * sizeof(float));
    if (!scr) return false;

    u32 q_dim  = c->n_head * c->head_dim;
    u32 kv_dim = c->n_kv_head * c->kv_head_dim;
    float scale = 1.0f / sqrtf((float)c->kv_head_dim);

    /* Walk the graph in build order (== topo order). */
    for (u32 i = 0; i < n_node; i++) {
        GraphNode *nd = &g->node[i];
        if (nd->op == OP_INPUT) continue;

        u32 od = dims[i];
        u32 r   = sink[i] ? 1u : n;             /* sink: last batch row only */
        u32 base = sink[i] ? n - 1 : 0;         /* batch row of local row 0  */
        if (od == 0 || r == 0) goto fail;
        if (!ensure_data(nd, (size_t)r * od * sizeof(float))) goto fail;
        float *dst = (float *)nd->data;

        switch (nd->op) {
            case OP_EMBED: {
                TensorInfo *te = nd->weights[0];
                bool te_trans = (te->dim[0] == (i64)c->n_vocab);
                u32 *tok = (u32 *)g->node[(u32)nd->src[0]].data;
                for (u32 p = 0; p < n; p++) {
                    float *dp = dst + (u64)p * od;
                    if (te_trans) {
                        tensor_get_f32_batch(te, (u64)tok[p] * od, od, dp);
                    } else {
                        for (u32 j = 0; j < od; j++)
                            dp[j] = tensor_get_f32(te, (u64)j * c->n_vocab + tok[p]);
                    }
                }
                break;
            }
            case OP_RMS_NORM: {
                TensorInfo *tw = nd->weights[0];
                float *src = (float *)g->node[(u32)nd->src[0]].data;
                for (u32 p = 0; p < r; p++)
                    rms_norm(dst + (u64)p * od, src + ((u64)base + p) * od,
                             tw, (int)od, DEFAULT_EPS);
                break;
            }
            case OP_MATMUL_T:
            case OP_MATMUL:
            case OP_MATMULARRY: {
                TensorInfo *w  = nd->weights[0];
                bool tr        = (nd->op == OP_MATMUL_T);
                u32 in         = (u32)(tr ? w->dim[0] : w->dim[1]);
                float *src     = (float *)g->node[(u32)nd->src[0]].data;
                float *x       = src + (u64)base * in;
                if (r == 1) {
                    if (!mat_vec_mul(dst, w, x, od, in, tr, s->pthreads)) goto fail;
                } else {
                    if (!mat_mat_mul(dst, w, x, r, od, in, tr, s->pthreads)) goto fail;
                }
                break;
            }
            case OP_BIAS: {
                TensorInfo *tb = nd->weights[0];
                float *src = (float *)g->node[(u32)nd->src[0]].data;
                u32 n = tb->n_element < (u64)od ? (u32)tb->n_element : od;
                for (u32 p = 0; p < r; p++) {
                    float *dp = dst + (u64)p * od;
                    memcpy(dp, src + ((u64)base + p) * od, (size_t)od * sizeof(float));
                    bias_add(dp, tb, n);
                }
                break;
            }
            case OP_ADD:
            case OP_MUL: {
                float *a = (float *)g->node[(u32)nd->src[0]].data;
                float *b = (float *)g->node[(u32)nd->src[1]].data;
                for (u32 p = 0; p < r; p++) {
                    float *dp = dst + (u64)p * od;
                    float *ap = a + ((u64)base + p) * od;
                    float *bp = b + ((u64)base + p) * od;
                    if (nd->op == OP_ADD)
                        for (u32 j = 0; j < od; j++) dp[j] = ap[j] + bp[j];
                    else
                        for (u32 j = 0; j < od; j++) dp[j] = ap[j] * bp[j];
                }
                break;
            }
            case OP_SILU:
                for (u32 p = 0; p < r; p++) {
                    float *dp = dst + (u64)p * od;
                    memcpy(dp, (float *)g->node[(u32)nd->src[0]].data +
                           ((u64)base + p) * od, (size_t)od * sizeof(float));
                    silu(dp, (int)od);
                }
                break;
            case OP_SOFTMAX:
                for (u32 p = 0; p < r; p++) {
                    float *dp = dst + (u64)p * od;
                    memcpy(dp, (float *)g->node[(u32)nd->src[0]].data +
                           ((u64)base + p) * od, (size_t)od * sizeof(float));
                    softmax(dp, od);
                }
                break;
            case OP_ROPE_NEOX: {
                float theta;
                memcpy(&theta, &nd->params[0], sizeof(theta));
                u32 heads = nd->params[1];
                u32 hdim  = nd->params[2];
                float *src = (float *)g->node[(u32)nd->src[0]].data;
                for (u32 p = 0; p < r; p++) {
                    u32 abspos = pos + base + p;   /* absolute position */
                    float *dp  = dst + (u64)p * od;
                    memcpy(dp, src + (u64)(base + p) * od,
                           (size_t)od * sizeof(float));
                    rope_neox(dp, heads, hdim, abspos, theta);
                }
                break;
            }
            case OP_ATTN: {
                /* Write the batch's K/V into the session cache at
                 * positions pos..pos+n-1, then attend causally over the
                 * cached keys (rows 0..pos+qi) — history included. */
                if (!s->cache.std) {
                    slog(WARN, "graph_compute: OP_ATTN requires the std KV cache");
                    goto fail;
                }
                u32 layer = nd->params[0];
                AttnKvCache *akc = &s->cache.std[layer];
                if (!akc->k || !akc->v) goto fail;

                GraphNode *nq = &g->node[(u32)nd->src[0]];
                GraphNode *nk = &g->node[(u32)nd->src[1]];
                GraphNode *nv = &g->node[(u32)nd->src[2]];
                float *qd = (float *)nq->data;
                float *kd = (float *)nk->data;
                float *vd = (float *)nv->data;
                u32 n_head = c->n_head;
                u32 n_kv   = c->n_kv_head;
                u32 hd     = c->head_dim;
                u32 khd    = c->kv_head_dim;
                u32 gqa    = n_head / n_kv;
                u32 hs     = akc->cap * khd;   /* stride between heads */

                /* K/V write-through (head-major: [head][pos][dim]). */
                for (u32 i = 0; i < n; i++) {
                    const float *kr = kd + (u64)i * kv_dim;
                    const float *vr = vd + (u64)i * kv_dim;
                    for (u32 h = 0; h < n_kv; h++) {
                        memcpy(akc->k + (u64)h * hs + (u64)(pos + i) * khd,
                               kr + (u64)h * khd, (size_t)khd * sizeof(float));
                        memcpy(akc->v + (u64)h * hs + (u64)(pos + i) * khd,
                               vr + (u64)h * khd, (size_t)khd * sizeof(float));
                    }
                }
                akc->n = pos + n;

                memset(dst, 0, (size_t)n * q_dim * sizeof(float));
                for (u32 h = 0; h < n_head; h++) {
                    u32 kv_h = h / gqa;
                    float *kh_base = akc->k + (u64)kv_h * hs;
                    float *vh_base = akc->v + (u64)kv_h * hs;
                    for (u32 qi = 0; qi < n; qi++) {
                        float *qh = qd + (u64)qi * q_dim + (u64)h * hd;
                        u32 n_keys = pos + qi + 1;   /* causal */

                        float *kt = kh_base;
                        for (u32 t = 0; t < n_keys; t++, kt += khd) {
                            float acc = 0.0f;
                            for (u32 d = 0; d < khd; d++)
                                acc += qh[d] * kt[d];
                            scr[t] = acc * scale;
                        }
                        softmax(scr, n_keys);

                        float *oh = dst + (u64)qi * q_dim + (u64)h * hd;
                        float *vt = vh_base;
                        for (u32 t = 0; t < n_keys; t++, vt += khd) {
                            float st = scr[t];
                            for (u32 d = 0; d < khd; d++)
                                oh[d] += st * vt[d];
                        }
                    }
                }
                break;
            }
            default:
                slog(WARN, "graph_compute: unhandled op %d", (int)nd->op);
                goto fail;
        }
    }

    /* Sink output (last row of the LM head) → session logits. */
    for (u32 i = 0; i < n_node; i++) {
        if (!sink[i] || g->node[i].op == OP_INPUT) continue;
        u32 n = dims[i] < c->n_vocab ? dims[i] : c->n_vocab;
        memcpy(s->logits, g->node[i].data, (size_t)n * sizeof(float));
        break;
    }

    sfree(scr);
    return true;

fail:
    sfree(scr);
    return false;
}
