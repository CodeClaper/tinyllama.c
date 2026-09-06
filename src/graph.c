#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "graph.h"
#include "core.h"
#include "mm.h"
#include "slog.h"

/* ---- Per-op-class timing (debug aid, dumped by graph_free) ------- */
enum {
    ST_EMBED, ST_RMS, ST_MM, ST_MM_T, ST_ATTN, ST_ROPE,
    ST_SILU, ST_BIAS, ST_BIN, ST_GATE, ST_SSM, ST_OTHER
};
static const char *const op_stat_name[] = {
    "embed", "rms_norm", "matmul", "matmul_T", "attn",
    "rope", "silu", "bias", "add/mul", "gate", "ssm", "other"
};
static double op_stat_secs[sizeof(op_stat_name) / sizeof(op_stat_name[0])];
static u64    op_stat_calls[sizeof(op_stat_name) / sizeof(op_stat_name[0])];

static double graph_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int op_stat_class(GraphOp op) {
    switch (op) {
        case OP_EMBED:          return ST_EMBED;
        case OP_RMS_NORM:
        case OP_RMS_NORM_HEADS: return ST_RMS;
        case OP_MATMUL:
        case OP_MATMULARRY:     return ST_MM;
        case OP_MATMUL_T:       return ST_MM_T;
        case OP_ATTN:           return ST_ATTN;
        case OP_ROPE_NEOX:      return ST_ROPE;
        case OP_SILU:           return ST_SILU;
        case OP_BIAS:           return ST_BIAS;
        case OP_ADD:
        case OP_MUL:            return ST_BIN;
        case OP_SIGMOID_GATE:   return ST_GATE;
        case OP_SSM_CONV:
        case OP_SSM_DELTA:      return ST_SSM;
        default:                return ST_OTHER;
    }
}

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
 * params[] = op-specific parameters, n_params = how many u32s of it are
 * valid (ops that need none pass NULL / 0).  The node's full params block
 * is always zeroed, so unspecified entries read back as 0.
 *
 * Every builder below hands node_add() a brace literal with exactly
 * GRAPH_NODE_MAX_SRC entries — reading past it would be out of bounds,
 * so the macro and the literals must move together. */
_Static_assert(GRAPH_NODE_MAX_SRC == 4,
               "graph builders pass src/weights literals with 4 entries");
static u32 node_add(Graph *g, GraphOp op, const int *src, TensorInfo *const *weights,
                    const u32 *params, u32 n_params) {
    if (!g) return GRAPH_NODE_NONE;
    if (n_params > sizeof(((GraphNode *)0)->params) / sizeof(u32)) return GRAPH_NODE_NONE;
    if (!grow((void **)&g->node, &g->cap, g->n_node + 1, sizeof(GraphNode))) return GRAPH_NODE_NONE;
    GraphNode *n = &g->node[g->n_node];
    n->op = op;
    for (int i = 0; i < GRAPH_NODE_MAX_SRC; i++) {
        n->src[i]     = src ? src[i] : -1;
        n->weights[i] = weights ? weights[i] : NULL;
    }
    memset(n->params, 0, sizeof(n->params));
    if (params && n_params) memcpy(n->params, params, n_params * sizeof(u32));
    return g->n_node++;
}

/* Leaf input: carries a run of token ids.  Shape (ne[0] = capacity,
 * nb[0] = element size) is recorded in the node itself; the data slot
 * lives in the graph arena and is filled by graph_compute() at
 * execution time with the concrete tokens. */
u32 graph_input(Graph *g, u32 n_tokens) {
    if (!g || n_tokens == 0) return GRAPH_NODE_NONE;

    u32 n = node_add(g, OP_INPUT, (int[]){ -1, -1, -1, -1 }, NULL, NULL, 0);
    if (n == GRAPH_NODE_NONE) return GRAPH_NODE_NONE;

    GraphNode *node = &g->node[n];
    node->ne[0] = n_tokens;       /* capacity: [n_tokens]     */
    node->nb[0] = sizeof(u32);    /* element stride           */
    return n;
}

u32 graph_rms_norm(Graph *g, u32 src, TensorInfo *weight) {
    if (!g || src >= g->n_node || !weight) return GRAPH_NODE_NONE;
    return node_add(g, OP_RMS_NORM, (int[]){ (int)src, -1, -1, -1 },
                    (TensorInfo *[]){ weight, NULL, NULL, NULL }, NULL, 0);
}

u32 graph_rms_norm_heads(Graph *g, u32 src, TensorInfo *weight,
                         u32 n_heads, u32 head_dim, u32 in_stride, u32 out_stride) {
    if (!g || src >= g->n_node || !weight) return GRAPH_NODE_NONE;
    if (n_heads == 0 || head_dim == 0) return GRAPH_NODE_NONE;
    if (in_stride < head_dim || out_stride < head_dim) return GRAPH_NODE_NONE;
    u32 params[] = { n_heads, head_dim, in_stride, out_stride };
    return node_add(g, OP_RMS_NORM_HEADS, (int[]){ (int)src, -1, -1, -1 },
                    (TensorInfo *[]){ weight, NULL, NULL, NULL }, params, 4);
}

u32 graph_mul_mat(Graph *g, u32 src, TensorInfo *weight, bool trans) {
    if (!g || src >= g->n_node || !weight || weight->ndim < 2) return GRAPH_NODE_NONE;
    return node_add(g, trans ? OP_MATMUL_T : OP_MATMUL, (int[]){ (int)src, -1, -1, -1 },
                    (TensorInfo *[]){ weight, NULL, NULL, NULL }, NULL, 0);
}

u32 graph_binary(Graph *g, GraphOp op, u32 a_id, u32 b_id) {
    if (op != OP_ADD && op != OP_MUL) return GRAPH_NODE_NONE;
    if (!g || a_id >= g->n_node || b_id >= g->n_node) return GRAPH_NODE_NONE;
    return node_add(g, op, (int[]){ (int)a_id, (int)b_id, -1, -1 }, NULL, NULL, 0);
}

u32 graph_silu(Graph *g, u32 src) {
    if (!g || src >= g->n_node) return GRAPH_NODE_NONE;
    return node_add(g, OP_SILU, (int[]){ (int)src, -1, -1, -1 }, NULL, NULL, 0);
}

u32 graph_softmax(Graph *g, u32 src) {
    if (!g || src >= g->n_node) return GRAPH_NODE_NONE;
    return node_add(g, OP_SOFTMAX, (int[]){ (int)src, -1, -1, -1 }, NULL, NULL, 0);
}

u32 graph_bias(Graph *g, u32 src, TensorInfo *bias) {
    if (!g || src >= g->n_node || !bias || bias->ndim < 1) return GRAPH_NODE_NONE;
    return node_add(g, OP_BIAS, (int[]){ (int)src, -1, -1, -1 },
                    (TensorInfo *[]){ bias, NULL, NULL, NULL }, NULL, 0);
}

u32 graph_sigmoid_gate(Graph *g, u32 a, u32 gate, u32 n_heads, u32 head_dim) {
    if (!g || a >= g->n_node || gate >= g->n_node) return GRAPH_NODE_NONE;
    if (n_heads == 0 || head_dim == 0) return GRAPH_NODE_NONE;
    u32 params[] = { n_heads, head_dim };
    return node_add(g, OP_SIGMOID_GATE, (int[]){ (int)a, (int)gate, -1, -1 },
                    NULL, params, 2);
}

u32 graph_ssm_conv(Graph *g, u32 src, TensorInfo *weight, u32 state, u32 kernel) {
    if (!g || src >= g->n_node || !weight || kernel == 0) return GRAPH_NODE_NONE;
    if (state >= g->n_state) return GRAPH_NODE_NONE;
    u32 params[] = { state, kernel };
    return node_add(g, OP_SSM_CONV, (int[]){ (int)src, -1, -1, -1 },
                    (TensorInfo *[]){ weight, NULL, NULL, NULL }, params, 2);
}

u32 graph_ssm_delta(Graph *g, u32 fused, u32 alpha, u32 beta,
                    TensorInfo *ssm_a, TensorInfo *dt_bias, TensorInfo *norm,
                    u32 state, u32 n_v_heads, u32 n_k_heads, u32 head_dim) {
    if (!g || fused >= g->n_node || alpha >= g->n_node || beta >= g->n_node)
        return GRAPH_NODE_NONE;
    if (!ssm_a || !dt_bias || n_v_heads == 0 || head_dim == 0) return GRAPH_NODE_NONE;
    if (n_k_heads == 0 || n_v_heads % n_k_heads != 0) return GRAPH_NODE_NONE;
    if (state >= g->n_state) return GRAPH_NODE_NONE;
    u32 params[] = { state, n_v_heads, n_k_heads, head_dim };
    return node_add(g, OP_SSM_DELTA,
                    (int[]){ (int)fused, (int)alpha, (int)beta, -1 },
                    (TensorInfo *[]){ ssm_a, dt_bias, norm, NULL }, params, 4);
}

/* Caller-owned state buffers.  The graph only borrows the pointer, so the
 * owner (the arch workspace) keeps control of allocation and reset. */
u32 graph_state(Graph *g, void *ptr) {
    if (!g || !ptr) return GRAPH_NODE_NONE;
    if (!grow((void **)&g->state, &g->state_cap, g->n_state + 1, sizeof(void *)))
        return GRAPH_NODE_NONE;
    g->state[g->n_state] = ptr;
    return g->n_state++;
}

u32 graph_attn(Graph *g, u32 q, u32 k, u32 v, u32 layer) {
    if (!g || q >= g->n_node || k >= g->n_node || v >= g->n_node) return GRAPH_NODE_NONE;
    u32 params[] = { layer };
    return node_add(g, OP_ATTN, (int[]){ (int)q, (int)k, (int)v, -1 }, NULL, params, 1);
}

u32 graph_rope(Graph *g, u32 src, float theta, u32 n_heads, u32 head_dim, u32 rope_dim) {
    if (!g || src >= g->n_node) return GRAPH_NODE_NONE;
    if (head_dim == 0 || rope_dim == 0 || rope_dim > head_dim) return GRAPH_NODE_NONE;
    u32 bits;
    memcpy(&bits, &theta, sizeof(bits));   /* float bits in params[0] */
    u32 params[] = { bits, n_heads, head_dim, rope_dim };
    return node_add(g, OP_ROPE_NEOX, (int[]){ (int)src, -1, -1, -1 }, NULL, params, 4);
}

u32 graph_embed(Graph *g, u32 src, TensorInfo *weight, u32 n_tokens) {
    if (!g || !weight || weight->ndim < 2 || n_tokens == 0) return GRAPH_NODE_NONE;
    if (src == GRAPH_NODE_NONE || src >= g->n_node) return GRAPH_NODE_NONE;
    const GraphNode *tok = &g->node[src];
    if (tok->op != OP_INPUT || tok->ne[0] != (int)n_tokens) return GRAPH_NODE_NONE;
    return node_add(g, OP_EMBED, (int[]){ (int)src, -1, -1, -1 },
                    (TensorInfo *[]){ weight, NULL, NULL, NULL }, NULL, 0);
}


/* ---------------------------------------------------------------- *
 * Graph creation
 * ---------------------------------------------------------------- */

Graph *graph_new(void) {
    return scalloc(1, sizeof(Graph));
}

void graph_free(Graph *g) {
    if (!g) return;
    /* Dump the per-op timing accumulated over the graph's lifetime. */
    double total = 0.0;
    u64    calls = 0;
    for (size_t i = 0; i < sizeof(op_stat_secs) / sizeof(op_stat_secs[0]); i++) {
        total += op_stat_secs[i];
        calls += op_stat_calls[i];
    }
    if (total > 1e-4) {
        fprintf(stderr, "graph timing: total %.3fs over %llu node execs\n",
                total, (unsigned long long)calls);
        for (size_t i = 0; i < sizeof(op_stat_secs) / sizeof(op_stat_secs[0]); i++)
            if (op_stat_secs[i] > 1e-5)
                fprintf(stderr, "  %-9s %8.3fs %6.1f%%  (%llu calls)\n",
                        op_stat_name[i], op_stat_secs[i],
                        100.0 * op_stat_secs[i] / total,
                        (unsigned long long)op_stat_calls[i]);
    }
    if (g->arena)
        fprintf(stderr, "graph arena: %.1f MB\n", (double)g->arena_size / 1048576.0);
    memset(op_stat_secs, 0, sizeof(op_stat_secs));
    memset(op_stat_calls, 0, sizeof(op_stat_calls));

    /* Every node->data aliases a slot inside the arena. */
    sfree(g->arena);
    g->arena = NULL;
    sfree(g->state);   /* borrowed pointers only: targets outlive us */
    g->state = NULL;
    sfree(g->node);
    sfree(g);
}

void graph_plan(Graph *g) {
    (void)g;
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

/* ---------------------------------------------------------------- *
 * Static arena plan
 * ---------------------------------------------------------------- *
 * Node output buffers are allocated once and reused across every
 * execution: a node's slot is released as soon as its last consumer
 * has run, so live memory tracks a few layers instead of the whole
 * graph.  The plan (slot offsets inside one peak-sized arena) is
 * computed lazily on the first execution and the node->data pointers
 * are baked into the graph; executions of any n <= ctx_size reuse the
 * same slots.  The sweep runs in build order (== execution order).  */

static size_t align16(size_t x) { return (x + 15) & ~(size_t)15; }

/* Capacity rows a node can ever be asked to produce. */
static u32 node_cap_rows(const GraphNode *node, bool sink, u32 ctx_size) {
    if (node->op == OP_INPUT) return (u32)node->ne[0];
    return sink ? 1u : ctx_size;
}

/* Elements per row of a node's output (float count).  dims[] must hold the
 * already-computed widths of every source node — guaranteed when nodes are
 * visited in build order, which is also execution order.  Shared by the
 * arena planner and the executor so the two can never disagree. */
static u32 node_out_dim(const GraphNode *node, const u32 *dims, const ArchConfig *c) {
    switch (node->op) {
        case OP_INPUT:
            return 0;
        case OP_MATMUL_T:
        case OP_MATMUL:
        case OP_MATMULARRY: {
            TensorInfo *w = node->weights[0];
            return (u32)(node->op == OP_MATMUL_T ? w->dim[1] : w->dim[0]);
        }
        case OP_EMBED: {
            TensorInfo *w = node->weights[0];
            return (u32)(w->dim[0] == (i64)c->n_vocab ? w->dim[1] : w->dim[0]);
        }
        case OP_RMS_NORM:
            return (u32)node->weights[0]->n_element;
        case OP_RMS_NORM_HEADS:
            return node->params[0] * node->params[3];   /* n_heads * out_stride  */
        case OP_SIGMOID_GATE:
            return node->params[0] * node->params[1];   /* n_heads * head_dim    */
        case OP_SSM_DELTA:
            return node->params[1] * node->params[3];   /* n_v_heads * head_dim  */
        case OP_ATTN:
            return c->n_head * c->head_dim;
        default:
            return node->src[0] >= 0 ? dims[(u32)node->src[0]] : 0;
    }
}

static bool arena_plan(Graph *g, Session *s) {
    u32 n_node = g->n_node;
    ArchConfig *c = &s->cfg;

    /* Sink marks and per-node output widths (mirrors graph_compute). */
    u32 sink[n_node], dims[n_node];
    for (u32 i = 0; i < n_node; i++) sink[i] = 1;
    for (u32 j = 0; j < n_node; j++)
        for (int k = 0; k < GRAPH_NODE_MAX_SRC; k++)
            if (g->node[j].src[k] >= 0) sink[(u32)g->node[j].src[k]] = 0;
    for (u32 i = 0; i < n_node; i++)
        dims[i] = node_out_dim(&g->node[i], dims, c);

    /* Byte needs at full capacity (INPUT slots hold u32 token ids). */
    size_t need[n_node];
    for (u32 i = 0; i < n_node; i++) {
        const GraphNode *node = &g->node[i];
        u32 rows = node_cap_rows(node, sink[i] != 0, s->ctx_size);
        need[i] = align16((node->op == OP_INPUT)
                              ? (size_t)node->ne[0] * sizeof(u32)
                              : (size_t)rows * dims[i] * sizeof(float));
    }

    /* release[i] = index of the last node consuming node i. */
    u32 release[n_node];
    for (u32 i = 0; i < n_node; i++) release[i] = i;
    for (u32 j = 0; j < n_node; j++)
        for (int k = 0; k < GRAPH_NODE_MAX_SRC; k++)
            if (g->node[j].src[k] >= 0) {
                u32 src = (u32)g->node[j].src[k];
                if (j > release[src]) release[src] = j;
            }

    /* Sweep: reclaim slots whose last consumer already ran, then give
     * the next node the smallest free chunk that fits (else extend). */
    u64  node_off[n_node];
    u64  foff[n_node];
    u64  fsz[n_node];
    u32  fcnt = 0;
    u64  bump = 0;

    for (u32 i = 0; i < n_node; i++) {
        /* Node j's slot becomes free exactly at step release[j] + 1. */
        if (i > 0) {
            for (u32 j = 0; j < i; j++) {
                if (release[j] + 1 != i) continue;
                foff[fcnt] = node_off[j];
                fsz[fcnt]  = (u64)need[j];
                fcnt++;
            }
        }

        u64 off;
        u32 best = fcnt;                 /* smallest chunk that fits */
        u64 best_sz = UINT64_MAX;
        for (u32 f = 0; f < fcnt; f++) {
            if (fsz[f] >= (u64)need[i] && fsz[f] < best_sz) {
                best    = f;
                best_sz = fsz[f];
            }
        }
        if (best < fcnt) {
            off = foff[best];
            fsz[best]  = fsz[fcnt - 1];   /* swap-remove */
            foff[best] = foff[fcnt - 1];
            fcnt--;
        } else {
            off = bump;
            bump += need[i];
        }
        node_off[i] = off;
    }

    if (bump == 0) return true;
    g->arena = smalloc((size_t)bump);
    if (!g->arena) return false;
    g->arena_size = (size_t)bump;

    for (u32 i = 0; i < n_node; i++) {
        GraphNode *node = &g->node[i];
        node->data      = (u8 *)g->arena + node_off[i];
        node->data_cap  = need[i];
    }
    return true;
}

/* ---------------------------------------------------------------- *
 * Operators — one function per GraphOp
 * ---------------------------------------------------------------- *
 * Every operator gets the same execution context: the node, its output
 * slot, and the row window it must fill (r rows starting at batch row
 * `base`, whose absolute position is pos + base).  Sources are resolved
 * through op_src() and are always already computed, since build order
 * is a valid topological order.  Returning false aborts the execution;
 * an operator must free whatever scratch it allocated before returning. */

typedef struct {
    Graph      *g;
    Session    *s;
    ArchConfig *cfg;
    GraphNode  *node;  /* node being executed                        */
    float      *dst;   /* node->data: output slot                    */
    u32         od;    /* output row width (floats)                  */
    u32         r;     /* rows to produce (sinks: 1)                 */
    u32         base;  /* batch row of local row 0                   */
    u32         pos;   /* absolute position of batch row 0           */
    u32         n;     /* batch rows                                 */
    float      *scr;   /* [pos + n] attention score scratch          */
    float       scale; /* 1 / sqrt(kv_head_dim)                      */
    u32         q_dim; /* n_head    * head_dim                       */
    u32         kv_dim;/* n_kv_head * kv_head_dim                    */
} OpCtx;

/* Row-major source tensor of edge `k`. */
static float *op_src(const OpCtx *c, int k) {
    return (float *)c->g->node[(u32)c->node->src[k]].data;
}

/* Op-specific parameter `k` of the node being executed. */
static u32 op_param(const OpCtx *c, int k) {
    return c->node->params[k];
}

static bool op_embed(OpCtx *c) {
    TensorInfo *te = c->node->weights[0];
    bool te_trans  = (te->dim[0] == (i64)c->cfg->n_vocab);
    u32 *tok = (u32 *)c->g->node[(u32)c->node->src[0]].data;

    /* Always the full batch: the embedding of every input row is needed. */
    for (u32 p = 0; p < c->n; p++) {
        float *dp = c->dst + (u64)p * c->od;
        if (te_trans) {
            tensor_get_f32_batch(te, (u64)tok[p] * c->od, c->od, dp);
        } else {
            for (u32 j = 0; j < c->od; j++)
                dp[j] = tensor_get_f32(te, (u64)j * c->cfg->n_vocab + tok[p]);
        }
    }
    return true;
}

static bool op_rms_norm(OpCtx *c) {
    TensorInfo *tw = c->node->weights[0];
    float *src = op_src(c, 0);
    for (u32 p = 0; p < c->r; p++)
        rms_norm(c->dst + (u64)p * c->od, src + ((u64)c->base + p) * c->od,
                 tw, (int)c->od, DEFAULT_EPS);
    return true;
}

static bool op_matmul(OpCtx *c) {
    TensorInfo *w = c->node->weights[0];
    bool tr       = (c->node->op == OP_MATMUL_T);
    u32 in        = (u32)(tr ? w->dim[0] : w->dim[1]);
    float *x      = op_src(c, 0) + (u64)c->base * in;

    if (c->r == 1)
        return mat_vec_mul(c->dst, w, x, c->od, in, tr, c->s->pthreads);
    return mat_mat_mul(c->dst, w, x, c->r, c->od, in, tr, c->s->pthreads);
}

static bool op_bias(OpCtx *c) {
    TensorInfo *tb = c->node->weights[0];
    float *src = op_src(c, 0);
    u32 nb = tb->n_element < (u64)c->od ? (u32)tb->n_element : c->od;
    for (u32 p = 0; p < c->r; p++) {
        float *dp = c->dst + (u64)p * c->od;
        memcpy(dp, src + ((u64)c->base + p) * c->od, (size_t)c->od * sizeof(float));
        bias_add(dp, tb, nb);
    }
    return true;
}

static bool op_binary(OpCtx *c) {
    bool add = (c->node->op == OP_ADD);
    float *a = op_src(c, 0);
    float *b = op_src(c, 1);
    for (u32 p = 0; p < c->r; p++) {
        float *dp = c->dst + (u64)p * c->od;
        float *ap = a + ((u64)c->base + p) * c->od;
        float *bp = b + ((u64)c->base + p) * c->od;
        if (add)
            for (u32 j = 0; j < c->od; j++) dp[j] = ap[j] + bp[j];
        else
            for (u32 j = 0; j < c->od; j++) dp[j] = ap[j] * bp[j];
    }
    return true;
}

/* Element-wise unary: copy the row, then apply f in place. */
static bool op_unary(OpCtx *c, void (*f)(float *, int)) {
    float *src = op_src(c, 0);
    for (u32 p = 0; p < c->r; p++) {
        float *dp = c->dst + (u64)p * c->od;
        memcpy(dp, src + ((u64)c->base + p) * c->od, (size_t)c->od * sizeof(float));
        f(dp, (int)c->od);
    }
    return true;
}

/* softmax() takes a u32 count; adapt it to the unary signature. */
static void softmax_n(float *x, int n) { softmax(x, (u32)n); }

static bool op_silu(OpCtx *c)    { return op_unary(c, silu); }
static bool op_softmax(OpCtx *c) { return op_unary(c, softmax_n); }

static bool op_rope_neox(OpCtx *c) {
    u32 bits = op_param(c, 0);
    float theta;
    memcpy(&theta, &bits, sizeof(theta));
    u32 heads = op_param(c, 1);
    u32 hdim  = op_param(c, 2);
    u32 rdim  = op_param(c, 3);
    float *src = op_src(c, 0);

    for (u32 p = 0; p < c->r; p++) {
        u32 abspos = c->pos + c->base + p;   /* absolute position */
        float *dp  = c->dst + (u64)p * c->od;
        memcpy(dp, src + (u64)(c->base + p) * c->od, (size_t)c->od * sizeof(float));
        /* rope_partial degenerates to rope_neox for rdim == hdim. */
        rope_partial(dp, heads, hdim, rdim, abspos, theta);
    }
    return true;
}

static bool op_rms_norm_heads(OpCtx *c) {
    /* Per-head RMS norm over [n_heads] slices of head_dim, read at
     * in_stride and written at out_stride.  Q-norm over a fused
     * [q, gate] projection uses in_stride == 2*head_dim, which also
     * drops the (unnormed) gate half. */
    u32 nh = op_param(c, 0), hd = op_param(c, 1);
    u32 is = op_param(c, 2), os = op_param(c, 3);
    float *src = op_src(c, 0);
    float *nw  = smalloc((u64)hd * sizeof(float));
    if (!nw) return false;
    tensor_get_f32_batch(c->node->weights[0], 0, hd, nw);

    for (u32 p = 0; p < c->r; p++) {
        const float *sp = src + ((u64)c->base + p) * (u64)nh * is;
        float *dp = c->dst + (u64)p * c->od;
        for (u32 h = 0; h < nh; h++) {
            float *o = dp + (u64)h * os;
            memcpy(o, sp + (u64)h * is, (size_t)hd * sizeof(float));
            rms_norm_inplace(o, nw, hd, DEFAULT_EPS);
        }
    }
    sfree(nw);
    return true;
}

static bool op_sigmoid_gate(OpCtx *c) {
    /* out = a * sigmoid(gate), gate taken from the second half of each
     * 2*head_dim block of the fused source. */
    u32 nh = op_param(c, 0), hd = op_param(c, 1);
    float *a  = op_src(c, 0);
    float *gt = op_src(c, 1);

    for (u32 p = 0; p < c->r; p++) {
        const float *ap = a  + ((u64)c->base + p) * c->od;
        const float *gp = gt + ((u64)c->base + p) * (u64)nh * 2 * hd;
        float *dp = c->dst + (u64)p * c->od;
        for (u32 h = 0; h < nh; h++)
            for (u32 d = 0; d < hd; d++)
                dp[(u64)h * hd + d] =
                    ap[(u64)h * hd + d] * sigmoid(gp[(u64)h * 2 * hd + hd + d]);
    }
    return true;
}

static bool op_ssm_conv(OpCtx *c) {
    /* Depthwise causal conv1d over the fused QKV projection, advancing
     * the per-layer ring buffer one row at a time. */
    u32 st = op_param(c, 0), ck = op_param(c, 1);
    if (st >= c->g->n_state) return false;
    float *state = (float *)c->g->state[st];
    float *src   = op_src(c, 0);
    float *cw    = smalloc((u64)ck * c->od * sizeof(float));
    if (!cw) return false;
    tensor_get_f32_batch(c->node->weights[0], 0, (u64)ck * c->od, cw);

    for (u32 p = 0; p < c->r; p++) {
        float *dp = c->dst + (u64)p * c->od;
        memcpy(dp, src + ((u64)c->base + p) * c->od, (size_t)c->od * sizeof(float));
        causal_conv1d_step(dp, dp, cw, state, c->od, ck);
    }
    sfree(cw);
    return true;
}

static bool op_ssm_delta(OpCtx *c) {
    /* Gated DeltaNet recurrence.  Sequential over rows by construction:
     * each row advances the recurrent state. */
    u32 st = op_param(c, 0), n_v = op_param(c, 1);
    u32 n_k = op_param(c, 2), hd = op_param(c, 3);
    if (st >= c->g->n_state) return false;
    u32 key_dim = n_k * hd;
    u64 val_dim = (u64)n_v * hd;
    if (c->od != (u32)val_dim) return false;
    float *state = (float *)c->g->state[st];
    float *fused = op_src(c, 0);
    float *alpha = op_src(c, 1);
    float *beta  = op_src(c, 2);

    /* Per-call scratch: gates, dequantised params, and a private
     * [q|k|v] copy (the fused source is shared with other consumers
     * and must not be mutated). */
    float *gv    = smalloc((u64)n_v * sizeof(float));
    float *bv    = smalloc((u64)n_v * sizeof(float));
    float *a_log = smalloc((u64)n_v * sizeof(float));
    float *dt_bi = smalloc((u64)n_v * sizeof(float));
    float *nw    = smalloc((u64)hd * sizeof(float));
    float *mem   = smalloc(val_dim * sizeof(float));
    float *qkv   = smalloc((2 * (u64)key_dim + val_dim) * sizeof(float));
    if (!gv || !bv || !a_log || !dt_bi || !nw || !mem || !qkv) {
        sfree(gv); sfree(bv); sfree(a_log); sfree(dt_bi);
        sfree(nw); sfree(mem); sfree(qkv);
        return false;
    }
    tensor_get_f32_batch(c->node->weights[0], 0, n_v, a_log);
    tensor_get_f32_batch(c->node->weights[1], 0, n_v, dt_bi);
    if (c->node->weights[2]) tensor_get_f32_batch(c->node->weights[2], 0, hd, nw);

    for (u32 p = 0; p < c->r; p++) {
        u64 fstride = 2 * (u64)key_dim + val_dim;
        const float *fp = fused + ((u64)c->base + p) * fstride;
        const float *ap = alpha + ((u64)c->base + p) * n_v;
        const float *bp = beta  + ((u64)c->base + p) * n_v;
        float *dp = c->dst + (u64)p * c->od;

        /* ssm.a already stores -exp(A_log) in the GGUF (converter:
         * A_log → -exp(A_log)), so the decay gate is
         * g = exp(ssm.a * softplus(alpha + dt_bias)). */
        for (u32 i = 0; i < n_v; i++) {
            gv[i] = expf(a_log[i] * softplus(ap[i] + dt_bi[i]));
            bv[i] = sigmoid(bp[i]);
        }

        memcpy(qkv, fp, fstride * sizeof(float));
        float *q_d = qkv, *k_d = qkv + key_dim, *v_d = qkv + 2 * key_dim;
        l2_norm_rows(q_d, n_k, hd, DEFAULT_EPS);
        l2_norm_rows(k_d, n_k, hd, DEFAULT_EPS);
        /* Q *= 1/sqrt(head_dim): the S^T q readout carries no 1/sqrt(d)
         * of its own, so the scale lives on q. */
        float q_scale = 1.0f / sqrtf((float)hd);
        for (u32 i = 0; i < key_dim; i++)
            q_d[i] *= q_scale;

        gated_delta_step(q_d, k_d, v_d, state, gv, bv, mem, dp, n_v, n_k, hd);

        /* Per-value-head RMS norm on the delta output. */
        if (c->node->weights[2])
            for (u32 grp = 0; grp < n_v; grp++)
                rms_norm_inplace(dp + (u64)grp * hd, nw, hd, DEFAULT_EPS);
    }
    sfree(gv); sfree(bv); sfree(a_log); sfree(dt_bi);
    sfree(nw); sfree(mem); sfree(qkv);
    return true;
}

static bool op_attn(OpCtx *c) {
    /* Write the batch's K/V into the session cache at positions
     * pos..pos+n-1, then attend causally over the cached keys (rows
     * 0..pos+qi) — history included. */
    if (!c->s->cache.std) {
        slog(WARN, "graph_compute: OP_ATTN requires the std KV cache");
        return false;
    }
    u32 layer = op_param(c, 0);
    AttnKvCache *akc = &c->s->cache.std[layer];
    if (!akc->k || !akc->v) return false;

    float *qd  = op_src(c, 0);
    float *kd  = op_src(c, 1);
    float *vd  = op_src(c, 2);
    u32 n_head = c->cfg->n_head;
    u32 n_kv   = c->cfg->n_kv_head;
    u32 hd     = c->cfg->head_dim;
    u32 khd    = c->cfg->kv_head_dim;
    u32 gqa    = n_head / n_kv;
    u32 hs     = akc->cap * khd;   /* stride between heads */

    /* K/V write-through (head-major: [head][pos][dim]). */
    for (u32 i = 0; i < c->n; i++) {
        const float *kr = kd + (u64)i * c->kv_dim;
        const float *vr = vd + (u64)i * c->kv_dim;
        for (u32 h = 0; h < n_kv; h++) {
            memcpy(akc->k + (u64)h * hs + (u64)(c->pos + i) * khd,
                   kr + (u64)h * khd, (size_t)khd * sizeof(float));
            memcpy(akc->v + (u64)h * hs + (u64)(c->pos + i) * khd,
                   vr + (u64)h * khd, (size_t)khd * sizeof(float));
        }
    }
    akc->n = c->pos + c->n;

    memset(c->dst, 0, (size_t)c->n * c->q_dim * sizeof(float));
    for (u32 h = 0; h < n_head; h++) {
        u32 kv_h = h / gqa;
        float *kh_base = akc->k + (u64)kv_h * hs;
        float *vh_base = akc->v + (u64)kv_h * hs;
        for (u32 qi = 0; qi < c->n; qi++) {
            float *qh = qd + (u64)qi * c->q_dim + (u64)h * hd;
            u32 n_keys = c->pos + qi + 1;   /* causal */

            float *kt = kh_base;
            for (u32 t = 0; t < n_keys; t++, kt += khd) {
                float acc = 0.0f;
                for (u32 d = 0; d < khd; d++)
                    acc += qh[d] * kt[d];
                c->scr[t] = acc * c->scale;
            }
            softmax(c->scr, n_keys);

            float *oh = c->dst + (u64)qi * c->q_dim + (u64)h * hd;
            float *vt = vh_base;
            for (u32 t = 0; t < n_keys; t++, vt += khd) {
                float st = c->scr[t];
                for (u32 d = 0; d < khd; d++)
                    oh[d] += st * vt[d];
            }
        }
    }
    return true;
}

/* Dispatch table, indexed by GraphOp.  NULL == not executable here
 * (OP_INPUT nothing to compute). */
typedef bool (*OpFn)(OpCtx *c);
static const OpFn op_table[] = {
    [OP_EMBED]          = op_embed,
    [OP_RMS_NORM]       = op_rms_norm,
    [OP_RMS_NORM_HEADS] = op_rms_norm_heads,
    [OP_MATMUL]         = op_matmul,
    [OP_MATMULARRY]     = op_matmul,
    [OP_MATMUL_T]       = op_matmul,
    [OP_BIAS]           = op_bias,
    [OP_ADD]            = op_binary,
    [OP_MUL]            = op_binary,
    [OP_SILU]           = op_silu,
    [OP_SOFTMAX]        = op_softmax,
    [OP_ROPE_NEOX]      = op_rope_neox,
    [OP_SIGMOID_GATE]   = op_sigmoid_gate,
    [OP_SSM_CONV]       = op_ssm_conv,
    [OP_SSM_DELTA]      = op_ssm_delta,
    [OP_ATTN]           = op_attn,
};

bool graph_compute(Graph *g, const GraphBatch *b, Session *s) {
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

    /* One-time static slot plan for the execution arena. */
    if (!g->arena && !arena_plan(g, s)) {
        slog(WARN, "graph_compute: arena plan failed");
        return false;
    }

    /* Bind the batch's token ids into each OP_INPUT leaf. */
    for (u32 i = 0; i < g->n_node; i++) {
        GraphNode *node = &g->node[i];
        if (node->op != OP_INPUT) continue;
        if (n > (u32)node->ne[0]) {
            slog(WARN, "graph_compute: batch of %u exceeds graph capacity", n);
            return false;
        }
        if (b->tokens) memcpy(node->data, b->tokens, (size_t)n * sizeof(u32));
    }

    ArchConfig *c = &s->cfg;
    u32 n_node = g->n_node;

    /* Sink detection: a node is a sink iff no other node reads it.
     * (Edges point strictly backward, so scanning all nodes is enough.) */
    u32 sink[n_node];
    for (u32 i = 0; i < n_node; i++) sink[i] = 1;
    for (u32 j = 0; j < n_node; j++)
        for (int k = 0; k < GRAPH_NODE_MAX_SRC; k++)
            if (g->node[j].src[k] >= 0) sink[(u32)g->node[j].src[k]] = 0;

    /* Elements per row of each node's output (float count). */
    u32 dims[n_node];
    for (u32 i = 0; i < n_node; i++)
        dims[i] = node_out_dim(&g->node[i], dims, c);

    /* Attention score scratch: the batch's last row attends every key
     * in the cache, positions 0..pos+n-1. */
    float *scr = smalloc((size_t)(pos + n) * sizeof(float));
    if (!scr) return false;

    u32 q_dim  = c->n_head * c->head_dim;
    u32 kv_dim = c->n_kv_head * c->kv_head_dim;
    float scale = 1.0f / sqrtf((float)c->kv_head_dim);

    /* Walk the graph in build order (== topo order). */
    for (u32 i = 0; i < n_node; i++) {
        GraphNode *node = &g->node[i];
        if (node->op == OP_INPUT) continue;
        u32 od   = dims[i];
        u32 r    = sink[i] ? 1u : n;            /* sink: last batch row only */
        u32 base = sink[i] ? n - 1 : 0;         /* batch row of local row 0  */
        if (od == 0 || r == 0) goto fail;
        if ((size_t)r * od * sizeof(float) > node->data_cap) goto fail;
        float *dst = (float *)node->data;

        int    st_cls = op_stat_class(node->op);
        double st_t0  = graph_now();
        OpCtx ctx = {
            .g = g, .s = s, .cfg = c, .node = node, .dst = dst,
            .od = od, .r = r, .base = base, .pos = pos, .n = n,
            .scr = scr, .scale = scale, .q_dim = q_dim, .kv_dim = kv_dim,
        };
        OpFn fn = (node->op < sizeof(op_table) / sizeof(op_table[0])) ? op_table[node->op] : NULL;
        if (!fn) {
            slog(WARN, "graph_compute: unhandled op %d", (int)node->op);
            goto fail;
        }
        if (!fn(&ctx)) goto fail;
        
        /* Graph stats. */
        op_stat_secs[st_cls] += graph_now() - st_t0;
        op_stat_calls[st_cls]++;
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
