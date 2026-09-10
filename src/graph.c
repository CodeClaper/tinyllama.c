#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "graph.h"
#include "core.h"
#include "mm.h"
#include "slog.h"
#include "backend/cpu/cpu_op.h"

#ifdef GPU_BUILD
#include <cuda_runtime.h>
#include "backend/gpu/gpu.h"
#include "backend/gpu/gpu_op.h"
#endif
#ifdef METAL_BUILD
#include "backend/metal/metal.h"
#endif

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

static void *arena_alloc(BackendType backend, size_t bytes);
static void arena_free(BackendType backend, void *arena);

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
    if (g->plan)
        fprintf(stderr, "graph arena: %.1f MB\n",
                (double)g->plan->arena_size / 1048576.0);
    memset(op_stat_secs, 0, sizeof(op_stat_secs));
    memset(op_stat_calls, 0, sizeof(op_stat_calls));

    /* Every node->data aliases a slot inside the plan's arena; its
     * placement (host vs VRAM) follows the plan's backend. */
    if (g->plan) {
        arena_free(g->plan->backend, g->plan->arena);
        sfree(g->plan);
        g->plan = NULL;
    }
    sfree(g->state);   /* borrowed pointers only: targets outlive us */
    g->state = NULL;
    sfree(g->node);
    sfree(g);
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

/* Arena placement follows the plan's backend: a CUDA plan lives in
 * VRAM so activations never leave the device; CPU (and Metal, until
 * its graph ops land) uses host memory. */
static void *arena_alloc(BackendType backend, size_t bytes) {
    switch (backend) {
#ifdef GPU_BUILD
        case BACKEND_CUDA: {
            void *d = NULL;
            if (cudaMalloc(&d, bytes) != cudaSuccess) return NULL;
            return d;
        }
#endif
        case BACKEND_CPU:
        case BACKEND_METAL:
        default:
            return smalloc(bytes);
    }
}

static void arena_free(BackendType backend, void *arena) {
    (void)backend;
    if (!arena) return;
    switch (backend) {
#ifdef GPU_BUILD
        case BACKEND_CUDA:
            cudaFree(arena);
            break;
#endif
        case BACKEND_CPU:
        case BACKEND_METAL:
        default:
            sfree(arena);
            break;
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
    /* Reuse the plan shell created by backend_plan (backend already
     * pinned); create one only when the caller skipped that step. */
    GraphPlan *p = g->plan;
    bool fresh = false;
    if (!p) {
        p = scalloc(1, sizeof(GraphPlan));
        if (!p) return false;
        p->backend = BACKEND_CPU;
        fresh = true;
    }
    p->arena      = arena_alloc(p->backend, (size_t)bump);
    if (!p->arena) { if (fresh) sfree(p); return false; }
    p->arena_size = (size_t)bump;
    g->plan = p;

    for (u32 i = 0; i < n_node; i++) {
        GraphNode *node = &g->node[i];
        node->data      = (u8 *)p->arena + node_off[i];
        node->data_cap  = need[i];
    }
    return true;
}

/* Auto-select the compute backend targeted by the graph's execution
 * plan; the selection must run BEFORE the arena sweep, because the
 * backend decides where the plan's arena is allocated (VRAM vs host).
 * A caller may pin the backend explicitly by pre-seeding the plan. */
bool backend_plan(Graph *g) {
    if (!g) return false;
    if (!g->plan) {
        g->plan = scalloc(1, sizeof(GraphPlan));
        if (!g->plan) return false;
        g->plan->backend = BACKEND_CPU;
    }
    /* Long-lived graphs keep their pinned backend; fresh plans take
     * the detected one. */
    if (!g->plan->arena) {
        BackendType backend = g->plan->backend;
#ifdef METAL_BUILD
        if (metal_available() && backend == BACKEND_CPU) backend = BACKEND_METAL;
#endif
#ifdef GPU_BUILD
        if (gpu_available() && backend == BACKEND_CPU) backend = BACKEND_CUDA;
#endif
        g->plan->backend = backend;
    }
    return true;
}

bool graph_plan(Graph *g, Session *s) {
    if (!g || !s) return false;
    if (g->plan) return true;          /* already planned */
    if (!backend_plan(g)) return false;   /* backend first: arena placement */
    return arena_plan(g, s);
}

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

    /* One-time static slot plan (generated by graph_plan()). */
    if (!g->plan && !graph_plan(g, s)) {
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
        if (b->tokens) {
#ifdef GPU_BUILD
            if (g->plan->backend == BACKEND_CUDA) {
                /* The only H2D of a GPU-path batch: token ids into the
                 * VRAM arena; every op reads them on-device. */
                CHECK(cudaMemcpy(node->data, b->tokens, (size_t)n * sizeof(u32),
                                 cudaMemcpyHostToDevice));
            } else
#endif
                memcpy(node->data, b->tokens, (size_t)n * sizeof(u32));
        }
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

        /* Device backends own data in VRAM: the CUDA plan fully
         * replaces the CPU op path, while the CPU path runs the
         * operators in backend/cpu/cpu_op.c via cpu_graph_op(). */
        bool ok;
#ifdef GPU_BUILD
        if (g->plan->backend == BACKEND_CUDA)
            ok = gpu_graph_op(&ctx);
        else
#endif
            ok = cpu_graph_op(&ctx);
        if (!ok) goto fail;
        
        /* Graph stats. */
        op_stat_secs[st_cls] += graph_now() - st_t0;
        op_stat_calls[st_cls]++;
    }

    /* Sink output (last row of the LM head) → session logits.
     * A CUDA arena hands the slot back through one D2H. */
    for (u32 i = 0; i < n_node; i++) {
        if (!sink[i] || g->node[i].op == OP_INPUT) continue;
        u32 n = dims[i] < c->n_vocab ? dims[i] : c->n_vocab;
#ifdef GPU_BUILD
        if (g->plan->backend == BACKEND_CUDA) {
            CHECK(cudaMemcpy(s->logits, g->node[i].data, (size_t)n * sizeof(float),
                             cudaMemcpyDeviceToHost));
        } else
#endif
            memcpy(s->logits, g->node[i].data, (size_t)n * sizeof(float));
        break;
    }

    sfree(scr);
    return true;

fail:
    sfree(scr);
    return false;
}

/* One forward step through the graph: it is built on first use (at ctx_size
 * capacity, so any later batch fits), the batch is appended to the session
 * ring buffer at s->n_tokens and executed, and the last row's logits are
 * published.  Replaces the old prefill / generate arch hooks — callers pass
 * the prompt ids for prefill and a single id for decode. */
bool graph_execute(Session *s, const u32 *tokens, u32 n_tokens, float *logits) {
    if (!s || !tokens || n_tokens == 0) {
        slog(WARN, "graph_execute: missing session / tokens");
        return false;
    }

    if (!s->graph) {
        if (!s->ops.graph_build) {
            slog(WARN, "graph_execute: no graph builder for this architecture");
            return false;
        }
        s->graph = s->ops.graph_build(s, s->ctx_size);
        if (!s->graph) {
            slog(WARN, "graph_execute: graph build failed");
            return false;
        }
    }

    u32 pos = s->n_tokens;
    if ((u64)pos + n_tokens > s->ctx_size) {
        slog(WARN, "graph_execute: batch of %u at pos %u exceeds ctx_size=%u",
             n_tokens, pos, s->ctx_size);
        return false;
    }
    if (!s->tokens) {
        slog(WARN, "graph_execute: session has no token buffer");
        return false;
    }

    memcpy(s->tokens + pos, tokens, (size_t)n_tokens * sizeof(u32));
    GraphBatch batch = {
        .tokens = s->tokens + pos,
        .pos    = pos,
        .n      = n_tokens
    };
    if (!graph_compute(s->graph, &batch, s)) return false;
    s->n_tokens = pos + n_tokens;

    /* graph_compute() leaves the logits in s->logits; mirror them out when
     * the caller asked for a different buffer. */
    if (logits && logits != s->logits)
        memcpy(logits, s->logits, (size_t)s->cfg.n_vocab * sizeof(float));

    return true;
}
