#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "graph.h"
#include "core.h"
#include "mm.h"
#include "model/model.h"
#include "slog.h"
#include "backend/cpu/cpu_op.h"

#ifdef GPU_BUILD
#include <cuda_runtime.h>
#include "backend/gpu/gpu.h"
#include "backend/gpu/gpu_op.h"
#endif
#ifdef METAL_BUILD
#include "backend/metal/metal.h"
#include "backend/metal/metal_op.h"
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
        case OP_ROPE_NEOX:
        case OP_ROPE_NORM:      return ST_ROPE;
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
 * src[]   = source nodes, weights[] = per-source weight tensors,
 * params[] = op-specific parameters, n_params = how many u32s of it are
 * valid (ops that need none pass NULL / 0).  The node's full params block
 * is always zeroed, so unspecified entries read back as 0.
 *
 * Every builder below hands node_add() a brace literal with exactly
 * GRAPH_NODE_MAX_SRC entries — reading past it would be out of bounds,
 * so the macro and the literals must move together. */
_Static_assert(GRAPH_NODE_MAX_SRC == 4,
               "graph builders pass src/weights literals with 4 entries");

static GraphNode *node_add_extern(Graph *g, GraphNode *n) {
    if (g->tail) g->tail->next = n; 
    else g->head = n;
    g->tail = n;
    g->n_node++;
    return n;
}

static GraphNode *node_add(Graph *g, GraphOp op, GraphNode *const *src,
                           TensorInfo *const *weights,
                           const u32 *params, u32 n_params) {
    if (!g) return GRAPH_NODE_NONE;
    if (n_params > sizeof(((GraphNode *)0)->params) / sizeof(u32)) return GRAPH_NODE_NONE;

    GraphNode *n = scalloc(1, sizeof(GraphNode));
    if (!n) return GRAPH_NODE_NONE;

    n->op      = op;
    n->idx     = g->n_node;
    n->layer   = g->cur_layer;
    n->backend = BACKEND_CPU;
    for (int i = 0; i < GRAPH_NODE_MAX_SRC; i++) {
        n->src[i]     = src ? src[i] : NULL;
        n->weights[i] = weights ? weights[i] : NULL;
    }
    if (params && n_params) 
        memcpy(n->params, params, n_params * sizeof(u32));
    return node_add_extern(g, n);
}

void graph_set_layer(Graph *g, u32 layer) {
    if (g) g->cur_layer = layer;
}

#if defined(GPU_BUILD)
/* True when `layer` is on the GPU side of the split.  A negative
 * n_gpu_layers is treated as "all" here; session_create() resolves it
 * to a concrete count before the model is initialised. */
static bool layer_on_gpu(const Session *s, u32 layer) {
    if (!s || s->n_gpu_layers <= 0) return false;
    return (u32)s->n_gpu_layers >= s->cfg.n_layer
           || layer < (u32)s->n_gpu_layers;
}
#endif

BackendType graph_layer_backend(const Session *s, u32 layer) {
#if defined(GPU_BUILD)
    if (layer_on_gpu(s, layer)) return BACKEND_CUDA;
#else
    /* CPU build, or Metal (whose op backend mirrors host state). */
    (void)s; (void)layer;
#endif
    return BACKEND_CPU;
}

static GraphNode *graph_copy(Graph *g, GraphNode *src, BackendType dst) {
    if (!g || !src) return GRAPH_NODE_NONE;
    GraphOp op = (dst == BACKEND_CPU) ? OP_D2H : OP_H2D;
    u32 params[] = { (u32)dst };
    return node_add(g, op, (GraphNode *[]){ src, NULL, NULL, NULL },
                    NULL, params, 1);
}

GraphNode *graph_layer_boundary(Graph *g, GraphNode *src, Session *s, u32 layer) {
    if (!g || !src || !s) return src;
    /* Only a split plan needs a boundary copy. */
    if (s->n_gpu_layers <= 0 || (u32)s->n_gpu_layers >= s->cfg.n_layer)
        return src;
    if (layer + 1 != (u32)s->n_gpu_layers) return src;
    return graph_copy(g, src, BACKEND_CPU);
}

void *graph_state_alloc(size_t bytes, BackendType backend) {
    if (bytes == 0) bytes = 1;
#if defined(GPU_BUILD)
    if (backend == BACKEND_CUDA) {
        void *d = NULL;
        if (cudaMalloc(&d, bytes) != cudaSuccess) {
            (void)cudaGetLastError();
            return NULL;
        }
        cudaMemset(d, 0, bytes);
        return d;
    }
#else
    (void)backend;
#endif
    return scalloc(bytes, 1);
}

void graph_state_free(void *ptr, BackendType backend) {
    if (!ptr) return;
#if defined(GPU_BUILD)
    if (backend == BACKEND_CUDA) { cudaFree(ptr); return; }
#else
    (void)backend;
#endif
    sfree(ptr);
}

void graph_state_zero(void *ptr, size_t bytes, BackendType backend) {
    if (!ptr || bytes == 0) return;
#if defined(GPU_BUILD)
    if (backend == BACKEND_CUDA) { cudaMemset(ptr, 0, bytes); return; }
#else
    (void)backend;
#endif
    memset(ptr, 0, bytes);
}

/* Leaf input: carries a run of token ids.  Shape (ne[0] = capacity,
 * nb[0] = element size) is recorded in the node itself; the data slot
 * lives in the graph arena and is filled by graph_compute() at
 * execution time with the concrete tokens. */
GraphNode *graph_input(Graph *g, u32 n_tokens) {
    if (!g || n_tokens == 0) return GRAPH_NODE_NONE;

    GraphNode *node = node_add(g, OP_INPUT, (GraphNode *[]){ NULL, NULL, NULL, NULL },
                               NULL, NULL, 0);
    if (!node) return GRAPH_NODE_NONE;

    node->ne[0] = n_tokens;       /* capacity: [n_tokens]     */
    node->nb[0] = sizeof(u32);    /* element stride           */
    return node;
}

GraphNode *graph_rms_norm(Graph *g, GraphNode *src, TensorInfo *weight) {
    if (!g || !src || !weight) return GRAPH_NODE_NONE;
    return node_add(g, OP_RMS_NORM, (GraphNode *[]){ src, NULL, NULL, NULL },
                    (TensorInfo *[]){ weight, NULL, NULL, NULL }, NULL, 0);
}

GraphNode *graph_rms_norm_heads(Graph *g, GraphNode *src, TensorInfo *weight,
                                u32 n_heads, u32 head_dim, u32 in_stride, u32 out_stride) {
    if (!g || !src || !weight) return GRAPH_NODE_NONE;
    if (n_heads == 0 || head_dim == 0) return GRAPH_NODE_NONE;
    if (in_stride < head_dim || out_stride < head_dim) return GRAPH_NODE_NONE;
    u32 params[] = { n_heads, head_dim, in_stride, out_stride };
    return node_add(g, OP_RMS_NORM_HEADS, (GraphNode *[]){ src, NULL, NULL, NULL },
                    (TensorInfo *[]){ weight, NULL, NULL, NULL }, params, 4);
}

GraphNode *graph_mul_mat(Graph *g, GraphNode *src, TensorInfo *weight, bool trans) {
    if (!g || !src || !weight || weight->ndim < 2) return GRAPH_NODE_NONE;
    return node_add(g, trans ? OP_MATMUL_T : OP_MATMUL, (GraphNode *[]){ src, NULL, NULL, NULL },
                    (TensorInfo *[]){ weight, NULL, NULL, NULL }, NULL, 0);
}

GraphNode *graph_binary(Graph *g, GraphOp op, GraphNode *a, GraphNode *b) {
    if (op != OP_ADD && op != OP_MUL) return GRAPH_NODE_NONE;
    if (!g || !a || !b) return GRAPH_NODE_NONE;
    return node_add(g, op, (GraphNode *[]){ a, b, NULL, NULL }, NULL, NULL, 0);
}

GraphNode *graph_silu(Graph *g, GraphNode *src) {
    if (!g || !src) return GRAPH_NODE_NONE;
    return node_add(g, OP_SILU, (GraphNode *[]){ src, NULL, NULL, NULL }, NULL, NULL, 0);
}

GraphNode *graph_gelu(Graph *g, GraphNode *src) {
    if (!g || !src) return GRAPH_NODE_NONE;
    return node_add(g, OP_GELU, (GraphNode *[]){ src, NULL, NULL, NULL }, NULL, NULL, 0);
}

GraphNode *graph_softmax(Graph *g, GraphNode *src) {
    if (!g || !src) return GRAPH_NODE_NONE;
    return node_add(g, OP_SOFTMAX, (GraphNode *[]){ src, NULL, NULL, NULL }, NULL, NULL, 0);
}

GraphNode *graph_scale(Graph *g, GraphNode *src, float scale) {
    if (!g || !src) return GRAPH_NODE_NONE;
    u32 bits;
    memcpy(&bits, &scale, sizeof(bits));   /* float bits in params[0] */
    u32 params[] = { bits };
    return node_add(g, OP_SCALE, (GraphNode *[]){ src, NULL, NULL, NULL }, NULL, params, 1);
}

GraphNode *graph_softcap(Graph *g, GraphNode *src, float cap) {
    if (!g || !src || cap <= 0.0f) return GRAPH_NODE_NONE;
    u32 bits;
    memcpy(&bits, &cap, sizeof(bits));     /* float bits in params[0] */
    u32 params[] = { bits };
    return node_add(g, OP_SOFTCAP, (GraphNode *[]){ src, NULL, NULL, NULL }, NULL, params, 1);
}

GraphNode *graph_bias(Graph *g, GraphNode *src, TensorInfo *bias) {
    if (!g || !src || !bias || bias->ndim < 1) return GRAPH_NODE_NONE;
    return node_add(g, OP_BIAS, (GraphNode *[]){ src, NULL, NULL, NULL },
                    (TensorInfo *[]){ bias, NULL, NULL, NULL }, NULL, 0);
}

GraphNode *graph_sigmoid_gate(Graph *g, GraphNode *a, GraphNode *gate,
                              u32 n_heads, u32 head_dim) {
    if (!g || !a || !gate) return GRAPH_NODE_NONE;
    if (n_heads == 0 || head_dim == 0) return GRAPH_NODE_NONE;
    u32 params[] = { n_heads, head_dim };
    return node_add(g, OP_SIGMOID_GATE, (GraphNode *[]){ a, gate, NULL, NULL },
                    NULL, params, 2);
}

GraphNode *graph_ssm_conv(Graph *g, GraphNode *src, TensorInfo *weight, u32 state, u32 kernel) {
    if (!g || !src || !weight || kernel == 0) return GRAPH_NODE_NONE;
    if (state >= g->n_state) return GRAPH_NODE_NONE;
    u32 params[] = { state, kernel };
    return node_add(g, OP_SSM_CONV, (GraphNode *[]){ src, NULL, NULL, NULL },
                    (TensorInfo *[]){ weight, NULL, NULL, NULL }, params, 2);
}

GraphNode *graph_ssm_delta(Graph *g, const GraphSsmDelta *args) {
    if (!g || !args) return GRAPH_NODE_NONE;
    if (!args->fused || !args->alpha || !args->beta) return GRAPH_NODE_NONE;
    if (!args->ssm_a || !args->dt_bias || args->n_v_heads == 0 || args->head_dim == 0) return GRAPH_NODE_NONE;
    if (args->n_k_heads == 0 || args->n_v_heads % args->n_k_heads != 0) return GRAPH_NODE_NONE;
    if (args->state >= g->n_state) return GRAPH_NODE_NONE;
    u32 params[] = { args->state, args->n_v_heads, args->n_k_heads, args->head_dim };
    return node_add(g, OP_SSM_DELTA,
                    (GraphNode *[]){ args->fused, args->alpha, args->beta, NULL },
                    (TensorInfo *[]){ args->ssm_a, args->dt_bias, args->norm, NULL },
                    params, 4);
}

/* Caller-owned state buffers.  The graph only borrows the pointer, so the
 * owner (the arch workspace) keeps control of allocation and reset. */
u32 graph_state(Graph *g, void *ptr) {
    if (!g || !ptr) return GRAPH_STATE_NONE;
    if (!grow((void **)&g->state, &g->state_cap, g->n_state + 1, sizeof(void *)))
        return GRAPH_STATE_NONE;
    g->state[g->n_state] = ptr;
    return g->n_state++;
}

GraphNode *graph_attn(Graph *g, GraphNode *q, GraphNode *k, GraphNode *v,
                      u32 layer, u32 window) {
    if (!g || !q || !k || !v) return GRAPH_NODE_NONE;
    u32 params[] = { layer, window };   /* window == 0: full causal attention */
    return node_add(g, OP_ATTN, (GraphNode *[]){ q, k, v, NULL }, NULL, params, 2);
}

GraphNode *graph_rope(Graph *g, GraphNode *src, float theta,
                      u32 n_heads, u32 head_dim, u32 rope_dim) {
    if (!g || !src) return GRAPH_NODE_NONE;
    if (head_dim == 0 || rope_dim == 0 || rope_dim > head_dim) return GRAPH_NODE_NONE;
    u32 bits;
    memcpy(&bits, &theta, sizeof(bits));   /* float bits in params[0] */
    u32 params[] = { bits, n_heads, head_dim, rope_dim };
    return node_add(g, OP_ROPE_NEOX, (GraphNode *[]){ src, NULL, NULL, NULL }, NULL, params, 4);
}

/* Consecutive-pair RoPE (GGML_ROPE_TYPE_NORM): rotates (x[i], x[i+1])
 * pairs, the convention used by Llama-family GGUF weights. */
GraphNode *graph_rope_norm(Graph *g, GraphNode *src, float theta,
                           u32 n_heads, u32 head_dim, u32 rope_dim) {
    if (!g || !src) return GRAPH_NODE_NONE;
    if (head_dim == 0 || rope_dim == 0 || rope_dim > head_dim) return GRAPH_NODE_NONE;
    u32 bits;
    memcpy(&bits, &theta, sizeof(bits));   /* float bits in params[0] */
    u32 params[] = { bits, n_heads, head_dim, rope_dim };
    return node_add(g, OP_ROPE_NORM, (GraphNode *[]){ src, NULL, NULL, NULL }, NULL, params, 4);
}

GraphNode *graph_embed(Graph *g, GraphNode *src, TensorInfo *weight, u32 n_tokens) {
    if (!g || !weight || weight->ndim < 2 || n_tokens == 0) return GRAPH_NODE_NONE;
    if (!src || src->op != OP_INPUT || src->ne[0] != (int)n_tokens) return GRAPH_NODE_NONE;
    return node_add(g, OP_EMBED, (GraphNode *[]){ src, NULL, NULL, NULL },
                    (TensorInfo *[]){ weight, NULL, NULL, NULL }, NULL, 0);
}


/* ---------------------------------------------------------------- *
 * Graph creation
 * ---------------------------------------------------------------- */

static void *arena_alloc(BackendType backend, size_t bytes);
static void arena_free(BackendType backend, void *arena);
static void cache_destroy(GraphPlan *p);
static void plan_reset(Graph *g);
static const char *backend_name(BackendType backend);

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
        fprintf(stderr, "  +-----------+----------+-------+-------------------------------+\n");
        fprintf(stderr, "  | %-9s | %8s | %5s | %-29s |\n",
                "op", "total", "pct", "calls / avg");
        fprintf(stderr, "  +-----------+----------+-------+-------------------------------+\n");
        for (size_t i = 0; i < sizeof(op_stat_secs) / sizeof(op_stat_secs[0]); i++) {
            if (op_stat_secs[i] <= 1e-5) continue;
            char extra[32];
            snprintf(extra, sizeof(extra), "%llu / %.3f ms",
                     (unsigned long long)op_stat_calls[i],
                     op_stat_calls[i] ? 1000.0 * op_stat_secs[i] / (double)op_stat_calls[i] : 0.0);
            fprintf(stderr, "  | %-9s | %7.3fs | %4.1f%% | %-29s |\n",
                    op_stat_name[i], op_stat_secs[i],
                    100.0 * op_stat_secs[i] / total, extra);
        }
        fprintf(stderr, "  +-----------+----------+-------+-------------------------------+\n");
    }
    if (g->plan) {
        for (u32 b = 0; b < BACKEND_COUNT; b++)
            if (g->plan->arena_size[b])
                fprintf(stderr, "graph arena[%s]: %.1f MB\n",
                        backend_name((BackendType)b),
                        (double)g->plan->arena_size[b] / 1048576.0);
    }
    memset(op_stat_secs, 0, sizeof(op_stat_secs));
    memset(op_stat_calls, 0, sizeof(op_stat_calls));

    /* Every node->data aliases a slot inside the plan's per-backend
     * arena; release all of them plus the KV caches. */
    if (g->plan) {
        plan_reset(g);
        sfree(g->plan);
        g->plan = NULL;
    }
    sfree(g->state);   /* borrowed pointers only: targets outlive us */
    g->state = NULL;
    for (GraphNode *n = g->head; n; ) {
        GraphNode *next = n->next;
        sfree(n);
        n = next;
    }
    g->head = g->tail = NULL;
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

/* Capacity rows a node can ever be asked to produce.  Non-sink
 * activations only need to cover one ubatch (forward pass), never the
 * whole context. */
static u32 node_cap_rows(const GraphNode *node, bool sink, u32 n_ubatch) {
    if (node->op == OP_INPUT) return (u32)node->ne[0];
    return sink ? 1u : n_ubatch;
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
            return node->src[0] ? dims[node->src[0]->idx] : 0;
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
            if (cudaMalloc(&d, bytes) != cudaSuccess) {
                (void)cudaGetLastError();
                return NULL;
            }
            return d;
        }
#endif
#ifdef METAL_BUILD
        case BACKEND_METAL:
            return metal_arena_alloc(bytes);
#endif
        case BACKEND_CPU:
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
#ifdef METAL_BUILD
        case BACKEND_METAL:
            metal_arena_free(arena);
            break;
#endif
        case BACKEND_CPU:
        default:
            sfree(arena);
            break;
    }
}

static bool arena_plan(Graph *g, Session *s) {
    u32 n_node = g->n_node;
    ArchConfig *c = &s->cfg;
    if (n_node == 0) return true;

    /* Materialise the linked list into build order for random access. */
    GraphNode *nodes[n_node];
    for (GraphNode *n = g->head; n; n = n->next) nodes[n->idx] = n;

    /* Sink marks and per-node output widths (mirrors graph_compute). */
    u32 sink[n_node], dims[n_node];
    for (u32 i = 0; i < n_node; i++) sink[i] = 1;
    for (u32 j = 0; j < n_node; j++)
        for (int k = 0; k < GRAPH_NODE_MAX_SRC; k++)
            if (nodes[j]->src[k]) sink[nodes[j]->src[k]->idx] = 0;
    for (u32 i = 0; i < n_node; i++)
        dims[i] = node_out_dim(nodes[i], dims, c);

    /* Byte needs at full capacity (INPUT slots hold u32 token ids). */
    size_t need[n_node];
    for (u32 i = 0; i < n_node; i++) {
        const GraphNode *node = nodes[i];
        u32 rows = node_cap_rows(node, sink[i] != 0, s->n_ubatch);
        need[i] = align16((node->op == OP_INPUT)
                              ? (size_t)node->ne[0] * sizeof(u32)
                              : (size_t)rows * dims[i] * sizeof(float));
    }

    /* release[i] = index of the last node consuming node i. */
    u32 release[n_node];
    for (u32 i = 0; i < n_node; i++) release[i] = i;
    for (u32 j = 0; j < n_node; j++)
        for (int k = 0; k < GRAPH_NODE_MAX_SRC; k++)
            if (nodes[j]->src[k]) {
                u32 src = nodes[j]->src[k]->idx;
                if (j > release[src]) release[src] = j;
            }

    /* Sweep: reclaim slots whose last consumer already ran, then give
     * the next node the smallest free chunk that fits (else extend).
     * Each backend has its own free list and bump pointer, because a
     * split plan keeps GPU and host activations in separate arenas.
     * Slot lifetimes are still computed globally: only the explicit
     * H2D/D2H copy nodes straddle the split. */
    u64  node_off[n_node];
    u64  foff[BACKEND_COUNT][n_node];
    u64  fsz[BACKEND_COUNT][n_node];
    u32  fcnt[BACKEND_COUNT] = {0};
    u64  bump[BACKEND_COUNT] = {0};

    for (u32 i = 0; i < n_node; i++) {
        /* Node j's slot becomes free exactly at step release[j] + 1. */
        if (i > 0) {
            for (u32 j = 0; j < i; j++) {
                if (release[j] + 1 != i) continue;
                u32 b = (u32)nodes[j]->backend;
                foff[b][fcnt[b]] = node_off[j];
                fsz[b][fcnt[b]]  = (u64)need[j];
                fcnt[b]++;
            }
        }

        u32 b = (u32)nodes[i]->backend;
        u64 off;
        u32 best = fcnt[b];              /* smallest chunk that fits */
        u64 best_sz = UINT64_MAX;
        for (u32 f = 0; f < fcnt[b]; f++) {
            if (fsz[b][f] >= (u64)need[i] && fsz[b][f] < best_sz) {
                best    = f;
                best_sz = fsz[b][f];
            }
        }
        if (best < fcnt[b]) {
            off = foff[b][best];
            fsz[b][best]  = fsz[b][fcnt[b] - 1];   /* swap-remove */
            foff[b][best] = foff[b][fcnt[b] - 1];
            fcnt[b]--;
        } else {
            off = bump[b];
            bump[b] += need[i];
        }
        node_off[i] = off;
    }

    /* Reuse the plan shell created by backend_plan; create one only
     * when the caller skipped that step. */
    GraphPlan *p = g->plan;
    bool fresh = false;
    if (!p) {
        p = scalloc(1, sizeof(GraphPlan));
        if (!p) return false;
        p->backend = BACKEND_CPU;
        fresh = true;
    }
    g->plan = p;

    for (u32 b = 0; b < BACKEND_COUNT; b++) {
        if (bump[b] == 0) continue;
        p->arena[b] = arena_alloc((BackendType)b, (size_t)bump[b]);
        if (!p->arena[b]) {
            plan_reset(g);
            if (fresh) { sfree(p); g->plan = NULL; }
            return false;
        }
        p->arena_size[b] = (size_t)bump[b];
    }

    for (u32 i = 0; i < n_node; i++) {
        GraphNode *node = nodes[i];
        u32 b = (u32)node->backend;
        node->data      = (u8 *)p->arena[b] + node_off[i];
        node->data_cap  = need[i];
    }
    return true;
}

/* Build the KV cache inside the plan.  All float storage is carved from
 * one contiguous block allocated on the plan's backend (host for CPU,
 * VRAM for CUDA/Metal) — the same allocator as the activation arena.
 * The KvCache metadata and the per-layer descriptor arrays stay on the
 * host because the executor reads them there before dispatching each
 * OP_ATTN. */
static bool cache_build(Graph *g, Session *s) {
    ArchConfig *c = &s->cfg;
    GraphPlan  *p = g->plan;
    if (p->cache) return true;              /* already built */

    KvCache *kc = scalloc(1, sizeof(KvCache));
    if (!kc) return false;
    kc->n_layer   = c->n_layer;
    kc->n_kv_head = c->n_kv_head;
    p->cache      = kc;

    bool mla = s->en && s->en->model && s->en->model->arch == ARCH_DEEPSEEK;
    kc->is_mla = mla;

    if (mla) {
        kc->head_dim = c->head_dim;
        u64 rank        = c->kv_lora_rank ? c->kv_lora_rank : 1;
        u64 raw_bytes   = (u64)s->ctx_size * c->head_dim * sizeof(float);
        u64 comp_bytes  = (u64)s->ctx_size * rank * sizeof(float);
        u64 score_bytes = (u64)s->ctx_size * sizeof(float);
        size_t total = (size_t)(raw_bytes + comp_bytes * 4 + score_bytes * 2) * c->n_layer;

        kc->mla = scalloc((u64)c->n_layer, sizeof(MLAKvCache));
        if (!kc->mla) return false;
        p->cache_arena[p->backend] = arena_alloc(p->backend, total);
        if (!p->cache_arena[p->backend]) return false;

        u8 *base = p->cache_arena[p->backend];
        for (u32 i = 0; i < c->n_layer; i++) {
            MLAKvCache *mc = &kc->mla[i];
            mc->cap_raw        = s->ctx_size;
            mc->compress_ratio = c->kv_lora_rank > 0 ? c->head_dim / c->kv_lora_rank : 2;
            mc->comp_cap       = s->ctx_size;
            mc->raw_kv            = (float *)base; base += raw_bytes;
            mc->attn_comp_kv      = (float *)base; base += comp_bytes;
            mc->attn_state_kv     = (float *)base; base += comp_bytes;
            mc->attn_state_score  = (float *)base; base += score_bytes;
            mc->index_comp_kv     = (float *)base; base += comp_bytes;
            mc->index_state_kv    = (float *)base; base += comp_bytes;
            mc->index_state_score = (float *)base; base += score_bytes;
        }
        p->kvcache_size[p->backend] = total;
        return true;
    }

    /* Standard attention: allocate K/V only for layers that actually
     * carry an OP_ATTN node (Qwen3.5 is hybrid, so most layers have no
     * KV cache at all), and place each layer's cache on the backend
     * that executes that layer. */
    kc->head_dim = c->kv_head_dim;
    size_t bytes = (size_t)s->ctx_size * c->n_kv_head * c->kv_head_dim
                   * sizeof(float);

    u8 used[c->n_layer];
    BackendType lb[c->n_layer];
    memset(used, 0, sizeof(used));
    memset(lb, 0, sizeof(lb));
    for (GraphNode *n = g->head; n; n = n->next) {
        if (n->op != OP_ATTN) continue;
        u32 l = n->params[0];
        if (l >= c->n_layer) continue;
        used[l] = 1;
        lb[l]   = n->backend;
    }

    kc->std = scalloc((u64)c->n_layer, sizeof(AttnKvCache));
    if (!kc->std) return false;

    /* Per-backend cache arena sizes. */
    for (u32 l = 0; l < c->n_layer; l++)
        if (used[l]) p->kvcache_size[lb[l]] += bytes * 2;

    u8 *base[BACKEND_COUNT] = {0};
    for (u32 b = 0; b < BACKEND_COUNT; b++) {
        if (p->kvcache_size[b] == 0) continue;
        p->cache_arena[b] = arena_alloc((BackendType)b, p->kvcache_size[b]);
        if (!p->cache_arena[b]) return false;
        base[b] = p->cache_arena[b];
    }

    for (u32 l = 0; l < c->n_layer; l++) {
        if (!used[l]) continue;
        u32 b = (u32)lb[l];
        kc->std[l].k   = (float *)base[b]; base[b] += bytes;
        kc->std[l].v   = (float *)base[b]; base[b] += bytes;
        kc->std[l].cap = s->ctx_size;
    }
    return true;
}

/* Release the plan's KV cache storage on every backend it used. */
static void cache_destroy(GraphPlan *p) {
    KvCache *kc = p->cache;
    if (!kc) return;
    if (kc->is_mla) sfree(kc->mla);
    else            sfree(kc->std);
    sfree(kc);
    for (u32 b = 0; b < BACKEND_COUNT; b++) {
        arena_free((BackendType)b, p->cache_arena[b]);
        p->cache_arena[b] = NULL;
        p->kvcache_size[b] = 0;
    }
    p->cache = NULL;
}

/* Free every allocation held by the plan but keep the shell so the
 * planner can report a clean failure and graph_free() can reuse it. */
static void plan_reset(Graph *g) {
    GraphPlan *p = g ? g->plan : NULL;
    if (!p) return;
    cache_destroy(p);
    for (u32 b = 0; b < BACKEND_COUNT; b++) {
        arena_free((BackendType)b, p->arena[b]);
        p->arena[b] = NULL;
        p->arena_size[b] = 0;
        p->weights_size[b] = 0;
    }
    p->ready = false;
}

/* Warmup: preload the weights referenced by GPU-resident nodes so the
 * first forward pass never pays an H2D cost mid-compute.  CPU weights
 * are the mmap itself and host-resident already, so they are skipped.
 * Returns false if a device upload fails (CUDA OOM). */
static bool graph_warmup(Graph *g) {
    GraphPlan *p = g ? g->plan : NULL;
    if (!p) return false;

    for (GraphNode *n = g->head; n; n = n->next) {
        for (int k = 0; k < GRAPH_NODE_MAX_SRC; k++) {
            TensorInfo *w = n->weights[k];
            if (!w) continue;
#if defined(GPU_BUILD)
            if (n->backend == BACKEND_CUDA && !gpu_weight_dev(w)) {
                slog(WARN, "graph_warmup: CUDA upload failed for %.*s",
                     (int)w->key.len, w->key.content ? w->key.content : "?");
                return false;
            }
#endif
#if defined(METAL_BUILD)
            if (n->backend == BACKEND_METAL) metal_weight_preload(w);
#endif
        }
    }
    return true;
}

/* Assign a compute backend to every node from the session's -ngl
 * setting.  Prefix layers go to the device, the rest to the host; the
 * final norm / LM head use layer == n_layer and therefore stay on the
 * host unless the whole model is offloaded. */
static bool assign_backends(Graph *g, Session *s) {
    if (!g || !s) return false;
    int ngl = s->n_gpu_layers;
    u32 L = s->cfg.n_layer;
    if (ngl < 0) ngl = 0;
    if ((u32)ngl > L) ngl = (int)L;
    s->n_gpu_layers = ngl;

    for (GraphNode *n = g->head; n; n = n->next) {
        n->backend = BACKEND_CPU;
#if defined(GPU_BUILD)
        if (ngl > 0 && ((u32)ngl >= L || n->layer < (u32)ngl))
            n->backend = BACKEND_CUDA;
#elif defined(METAL_BUILD)
        /* Metal keeps the historical all-or-nothing behaviour. */
        if (ngl > 0) n->backend = BACKEND_METAL;
#else
        (void)L;
#endif
        if (n->op == OP_H2D || n->op == OP_D2H)
            n->backend = (BackendType)n->params[0];
    }

    /* Primary backend, used by the MLA cache path and reporting. */
    BackendType primary = BACKEND_CPU;
#if defined(GPU_BUILD)
    if (ngl > 0) primary = BACKEND_CUDA;
#elif defined(METAL_BUILD)
    if (ngl > 0) primary = BACKEND_METAL;
#endif
    if (!g->plan) {
        g->plan = scalloc(1, sizeof(GraphPlan));
        if (!g->plan) return false;
    }
    g->plan->backend = primary;
    return true;
}

/* Tally each distinct weight tensor once per backend that uses it. */
static void account_weights(Graph *g) {
    GraphPlan *p = g->plan;
    for (u32 b = 0; b < BACKEND_COUNT; b++) p->weights_size[b] = 0;
    for (GraphNode *n = g->head; n; n = n->next) {
        for (int k = 0; k < GRAPH_NODE_MAX_SRC; k++) {
            TensorInfo *w = n->weights[k];
            if (!w) continue;
            bool seen = false;
            for (GraphNode *m = g->head; m && !seen; m = m->next) {
                if (m->idx > n->idx) break;
                if (m->backend != n->backend) continue;
                int lim = (m == n) ? k : GRAPH_NODE_MAX_SRC;
                for (int t = 0; t < lim; t++)
                    if (m->weights[t] == w) { seen = true; break; }
            }
            if (!seen) p->weights_size[n->backend] += (size_t)w->bytes;
        }
    }
}

static const char *backend_name(BackendType backend) {
    switch (backend) {
        case BACKEND_CPU:   return "cpu";
        case BACKEND_CUDA:  return "cuda";
        case BACKEND_METAL: return "metal";
        default:            return "unknown";
    }
}

static const char *op_name(GraphOp op) {
    switch (op) {
        case OP_INPUT:          return "input";
        case OP_ADD:            return "add";
        case OP_MUL:            return "mul";
        case OP_MATMUL:         return "matmul";
        case OP_MATMULARRY:     return "matmul_arr";
        case OP_MATMUL_T:       return "matmul_t";
        case OP_SCALE:          return "scale";
        case OP_RMS_NORM:       return "rms_norm";
        case OP_RMS_NORM_HEADS: return "rms_heads";
        case OP_SIGMOID_GATE:   return "sigmoid_gate";
        case OP_ROPE_NEOX:      return "rope_neox";
        case OP_ROPE_NORM:      return "rope_norm";
        case OP_SOFTMAX:        return "softmax";
        case OP_SILU:           return "silu";
        case OP_GELU:           return "gelu";
        case OP_EMBED:          return "embed";
        case OP_BIAS:           return "bias";
        case OP_ATTN:           return "attn";
        case OP_SSM_CONV:       return "ssm_conv";
        case OP_SSM_DELTA:      return "ssm_delta";
        case OP_SOFTCAP:        return "softcap";
        case OP_H2D:            return "h2d";
        case OP_D2H:            return "d2h";
        default:                return "?";
    }
}

/* Render the static graph as a linear workflow.  Nodes are listed in
 * build order (== topo / execution order), and every edge is shown as
 * the index of its source, so a split plan's H2D/D2H boundaries and
 * each op's weights are visible at a glance. */
void graph_drawer(const Graph *g) {
    if (!g) {
        slog(WARN, "graph_drawer: no graph to draw");
        return;
    }

    u32 n_sink = 0;
    for (const GraphNode *n = g->head; n; n = n->next) {
        bool used = false;
        for (const GraphNode *m = g->head; m && !used; m = m->next)
            for (int k = 0; k < GRAPH_NODE_MAX_SRC; k++)
                if (m->src[k] == n) { used = true; break; }
        if (!used) n_sink++;
    }

    printf("\n");
    printf("Graph workflow: %u nodes, %u states, %u sinks\n",
           g->n_node, g->n_state, n_sink);
    printf("  +-------+-------+-------+--------------+--------------------------+------------------------------------------------+\n");
    printf("  | %5s | %5s | %-5s | %-12s | %-24s | %-46s |\n",
           "idx", "layer", "bkd", "op", "inputs", "weights");
    printf("  +-------+-------+-------+--------------+--------------------------+------------------------------------------------+\n");

    char src[64];
    char wts[512];
    for (const GraphNode *n = g->head; n; n = n->next) {
        int so = 0;
        src[0] = '\0';
        for (int k = 0; k < GRAPH_NODE_MAX_SRC; k++) {
            if (!n->src[k]) continue;
            so += snprintf(src + so, sizeof(src) - (size_t)so, "%s%u",
                           so ? "," : "", n->src[k]->idx);
            if ((size_t)so >= sizeof(src)) break;
        }
        if (!src[0]) snprintf(src, sizeof(src), "-");

        int wo = 0;
        wts[0] = '\0';
        for (int k = 0; k < GRAPH_NODE_MAX_SRC; k++) {
            TensorInfo *w = n->weights[k];
            if (!w) continue;
            wo += snprintf(wts + wo, sizeof(wts) - (size_t)wo, "%s%.*s",
                           wo ? "," : "",
                           (int)w->key.len, w->key.content ? w->key.content : "?");
            if ((size_t)wo >= sizeof(wts)) break;
        }
        if (!wts[0]) snprintf(wts, sizeof(wts), "-");

        printf("  | %5u | %5u | %-5s | %-12s | %-24s | %-46s |\n",
               n->idx, n->layer, backend_name(n->backend),
               op_name(n->op), src, wts);
    }
    printf("  +-------+-------+-------+--------------+--------------------------+------------------------------------------------+\n");

    printf("Sinks:");
    for (const GraphNode *n = g->head; n; n = n->next) {
        bool used = false;
        for (const GraphNode *m = g->head; m && !used; m = m->next)
            for (int k = 0; k < GRAPH_NODE_MAX_SRC; k++)
                if (m->src[k] == n) { used = true; break; }
        if (!used) printf(" %u(%s)", n->idx, op_name(n->op));
    }
    printf("\n\n");
}

/* Dump the static execution plan: node count and the sizes of the three
 * big allocations (weights, activation arena, KV cache) per backend. */
static void graph_plan_print(const Graph *g, const Session *s) {
    if (!g || !g->plan) return;
    const GraphPlan *p = g->plan;
    slog(INFO, "GraphPlan: backend=%s n_gpu_layers=%d nodes=%u state=%u "
         "ctx_size=%u n_ubatch=%u",
         backend_name(p->backend), s ? s->n_gpu_layers : 0,
         g->n_node, g->n_state, s ? s->ctx_size : 0,
         s ? s->n_ubatch : 0);
    for (u32 b = 0; b < BACKEND_COUNT; b++) {
        if (p->weights_size[b] == 0 && p->arena_size[b] == 0 &&
            p->kvcache_size[b] == 0) continue;
        slog(INFO, "GraphPlan[%s]: weights=%.1f MB arena=%.1f MB kvcache=%.1f MB total=%.1f MB",
             backend_name((BackendType)b),
             (double)p->weights_size[b] / 1048576.0,
             (double)p->arena_size[b] / 1048576.0,
             (double)p->kvcache_size[b] / 1048576.0,
             (double)(p->weights_size[b] + p->arena_size[b] + p->kvcache_size[b]) / 1048576.0);
    }
}

static bool graph_plan(Graph *g, Session *s) {
    if (!g || !s) return false;
    if (g->plan && g->plan->ready) return true;

    /* The offload split (and the recurrent-state placement it implies)
     * is fixed when the graph is built, so planning is a single
     * attempt: session_resolve_gpu_layers() has already clamped the
     * requested layer count to what the device can hold. */
    if (s->n_gpu_layers < 0) s->n_gpu_layers = 0;
    if (!assign_backends(g, s)) return false;
    account_weights(g);
    if (!arena_plan(g, s) || !cache_build(g, s) || !graph_warmup(g)) {
        plan_reset(g);
        slog(WARN, "GraphPlan: allocation failed (try fewer -ngl layers or a smaller -c)");
        return false;
    }
    g->plan->ready = true;
    graph_plan_print(g, s);
    return true;
}

/* Execute an explicit backend-boundary copy (OP_H2D / OP_D2H). */
static bool graph_copy_op(const GraphNode *node, float *dst, u64 bytes) {
    const float *src = (const float *)node->src[0]->data;
    if (!src || !dst) return false;
#if defined(GPU_BUILD)
    if (node->op == OP_D2H)
        return cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
    if (node->op == OP_H2D)
        return cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice) == cudaSuccess;
#endif
    memcpy(dst, src, bytes);
    return true;
}

static bool graph_op_compute(OpCtx ctx) {
    /* Explicit backend boundary (split plan). */
    if (ctx.node->op == OP_H2D || ctx.node->op == OP_D2H)
        return graph_copy_op(ctx.node, ctx.dst,
                             (u64)ctx.r * ctx.od * sizeof(float));

    /* Dispatch per node, so a split plan can run some layers on the
     * device and the rest on the host. */
    bool ok;
#ifdef METAL_BUILD
    if (ctx.node->backend == BACKEND_METAL)
        ok = metal_graph_op(&ctx);
    else
#endif
#ifdef GPU_BUILD
    if (ctx.node->backend == BACKEND_CUDA)
        ok = gpu_graph_op(&ctx);
    else
#endif
        ok = cpu_graph_op(&ctx);
    return ok;
}

static bool graph_compute(Graph *g, const GraphBatch *b, Session *s) {
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
    if ((!g->plan || !g->plan->ready) && !graph_plan(g, s)) {
        slog(WARN, "graph_compute: arena plan failed");
        return false;
    }

    /* Bind the batch's token ids into each OP_INPUT leaf. */
    for (GraphNode *node = g->head; node; node = node->next) {
        if (node->op != OP_INPUT) continue;
        if (n > (u32)node->ne[0]) {
            slog(WARN, "graph_compute: batch of %u exceeds graph capacity", n);
            return false;
        }
        if (b->tokens) {
#ifdef GPU_BUILD
            if (node->backend == BACKEND_CUDA) {
                /* Token ids live in the same arena as the embedding op. */
                CHECK(cudaMemcpy(node->data, b->tokens, (size_t)n * sizeof(u32),
                                 cudaMemcpyHostToDevice));
            } else
#endif
                memcpy(node->data, b->tokens, (size_t)n * sizeof(u32));
        }
    }

    ArchConfig *c = &s->cfg;
    u32 n_node = g->n_node;

    /* Materialise the linked list into build order for random access. */
    GraphNode *nodes[n_node];
    for (GraphNode *it = g->head; it; it = it->next) nodes[it->idx] = it;

    /* Sink detection: a node is a sink iff no other node reads it.
     * (Edges point strictly backward, so scanning all nodes is enough.) */
    u32 sink[n_node];
    for (u32 i = 0; i < n_node; i++) sink[i] = 1;
    for (u32 j = 0; j < n_node; j++)
        for (int k = 0; k < GRAPH_NODE_MAX_SRC; k++)
            if (nodes[j]->src[k]) sink[nodes[j]->src[k]->idx] = 0;

    /* Elements per row of each node's output (float count). */
    u32 dims[n_node];
    for (u32 i = 0; i < n_node; i++)
        dims[i] = node_out_dim(nodes[i], dims, c);

    /* Attention score scratch: the batch's last row attends every key
     * in the cache, positions 0..pos+n-1. */
    float *scr = smalloc((size_t)(pos + n) * sizeof(float));
    if (!scr) return false;

    u32 q_dim  = c->n_head * c->head_dim;
    u32 kv_dim = c->n_kv_head * c->kv_head_dim;
    float scale = 1.0f / sqrtf((float)c->kv_head_dim);

    /* Walk the graph in build order (== topo order). */
    for (u32 i = 0; i < n_node; i++) {
        GraphNode *node = nodes[i];
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
            
        bool ok = graph_op_compute(ctx);
        if (!ok) goto fail;
        
        /* Graph stats. */
        op_stat_secs[st_cls] += graph_now() - st_t0;
        op_stat_calls[st_cls]++;
    }

    /* Metal batches every dispatch of this walk into one command buffer;
     * commit and wait before the sink is read back. */
#ifdef METAL_BUILD
    if (g->plan->backend == BACKEND_METAL) metal_graph_flush();
#endif

    /* Sink output (last row of the LM head) → session logits.
     * A CUDA arena hands the slot back through one D2H. */
    for (u32 i = 0; i < n_node; i++) {
        if (!sink[i] || nodes[i]->op == OP_INPUT) continue;
        u32 n = dims[i] < c->n_vocab ? dims[i] : c->n_vocab;
#ifdef GPU_BUILD
        if (nodes[i]->backend == BACKEND_CUDA) {
            CHECK(cudaMemcpy(s->logits, nodes[i]->data, (size_t)n * sizeof(float),
                             cudaMemcpyDeviceToHost));
        } else
#endif
            memcpy(s->logits, nodes[i]->data, (size_t)n * sizeof(float));
        break;
    }

    sfree(scr);
    return true;

fail:
#ifdef METAL_BUILD
    if (g->plan->backend == BACKEND_METAL) metal_graph_flush();
#endif
    sfree(scr);
    return false;
}

/* Append tokens to the session and run them through the graph in
 * n_ubatch-sized chunks.  The graph is built once at n_ubatch capacity,
 * so activation slots cover one forward pass rather than the whole
 * context; prompt prefill is split the same way llama.cpp splits a
 * logical batch into micro-batches.  KV/recurrent state accumulates
 * across chunks, and the final chunk's last row publishes the logits.
 * Replaces the old prefill / generate arch hooks — callers pass the
 * prompt ids for prefill and a single id for decode. */
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
        s->graph = s->ops.graph_build(s, s->n_ubatch);
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

    u32 done = 0;
    while (done < n_tokens) {
        u32 chunk = n_tokens - done;
        if (chunk > s->n_ubatch) chunk = s->n_ubatch;

        memcpy(s->tokens + pos + done, tokens + done,
               (size_t)chunk * sizeof(u32));
        GraphBatch batch = {
            .tokens = s->tokens + pos + done,
            .pos    = pos + done,
            .n      = chunk
        };
        if (!graph_compute(s->graph, &batch, s)) return false;
        done += chunk;
    }
    s->n_tokens = pos + n_tokens;

    /* graph_compute() leaves the logits in s->logits; mirror them out when
     * the caller asked for a different buffer. */
    if (logits && logits != s->logits)
        memcpy(logits, s->logits, (size_t)s->cfg.n_vocab * sizeof(float));

    return true;
}

/* Build + plan + preload up front.  graph_execute() does all of this
 * lazily on first call; calling this at startup moves the one-time
 * graph-build, arena planning and weight upload (warmup) off the first
 * request's critical path, so TTFT reflects compute only. */
bool graph_prepare(Session *s) {
    if (!s) return false;
    if (!s->graph) {
        if (!s->ops.graph_build) {
            slog(WARN, "graph_prepare: no graph builder for this architecture");
            return false;
        }
        s->graph = s->ops.graph_build(s, s->n_ubatch);
        if (!s->graph) {
            slog(WARN, "graph_prepare: graph build failed");
            return false;
        }
    }
    if (!s->graph->plan && !graph_plan(s->graph, s)) {
        slog(WARN, "graph_prepare: plan / warmup failed");
        return false;
    }
    return true;
}
