/*
 * metal_op.m — Metal graph operators (device twins of graph.c's
 * op_table).
 *
 * Mirrors backend/gpu/gpu_op.cu.  The executor's arena is one shared
 * MTLBuffer (see metal_arena_alloc); because shared storage is
 * CPU- and GPU-visible on Apple silicon, activations never leave the
 * device and the executor's token/logit memcpys keep working.  Weights
 * are the only host-originated data and are served from metal.m's
 * persistent raw-weight cache; small 1-D weights (norms, biases, SSM
 * params) are dequantized once into cached f32 buffers.
 *
 * All dispatches for one graph_compute() are encoded into a single
 * command buffer and committed by metal_graph_flush(), so a token is
 * one submit + one wait instead of one per matmul.  Per-op scalars are
 * staged in an auxiliary shared buffer at 256-byte-aligned offsets
 * (constant-buffer alignment).
 *
 * The KV cache lives in host memory allocated by the arch init; this
 * backend keeps a per-layer device mirror that op_kv_write fills
 * directly from the arena, so attention runs device-resident.  Gated
 * DeltaNet state gets a zero-initialized device mirror, reset when a
 * batch starts at position 0.
 */

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../def.h"
#include "metal.h"
#include "metal_internal.h"
#include "metal_op.h"
#include "../../slog.h"

#define OP_THREADS 256
#define OP_MAX_GRID 65535u

static inline NSUInteger op_blocks(u64 n) {
    u64 want = (n + OP_THREADS - 1) / OP_THREADS;
    if (want == 0) want = 1;
    return (NSUInteger)(want < OP_MAX_GRID ? want : OP_MAX_GRID);
}

/* ================================================================
 * Command-buffer batching + aux argument staging
 * ================================================================ */

static id<MTLCommandBuffer>          g_cb;
static id<MTLComputeCommandEncoder>  g_enc;
static id<MTLBuffer>                 g_args;
static NSUInteger                    g_args_cap;
static NSUInteger                    g_args_used;
static NSMutableArray               *g_args_retired;

static void op_encoder_begin(void) {
    if (g_enc) return;
    if (!g_args_retired) g_args_retired = [NSMutableArray array];
    g_cb  = [metal_i_queue() commandBuffer];
    g_enc = [g_cb computeCommandEncoder];
}

/* Reserve a 256-byte-aligned slice of the shared argument buffer and
 * return its byte offset (or (NSUInteger)-1 on allocation failure). */
static NSUInteger args_alloc(size_t bytes) {
    NSUInteger aligned = ((NSUInteger)bytes + 255u) & ~(NSUInteger)255u;
    if (aligned == 0) aligned = 256u;
    if (!g_args || g_args_used + aligned > g_args_cap) {
        NSUInteger cap = g_args_cap ? g_args_cap * 2u : (1u << 20);
        while (cap < aligned) cap *= 2u;
        id<MTLBuffer> nb = [metal_i_dev() newBufferWithLength:cap
                                                     options:MTLResourceStorageModeShared];
        if (!nb) return (NSUInteger)-1;
        if (g_args) [g_args_retired addObject:g_args];
        g_args = nb;
        g_args_cap = cap;
        g_args_used = 0;
    }
    NSUInteger off = g_args_used;
    g_args_used += aligned;
    return off;
}

static uint32_t *args_ptr(NSUInteger off) {
    return (uint32_t *)((uint8_t *)g_args.contents + off);
}

static inline void put_u64(uint32_t *p, int i, uint64_t v) {
    p[i]     = (uint32_t)v;
    p[i + 1] = (uint32_t)(v >> 32);
}

static void dispatch(const char *name,
                     id<MTLBuffer> __strong *bufs, NSUInteger *offs, int nbuf,
                     NSUInteger ntg) {
    if (!g_enc) op_encoder_begin();
    static NSMutableDictionary *pipes;
    if (!pipes) pipes = [NSMutableDictionary dictionary];
    NSString *key = [NSString stringWithUTF8String:name];
    id<MTLComputePipelineState> pipe = pipes[key];
    if (!pipe) {
        pipe = metal_i_pipe(key);
        if (!pipe) { fprintf(stderr, "metal_op: no pipeline %s\n", name); exit(1); }
        pipes[key] = pipe;
    }
    [g_enc setComputePipelineState:pipe];
    for (int i = 0; i < nbuf; i++)
        [g_enc setBuffer:bufs[i] offset:offs[i] atIndex:(NSUInteger)i];
    [g_enc dispatchThreadgroups:MTLSizeMake(ntg, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(OP_THREADS, 1, 1)];
    [g_enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

void metal_graph_flush(void) {
    if (g_enc) {
        [g_enc endEncoding];
        [g_cb commit];
        [g_cb waitUntilCompleted];
        if (g_cb.status == MTLCommandBufferStatusError) {
            fprintf(stderr, "Error: metal_op: command buffer: %s\n",
                    g_cb.error ? g_cb.error.localizedDescription.UTF8String : "unknown");
            exit(1);
        }
        g_enc = nil;
        g_cb  = nil;
    }
    g_args_used = 0;
    if (g_args_retired) [g_args_retired removeAllObjects];
}

/* ================================================================
 * Buffer binding helpers
 * ================================================================ */

/* Resolve a raw activation pointer to (arena buffer, offset). */
static bool bind_ptr(const void *p, id<MTLBuffer> __strong *b, NSUInteger *o) {
    return metal_i_lookup(p, b, o) == 0;
}

/* ================================================================
 * Dequantized f32 weight cache (small 1-D tensors)
 * ================================================================ */

#define OP_WCACHE_MAX 1024
static TensorInfo   *g_wc_ti[OP_WCACHE_MAX];
static id<MTLBuffer> g_wc_buf[OP_WCACHE_MAX];
static int           g_wc_n;

static id<MTLBuffer> weight_f32(TensorInfo *ti) {
    if (!ti || !ti->data || ti->n_element == 0) return nil;
    for (int i = 0; i < g_wc_n; i++)
        if (g_wc_ti[i] == ti) return g_wc_buf[i];
    if (g_wc_n >= OP_WCACHE_MAX) return nil;

    id<MTLBuffer> raw = metal_i_weight(ti);
    if (!raw) return nil;
    u64 n = ti->n_element;
    id<MTLBuffer> f = [metal_i_dev() newBufferWithLength:(NSUInteger)(n * sizeof(float))
                                                options:MTLResourceStorageModeShared];
    if (!f) return nil;

    NSUInteger poff = args_alloc(8 * sizeof(uint32_t));
    if (poff == (NSUInteger)-1) return nil;
    uint32_t *p = args_ptr(poff);
    p[0] = ti->type; p[1] = 0;       /* type, pad          */
    put_u64(p, 2, 0);                /* start_i            */
    put_u64(p, 4, 0);                /* i0                 */
    put_u64(p, 6, n);                /* nb                 */

    id<MTLBuffer> bufs[4] = { raw, f, metal_i_tables(), g_args };
    NSUInteger    offs[4] = { 0, 0, 0, poff };
    dispatch("dequant_range", bufs, offs, 4, op_blocks(n));

    g_wc_ti[g_wc_n]  = ti;
    g_wc_buf[g_wc_n] = f;
    g_wc_n++;
    return f;
}

/* ================================================================
 * KV-cache device mirror
 * ================================================================ */

#define OP_KV_MAX 256
typedef struct {
    const float *host;   /* akc->k — cache key                  */
    id<MTLBuffer> k, v;  /* [cap * n_kv_head * kv_head_dim]     */
    u64 cap;             /* float capacity of each buffer       */
} OpKv;
static OpKv g_kv[OP_KV_MAX];
static int  g_kv_n;

static OpKv *kv_get(const float *host, u64 need) {
    for (int i = 0; i < g_kv_n; i++) {
        if (g_kv[i].host != host) continue;
        if (g_kv[i].cap >= need) return &g_kv[i];
        g_kv[i].k = [metal_i_dev() newBufferWithLength:(NSUInteger)(need * sizeof(float))
                                               options:MTLResourceStorageModeShared];
        g_kv[i].v = [metal_i_dev() newBufferWithLength:(NSUInteger)(need * sizeof(float))
                                               options:MTLResourceStorageModeShared];
        g_kv[i].cap = need;
        return &g_kv[i];
    }
    if (g_kv_n >= OP_KV_MAX) return NULL;
    OpKv *e = &g_kv[g_kv_n];
    e->host = host;
    e->cap  = need;
    e->k = [metal_i_dev() newBufferWithLength:(NSUInteger)(need * sizeof(float))
                                      options:MTLResourceStorageModeShared];
    e->v = [metal_i_dev() newBufferWithLength:(NSUInteger)(need * sizeof(float))
                                      options:MTLResourceStorageModeShared];
    g_kv_n++;
    return e;
}

/* ================================================================
 * Gated DeltaNet device state mirror
 * ================================================================ */

#define OP_STATE_MAX 256
typedef struct {
    const void   *host;   /* graph_state() pointer — cache key */
    id<MTLBuffer> buf;
    size_t        bytes;
} OpState;
static OpState g_state[OP_STATE_MAX];
static int     g_state_n;

/* Returns the device mirror, zero-initializing it when fresh or when a
 * new sequence starts at position 0. */
static id<MTLBuffer> state_get(const void *host, size_t bytes, bool reset) {
    for (int i = 0; i < g_state_n; i++) {
        if (g_state[i].host != host) continue;
        if (reset) memset(g_state[i].buf.contents, 0, g_state[i].bytes);
        return g_state[i].buf;
    }
    if (g_state_n >= OP_STATE_MAX || bytes == 0) return nil;
    id<MTLBuffer> b = [metal_i_dev() newBufferWithLength:(NSUInteger)bytes
                                                 options:MTLResourceStorageModeShared];
    if (!b) return nil;
    memset(b.contents, 0, bytes);
    g_state[g_state_n].host  = host;
    g_state[g_state_n].buf   = b;
    g_state[g_state_n].bytes = bytes;
    g_state_n++;
    return b;
}

/* ================================================================
 * Operators
 * ================================================================ */

static float *op_src(const OpCtx *c, int k) {
    return (float *)c->g->node[(u32)c->node->src[k]].data;
}
static u32 op_param(const OpCtx *c, int k) {
    return c->node->params[k];
}

/* OP_EMBED */
static bool metal_op_embed(OpCtx *c) {
    TensorInfo *te = c->node->weights[0];
    bool te_trans  = (te->dim[0] == (i64)c->cfg->n_vocab);
    id<MTLBuffer> raw = metal_i_weight(te);
    if (!raw) return false;
    id<MTLBuffer> ids, bdst;
    NSUInteger ioff, doff;
    if (!bind_ptr(op_src(c, 0), &ids, &ioff) || !bind_ptr(c->dst, &bdst, &doff))
        return false;
    NSUInteger aoff = args_alloc(7 * sizeof(uint32_t));
    if (aoff == (NSUInteger)-1) return false;
    uint32_t *p = args_ptr(aoff);
    p[0] = te->type;
    p[1] = c->n;
    p[2] = c->od;
    put_u64(p, 3, te_trans ? (u64)c->od : 1);          /* base_mul */
    put_u64(p, 5, te_trans ? 1 : (u64)c->cfg->n_vocab); /* stride   */
    id<MTLBuffer> bufs[5] = { raw, ids, bdst, metal_i_tables(), g_args };
    NSUInteger    offs[5] = { 0, ioff, doff, 0, aoff };
    dispatch("op_dequant_gather", bufs, offs, 5, op_blocks((u64)c->n * c->od));
    return true;
}

/* OP_RMS_NORM */
static bool metal_op_rms_norm(OpCtx *c) {
    TensorInfo *tw = c->node->weights[0];
    id<MTLBuffer> w = weight_f32(tw);
    if (!w) return false;
    id<MTLBuffer> src, dst;
    NSUInteger soff, doff;
    if (!bind_ptr(op_src(c, 0), &src, &soff) || !bind_ptr(c->dst, &dst, &doff))
        return false;
    NSUInteger aoff = args_alloc(4 * sizeof(uint32_t));
    if (aoff == (NSUInteger)-1) return false;
    uint32_t *p = args_ptr(aoff);
    p[0] = c->od;
    p[1] = c->base;
    p[2] = (u32)tw->n_element;
    memcpy(&p[3], &(float){DEFAULT_EPS}, sizeof(float));
    id<MTLBuffer> bufs[4] = { src, dst, w, g_args };
    NSUInteger    offs[4] = { soff, doff, 0, aoff };
    dispatch("op_rms_norm", bufs, offs, 4, c->r);
    return true;
}

/* OP_RMS_NORM_HEADS */
static bool metal_op_rms_norm_heads(OpCtx *c) {
    u32 nh = op_param(c, 0), hd = op_param(c, 1);
    u32 is = op_param(c, 2), os = op_param(c, 3);
    id<MTLBuffer> nw = weight_f32(c->node->weights[0]);
    if (!nw) return false;
    id<MTLBuffer> src, dst;
    NSUInteger soff, doff;
    if (!bind_ptr(op_src(c, 0), &src, &soff) || !bind_ptr(c->dst, &dst, &doff))
        return false;
    NSUInteger aoff = args_alloc(7 * sizeof(uint32_t));
    if (aoff == (NSUInteger)-1) return false;
    uint32_t *p = args_ptr(aoff);
    p[0] = nh; p[1] = hd; p[2] = is; p[3] = os; p[4] = c->od; p[5] = c->base;
    memcpy(&p[6], &(float){DEFAULT_EPS}, sizeof(float));
    id<MTLBuffer> bufs[4] = { src, dst, nw, g_args };
    NSUInteger    offs[4] = { soff, doff, 0, aoff };
    dispatch("op_rms_norm_heads", bufs, offs, 4, (NSUInteger)c->r * nh);
    return true;
}

/* OP_MATMUL / OP_MATMULARRY / OP_MATMUL_T */
static bool metal_op_matmul(OpCtx *c) {
    TensorInfo *w  = c->node->weights[0];
    bool tr        = (c->node->op == OP_MATMUL_T);
    u64 rows       = tr ? (u64)w->dim[1] : (u64)w->dim[0];
    u64 cols       = tr ? (u64)w->dim[0] : (u64)w->dim[1];
    id<MTLBuffer> wb = metal_i_weight(w);
    if (!wb) return false;
    id<MTLBuffer> xb, yb;
    NSUInteger xoff, yoff;
    const float *x = op_src(c, 0) + (u64)c->base * cols;
    if (!bind_ptr(x, &xb, &xoff) || !bind_ptr(c->dst, &yb, &yoff))
        return false;
    NSUInteger aoff = args_alloc(6 * sizeof(uint32_t));
    if (aoff == (NSUInteger)-1) return false;
    uint32_t *p = args_ptr(aoff);
    put_u64(p, 0, rows);
    put_u64(p, 2, cols);
    put_u64(p, 4, c->r);
    id<MTLBuffer> bufs[5] = { wb, xb, yb, metal_i_tables(), g_args };
    NSUInteger    offs[5] = { 0, xoff, yoff, 0, aoff };
    NSUInteger    ntg = tr ? (((rows + 255) / 256) * c->r)
                           : (((rows + 7) / 8) * c->r);
    static const char *suffix[31] = {
        "F32", "F16", "Q4_0", "Q4_1", NULL, NULL, "Q5_0", "Q5_1",
        "Q8_0", "Q8_1", "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_K",
        "IQ2_XXS", "IQ2_XS", "IQ3_XXS", "IQ1_S", "IQ4_NL", "IQ3_S",
        "IQ2_S", "IQ4_XS", "I8", "I16", "I32", "I64", "F64", "IQ1_M", "BF16"
    };
    if (w->type >= 31 || !suffix[w->type]) return false;
    char kname[64];
    snprintf(kname, sizeof(kname), "matmul_%s_%s", tr ? "t" : "nt", suffix[w->type]);
    dispatch(kname, bufs, offs, 5, ntg);
    return true;
}

/* OP_BIAS */
static bool metal_op_bias(OpCtx *c) {
    TensorInfo *tb = c->node->weights[0];
    id<MTLBuffer> bw = weight_f32(tb);
    if (!bw) return false;
    id<MTLBuffer> src, dst;
    NSUInteger soff, doff;
    if (!bind_ptr(op_src(c, 0), &src, &soff) || !bind_ptr(c->dst, &dst, &doff))
        return false;
    u32 nb = tb->n_element < (u64)c->od ? (u32)tb->n_element : c->od;
    NSUInteger aoff = args_alloc(5 * sizeof(uint32_t));
    if (aoff == (NSUInteger)-1) return false;
    uint32_t *p = args_ptr(aoff);
    p[0] = nb; p[1] = c->od; p[2] = c->base;
    put_u64(p, 3, (u64)c->r * c->od);
    id<MTLBuffer> bufs[4] = { src, dst, bw, g_args };
    NSUInteger    offs[4] = { soff, doff, 0, aoff };
    dispatch("op_bias", bufs, offs, 4, op_blocks((u64)c->r * c->od));
    return true;
}

/* OP_ADD / OP_MUL */
static bool metal_op_binary(OpCtx *c) {
    bool add = (c->node->op == OP_ADD);
    id<MTLBuffer> a, b, dst;
    NSUInteger ao, bo, doff;
    if (!bind_ptr(op_src(c, 0), &a, &ao) || !bind_ptr(op_src(c, 1), &b, &bo) ||
        !bind_ptr(c->dst, &dst, &doff))
        return false;
    NSUInteger aoff = args_alloc(5 * sizeof(uint32_t));
    if (aoff == (NSUInteger)-1) return false;
    uint32_t *p = args_ptr(aoff);
    p[0] = c->od; p[1] = c->base;
    put_u64(p, 2, (u64)c->r * c->od);
    p[4] = add ? 1 : 0;
    id<MTLBuffer> bufs[4] = { a, b, dst, g_args };
    NSUInteger    offs[4] = { ao, bo, doff, aoff };
    dispatch("op_binary", bufs, offs, 4, op_blocks((u64)c->r * c->od));
    return true;
}

/* OP_SILU */
static bool metal_op_silu(OpCtx *c) {
    id<MTLBuffer> src, dst;
    NSUInteger soff, doff;
    if (!bind_ptr(op_src(c, 0), &src, &soff) || !bind_ptr(c->dst, &dst, &doff))
        return false;
    NSUInteger aoff = args_alloc(4 * sizeof(uint32_t));
    if (aoff == (NSUInteger)-1) return false;
    uint32_t *p = args_ptr(aoff);
    p[0] = c->od; p[1] = c->base;
    put_u64(p, 2, (u64)c->r * c->od);
    id<MTLBuffer> bufs[3] = { src, dst, g_args };
    NSUInteger    offs[3] = { soff, doff, aoff };
    dispatch("op_silu", bufs, offs, 3, op_blocks((u64)c->r * c->od));
    return true;
}

/* OP_SOFTMAX */
static bool metal_op_softmax(OpCtx *c) {
    id<MTLBuffer> src, dst;
    NSUInteger soff, doff;
    if (!bind_ptr(op_src(c, 0), &src, &soff) || !bind_ptr(c->dst, &dst, &doff))
        return false;
    NSUInteger aoff = args_alloc(2 * sizeof(uint32_t));
    if (aoff == (NSUInteger)-1) return false;
    uint32_t *p = args_ptr(aoff);
    p[0] = c->od; p[1] = c->base;
    id<MTLBuffer> bufs[3] = { src, dst, g_args };
    NSUInteger    offs[3] = { soff, doff, aoff };
    dispatch("op_softmax", bufs, offs, 3, c->r);
    return true;
}

/* OP_ROPE_NEOX */
static bool metal_op_rope_neox(OpCtx *c) {
    u32 bits = op_param(c, 0);
    float theta;
    memcpy(&theta, &bits, sizeof(theta));
    u32 heads = op_param(c, 1);
    u32 hdim  = op_param(c, 2);
    u32 rdim  = op_param(c, 3);
    if (c->od != heads * hdim || hdim == 0) return false;
    id<MTLBuffer> src, dst;
    NSUInteger soff, doff;
    if (!bind_ptr(op_src(c, 0), &src, &soff) || !bind_ptr(c->dst, &dst, &doff))
        return false;
    NSUInteger aoff = args_alloc(9 * sizeof(uint32_t));
    if (aoff == (NSUInteger)-1) return false;
    uint32_t *p = args_ptr(aoff);
    memcpy(&p[0], &theta, sizeof(theta));
    p[1] = heads; p[2] = hdim; p[3] = rdim; p[4] = c->od; p[5] = c->base; p[6] = c->pos;
    u64 total = (u64)c->r * heads * (hdim / 2);
    put_u64(p, 7, total);
    id<MTLBuffer> bufs[3] = { src, dst, g_args };
    NSUInteger    offs[3] = { soff, doff, aoff };
    dispatch("op_rope_neox", bufs, offs, 3, op_blocks(total));
    return true;
}

/* OP_SIGMOID_GATE */
static bool metal_op_sigmoid_gate(OpCtx *c) {
    u32 nh = op_param(c, 0), hd = op_param(c, 1);
    if (c->od != nh * hd) return false;
    id<MTLBuffer> a, g, dst;
    NSUInteger ao, go, doff;
    if (!bind_ptr(op_src(c, 0), &a, &ao) || !bind_ptr(op_src(c, 1), &g, &go) ||
        !bind_ptr(c->dst, &dst, &doff))
        return false;
    NSUInteger aoff = args_alloc(6 * sizeof(uint32_t));
    if (aoff == (NSUInteger)-1) return false;
    uint32_t *p = args_ptr(aoff);
    p[0] = hd; p[1] = c->od; p[2] = (u32)((u64)nh * 2 * hd); p[3] = c->base;
    put_u64(p, 4, (u64)c->r * c->od);
    id<MTLBuffer> bufs[4] = { a, g, dst, g_args };
    NSUInteger    offs[4] = { ao, go, doff, aoff };
    dispatch("op_sigmoid_gate", bufs, offs, 4, op_blocks((u64)c->r * c->od));
    return true;
}

/* OP_SSM_CONV */
static bool metal_op_ssm_conv(OpCtx *c) {
    u32 st = op_param(c, 0), ck = op_param(c, 1);
    if (st >= c->g->n_state || ck == 0) return false;
    const void *host = c->g->state[st];
    if (!host) return false;
    size_t sn = (size_t)(ck > 1 ? ck - 1 : 1) * c->od * sizeof(float);
    id<MTLBuffer> state = state_get(host, sn, c->pos == 0);
    id<MTLBuffer> cw = weight_f32(c->node->weights[0]);
    if (!state || !cw) return false;
    id<MTLBuffer> src, dst;
    NSUInteger soff, doff;
    if (!bind_ptr(op_src(c, 0), &src, &soff) || !bind_ptr(c->dst, &dst, &doff))
        return false;
    NSUInteger aoff = args_alloc(4 * sizeof(uint32_t));
    if (aoff == (NSUInteger)-1) return false;
    uint32_t *p = args_ptr(aoff);
    p[0] = c->od; p[1] = ck; p[2] = c->r; p[3] = c->base;
    id<MTLBuffer> bufs[5] = { src, dst, cw, state, g_args };
    NSUInteger    offs[5] = { soff, doff, 0, 0, aoff };
    dispatch("op_ssm_conv", bufs, offs, 5, 1);
    return true;
}

/* OP_SSM_DELTA */
static bool metal_op_ssm_delta(OpCtx *c) {
    u32 st = op_param(c, 0), n_v = op_param(c, 1);
    u32 n_k = op_param(c, 2), hd = op_param(c, 3);
    if (st >= c->g->n_state || hd == 0 || hd > OP_THREADS) return false;
    u32 key_dim = n_k * hd;
    u64 val_dim = (u64)n_v * hd;
    if (c->od != (u32)val_dim) return false;
    const void *host = c->g->state[st];
    if (!host) return false;
    id<MTLBuffer> state = state_get(host, (size_t)n_v * hd * hd * sizeof(float), c->pos == 0);
    id<MTLBuffer> a_log = weight_f32(c->node->weights[0]);
    id<MTLBuffer> dt_bi = weight_f32(c->node->weights[1]);
    id<MTLBuffer> nw    = c->node->weights[2] ? weight_f32(c->node->weights[2]) : nil;
    if (!state || !a_log || !dt_bi || (c->node->weights[2] && !nw)) return false;

    id<MTLBuffer> fused, alpha, beta, dst;
    NSUInteger fo, ao, bo, doff;
    if (!bind_ptr(op_src(c, 0), &fused, &fo) ||
        !bind_ptr(op_src(c, 1), &alpha, &ao) ||
        !bind_ptr(op_src(c, 2), &beta,  &bo) ||
        !bind_ptr(c->dst, &dst, &doff))
        return false;
    NSUInteger aoff = args_alloc(9 * sizeof(uint32_t));
    if (aoff == (NSUInteger)-1) return false;
    uint32_t *p = args_ptr(aoff);
    p[0] = n_v; p[1] = n_k; p[2] = hd; p[3] = key_dim; p[4] = c->od;
    p[5] = c->base; p[6] = c->r; p[7] = nw ? 1 : 0;
    memcpy(&p[8], &(float){DEFAULT_EPS}, sizeof(float));

    id<MTLBuffer> bufs[9] = { fused, alpha, beta, a_log, dt_bi,
                              nw ? nw : metal_i_tables(), state, dst, g_args };
    NSUInteger    offs[9] = { fo, ao, bo, 0, 0, 0, 0, doff, aoff };
    dispatch("op_ssm_delta", bufs, offs, 9, n_v);
    return true;
}

/* OP_ATTN */
static bool metal_op_attn(OpCtx *c) {
    if (!c->s->cache.std) {
        slog(WARN, "metal_op_attn: OP_ATTN requires the std KV cache");
        return false;
    }
    u32 layer = op_param(c, 0);
    AttnKvCache *akc = &c->s->cache.std[layer];
    if (!akc->k || !akc->v) return false;

    u32 n_head = c->cfg->n_head;
    u32 n_kv   = c->cfg->n_kv_head;
    u32 khd    = c->cfg->kv_head_dim;
    u32 gqa    = n_head / n_kv;
    u64 hs     = (u64)akc->cap * khd;

    OpKv *e = kv_get(akc->k, (u64)akc->cap * n_kv * khd);
    if (!e || !e->k || !e->v) return false;

    id<MTLBuffer> kd, vd, dst;
    NSUInteger ko, vo, doff;
    if (!bind_ptr(op_src(c, 1), &kd, &ko) ||
        !bind_ptr(op_src(c, 2), &vd, &vo) ||
        !bind_ptr(c->dst, &dst, &doff))
        return false;

    /* Phase 1: write-through into the device mirror. */
    u64 total = (u64)c->n * n_kv * khd;
    NSUInteger aoff = args_alloc(8 * sizeof(uint32_t));
    if (aoff == (NSUInteger)-1) return false;
    uint32_t *p = args_ptr(aoff);
    p[0] = n_kv; p[1] = khd; p[2] = c->kv_dim;
    put_u64(p, 3, hs);
    p[5] = c->pos;
    put_u64(p, 6, total);
    id<MTLBuffer> bufs1[5] = { e->k, e->v, kd, vd, g_args };
    NSUInteger    offs1[5] = { 0, 0, ko, vo, aoff };
    dispatch("op_kv_write", bufs1, offs1, 5, op_blocks(total));
    c->s->cache.std[layer].n = c->pos + c->n;

    /* Phase 2: causal attention over the mirror. */
    NSUInteger aoff2 = args_alloc(9 * sizeof(uint32_t));
    if (aoff2 == (NSUInteger)-1) return false;
    uint32_t *q = args_ptr(aoff2);
    q[0] = n_head; q[1] = gqa; q[2] = c->cfg->head_dim; q[3] = khd; q[4] = c->q_dim;
    put_u64(q, 5, hs);
    q[7] = c->pos;
    memcpy(&q[8], &c->scale, sizeof(float));
    id<MTLBuffer> qd;
    NSUInteger qoff;
    if (!bind_ptr(op_src(c, 0), &qd, &qoff)) return false;
    id<MTLBuffer> bufs2[5] = { qd, e->k, e->v, dst, g_args };
    NSUInteger    offs2[5] = { qoff, 0, 0, doff, aoff2 };
    dispatch("op_attn", bufs2, offs2, 5, (NSUInteger)c->r * n_head);
    return true;
}

/* ================================================================
 * Dispatch — the Metal twin of graph.c's op_table
 * ================================================================ */

bool metal_graph_op(OpCtx *c) {
    if (!c || !c->g || !c->s || !c->cfg || !c->node || !c->dst || c->od == 0)
        return false;
    if (metal_i_ready() != 0) return false;
    if (!g_args_retired) g_args_retired = [NSMutableArray array];
    switch (c->node->op) {
        case OP_EMBED:          return metal_op_embed(c);
        case OP_RMS_NORM:       return metal_op_rms_norm(c);
        case OP_RMS_NORM_HEADS: return metal_op_rms_norm_heads(c);
        case OP_MATMUL:
        case OP_MATMULARRY:
        case OP_MATMUL_T:       return metal_op_matmul(c);
        case OP_BIAS:           return metal_op_bias(c);
        case OP_ADD:
        case OP_MUL:            return metal_op_binary(c);
        case OP_SILU:           return metal_op_silu(c);
        case OP_SOFTMAX:        return metal_op_softmax(c);
        case OP_ROPE_NEOX:      return metal_op_rope_neox(c);
        case OP_SIGMOID_GATE:   return metal_op_sigmoid_gate(c);
        case OP_SSM_CONV:       return metal_op_ssm_conv(c);
        case OP_SSM_DELTA:      return metal_op_ssm_delta(c);
        case OP_ATTN:           return metal_op_attn(c);
        default:                return false;
    }
}

void metal_op_shutdown(void) {
    metal_graph_flush();
    for (int i = 0; i < g_wc_n; i++) g_wc_buf[i] = nil;
    g_wc_n = 0;
    for (int i = 0; i < g_kv_n; i++) { g_kv[i].k = nil; g_kv[i].v = nil; }
    g_kv_n = 0;
    for (int i = 0; i < g_state_n; i++) g_state[i].buf = nil;
    g_state_n = 0;
    g_args = nil;
    g_args_cap = g_args_used = 0;
    if (g_args_retired) [g_args_retired removeAllObjects];
}
