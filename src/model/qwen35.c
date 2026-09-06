#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "model.h"
#include "../core.h"
#include "../mm.h"
#include "../slog.h"
#include "../utils.h"
#include "../graph.h"

typedef struct {
    float *x;               /* [n_embd] hidden state                        */
    float *xb;              /* [n_embd] residual / scratch                  */
    float *xb2;             /* [n_embd] second scratch                      */
    float *q;               /* [n_head * head_dim] query buffer             */
    float *k;               /* [n_kv_head * head_dim] key buffer            */
    float *v;               /* [n_kv_head * head_dim] value buffer          */
    float *qkv_fused;       /* [max_fused] fused QKV buffer                 */
    float *scores;          /* [ctx_size] attention scores (per head)       */
    float *hb;              /* [ffn_hidden] FFN hidden buffer               */
    float *hb2;             /* [ffn_hidden] FFN hidden buffer 2             */
    u32    ffn_hidden;
    float  rope_theta;      /* RoPE frequency base                          */

    /* SSM (Gated DeltaNet) per-layer state. */
    float **ssm_state;      /* [n_layer][n_v * head_v * head_v]             */
    float **conv_state;     /* [n_layer][(conv_kernel-1) * fused]           */

    /* SSM geometry, from the GGUF metadata (same decomposition as the
     * reference loader in llama.cpp src/models/qwen35.cpp):
     *   head_k = ssm.state_size          n_k = ssm.group_count
     *   n_v    = ssm.time_step_rank      head_v = ssm.inner_size / n_v
     *   key_dim = head_k * n_k     val_dim = head_v * n_v
     *   fused   = 2 * key_dim + val_dim   (== conv1d channels)
     * n_k and n_v differ on wider models (16 vs 32 on Qwen3.5-9B), where
     * Q/K heads are reused across value heads (GVA). */
    u32    ssm_head_k;      /* q/k head dim      (ssm.state_size)           */
    u32    ssm_head_v;      /* value head dim    (inner_size / n_v)         */
    u32    ssm_n_k;         /* q/k head count    (ssm.group_count)          */
    u32    ssm_n_v;         /* value head count  (ssm.time_step_rank)       */
    u32    ssm_key_dim;     /* head_k * n_k                                 */
    u32    ssm_val_dim;     /* head_v * n_v      (== ssm.inner_size)        */
    u32    ssm_fused;       /* 2*key_dim + val_dim (fused QKV / conv width) */
    u32    conv_kernel;     /* conv1d kernel size (ssm.conv_kernel)         */
} Qwen35Workspace;

/* ---- Arch ops --------------------------------------------------- */

/* Decide whether a 2-D weight has to be read transposed for y = W @ x
 * producing `out` rows from `in` columns.  Qwen3.5 GGUFs store weights as
 * [out, in], but the usual `dim[0] == in` shortcut is ambiguous whenever
 * out == in — Qwen3.5-9B's attn_gate and ssm_out are both square
 * [4096, 4096] — and then wrongly requests a transpose, which silently
 * corrupts the SSM gate and output projections.  Match [out, in] first. */
static bool ssm_mm_trans(TensorInfo *t, u64 out, u64 in) {
    if (!t || t->ndim < 2) return false;
    u64 d0 = (u64)t->dim[0], d1 = (u64)t->dim[1];
    if (d0 == out && d1 == in) return false;   /* stored [out, in] */
    if (d0 == in  && d1 == out) return true;   /* stored [in, out] */
    return d0 == in;                            /* unknown: old heuristic */
}

static bool qwen35_init(Session *s) {
    ArchConfig *c = &s->cfg;

    u32 q_head_dim  = c->head_dim;
    u32 kv_head_dim = c->kv_head_dim;
    u32 q_dim       = c->n_head * q_head_dim;
    u32 kv_dim      = c->n_kv_head * kv_head_dim;
    u32 max_fused   = q_dim + 2 * kv_dim;

    /* Allocate standard KV cache. */
    KvCache *kc   = &s->cache;
    kc->n_layer   = c->n_layer;
    kc->head_dim  = kv_head_dim;
    kc->n_kv_head = c->n_kv_head;
    kc->std       = scalloc((u64)c->n_layer, sizeof(AttnKvCache));

    u64 per_layer = (u64)s->ctx_size * (u64)c->n_kv_head * (u64)kv_head_dim;
    for (u32 i = 0; i < c->n_layer; i++) {
        kc->std[i].k   = scalloc(per_layer, sizeof(float));
        kc->std[i].v   = scalloc(per_layer, sizeof(float));
        kc->std[i].cap = (u32)s->ctx_size;
    }

    s->tokens = scalloc((u64)s->ctx_size, sizeof(u32));
    s->logits = scalloc((u64)c->n_vocab, sizeof(float));

    /* Scan all layers for max fused QKV width (covers SSM layers
     * whose fused output can be much wider than standard attention,
     * as well as Q-norm layers that use qkv_fused as temp storage). */
    Weights *w = s->en->weights;
    for (u32 i = 0; i < c->n_layer; i++) {
        TensorInfo *t_qkv = w->layers[i].tensors[TENSOR_ATTN_QKV];
        if (t_qkv && t_qkv->ndim >= 2) {
            u64 total = t_qkv->dim[0] > t_qkv->dim[1]
                      ? (u32)t_qkv->dim[0] : (u32)t_qkv->dim[1];
            if (total > max_fused) max_fused = (u32)total;
        }
        TensorInfo *t_q = w->layers[i].tensors[TENSOR_ATTN_Q];
        if (t_q && t_q->ndim >= 2) {
            u32 q_sz = t_q->dim[0] > t_q->dim[1]
                     ? (u32)t_q->dim[0] : (u32)t_q->dim[1];
            if (q_sz > max_fused) max_fused = q_sz;
        }
    }

    /* Allocate workspace buffers. */
    Qwen35Workspace *ws = scalloc(1, sizeof(Qwen35Workspace));
    ws->x         = scalloc((u64)c->n_embd, sizeof(float));
    ws->xb        = scalloc((u64)c->n_embd, sizeof(float));
    ws->xb2       = scalloc((u64)c->n_embd, sizeof(float));
    ws->q         = scalloc((u64)q_dim, sizeof(float));
    ws->k         = scalloc((u64)kv_dim, sizeof(float));
    ws->v         = scalloc((u64)kv_dim, sizeof(float));
    ws->qkv_fused = scalloc((u64)max_fused, sizeof(float));
    ws->scores    = scalloc((u64)s->ctx_size, sizeof(float));

    /* Derive FFN hidden dim from layer 0 ffn_gate tensor. */
    TensorInfo *ffn_gate = w->layers[0].tensors[TENSOR_FFN_GATE];
    ws->ffn_hidden = (ffn_gate && ffn_gate->ndim >= 1)
                     ? (ffn_gate->dim[0] > ffn_gate->dim[1]
                        ? (u32)ffn_gate->dim[0]
                        : (u32)(ffn_gate->ndim >= 2 ? ffn_gate->dim[1] : ffn_gate->dim[0]))
                     : c->n_embd * 4;
    ws->hb  = scalloc((u64)ws->ffn_hidden, sizeof(float));
    ws->hb2 = scalloc((u64)ws->ffn_hidden, sizeof(float));

    /* Read rope theta from metadata (default 1e6 for Qwen2).
     * Try arch_name prefix first, then generic "qwen3" prefix. */
    ws->rope_theta = 1000000.0f;
    {
        float tmp = 0.0f;
        const char *pfx = s->en->model->arch_name[0]
                          ? s->en->model->arch_name : "qwen3";
        char k[96];
        if (snprintf(k, sizeof(k), "%s.rope.freq_base", pfx) > 0
            && model_get_f32(s->en->model, k, &tmp))
            ws->rope_theta = tmp;
        else if (strcmp(pfx, "qwen3") != 0
                 && snprintf(k, sizeof(k), "qwen3.rope.freq_base") > 0
                 && model_get_f32(s->en->model, k, &tmp))
            ws->rope_theta = tmp;
    }

    /* ---- SSM (Gated DeltaNet) geometry ----
     * Metadata first; tensor shapes fill in whatever is missing.  Note
     * n_v (value heads) is ssm.time_step_rank and is NOT ssm.group_count:
     * they differ on wider models (16 vs 32 on Qwen3.5-9B), which is what
     * makes Q/K head reuse (GVA) necessary. */
    ws->ssm_head_k   = 128;
    ws->ssm_n_k      = 16;
    ws->conv_kernel  = 4;
    i32 n_v_i = 0, inner_i = 0;
    {
        const char *pfx = s->en->model->arch_name[0]
                          ? s->en->model->arch_name : "qwen3";
        i32 v32 = 0;
        #define TRY_SSM_I32(suffix, field) do {                         \
            char _k[96];                                                \
            int _n = snprintf(_k, sizeof(_k), "%s.ssm.%s", pfx, suffix);\
            if (_n > 0 && (size_t)_n < sizeof(_k)                       \
                && model_get_i32(s->en->model, _k, &v32))               \
                field = (i32)v32;                                       \
            else if (strcmp(pfx, "qwen3") != 0                          \
                     && snprintf(_k, sizeof(_k), "qwen3.ssm.%s",        \
                                 suffix) > 0                            \
                     && model_get_i32(s->en->model, _k, &v32))          \
                field = (i32)v32;                                       \
        } while(0)
        TRY_SSM_I32("state_size",      ws->ssm_head_k);
        TRY_SSM_I32("group_count",     ws->ssm_n_k);
        TRY_SSM_I32("conv_kernel",     ws->conv_kernel);
        TRY_SSM_I32("time_step_rank",  n_v_i);
        TRY_SSM_I32("inner_size",      inner_i);
        #undef TRY_SSM_I32
    }

    /* Fall back to tensor shapes when metadata is missing: alpha/beta are
     * [n_v, n_embd], ssm_out is [n_embd, val_dim] and conv1d is
     * [fused, conv_kernel]. */
    {
        u32 n_v = (u32)n_v_i, val_dim = (u32)inner_i;
        for (u32 i = 0; i < c->n_layer; i++) {
            LayerWeights *lwp = &w->layers[i];
            if (!lwp->tensors[TENSOR_SSM_CONV1D]) continue;
            TensorInfo *ta = lwp->tensors[TENSOR_SSM_ALPHA];
            TensorInfo *to = lwp->tensors[TENSOR_SSM_OUT];
            if (!n_v && ta && ta->ndim >= 2) {
                u64 d0 = (u64)ta->dim[0], d1 = (u64)ta->dim[1];
                n_v = (u32)(d1 == c->n_embd ? d0 : d1);
            }
            if (!val_dim && to && to->ndim >= 2) {
                u64 d0 = (u64)to->dim[0], d1 = (u64)to->dim[1];
                val_dim = (u32)(d0 == c->n_embd ? d1 : d0);
            }
            break;
        }
        if (!n_v) n_v = ws->ssm_n_k;
        if (!val_dim) val_dim = n_v * ws->ssm_head_k;
        if (!ws->ssm_head_k) ws->ssm_head_k = 128;
        if (!ws->ssm_n_k) ws->ssm_n_k = 16;

        ws->ssm_n_v      = n_v;
        ws->ssm_head_v   = n_v ? val_dim / n_v : ws->ssm_head_k;
        if (!ws->ssm_head_v) ws->ssm_head_v = ws->ssm_head_k;
        ws->ssm_key_dim  = ws->ssm_head_k * ws->ssm_n_k;
        ws->ssm_val_dim  = ws->ssm_head_v * ws->ssm_n_v;
        ws->ssm_fused    = 2 * ws->ssm_key_dim + ws->ssm_val_dim;
    }

    /* The fused QKV scratch must hold the widest SSM projection too. */
    if (ws->ssm_fused > max_fused) {
        sfree(ws->qkv_fused);
        ws->qkv_fused = scalloc((u64)ws->ssm_fused, sizeof(float));
        max_fused     = ws->ssm_fused;
    }

    /* hb/hb2 double as the delta-net output / gate scratch, whose width is
     * val_dim.  Widen them when needed, but leave ffn_hidden alone: it is
     * the FFN's output dimension, not a buffer size. */
    if (ws->ssm_val_dim > ws->ffn_hidden) {
        sfree(ws->hb);
        sfree(ws->hb2);
        ws->hb  = scalloc((u64)ws->ssm_val_dim, sizeof(float));
        ws->hb2 = scalloc((u64)ws->ssm_val_dim, sizeof(float));
    }

    /* The fused width must match the conv1d channel count, otherwise the
     * conv weight dequant below reads the wrong span. */
    for (u32 i = 0; i < c->n_layer; i++) {
        TensorInfo *tc = w->layers[i].tensors[TENSOR_SSM_CONV1D];
        if (!tc) continue;
        u64 ch = (u64)(tc->dim[1] == (i64)ws->conv_kernel ? tc->dim[0] : tc->dim[1]);
        if (ch != ws->ssm_fused)
            slog(WARN, "Qwen3 SSM: conv1d channels=%llu but computed fused=%u",
                 (unsigned long long)ch, ws->ssm_fused);
        break;
    }

    /* Allocate per-layer SSM state for Gated DeltaNet layers.
     * Only layers with ssm_conv1d tensors use this state. */
    ws->ssm_state  = scalloc((u64)c->n_layer, sizeof(float *));
    ws->conv_state = scalloc((u64)c->n_layer, sizeof(float *));
    {
        u64 state_sz  = (u64)ws->ssm_n_v * ws->ssm_head_v * ws->ssm_head_v;
        u64 conv_sz   = (u64)(ws->conv_kernel - 1) * ws->ssm_fused;
        u32 n_ssm     = 0;
        for (u32 i = 0; i < c->n_layer; i++) {
            if (w->layers[i].tensors[TENSOR_SSM_CONV1D]) {
                ws->ssm_state[i]  = scalloc(state_sz, sizeof(float));
                ws->conv_state[i] = scalloc(conv_sz, sizeof(float));
                n_ssm++;
            }
        }
        slog(INFO, "Qwen3 SSM: n_k=%u n_v=%u head_k=%u head_v=%u key_dim=%u val_dim=%u "
             "fused=%u conv_k=%u layers=%u (state_per_layer=%.1f MB)",
             ws->ssm_n_k, ws->ssm_n_v, ws->ssm_head_k, ws->ssm_head_v,
             ws->ssm_key_dim, ws->ssm_val_dim, ws->ssm_fused, ws->conv_kernel, n_ssm,
             (double)(state_sz + conv_sz) * sizeof(float) / 1048576.0);
    }

    slog(INFO, "Qwen3 init: n_embd=%u n_head=%u n_kv_head=%u head_dim=%u kv_head_dim=%u "
         "n_layer=%u n_vocab=%u ctx_size=%u",
         c->n_embd, c->n_head, c->n_kv_head, q_head_dim, kv_head_dim,
         c->n_layer, c->n_vocab, s->ctx_size);
    slog(INFO, "Qwen3 init: rope_theta=%.6f", ws->rope_theta);

    s->arch_data = ws;
    return true;
}

static void qwen35_reset(Session *s) {
    KvCache *kc = &s->cache;
    for (u32 i = 0; i < kc->n_layer; i++)
        kc->std[i].n = 0;
    s->n_tokens = 0;

    /* Zero SSM recurrent state and conv state. */
    Qwen35Workspace *ws = (Qwen35Workspace *)s->arch_data;
    if (ws && ws->ssm_state) {
        u64 state_sz = (u64)ws->ssm_n_v * ws->ssm_head_v * ws->ssm_head_v;
        u64 conv_sz  = (u64)(ws->conv_kernel - 1) * ws->ssm_fused;
        for (u32 i = 0; i < s->cfg.n_layer; i++) {
            if (ws->ssm_state[i])
                memset(ws->ssm_state[i], 0, state_sz * sizeof(float));
            if (ws->conv_state[i])
                memset(ws->conv_state[i], 0, conv_sz * sizeof(float));
        }
    }
}

static void qwen35_free(Session *s) {
    KvCache *kc = &s->cache;
    if (kc->std) {
        for (u32 i = 0; i < kc->n_layer; i++) {
            sfree(kc->std[i].k);
            sfree(kc->std[i].v);
        }
        sfree(kc->std);
        kc->std = NULL;
    }
    sfree(s->tokens);
    sfree(s->logits);
    s->tokens = NULL;
    s->logits = NULL;

    Qwen35Workspace *ws = (Qwen35Workspace *)s->arch_data;
    if (ws) {
        sfree(ws->x); sfree(ws->xb); sfree(ws->xb2);
        sfree(ws->q); sfree(ws->k); sfree(ws->v);
        sfree(ws->qkv_fused); sfree(ws->scores);
        sfree(ws->hb); sfree(ws->hb2);
        if (ws->ssm_state) {
            for (u32 i = 0; i < s->cfg.n_layer; i++) {
                sfree(ws->ssm_state[i]);
                sfree(ws->conv_state[i]);
            }
            sfree(ws->ssm_state);
            sfree(ws->conv_state);
        }
        sfree(ws);
        s->arch_data = NULL;
    }
}

static int qwen35_decode(const u8 *raw, int raw_len, char *out, int max_len) {
    int w = 0;
    for (int b = 0; b < raw_len && w < max_len; b++) {
        unsigned char c = raw[b];
        if (c >= 0x21 && c <= 0x7E) {
            out[w++] = (char)c;
        } else if (c >= 0xC2 && c <= 0xC5 && b + 1 < raw_len) {
            unsigned char nc = raw[b + 1];
            int byte = -1;
            if (nc >= 0x80 && nc <= 0xBF) {
                if      (c == 0xC2) byte = (int)nc;
                else if (c == 0xC3) byte = (int)nc + 0x40;
                else if (c == 0xC4) {
                    if      (nc <= 0xA0) byte = nc - 0x80;
                    else if (nc == 0xA1) byte = 0x7F;
                    else                 byte = nc - 0x22;
                } else if (c == 0xC5) {
                    if      (nc == 0x80) byte = 0x9E;
                    else if (nc == 0x81) byte = 0x9F;
                    else if (nc == 0x82) byte = 0xA0;
                    else if (nc == 0x83) byte = 0xAD;
                }
            }
            if (byte >= 0 && byte <= 255) {
                out[w++] = (char)(unsigned char)byte;
                b++;
            } else {
                out[w++] = (char)c;
            }
        } else {
            out[w++] = (char)c;
        }
    }
    if (w < max_len) out[w] = '\0';
    return w;
}

/* Build a static execution graph for a batch of n_tokens tokens,
 * covering both Qwen3.5 layer kinds:
 *
 *   - SSM (Gated DeltaNet) layers: fused QKV projection, depthwise
 *     causal Conv1d, SiLU, the gated delta recurrence, the per-group
 *     RMS norm, the SiLU output gate and the ssm_out projection;
 *
 *   - dense attention layers: separate Q/K/V projections (+bias,
 *     optional per-head Q/K-norm, partial RoPE), causal GQA attention,
 *     an optional SiLU output gate and/or the sigmoid gate folded into
 *     a fused [q, gate] Q projection, then the output projection.
 *
 * Both branches are followed by the residual add, the post-attention
 * norm, the SwiGLU FFN and its residual.  OP_ATTN writes the batch's
 * K/V through to the session KV cache and attends causally, so the
 * graph is reusable for any n <= n_tokens.
 *
 * The two SSM state buffers per layer (the conv ring and the recurrent
 * [groups * d_state * d_state] matrix) are borrowed from the workspace
 * via graph_state(): the graph only holds the pointer, so qwen35_reset
 * keeps zeroing them and qwen35_free keeps owning them.
 *
 * The one configuration still rejected (build fails) is a gated
 * attention output weight (2*n_embd rows with q_dim columns): the
 * SiLU-split gate it implies has no graph op. */
static Graph *qwen35_graph_build(Session *s, u32 n_tokens) {
    Qwen35Workspace *ws   = (Qwen35Workspace *)s->arch_data;
    ArchConfig      *c    = &s->cfg;
    Weights         *w    = s->en->weights;

    if (!s || !w || !ws || n_tokens == 0 || n_tokens > s->ctx_size) return NULL;

    /* Dims for the matmul direction flags.  Each trans below must mirror
     * the same weight, otherwise the executor dequantises the wrong axis. */
    u32 n_embd  = c->n_embd;
    u32 q_dim   = c->n_head * c->head_dim;
    u32 fh      = ws->ffn_hidden;
    float theta = ws->rope_theta;
    u32 n_k     = ws->ssm_n_k;
    u32 n_v     = ws->ssm_n_v;
    u32 hd_v    = ws->ssm_head_v;
    u32 val_dim = ws->ssm_val_dim;

    /* rope_partial needs a sane rotation width (it degenerates to
     * rope_neox when rope_dim == head_dim). */
    if (c->rope_dim == 0 || c->rope_dim > c->head_dim) {
        slog(WARN, "graph_build: invalid rope_dim=%u for head_dim=%u",
             c->rope_dim, c->head_dim);
        return NULL;
    }

    TensorInfo *te = w->tensors[TENSOR_TOKEN_EMBD];
    if (!te || te->ndim < 2) {
        slog(WARN, "graph_build: missing token embedding");
        return NULL;
    }

    Graph *g = graph_new();
    if (!g) return NULL;

    /* Input tokens → OP_INPUT leaf (shape only; values bound at compute). */
    u32 in = graph_input(g, n_tokens);
    if (in == GRAPH_NODE_NONE) goto fail;

    u32 cur = graph_embed(g, in, te, n_tokens);
    if (cur == GRAPH_NODE_NONE) goto fail;

    for (u32 l = 0; l < c->n_layer; l++) {
        LayerWeights *lw = &w->layers[l];

        TensorInfo *t_qkv = lw->tensors[TENSOR_ATTN_QKV];
        TensorInfo *t_conv = lw->tensors[TENSOR_SSM_CONV1D];
        TensorInfo *t_q   = lw->tensors[TENSOR_ATTN_Q];
        TensorInfo *t_k   = lw->tensors[TENSOR_ATTN_K];
        TensorInfo *t_v   = lw->tensors[TENSOR_ATTN_V];
        TensorInfo *t_o   = lw->tensors[TENSOR_ATTN_OUT];
        TensorInfo *t_ag  = lw->tensors[TENSOR_ATTN_GATE];
        TensorInfo *t_g   = lw->tensors[TENSOR_FFN_GATE];
        TensorInfo *t_u   = lw->tensors[TENSOR_FFN_UP];
        TensorInfo *t_d   = lw->tensors[TENSOR_FFN_DOWN];
        if (!t_g || !t_u || !t_d) {
            slog(WARN, "graph_build: layer %u missing FFN tensors", l);
            goto fail;
        }

        /* ---- Attention norm ---- */
        u32 n = graph_rms_norm(g, cur, lw->tensors[TENSOR_ATTN_NORM]);
        if (n == GRAPH_NODE_NONE) goto fail;

        if (t_conv) {
            /* ======== SSM (Gated DeltaNet) layer ======== */
            TensorInfo *t_a  = lw->tensors[TENSOR_SSM_ALPHA];
            TensorInfo *t_b  = lw->tensors[TENSOR_SSM_BETA];
            TensorInfo *t_A  = lw->tensors[TENSOR_SSM_A];
            TensorInfo *t_dt = lw->tensors[TENSOR_SSM_DT_BIAS];
            TensorInfo *t_sn = lw->tensors[TENSOR_SSM_NORM];
            TensorInfo *t_so = lw->tensors[TENSOR_SSM_OUT];
            if (!t_qkv || !t_a || !t_b || !t_A || !t_dt || !t_so) {
                slog(WARN, "graph_build: layer %u is an SSM layer with missing tensors", l);
                goto fail;
            }
            if (!ws->conv_state[l] || !ws->ssm_state[l]) {
                slog(WARN, "graph_build: layer %u has no SSM state", l);
                goto fail;
            }

            /* Borrow the per-layer conv ring + recurrent state. */
            u32 conv_st = graph_state(g, ws->conv_state[l]);
            u32 ssm_st  = graph_state(g, ws->ssm_state[l]);
            if (conv_st == GRAPH_NODE_NONE || ssm_st == GRAPH_NODE_NONE) goto fail;

            /* Fused QKV → conv → SiLU (reference: conv → silu → split). */
            u32 f = graph_mul_mat(g, n, t_qkv, t_qkv->dim[0] == (i64)n_embd);
            if (f == GRAPH_NODE_NONE) goto fail;
            f = graph_ssm_conv(g, f, t_conv, conv_st, ws->conv_kernel);
            if (f == GRAPH_NODE_NONE) goto fail;
            f = graph_silu(g, f);
            if (f == GRAPH_NODE_NONE) goto fail;

            /* Decay / update gates are projected from the same normed input. */
            u32 a = graph_mul_mat(g, n, t_a, ssm_mm_trans(t_a, n_v, n_embd));
            u32 b = graph_mul_mat(g, n, t_b, ssm_mm_trans(t_b, n_v, n_embd));
            if (a == GRAPH_NODE_NONE || b == GRAPH_NODE_NONE) goto fail;

            u32 d = graph_ssm_delta(g, f, a, b, t_A, t_dt, t_sn, ssm_st,
                                    n_v, n_k, hd_v);
            if (d == GRAPH_NODE_NONE) goto fail;

            /* Output gate: silu(gate @ n) * delta_out. */
            if (t_ag) {
                u32 ag = graph_silu(g, graph_mul_mat(g, n, t_ag,
                                     ssm_mm_trans(t_ag, val_dim, n_embd)));
                if (ag == GRAPH_NODE_NONE) goto fail;
                d = graph_binary(g, OP_MUL, d, ag);
                if (d == GRAPH_NODE_NONE) goto fail;
            }

            u32 y = graph_mul_mat(g, d, t_so, ssm_mm_trans(t_so, n_embd, val_dim));
            if (y == GRAPH_NODE_NONE) goto fail;
            cur = graph_binary(g, OP_ADD, cur, y);
            if (cur == GRAPH_NODE_NONE) goto fail;

        } else if (t_q && t_k && t_v) {
            /* ======== Dense attention layer ======== */
            TensorInfo *t_qn = lw->tensors[TENSOR_ATTN_Q_NORM];
            TensorInfo *t_kn = lw->tensors[TENSOR_ATTN_K_NORM];
            TensorInfo *tb_q = lw->tensors[TENSOR_ATTN_Q_BIAS];
            TensorInfo *tb_k = lw->tensors[TENSOR_ATTN_K_BIAS];
            TensorInfo *tb_v = lw->tensors[TENSOR_ATTN_V_BIAS];
            if (!t_o) {
                slog(WARN, "graph_build: layer %u missing attn output tensor", l);
                goto fail;
            }
            if (t_o->ndim >= 2 && t_o->dim[0] == 2 * (i64)n_embd
                && t_o->dim[1] == (i64)q_dim) {
                slog(WARN, "graph_build: layer %u uses a gated attention output weight, unsupported", l);
                goto fail;
            }

            /* Q projection.  Qwen3.5 dense layers emit n_head * 2 *
             * head_dim columns: the first head_dim of each head is Q,
             * the second half is the raw attention gate (fused QG). */
            u32 q_proj = (u32)(t_q->dim[0] == (i64)n_embd ? t_q->dim[1] : t_q->dim[0]);
            bool q_tr  = (q_proj != n_embd) && t_q->dim[0] == (i64)n_embd;
            u32 fq = graph_mul_mat(g, n, t_q, q_tr);
            if (fq == GRAPH_NODE_NONE) goto fail;
            /* Bias goes on before the norm, as in the eager path. */
            if (tb_q) {
                fq = graph_bias(g, fq, tb_q);
                if (fq == GRAPH_NODE_NONE) goto fail;
            }

            u32 q;
            if (t_qn) {
                if (q_proj != 2 * q_dim) {
                    slog(WARN, "graph_build: layer %u has Q-norm but q_proj=%u (expected %u)",
                         l, q_proj, 2 * q_dim);
                    goto fail;
                }
                /* Norm the Q half, drop the gate half (read back later). */
                q = graph_rms_norm_heads(g, fq, t_qn, c->n_head, c->head_dim,
                                         2 * c->head_dim, c->head_dim);
            } else {
                if (q_proj != q_dim) {
                    slog(WARN, "graph_build: layer %u has q_proj=%u (expected q_dim=%u)",
                         l, q_proj, q_dim);
                    goto fail;
                }
                q = fq;
            }
            if (q == GRAPH_NODE_NONE) goto fail;

            u32 k = graph_mul_mat(g, n, t_k, t_k->dim[0] == (i64)n_embd);
            u32 v = graph_mul_mat(g, n, t_v, t_v->dim[0] == (i64)n_embd);
            if (k == GRAPH_NODE_NONE || v == GRAPH_NODE_NONE) goto fail;
            if (tb_k) {
                k = graph_bias(g, k, tb_k);
                if (k == GRAPH_NODE_NONE) goto fail;
            }
            if (tb_v) {
                v = graph_bias(g, v, tb_v);
                if (v == GRAPH_NODE_NONE) goto fail;
            }
            if (t_kn) {
                k = graph_rms_norm_heads(g, k, t_kn, c->n_kv_head, c->kv_head_dim,
                                         c->kv_head_dim, c->kv_head_dim);
                if (k == GRAPH_NODE_NONE) goto fail;
            }

            q = graph_rope(g, q, theta, c->n_head, c->head_dim, c->rope_dim);
            k = graph_rope(g, k, theta, c->n_kv_head, c->kv_head_dim, c->rope_dim);
            if (q == GRAPH_NODE_NONE || k == GRAPH_NODE_NONE) goto fail;

            u32 attn = graph_attn(g, q, k, v, l);
            if (attn == GRAPH_NODE_NONE) goto fail;

            /* Attention output gate: silu(gate @ n) * attn. */
            if (t_ag) {
                u32 ag = graph_silu(g, graph_mul_mat(g, n, t_ag,
                                     ssm_mm_trans(t_ag, q_dim, n_embd)));
                if (ag == GRAPH_NODE_NONE) goto fail;
                attn = graph_binary(g, OP_MUL, attn, ag);
                if (attn == GRAPH_NODE_NONE) goto fail;
            }
            /* Fused QG: attn *= sigmoid(gate half of the Q projection). */
            if (t_qn) {
                attn = graph_sigmoid_gate(g, attn, fq, c->n_head, c->head_dim);
                if (attn == GRAPH_NODE_NONE) goto fail;
            }

            u32 o = graph_mul_mat(g, attn, t_o, (n_embd != q_dim) && t_o->dim[0] == (i64)q_dim);
            if (o == GRAPH_NODE_NONE) goto fail;
            cur = graph_binary(g, OP_ADD, cur, o);
            if (cur == GRAPH_NODE_NONE) goto fail;

        } else {
            slog(WARN, "graph_build: layer %u missing QKV / Q,K,V tensors", l);
            goto fail;
        }

        /* ---- SwiGLU FFN ---- */
        u32 hn = graph_rms_norm(g, cur, lw->tensors[TENSOR_POST_ATTN_NORM]);
        if (hn == GRAPH_NODE_NONE) goto fail;

        u32 gate = graph_silu(g, graph_mul_mat(g, hn, t_g, t_g->dim[0] == (i64)n_embd));
        u32 up   = graph_mul_mat(g, hn, t_u, t_u->dim[0] == (i64)n_embd);
        if (gate == GRAPH_NODE_NONE || up == GRAPH_NODE_NONE) goto fail;

        u32 mul = graph_binary(g, OP_MUL, gate, up);
        if (mul == GRAPH_NODE_NONE) goto fail;

        u32 down = graph_mul_mat(g, mul, t_d, t_d->dim[0] == (i64)fh);
        if (down == GRAPH_NODE_NONE) goto fail;

        cur = graph_binary(g, OP_ADD, cur, down);
        if (cur == GRAPH_NODE_NONE) goto fail;
    }

    /* ---- Final norm ---- */
    TensorInfo *t_norm = w->tensors[TENSOR_OUTPUT_NORM];
    if (!t_norm) t_norm = w->layers[c->n_layer - 1].tensors[TENSOR_POST_ATTN_NORM];
    u32 fn = graph_rms_norm(g, cur, t_norm);
    if (fn == GRAPH_NODE_NONE) goto fail;

    /* ---- LM head (tied to token embeddings when absent) ---- */
    TensorInfo *t_out = w->tensors[TENSOR_OUTPUT];
    if (!t_out) t_out = te;
    if (graph_mul_mat(g, fn, t_out, t_out->dim[0] == (i64)n_embd) == GRAPH_NODE_NONE) goto fail;

    return g;
fail:
    graph_free(g);
    return NULL;
}

const ArchOps qwen35_ops = {
    .init          = qwen35_init,
    .free          = qwen35_free,
    .reset         = qwen35_reset,
    .decode        = qwen35_decode,
    .graph_build   = qwen35_graph_build,
    .graph_execute = graph_execute,
};
