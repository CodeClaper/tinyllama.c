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
    float **ssm_state;      /* [n_layer][n_groups * d_state * d_state]      */
    float **conv_state;     /* [n_layer][(conv_kernel-1) * fused_dim]       */
    u32    ssm_groups;      /* n_groups (16 for Qwen3.5-0.8B)               */
    u32    ssm_dim;         /* d_state  (128 for Qwen3.5-0.8B)              */
    u32    ssm_fused;       /* fused QKV output width (6144)                */
    u32    conv_kernel;     /* conv1d kernel size (4)                       */
} Qwen35Workspace;

/* ---- Arch ops --------------------------------------------------- */

/* In-place RMS normalization of a single head_dim slice.
 * Equivalent to rms_norm(o, x, weight, epsilon) but operates
 * directly on the input buffer (o == x).  The caller has already
 * dequantised the weight array. */
static void rms_norm_inplace(float *x, const float *weight, u32 n, float eps) {
    float ss = 0.0f;
    for (u32 j = 0; j < n; j++)
        ss += x[j] * x[j];
    ss = 1.0f / sqrtf(ss / (float)n + eps);
    for (u32 j = 0; j < n; j++)
        x[j] = x[j] * ss * weight[j];
}

/* Qwen3.5 K-norm: the K projection output has shape [n_kv_head, head_dim].
 * Apply RMS norm to each head_dim slice in-place. */
static void qwen_k_norm(float *k, u32 n_kv_head, u32 head_dim,
                        const float *norm_weight, float eps) {
    for (u32 h = 0; h < n_kv_head; h++) {
        float *hk = k + (u64)h * head_dim;
        rms_norm_inplace(hk, norm_weight, head_dim, eps);
    }
}

/* L2-normalize each row of a [n_rows, dim] matrix in-place.
 * Used for Q and K in the Gated DeltaNet recurrence. */
static void l2_norm_rows(float *x, u32 n_rows, u32 dim, float eps) {
    for (u32 r = 0; r < n_rows; r++) {
        float *row = x + (u64)r * dim;
        float ss = 0.0f;
        for (u32 i = 0; i < dim; i++)
            ss += row[i] * row[i];
        float inv = 1.0f / sqrtf(ss + eps);
        for (u32 i = 0; i < dim; i++)
            row[i] *= inv;
    }
}

/* Depthwise causal Conv1d step for single-token inference.
 * Processes fused QKV output (channels) through a depthwise conv
 * with kernel_size taps, maintaining a ring buffer of previous inputs.
 * weight – [channels * kernel_size] dequantised weight (GGUF layout,
 *          PyTorch Conv1d convention), weight[c * kernel_size + k] for
 *          channel c, tap k.
 *          Tap k multiplies the input from k taps ago: tap 0 is the
 *          OLDEST input (x[t-(K-1)]), tap K-1 is the current input,
 *          matching ggml_ssm_conv and the reference implementation.
 * state  – [(kernel_size-1) * channels] ring buffer (oldest first).
 * On return, state is shifted and the new input is stored. */
static void causal_conv1d_step(float *output, const float *input,
                               const float *weight, float *state,
                               u32 channels, u32 kernel_size) {
    /* Callers run this in-place (output == input), which clobbers the
     * raw inputs before the state update below.  The state must hold
     * the RAW pre-conv inputs — storing the conv outputs there instead
     * corrupts every subsequent token's window and silently diverges
     * the whole recurrence.  Save a copy only when aliased. */
    float *saved = NULL;
    if (input == output) {
        saved = smalloc((u64)channels * sizeof(float));
        memcpy(saved, input, (u64)channels * sizeof(float));
    }
    const float *raw = saved ? saved : input;

    /* State is stored [oldest, ..., newest] = [x[t-(K-1)], ..., x[t-1]]
     * at positions [0, ..., K-2].  Weight tap k (GGUF convention)
     * multiplies the input at lag (K-1-k): state position k for
     * k < K-1, and the current input for k == K-1. */
    for (u32 c = 0; c < channels; c++) {
        float sum = 0.0f;
        for (u32 k = 0; k < kernel_size; k++) {
            float x = (k == kernel_size - 1) ? raw[c]
                                             : state[k * channels + c];
            sum += x * weight[c * kernel_size + k];
        }
        output[c] = sum;
    }
    /* Shift state: drop oldest (index 0), shift remaining left,
     * store current input at the end. */
    if (kernel_size > 1) {
        u32 shift_len = (kernel_size - 2) * channels;
        if (shift_len > 0)
            memmove(state, state + channels, shift_len * sizeof(float));
        memcpy(state + shift_len, raw, channels * sizeof(float));
    }
    if (saved) sfree(saved);
}

/* Single-step Gated DeltaNet recurrence.
 * Q, K, V     – [n_groups * d_ssm] each
 * state       – [n_groups * d_ssm * d_ssm] recurrent state (updated in-place)
 * g           – [n_groups] decay gate, pre-computed as exp(g_raw)
 * beta        – [n_groups] update gate, pre-computed as sigmoid(beta_raw)
 * kv_mem      – [n_groups * d_ssm] scratch buffer
 * output      – [n_groups * d_ssm] result */
static void gated_delta_step(const float *Q, const float *K, const float *V,
                             float *state, const float *g, const float *beta,
                             float *kv_mem, float *output,
                             u32 n_groups, u32 d_ssm) {
    u64 d2 = (u64)d_ssm * d_ssm;
    for (u32 grp = 0; grp < n_groups; grp++) {
        const float *Qg = Q + (u64)grp * d_ssm;
        const float *Kg = K + (u64)grp * d_ssm;
        const float *Vg = V + (u64)grp * d_ssm;
        float *Sg       = state + (u64)grp * d2;
        float *mem      = kv_mem + (u64)grp * d_ssm;

        float decay = g[grp];
        float upd   = beta[grp];

        /* 1. Decay: S *= decay */
        for (u64 i = 0; i < d2; i++)
            Sg[i] *= decay;

        /* 2. kv_mem = S^T @ K  →  mem[j] = sum_i Sg[i][j] * Kg[i] */
        memset(mem, 0, d_ssm * sizeof(float));
        for (u32 i = 0; i < d_ssm; i++) {
            float ki = Kg[i];
            float *row = Sg + (u64)i * d_ssm; /* Sg[i][:] */
            for (u32 j = 0; j < d_ssm; j++)
                mem[j] += row[j] * ki;
        }

        /* 3. Compute delta vector: delta[j] = (Vg[j] - mem[j]) * beta */
        float delta_j[d_ssm];
        for (u32 j = 0; j < d_ssm; j++)
            delta_j[j] = (Vg[j] - mem[j]) * upd;

        /* 4. S += K @ delta^T (outer product)
         * 5. output = S^T @ Q (readout) */
        float out_j[d_ssm];
        memset(out_j, 0, d_ssm * sizeof(float));
        for (u32 i = 0; i < d_ssm; i++) {
            float qi = Qg[i];
            float ki = Kg[i];
            float *row = Sg + (u64)i * d_ssm;
            for (u32 j = 0; j < d_ssm; j++) {
                row[j] += ki * delta_j[j];         /* 4. S[i][j] += K[i] * delta[j] */
                out_j[j] += row[j] * qi;            /* 5. readout */
            }
        }
        memcpy(output + (u64)grp * d_ssm, out_j, d_ssm * sizeof(float));
    }
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

    /* ---- SSM (Gated DeltaNet) state setup ---- */
    ws->ssm_groups  = 16;
    ws->ssm_dim     = 128;
    ws->ssm_fused   = max_fused;  /* full fused QKV output width (6144) */
    ws->conv_kernel = 4;
    {
        const char *pfx = s->en->model->arch_name[0]
                          ? s->en->model->arch_name : "qwen3";
        i32 v32 = 0;
        #define TRY_SSM_I32(suffix, field) do {                         \
            char _k[96];                                                \
            int _n = snprintf(_k, sizeof(_k), "%s.ssm.%s", pfx, suffix);\
            if (_n > 0 && (size_t)_n < sizeof(_k)                       \
                && model_get_i32(s->en->model, _k, &v32))               \
                field = (u32)v32;                                       \
            else if (strcmp(pfx, "qwen3") != 0                          \
                     && snprintf(_k, sizeof(_k), "qwen3.ssm.%s",        \
                                 suffix) > 0                            \
                     && model_get_i32(s->en->model, _k, &v32))          \
                field = (u32)v32;                                       \
        } while(0)
        TRY_SSM_I32("group_count",  ws->ssm_groups);
        TRY_SSM_I32("state_size",   ws->ssm_dim);
        TRY_SSM_I32("conv_kernel",  ws->conv_kernel);
        #undef TRY_SSM_I32
    }

    /* Allocate per-layer SSM state for Gated DeltaNet layers.
     * Only layers with ssm_conv1d tensors use this state. */
    ws->ssm_state  = scalloc((u64)c->n_layer, sizeof(float *));
    ws->conv_state = scalloc((u64)c->n_layer, sizeof(float *));
    {
        u64 state_sz  = (u64)ws->ssm_groups * ws->ssm_dim * ws->ssm_dim;
        u64 conv_sz   = (u64)(ws->conv_kernel - 1) * ws->ssm_fused;
        u32 n_ssm     = 0;
        for (u32 i = 0; i < c->n_layer; i++) {
            if (w->layers[i].tensors[TENSOR_SSM_CONV1D]) {
                ws->ssm_state[i]  = scalloc(state_sz, sizeof(float));
                ws->conv_state[i] = scalloc(conv_sz, sizeof(float));
                n_ssm++;
            }
        }
        slog(INFO, "Qwen3 SSM: groups=%u dim=%u fused=%u conv_k=%u layers=%u "
             "(state_per_layer=%.1f MB)",
             ws->ssm_groups, ws->ssm_dim, ws->ssm_fused, ws->conv_kernel, n_ssm,
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

static bool qwen35_generate(Session *s, u32 token, float *logits) {
    Qwen35Workspace *ws = (Qwen35Workspace *)s->arch_data;
    ArchConfig     *c  = &s->cfg;
    Weights        *w  = s->en->weights;
    
    u32 n_head      = c->n_head;
    u32 n_kv_head   = c->n_kv_head;
    u32 q_head_dim  = c->head_dim;
    u32 kv_head_dim = c->kv_head_dim;
    u32 q_dim       = n_head * q_head_dim;
    u32 kv_dim      = n_kv_head * kv_head_dim;
    u32 n_embd      = c->n_embd;
    u32 n_layer     = c->n_layer;
    u32 pos         = s->n_tokens;
    u32 gqa_ratio   = n_head / n_kv_head;
    float scale     = 1.0f / sqrtf((float)kv_head_dim);
    float eps       = DEFAULT_EPS;

    /* ---- 1. Token embedding ---- */
    {
        TensorInfo *te = w->tensors[TENSOR_TOKEN_EMBD];
        if (te->dim[0] == c->n_vocab) {
            /* [n_vocab, n_embd] — row token is contiguous */
            tensor_get_f32_batch(te,  (u64)token * n_embd, n_embd, ws->x);
        } else {
            /* [n_embd, n_vocab] */
            for (u32 i = 0; i < n_embd; i++)
                ws->x[i] = tensor_get_f32(te,  (u64)i * c->n_vocab + token);
        }
    }

    /* ---- 2. Per-layer ---- */
    for (u32 l = 0; l < n_layer; l++) {
        LayerWeights *lw  = &w->layers[l];
        AttnKvCache  *akc = &s->cache.std[l];

        /* ---- 2a. Attention norm ---- */
        rms_norm(ws->xb, ws->x, lw->tensors[TENSOR_ATTN_NORM],  n_embd, eps);

        /* ---- 2b. Q / K / V projections ---- */
        TensorInfo *t_qkv = lw->tensors[TENSOR_ATTN_QKV];
        TensorInfo *t_q   = lw->tensors[TENSOR_ATTN_Q];
        TensorInfo *t_k   = lw->tensors[TENSOR_ATTN_K];
        TensorInfo *t_v   = lw->tensors[TENSOR_ATTN_V];
        TensorInfo *t_ssm_conv = lw->tensors[TENSOR_SSM_CONV1D];
        u32 fused_total   = 0;

        if (t_ssm_conv) {
            /* ======== SSM (Gated DeltaNet) pathway ======== */

            /* Fused QKV projection → qkv_fused [ssm_fused = 6144]. */
            bool qkv_trans = (t_qkv->dim[0] == n_embd);
            fused_total = (u32)(qkv_trans ? t_qkv->dim[1] : t_qkv->dim[0]);
            if (!mat_vec_mul(ws->qkv_fused, t_qkv,  ws->xb, fused_total, n_embd, qkv_trans, s->pthreads))
                return false;

            /* Depthwise causal Conv1d (in-place on qkv_fused). */
            {
                u32 cch = ws->ssm_fused;   /* 6144 */
                u32 ck  = ws->conv_kernel; /* 4 */
                float *cw = smalloc((u64)ck * cch * sizeof(float));
                tensor_get_f32_batch(t_ssm_conv,  0, (u64)ck * cch, cw);
                causal_conv1d_step(ws->qkv_fused, ws->qkv_fused, cw, ws->conv_state[l], cch, ck);
                sfree(cw);
            }

            /* SiLU on the conv output (reference: conv → silu → split). */
            silu(ws->qkv_fused, (int)ws->ssm_fused);

            /* Split conv output: Q, K, V  [n_groups * d_ssm = 2048 each]. */
            u32 n_g = ws->ssm_groups;
            u32 d_s = ws->ssm_dim;
            u32 qkv_d = n_g * d_s;  /* 2048 */
            float *q_d = ws->qkv_fused;
            float *k_d = ws->qkv_fused + qkv_d;
            float *v_d = ws->qkv_fused + qkv_d * 2;

            /* L2-normalise Q and K (per group / row). */
            l2_norm_rows(q_d, n_g, d_s, DEFAULT_EPS);
            l2_norm_rows(k_d, n_g, d_s, DEFAULT_EPS);

            /* Scale Q by 1/sqrt(d_ssm) (reference: Gated DeltaNet). */
            {
                float q_scale = 1.0f / sqrtf((float)d_s);
                for (u32 i = 0; i < qkv_d; i++)
                    q_d[i] *= q_scale;
            }

            /* Compute g (decay) and beta (update) from residual. */
            float g_stack[16], beta_stack[16];
            float *gv = n_g <= 16 ? g_stack : smalloc((u64)n_g * sizeof(float));
            float *bv = n_g <= 16 ? beta_stack : smalloc((u64)n_g * sizeof(float));
            {
                TensorInfo *ta = lw->tensors[TENSOR_SSM_ALPHA];
                TensorInfo *tb = lw->tensors[TENSOR_SSM_BETA];
                TensorInfo *tA = lw->tensors[TENSOR_SSM_A];
                TensorInfo *td = lw->tensors[TENSOR_SSM_DT_BIAS];

                float *alpha   = smalloc((u64)n_g * sizeof(float));
                float *abuf    = smalloc((u64)n_g * sizeof(float));
                tensor_get_f32_batch(td,  0, n_g, abuf); /* dt_bias */
                tensor_get_f32_batch(tA,  0, n_g, gv);   /* ssm.a = -exp(A_log) */

                if (!mat_vec_mul(alpha, ta,  ws->xb, n_g, n_embd, ta->dim[0] == n_embd, s->pthreads)) {
                    sfree(alpha); sfree(abuf);
                    if (gv != g_stack) sfree(gv);
                    if (bv != beta_stack) sfree(bv);
                    return false;
                }
                /* ssm.a already stores -exp(A_log) in the GGUF
                 * (converter: A_log → -exp(A_log)), so the decay gate
                 * is g = ssm.a * softplus(alpha + dt_bias). */
                for (u32 i = 0; i < n_g; i++)
                    gv[i] = gv[i] * softplus(alpha[i] + abuf[i]);
                sfree(alpha); sfree(abuf);

                if (!mat_vec_mul(bv, tb,  ws->xb, n_g, n_embd, tb->dim[0] == n_embd, s->pthreads)) {
                    if (gv != g_stack) sfree(gv);
                    if (bv != beta_stack) sfree(bv);
                    return false;
                }
                /* beta = sigmoid */
                for (u32 i = 0; i < n_g; i++)
                    bv[i] = sigmoid(bv[i]);

                /* Convert g to multiplicative decay: g = exp(g). */
                for (u32 i = 0; i < n_g; i++)
                    gv[i] = expf(gv[i]);
            }

            /* Gated DeltaNet recurrence → hb2 [qkv_d = 2048]. */
            {
                float *mem = smalloc((u64)qkv_d * sizeof(float));
                gated_delta_step(q_d, k_d, v_d, ws->ssm_state[l], gv, bv, mem, ws->hb2, n_g, d_s);
                sfree(mem);
            }
            if (gv != g_stack) sfree(gv);
            if (bv != beta_stack) sfree(bv);

            /* Apply SSM norm FIRST (per-group RMS norm on delta output),
             * THEN apply output gating.  This matches the Qwen3.5
             * reference implementation. */
            {
                TensorInfo *t_sn = lw->tensors[TENSOR_SSM_NORM];
                if (t_sn) {
                    float *snw = smalloc((u64)d_s * sizeof(float));
                    tensor_get_f32_batch(t_sn,  0, d_s, snw);
                    float *dp = ws->hb2;
                    for (u32 g = 0; g < n_g; g++)
                        rms_norm_inplace(dp + (u64)g * d_s, snw, d_s, eps);
                    sfree(snw);
                }
            }

            /* Output gating: gate = silu(attn_gate @ xb). */
            TensorInfo *t_g = lw->tensors[TENSOR_ATTN_GATE];
            if (t_g) {
                if (!mat_vec_mul(ws->qkv_fused, t_g,  ws->xb, qkv_d, n_embd, t_g->dim[0] == n_embd, s->pthreads)) return false;
                silu(ws->qkv_fused, (int)qkv_d);
                for (u32 i = 0; i < qkv_d; i++)
                    ws->hb2[i] *= ws->qkv_fused[i];
            }

            /* Output projection: ssm_out @ delta_out → xb2 [n_embd]. */
            {
                TensorInfo *t_so = lw->tensors[TENSOR_SSM_OUT];
                if (t_so) {
                    if (!mat_vec_mul(ws->xb2, t_so,  ws->hb2, n_embd, qkv_d, t_so->dim[0] == qkv_d, s->pthreads))
                        return false;
                } else {
                    memcpy(ws->xb2, ws->hb2, n_embd * sizeof(float));
                }
            }

            /* SSM residual and skip standard attention pathway. */
            for (u32 i = 0; i < n_embd; i++)
                ws->x[i] += ws->xb2[i];
            goto ssm_attn_done;

        } else if (t_qkv) {
            bool qkv_trans = (t_qkv->dim[0] == n_embd);
            fused_total = (u32)(qkv_trans ? t_qkv->dim[1] : t_qkv->dim[0]);
            if (!mat_vec_mul(ws->qkv_fused, t_qkv,  ws->xb, fused_total, n_embd, qkv_trans, s->pthreads)) return false;
            memcpy(ws->q, ws->qkv_fused, q_dim * sizeof(float));
            memcpy(ws->k, ws->qkv_fused + q_dim, kv_dim * sizeof(float));
            memcpy(ws->v, ws->qkv_fused + q_dim + kv_dim, kv_dim * sizeof(float));
        } else if (t_q && t_k && t_v) {
            TensorInfo *t_q_norm = lw->tensors[TENSOR_ATTN_Q_NORM];
            TensorInfo *t_k_norm = lw->tensors[TENSOR_ATTN_K_NORM];

            /* Q projection with optional Q-norm (Qwen3.5).
             * Q-norm mode: Q weight produces n_head * 2 * head_dim outputs
             * (fused Q + gate).  Only the first head_dim half of each head
             * is RMS-normed and kept as Q; the second half is the raw
             * gate, consumed later (sigmoid) in the output projection. */
            if (t_q_norm) {
                u32 q_proj_dim = (t_q->dim[0] == n_embd)
                                 ? (u32)t_q->dim[1] : (u32)t_q->dim[0];
                u32 stride = 2 * q_head_dim;
                if (!mat_vec_mul(ws->qkv_fused, t_q,  ws->xb,
                                 q_proj_dim, n_embd,
                                 (q_proj_dim != n_embd) && t_q->dim[0] == n_embd,
                                 s->pthreads)) return false;
                /* Add Q bias before norm (applied to full projection). */
                {
                    TensorInfo *tb_q = lw->tensors[TENSOR_ATTN_Q_BIAS];
                    if (tb_q) bias_add(ws->qkv_fused, tb_q,  q_proj_dim);
                }
                /* Dequantise norm weight once, then apply per head.
                 * The gate half is NOT normed (matches the reference). */
                {
                    float *nw = smalloc((u64)q_head_dim * sizeof(float));
                    tensor_get_f32_batch(t_q_norm,  0, q_head_dim, nw);
                    for (u32 h = 0; h < n_head; h++) {
                        float *hsrc = ws->qkv_fused + (u64)h * stride;
                        rms_norm_inplace(hsrc, nw, q_head_dim, eps);
                        memcpy(ws->q + (u64)h * q_head_dim, hsrc,
                               q_head_dim * sizeof(float));
                    }
                    sfree(nw);
                }
            } else {
                if (!mat_vec_mul(ws->q, t_q,  ws->xb, q_dim, n_embd, (q_dim != n_embd) && t_q->dim[0] == n_embd, s->pthreads)) return false;
                {
                    TensorInfo *tb_q = lw->tensors[TENSOR_ATTN_Q_BIAS];
                    if (tb_q) bias_add(ws->q, tb_q,  q_dim);
                }
            }

            /* K projection with optional K-norm (Qwen3.5). */
            {
                if (!mat_vec_mul(ws->k, t_k,  ws->xb, kv_dim, n_embd, t_k->dim[0] == n_embd, s->pthreads)) return false;
                TensorInfo *tb_k = lw->tensors[TENSOR_ATTN_K_BIAS];
                if (tb_k) bias_add(ws->k, tb_k,  kv_dim);
                if (t_k_norm) {
                    float *nw = smalloc((u64)kv_head_dim * sizeof(float));
                    tensor_get_f32_batch(t_k_norm,  0, kv_head_dim, nw);
                    qwen_k_norm(ws->k, n_kv_head, kv_head_dim, nw, eps);
                    sfree(nw);
                }
            }

            /* V projection (no norm). */
            {
                if (!mat_vec_mul(ws->v, t_v,  ws->xb, kv_dim, n_embd, t_v->dim[0] == n_embd, s->pthreads)) return false;
                TensorInfo *tb_v = lw->tensors[TENSOR_ATTN_V_BIAS];
                if (tb_v) bias_add(ws->v, tb_v,  kv_dim);
            }
        } else {
            slog(WARN, "Layer %u: missing QKV / Q,K,V tensors", l);
            return false;
        }

        /* ---- 2c. RoPE ---- */
        rope_partial(ws->q, n_head, q_head_dim, c->rope_dim, pos, ws->rope_theta);
        rope_partial(ws->k, n_kv_head, kv_head_dim, c->rope_dim, pos, ws->rope_theta);

        /* ---- 2d. KV cache write (head-major: [head][pos][dim]) ---- */
        {
            u32 hs = akc->cap * kv_head_dim; /* stride between heads in cache */
            for (u32 h = 0; h < n_kv_head; h++) {
                u64 dst_off = (u64)h * hs + (u64)pos * kv_head_dim;
                u64 src_off = (u64)h * kv_head_dim;
                memcpy(akc->k + dst_off, ws->k + src_off, kv_head_dim * sizeof(float));
                memcpy(akc->v + dst_off, ws->v + src_off, kv_head_dim * sizeof(float));
            }
        }
        akc->n = pos + 1;

        /* ---- 2e. Attention ---- */
        u32 n_cached = akc->n;
        u32 hs = akc->cap * kv_head_dim; /* head stride in cache */
        float *attn_out = ws->hb;
        memset(attn_out, 0, q_dim * sizeof(float));

        for (u32 h = 0; h < n_head; h++) {
            u32 kv_h   = h / gqa_ratio;
            float *qh  = ws->q + (u64)h * q_head_dim;
            float *kh_base = akc->k + (u64)kv_h * hs;
            float *vh_base = akc->v + (u64)kv_h * hs;

            /* Q·K^T scores. */
            float *kt = kh_base;
            for (u32 t = 0; t < n_cached; t++, kt += kv_head_dim) {
                float s = 0.0f;
                for (u32 d = 0; d < kv_head_dim; d++)
                    s += qh[d] * kt[d];
                ws->scores[t] = s * scale;
            }
            softmax(ws->scores, n_cached);

            /* Weighted V sum. */
            float *oh = attn_out + (u64)h * q_head_dim;
            float *vt = vh_base;
            for (u32 t = 0; t < n_cached; t++, vt += kv_head_dim) {
                float st = ws->scores[t];
                for (u32 d = 0; d < kv_head_dim; d++)
                    oh[d] += st * vt[d];
            }
        }

        /* ---- 2f. Attention output projection ---- */
        {
            TensorInfo *t_out  = lw->tensors[TENSOR_ATTN_OUT];
            TensorInfo *t_gate = lw->tensors[TENSOR_ATTN_GATE];

            /* Qwen3.5 gating: project residual → gate, apply SiLU,
             * then multiply element-wise with attention output. */
            if (t_gate) {
                if (!mat_vec_mul(ws->hb, t_gate,  ws->xb, q_dim, n_embd, t_gate->dim[0] == n_embd, s->pthreads)) return false;
                silu(ws->hb, (int)q_dim);
                for (u32 i = 0; i < q_dim; i++)
                    attn_out[i] *= ws->hb[i];
            }

            /* Qwen3.5 fused QG mode: the second half of the fused
             * Q projection (still in qkv_fused) is the attention gate.
             * Gate = sigmoid(gate_proj), multiplied onto the attention
             * output before the out projection (reference: gate_reshaped
             * → sigmoid → mul on attn output).  The Q half was already
             * separated and normed in 2b; qkv_fused is untouched since. */
            {
                TensorInfo *t_q_norm = lw->tensors[TENSOR_ATTN_Q_NORM];
                TensorInfo *t_q2     = lw->tensors[TENSOR_ATTN_Q];
                if (t_q_norm && t_q2) {
                    u32 q_proj_dim = (t_q2->dim[0] == n_embd) ? (u32)t_q2->dim[1] : (u32)t_q2->dim[0];
                    if (q_proj_dim == 2 * q_dim) {
                        u32 stride = 2 * q_head_dim;
                        for (u32 h = 0; h < n_head; h++) {
                            float *gate = ws->qkv_fused + (u64)h * stride + q_head_dim;
                            float *oh = attn_out + (u64)h * q_head_dim;
                            for (u32 d = 0; d < q_head_dim; d++)
                                oh[d] *= sigmoid(gate[d]);
                        }
                    }
                }
            }

            if (t_out) {
                if (t_out->ndim >= 2 && t_out->dim[0] == 2 * n_embd && t_out->dim[1] == q_dim) {
                    /* Gated output weight (Qwen2.5): 2*n_embd outputs,
                     * split into gate (first n_embd) and value. */
                    for (u32 i = 0; i < 2 * n_embd; i++) {
                        u64 off = (u64)i * q_dim;
                        float sum = 0.0f;
                        for (u32 j = 0; j < q_dim; j++)
                            sum += tensor_get_f32(t_out,  off + j) * attn_out[j];
                        ws->hb2[i] = sum;
                    }
                    silu(ws->hb2, (int)n_embd);
                    for (u32 i = 0; i < n_embd; i++)
                        ws->xb2[i] = ws->hb2[i] * ws->hb2[i + n_embd];
                } else {
                    if (!mat_vec_mul(ws->xb2, t_out,  attn_out, n_embd, q_dim, (n_embd != q_dim) && t_out->dim[0] == q_dim, s->pthreads)) 
                        return false;
                }
            } else {
                if (q_dim == n_embd) memcpy(ws->xb2, attn_out, q_dim * sizeof(float));
                else memcpy(ws->xb2, attn_out, n_embd * sizeof(float));
            }
        }

        /* ---- 2g. Attention residual ---- */
        for (u32 i = 0; i < n_embd; i++)
            ws->x[i] += ws->xb2[i];

        ssm_attn_done:
        /* ---- 2h. Pre-FFN norm ---- */
        {
            TensorInfo *ti = lw->tensors[TENSOR_POST_ATTN_NORM];
            if (ti) rms_norm(ws->xb, ws->x, ti,  n_embd, eps);
        }

        /* ---- 2i. SwiGLU FFN ---- */
        {
            TensorInfo *t_gate = lw->tensors[TENSOR_FFN_GATE];
            TensorInfo *t_up   = lw->tensors[TENSOR_FFN_UP];
            TensorInfo *t_down = lw->tensors[TENSOR_FFN_DOWN];
            u32 fh = ws->ffn_hidden;

            if (!mat_vec_mul(ws->hb, t_gate,  ws->xb, fh, n_embd, t_gate->dim[0] == n_embd, s->pthreads)) return false;
            if (!mat_vec_mul(ws->hb2, t_up,  ws->xb, fh, n_embd, t_up->dim[0] == n_embd, s->pthreads)) return false;

            silu(ws->hb, fh);
            for (u32 i = 0; i < fh; i++)
                ws->hb[i] *= ws->hb2[i];

            if (!mat_vec_mul(ws->xb, t_down,  ws->hb, n_embd, fh, t_down->dim[0] == fh, s->pthreads)) return false;

            /* Residual. */
            for (u32 i = 0; i < n_embd; i++)
                ws->x[i] += ws->xb[i];
        }

    }

    /* ---- 3. Final RMS norm ---- */
    {
        TensorInfo *t_norm = w->tensors[TENSOR_OUTPUT_NORM];
        /* Some Qwen2 models use TENSOR_POST_ATTN_NORM from layer 0
         * as the final norm; try output_norm first, then fall back. */
        if (!t_norm) t_norm = w->layers[n_layer - 1].tensors[TENSOR_POST_ATTN_NORM];
        rms_norm(ws->xb, ws->x, t_norm,  n_embd, eps);
    }

    /* ---- 4. LM head ---- */
    {
        TensorInfo *t_out = w->tensors[TENSOR_OUTPUT];
        if (!t_out) t_out = w->tensors[TENSOR_TOKEN_EMBD];  /* tied weights */
        float *dst = logits ? logits : s->logits;
        if (!mat_vec_mul(dst, t_out,  ws->xb, c->n_vocab, n_embd, t_out->dim[0] == n_embd, s->pthreads)) return false;
    }

    /* ---- 5. Update session state ---- */
    s->tokens[pos] = token;
    s->n_tokens    = pos + 1;
    return true;
}

static bool qwen35_prefill(Session *s, u32 *tokens, u32 n_tokens, float *logits) {
    if (n_tokens == 0) return true;
    if (n_tokens == 1) return qwen35_generate(s, tokens[0], logits);

    Qwen35Workspace *ws   = (Qwen35Workspace *)s->arch_data;
    ArchConfig      *c    = &s->cfg;
    Weights         *w    = s->en->weights;
    
    u32 n_head      = c->n_head;
    u32 n_kv_head   = c->n_kv_head;
    u32 q_head_dim  = c->head_dim;
    u32 kv_head_dim = c->kv_head_dim;
    u32 q_dim       = n_head * q_head_dim;
    u32 kv_dim      = n_kv_head * kv_head_dim;
    u32 n_embd      = c->n_embd;
    u32 n_layer     = c->n_layer;
    u32 cache_start = s->n_tokens;
    u32 gqa_ratio   = n_head / n_kv_head;
    float scale     = 1.0f / sqrtf((float)kv_head_dim);
    float eps       = DEFAULT_EPS;

    /* Determine max fused QKV output width across layers.
     * This buffer is also used as temporary scratch for Q-norm
     * processing (Qwen3.5), so we must also account for the
     * largest attn_q tensor when Q-norm is present. */
    u32 max_fused = q_dim + 2 * kv_dim;
    for (u32 l = 0; l < n_layer; l++) {
        TensorInfo *t_qkv = w->layers[l].tensors[TENSOR_ATTN_QKV];
        if (t_qkv && t_qkv->ndim >= 2) {
            u32 sz = (t_qkv->dim[0] > t_qkv->dim[1])
                   ? (u32)t_qkv->dim[0] : (u32)t_qkv->dim[1];
            if (sz > max_fused) max_fused = sz;
        }
        /* Also check attn_q for Q-norm layers (Qwen3.5 hybrid). */
        TensorInfo *t_q = w->layers[l].tensors[TENSOR_ATTN_Q];
        if (t_q && t_q->ndim >= 2) {
            u32 q_sz = (t_q->dim[0] > t_q->dim[1])
                     ? (u32)t_q->dim[0] : (u32)t_q->dim[1];
            /* Need space for Q-proj + K + V in qkv_buf. */
            u32 need = q_sz + 2 * kv_dim;
            if (need > max_fused) max_fused = need;
        }
    }

    u64 row_x   = (u64)n_tokens * n_embd;
    u64 row_q   = (u64)n_tokens * q_dim;
    u64 row_kv  = (u64)n_tokens * max_fused;
    u64 row_fh  = (u64)n_tokens * ws->ffn_hidden;
    u64 row_max = row_x > row_q ? row_x : row_q;

    /* ---- Temporary buffers ---- */
    float *xs       = smalloc(row_x * sizeof(float));   /* hidden states          */
    float *norm_buf = smalloc(row_max * sizeof(float)); /* RMS-norm scratch       */
    float *qkv_buf  = smalloc(row_kv * sizeof(float));  /* batched QKV output     */
    float *qbuf     = smalloc(row_q * sizeof(float));   /* Q buffer for attention */
    float *attn_buf = smalloc(row_q * sizeof(float));   /* attention output       */
    float *gate_buf = smalloc(row_fh * sizeof(float));  /* FFN gate / silu*up     */
    float *up_buf   = smalloc(row_fh * sizeof(float));  /* FFN up                 */
    if (!xs || !norm_buf || !qkv_buf || !qbuf || !attn_buf || !gate_buf || !up_buf) {
        sfree(xs); sfree(norm_buf); sfree(qkv_buf); sfree(qbuf);
        sfree(attn_buf); sfree(gate_buf); sfree(up_buf);
        slog(WARN, "forward: smalloc failed for %u tokens", n_tokens);
        return false;
    }

    /* ---- 1. Token embeddings (all positions) ---- */
    {
        TensorInfo *te = w->tensors[TENSOR_TOKEN_EMBD];
        bool te_trans = (te->dim[0] == c->n_vocab);
        if (te_trans) {
            /* [n_vocab, n_embd] — each token's row is contiguous */
            for (u32 p = 0; p < n_tokens; p++) {
                float *xp = xs + (u64)p * n_embd;
                tensor_get_f32_batch(te,  (u64)tokens[p] * n_embd, n_embd, xp);
            }
        } else {
            for (u32 p = 0; p < n_tokens; p++) {
                float *xp = xs + (u64)p * n_embd;
                u32 tok = tokens[p];
                for (u32 i = 0; i < n_embd; i++)
                    xp[i] = tensor_get_f32(te,  (u64)i * c->n_vocab + tok);
            }
        }
    }

    /* ---- 2. Per-layer ---- */
    for (u32 l = 0; l < n_layer; l++) {
        LayerWeights *lw  = &w->layers[l];
        AttnKvCache  *akc = &s->cache.std[l];

        TensorInfo *t_qkv    = lw->tensors[TENSOR_ATTN_QKV];
        TensorInfo *t_q      = lw->tensors[TENSOR_ATTN_Q];
        TensorInfo *t_k      = lw->tensors[TENSOR_ATTN_K];
        TensorInfo *t_v      = lw->tensors[TENSOR_ATTN_V];
        TensorInfo *tb_q     = lw->tensors[TENSOR_ATTN_Q_BIAS];
        TensorInfo *tb_k     = lw->tensors[TENSOR_ATTN_K_BIAS];
        TensorInfo *tb_v     = lw->tensors[TENSOR_ATTN_V_BIAS];
        TensorInfo *t_attn   = lw->tensors[TENSOR_ATTN_NORM];
        TensorInfo *t_out    = lw->tensors[TENSOR_ATTN_OUT];
        TensorInfo *t_post   = lw->tensors[TENSOR_POST_ATTN_NORM];
        TensorInfo *t_gate   = lw->tensors[TENSOR_FFN_GATE];
        TensorInfo *t_up     = lw->tensors[TENSOR_FFN_UP];
        TensorInfo *t_down   = lw->tensors[TENSOR_FFN_DOWN];
        u32 fh = ws->ffn_hidden;

        /* ---- 2a. RMS norm on all xs → norm_buf ---- */
        for (u32 p = 0; p < n_tokens; p++)
            rms_norm(norm_buf + (u64)p * n_embd,
                     xs + (u64)p * n_embd, t_attn, n_embd, eps);

        /* ---- 2b. Batched Q/K/V projection ---- */
        {
            u32 fused_total;
            bool qkv_trans;

            u32 n_q = 0, n_k = 0, n_v = 0;
            float  q_bias_stack[BIAS_BUF_STACK], k_bias_stack[BIAS_BUF_STACK], v_bias_stack[BIAS_BUF_STACK];
            float *q_bias = NULL, *k_bias = NULL, *v_bias = NULL;

            if (tb_q) {
                n_q = tb_q->n_element < (u64)q_dim ? (u32)tb_q->n_element : q_dim;
                q_bias = n_q <= BIAS_BUF_STACK ? q_bias_stack : smalloc((u64)n_q * sizeof(float));
                tensor_get_f32_batch(tb_q,  0, n_q, q_bias);
            }
            if (tb_k) {
                n_k = tb_k->n_element < (u64)kv_dim ? (u32)tb_k->n_element : kv_dim;
                k_bias = n_k <= BIAS_BUF_STACK ? k_bias_stack : smalloc((u64)n_k * sizeof(float));
                tensor_get_f32_batch(tb_k,  0, n_k, k_bias);
            }
            if (tb_v) {
                n_v = tb_v->n_element < (u64)kv_dim ? (u32)tb_v->n_element : kv_dim;
                v_bias = n_v <= BIAS_BUF_STACK ? v_bias_stack : smalloc((u64)n_v * sizeof(float));
                tensor_get_f32_batch(tb_v,  0, n_v, v_bias);
            }

            TensorInfo *t_ssm_conv_pf = lw->tensors[TENSOR_SSM_CONV1D];
            if (t_ssm_conv_pf) {
                /* ======== Prefill SSM (Gated DeltaNet) ========
                 * The recurrence is inherently sequential, so we
                 * process tokens one-by-one even during prefill. */
                u32 n_g = ws->ssm_groups;
                u32 d_s = ws->ssm_dim;
                u32 qkv_d = n_g * d_s;  /* 2048 */
                u32 f_dim = ws->ssm_fused;
                u32 ck    = ws->conv_kernel;

                /* Dequantise conv weight once for all tokens. */
                float *cw = smalloc((u64)ck * f_dim * sizeof(float));
                tensor_get_f32_batch(t_ssm_conv_pf,  0, (u64)ck * f_dim, cw);

                /* Dequantise SSM params once. */
                TensorInfo *ta  = lw->tensors[TENSOR_SSM_ALPHA];
                TensorInfo *tb  = lw->tensors[TENSOR_SSM_BETA];
                TensorInfo *tA  = lw->tensors[TENSOR_SSM_A];
                TensorInfo *tdt = lw->tensors[TENSOR_SSM_DT_BIAS];
                TensorInfo *t_gate_pf = lw->tensors[TENSOR_ATTN_GATE];
                TensorInfo *t_so_pf   = lw->tensors[TENSOR_SSM_OUT];

                float *dt_bi = smalloc((u64)n_g * sizeof(float));
                float *a_log = smalloc((u64)n_g * sizeof(float));
                tensor_get_f32_batch(tdt,  0, n_g, dt_bi);
                tensor_get_f32_batch(tA,  0, n_g, a_log);

                for (u32 p = 0; p < n_tokens; p++) {
                    float *xp = norm_buf + (u64)p * n_embd;

                    /* Fused QKV projection. */
                    bool qkv_tr = (t_qkv->dim[0] == n_embd);
                    u32 ft = (u32)(qkv_tr ? t_qkv->dim[1] : t_qkv->dim[0]);
                    if (!mat_vec_mul(ws->qkv_fused, t_qkv,  xp,
                                     ft, n_embd, qkv_tr, s->pthreads)) {
                        sfree(cw); sfree(dt_bi); sfree(a_log); goto fail;
                    }

                    /* Conv1d (in-place). */
                    causal_conv1d_step(ws->qkv_fused, ws->qkv_fused, cw,
                                       ws->conv_state[l], f_dim, ck);
                    /* SiLU on the conv output (reference: conv → silu → split). */
                    silu(ws->qkv_fused, (int)f_dim);

                    /* Split Q, K, V. */
                    float *qd = ws->qkv_fused;
                    float *kd = ws->qkv_fused + qkv_d;
                    float *vd = ws->qkv_fused + qkv_d * 2;

                    /* L2-norm Q, K. */
                    l2_norm_rows(qd, n_g, d_s, DEFAULT_EPS);
                    l2_norm_rows(kd, n_g, d_s, DEFAULT_EPS);
                    /* Q *= 1/sqrt(d_ssm) */
                    {
                        float qs = 1.0f / sqrtf((float)d_s);
                        for (u32 i = 0; i < qkv_d; i++)
                            qd[i] *= qs;
                    }

                    /* Compute g, beta. */
                    float g_stk[16], b_stk[16];
                    float *gp = n_g <= 16 ? g_stk : smalloc((u64)n_g * sizeof(float));
                    float *bp = n_g <= 16 ? b_stk : smalloc((u64)n_g * sizeof(float));
                    {
                        float *al = smalloc((u64)n_g * sizeof(float));
                        if (!mat_vec_mul(al, ta,  xp, n_g, n_embd,
                                         ta->dim[0] == n_embd, s->pthreads)) {
                            sfree(al); sfree(cw); sfree(dt_bi); sfree(a_log);
                            if (gp != g_stk) sfree(gp);
                            if (bp != b_stk) sfree(bp);
                            goto fail;
                        }
                        /* ssm.a already stores -exp(A_log) in the GGUF
                         * (converter: A_log → -exp(A_log)), so the decay
                         * gate is g = ssm.a * softplus(alpha + dt_bias). */
                        for (u32 i = 0; i < n_g; i++) {
                            gp[i] = a_log[i] * softplus(al[i] + dt_bi[i]);
                            gp[i] = expf(gp[i]); /* multiplicative form */
                        }
                        sfree(al);
                    }
                    {
                        if (!mat_vec_mul(bp, tb,  xp, n_g, n_embd,
                                         tb->dim[0] == n_embd, s->pthreads)) {
                            sfree(cw); sfree(dt_bi); sfree(a_log);
                            if (gp != g_stk) sfree(gp);
                            if (bp != b_stk) sfree(bp);
                            goto fail;
                        }
                        for (u32 i = 0; i < n_g; i++)
                            bp[i] = sigmoid(bp[i]);
                    }

                    /* Delta recurrence → hb2. */
                    {
                        float *mem = smalloc((u64)qkv_d * sizeof(float));
                        gated_delta_step(qd, kd, vd, ws->ssm_state[l],
                                         gp, bp, mem, ws->hb2, n_g, d_s);
                        sfree(mem);
                    }
                    if (gp != g_stk) sfree(gp);
                    if (bp != b_stk) sfree(bp);

                    /* Apply SSM norm. */
                    {
                        TensorInfo *t_sn_pf = lw->tensors[TENSOR_SSM_NORM];
                        if (t_sn_pf) {
                            float *snw = smalloc((u64)d_s * sizeof(float));
                            tensor_get_f32_batch(t_sn_pf,  0, d_s, snw);
                            float *dp = ws->hb2;
                            for (u32 g = 0; g < n_g; g++)
                                rms_norm_inplace(dp + (u64)g * d_s, snw, d_s, eps);
                            sfree(snw);
                        }
                    }

                    /* Gate. */
                    if (t_gate_pf) {
                        if (!mat_vec_mul(ws->hb, t_gate_pf,  xp,
                                         qkv_d, n_embd,
                                         t_gate_pf->dim[0] == n_embd, s->pthreads)) {
                            sfree(cw); sfree(dt_bi); sfree(a_log); goto fail;
                        }
                        silu(ws->hb, (int)qkv_d);
                        for (u32 i = 0; i < qkv_d; i++)
                            ws->hb2[i] *= ws->hb[i];
                    }

                    /* Output projection → norm_buf[p]. */
                    float *yp = norm_buf + (u64)p * n_embd;
                    if (t_so_pf) {
                        if (!mat_vec_mul(yp, t_so_pf,  ws->hb2,
                                         n_embd, qkv_d,
                                         t_so_pf->dim[0] == qkv_d, s->pthreads)) {
                            sfree(cw); sfree(dt_bi); sfree(a_log); goto fail;
                        }
                    } else {
                        memcpy(yp, ws->hb2, n_embd * sizeof(float));
                    }
                }
                sfree(cw); sfree(dt_bi); sfree(a_log);
                akc->n = cache_start + n_tokens;

                /* Release bias buffers (unused in SSM path). */
                if (q_bias != q_bias_stack) sfree(q_bias);
                if (k_bias != k_bias_stack) sfree(k_bias);
                if (v_bias != v_bias_stack) sfree(v_bias);

                /* Skip the remaining attention code (2c, 2d) for this layer.
                 * The SSM output is already in norm_buf. */
                goto ssm_layer_done;
            }

            if (t_qkv) {
                /* Fused QKV: one mat_mat_mul, then split per token. */
                qkv_trans   = (t_qkv->dim[0] == n_embd);
                fused_total = (u32)(qkv_trans ? t_qkv->dim[1] : t_qkv->dim[0]);
                if (!mat_mat_mul(qkv_buf, t_qkv,  norm_buf, n_tokens, fused_total, n_embd, qkv_trans, s->pthreads)) goto fail;
                for (u32 p = 0; p < n_tokens; p++) {
                    u32 pos = cache_start + p;
                    float *row = qkv_buf + (u64)p * fused_total;

                    /* Copy Q to qbuf (stride q_dim) for batched attention. */
                    memcpy(qbuf + (u64)p * q_dim, row, q_dim * sizeof(float));
                    /* Copy K/V to workspace for RoPE + cache write. */
                    memcpy(ws->k, row + q_dim, kv_dim * sizeof(float));
                    memcpy(ws->v, row + q_dim + kv_dim, kv_dim * sizeof(float));

                    /* Apply pre-dequantised bias (f32 addition, trivially vectorisable). */
                    if (tb_q) {
                        float *qp = qbuf + (u64)p * q_dim;
                        for (u32 i = 0; i < n_q; i++) qp[i] += q_bias[i];
                    }
                    if (tb_k) {
                        for (u32 i = 0; i < n_k; i++) ws->k[i] += k_bias[i];
                    }
                    if (tb_v) {
                        for (u32 i = 0; i < n_v; i++) ws->v[i] += v_bias[i];
                    }

                    rope_partial(qbuf + (u64)p * q_dim, n_head, q_head_dim, c->rope_dim, pos, ws->rope_theta);
                    rope_partial(ws->k, n_kv_head, kv_head_dim, c->rope_dim, pos, ws->rope_theta);

                    {
                        u32 hs = akc->cap * kv_head_dim;
                        for (u32 h = 0; h < n_kv_head; h++) {
                            u64 dst = (u64)h * hs + (u64)pos * kv_head_dim;
                            u64 src = (u64)h * kv_head_dim;
                            memcpy(akc->k + dst, ws->k + src, kv_head_dim * sizeof(float));
                            memcpy(akc->v + dst, ws->v + src, kv_head_dim * sizeof(float));
                        }
                    }
                }
            } else if (t_q && t_k && t_v) {
                TensorInfo *t_q_norm = lw->tensors[TENSOR_ATTN_Q_NORM];
                TensorInfo *t_k_norm = lw->tensors[TENSOR_ATTN_K_NORM];

                /* Q projection with optional Q-norm (Qwen3.5).
                 * Use qkv_buf as temporary space for the full Q projection
                 * (q_proj_dim per token), then apply Q-norm → qbuf. */
                if (t_q_norm) {
                    u32 q_proj_dim = (t_q->dim[0] == n_embd)
                                     ? (u32)t_q->dim[1] : (u32)t_q->dim[0];
                    u32 stride = 2 * q_head_dim;
                    if (!mat_mat_mul(qkv_buf, t_q,  norm_buf, n_tokens,
                                     q_proj_dim, n_embd,
                                     (q_proj_dim != n_embd) && t_q->dim[0] == n_embd,
                                     s->pthreads)) goto fail;
                    /* Dequantise Q-norm weight once. */
                    float *qnw = smalloc((u64)q_head_dim * sizeof(float));
                    tensor_get_f32_batch(t_q_norm,  0, q_head_dim, qnw);
                    for (u32 p = 0; p < n_tokens; p++) {
                        float *src = qkv_buf + (u64)p * q_proj_dim;
                        float *dst = qbuf   + (u64)p * q_dim;
                        /* Add Q bias before norm. */
                        if (tb_q) {
                            for (u32 i = 0; i < n_q; i++)
                                src[i] += q_bias[i];
                        }
                        /* Only the Q half is normed; the gate half of the
                         * fused projection is left raw (sigmoid later). */
                        for (u32 h = 0; h < n_head; h++) {
                            float *hsrc = src + (u64)h * stride;
                            rms_norm_inplace(hsrc, qnw, q_head_dim, eps);
                            memcpy(dst + (u64)h * q_head_dim, hsrc,
                                   q_head_dim * sizeof(float));
                        }
                    }
                    sfree(qnw);
                } else {
                    if (!mat_mat_mul(qbuf, t_q,  norm_buf, n_tokens,
                                     q_dim, n_embd,
                                     (q_dim != n_embd) && t_q->dim[0] == n_embd,
                                     s->pthreads)) goto fail;
                    if (tb_q) {
                        for (u32 p = 0; p < n_tokens; p++) {
                            float *qp = qbuf + (u64)p * q_dim;
                            for (u32 i = 0; i < n_q; i++) qp[i] += q_bias[i];
                        }
                    }
                }

                /* K + V: use area of qkv_buf beyond the Q-proj region. */
                {
                    u32 q_use = t_q_norm
                        ? (t_q->dim[0] == n_embd ? (u32)t_q->dim[1] : (u32)t_q->dim[0])
                        : q_dim;
                    float *kbuf = qkv_buf + (u64)n_tokens * q_use;
                    float *vbuf = kbuf + (u64)n_tokens * kv_dim;
                    if (!mat_mat_mul(kbuf, t_k,  norm_buf, n_tokens, kv_dim,
                                     n_embd, t_k->dim[0] == n_embd, s->pthreads)) goto fail;
                    if (!mat_mat_mul(vbuf, t_v,  norm_buf, n_tokens, kv_dim,
                                     n_embd, t_v->dim[0] == n_embd, s->pthreads)) goto fail;

                    /* Dequantise K-norm weight once if present. */
                    float *knw = NULL;
                    if (t_k_norm) {
                        knw = smalloc((u64)kv_head_dim * sizeof(float));
                        tensor_get_f32_batch(t_k_norm,  0, kv_head_dim, knw);
                    }
                    for (u32 p = 0; p < n_tokens; p++) {
                        u32 pos = cache_start + p;

                        memcpy(ws->k, kbuf + (u64)p * kv_dim, kv_dim * sizeof(float));
                        memcpy(ws->v, vbuf + (u64)p * kv_dim, kv_dim * sizeof(float));

                        if (tb_k) {
                            for (u32 i = 0; i < n_k; i++) ws->k[i] += k_bias[i];
                        }
                        if (t_k_norm) {
                            qwen_k_norm(ws->k, n_kv_head, kv_head_dim, knw, eps);
                        }
                        if (tb_v) {
                            for (u32 i = 0; i < n_v; i++) ws->v[i] += v_bias[i];
                        }

                        rope_partial(qbuf + (u64)p * q_dim, n_head, q_head_dim, c->rope_dim, pos, ws->rope_theta);
                        rope_partial(ws->k, n_kv_head, kv_head_dim, c->rope_dim, pos, ws->rope_theta);

                        {
                            u32 hs = akc->cap * kv_head_dim;
                            for (u32 h = 0; h < n_kv_head; h++) {
                                u64 dst = (u64)h * hs + (u64)pos * kv_head_dim;
                                u64 src = (u64)h * kv_head_dim;
                                memcpy(akc->k + dst, ws->k + src, kv_head_dim * sizeof(float));
                                memcpy(akc->v + dst, ws->v + src, kv_head_dim * sizeof(float));
                            }
                        }
                    }
                    sfree(knw);
                }
            } else {
                slog(WARN, "Layer %u: missing QKV / Q,K,V tensors", l);
                goto fail;
            }
            akc->n = cache_start + n_tokens;

            /* Release heap-allocated bias buffers. */
            if (q_bias != q_bias_stack) sfree(q_bias);
            if (k_bias != k_bias_stack) sfree(k_bias);
            if (v_bias != v_bias_stack) sfree(v_bias);
        }

        /* ---- 2c. Batched causal attention ---- */
        memset(attn_buf, 0, row_q * sizeof(float));
        {
            u32 hs = s->cache.std->cap * kv_head_dim;
            for (u32 h = 0; h < n_head; h++) {
                u32 kv_h = h / gqa_ratio;
                float *kh_base = akc->k + (u64)kv_h * hs;
                float *vh_base = akc->v + (u64)kv_h * hs;
                for (u32 qi = 0; qi < n_tokens; qi++) {
                    float *qh = qbuf + (u64)qi * q_dim + (u64)h * q_head_dim;
                    u32 n_keys = cache_start + qi + 1;

                    float *kt = kh_base;
                    for (u32 kj = 0; kj < n_keys; kj++, kt += kv_head_dim) {
                        float s = 0.0f;
                        for (u32 d = 0; d < kv_head_dim; d++)
                            s += qh[d] * kt[d];
                        ws->scores[kj] = s * scale;
                    }
                    softmax(ws->scores, n_keys);

                    float *oh = attn_buf + (u64)qi * q_dim + (u64)h * q_head_dim;
                    float *vt = vh_base;
                    for (u32 kj = 0; kj < n_keys; kj++, vt += kv_head_dim) {
                        float st = ws->scores[kj];
                        for (u32 d = 0; d < kv_head_dim; d++)
                            oh[d] += st * vt[d];
                    }
                }
            }
        }

        /* ---- 2d. Batched attention output projection ---- */
        {
            TensorInfo *t_gate_attn = lw->tensors[TENSOR_ATTN_GATE];

            /* Qwen3.5 gating: project residual → gate, apply SiLU,
             * then multiply element-wise with attention output. */
            if (t_gate_attn) {
                /* gate_buf has n_tokens * ffn_hidden elements (3584 per token),
                 * large enough for q_dim = 2048 per token. */
                if (!mat_mat_mul(gate_buf, t_gate_attn,  norm_buf, n_tokens,
                                 q_dim, n_embd,
                                 t_gate_attn->dim[0] == n_embd, s->pthreads)) goto fail;
                for (u32 p = 0; p < n_tokens; p++) {
                    float *gp = gate_buf + (u64)p * q_dim;
                    float *ap = attn_buf + (u64)p * q_dim;
                    silu(gp, (int)q_dim);
                    for (u32 i = 0; i < q_dim; i++)
                        ap[i] *= gp[i];
                }
            }

            /* Qwen3.5 fused QG mode: gate = sigmoid of the raw second
             * half of the fused Q projection (still in qkv_buf). */
            {
                TensorInfo *t_q_norm = lw->tensors[TENSOR_ATTN_Q_NORM];
                if (t_q_norm && t_q) {
                    u32 q_proj_dim = (t_q->dim[0] == n_embd)
                                     ? (u32)t_q->dim[1] : (u32)t_q->dim[0];
                    if (q_proj_dim == 2 * q_dim) {
                        u32 stride = 2 * q_head_dim;
                        for (u32 p = 0; p < n_tokens; p++) {
                            float *src = qkv_buf + (u64)p * q_proj_dim;
                            float *ap  = attn_buf + (u64)p * q_dim;
                            for (u32 h = 0; h < n_head; h++) {
                                float *gate = src + (u64)h * stride + q_head_dim;
                                float *oh   = ap + (u64)h * q_head_dim;
                                for (u32 d = 0; d < q_head_dim; d++)
                                    oh[d] *= sigmoid(gate[d]);
                            }
                        }
                    }
                }
            }

            if (t_out) {
                if (t_out->ndim >= 2 && t_out->dim[0] == 2 * n_embd && t_out->dim[1] == q_dim) {
                    /* Gated output weight: 2*n_embd outputs,
                     * split into gate (first n_embd) and value. */
                    for (u32 p = 0; p < n_tokens; p++) {
                        float *y_row = norm_buf + (u64)p * n_embd;
                        const float *x_row = attn_buf + (u64)p * q_dim;
                        float *tmp = gate_buf + (u64)p * 2 * n_embd;
                        for (u32 i = 0; i < 2 * n_embd; i++) {
                            u64 off = (u64)i * q_dim;
                            float sum = 0.0f;
                            for (u32 j = 0; j < q_dim; j++)
                                sum += tensor_get_f32(t_out,  off + j) * x_row[j];
                            tmp[i] = sum;
                        }
                        silu(tmp, (int)n_embd);
                        for (u32 i = 0; i < n_embd; i++)
                            y_row[i] = tmp[i] * tmp[i + n_embd];
                    }
                } else if (!mat_mat_mul(norm_buf, t_out,  attn_buf, n_tokens,
                                        n_embd, q_dim,
                                        (n_embd != q_dim) && t_out->dim[0] == q_dim,
                                        s->pthreads)) { goto fail; }
            } else {
                if (q_dim == n_embd) {
                    memcpy(norm_buf, attn_buf, row_q * sizeof(float));
                } else {
                    for (u32 p = 0; p < n_tokens; p++)
                        memcpy(norm_buf + (u64)p * n_embd, attn_buf + (u64)p * q_dim, n_embd * sizeof(float));
                }
            }
        }

        ssm_layer_done:
        /* ---- 2e. Residual ---- */
        for (u32 p = 0; p < n_tokens; p++) {
            float *xp = xs + (u64)p * n_embd;
            float *ap = norm_buf + (u64)p * n_embd;
            for (u32 i = 0; i < n_embd; i++)
                xp[i] += ap[i];
        }

        /* ---- 2f. Pre-FFN norm → norm_buf ---- */
        if (t_post) {
            for (u32 p = 0; p < n_tokens; p++)
                rms_norm(norm_buf + (u64)p * n_embd, xs + (u64)p * n_embd, t_post,  n_embd, eps);
        }

        /* ---- 2g. Batched FFN (SwiGLU) ---- */
        {
            if (!mat_mat_mul(gate_buf, t_gate,  norm_buf, n_tokens, fh, n_embd, t_gate->dim[0] == n_embd, s->pthreads)) goto fail;
            if (!mat_mat_mul(up_buf, t_up,  norm_buf, n_tokens, fh, n_embd, t_up->dim[0] == n_embd, s->pthreads)) goto fail;

            for (u32 p = 0; p < n_tokens; p++) {
                float *gp = gate_buf + (u64)p * fh;
                float *up = up_buf   + (u64)p * fh;
                silu(gp, fh);
                for (u32 i = 0; i < fh; i++)
                    gp[i] *= up[i];
            }

            if (!mat_mat_mul(norm_buf, t_down,  gate_buf, n_tokens, n_embd, fh, t_down->dim[0] == fh, s->pthreads)) goto fail;
        }

        /* ---- 2h. Residual ---- */
        for (u32 p = 0; p < n_tokens; p++) {
            float *xp = xs + (u64)p * n_embd;
            float *dp = norm_buf + (u64)p * n_embd;
            for (u32 i = 0; i < n_embd; i++)
                xp[i] += dp[i];
        }
    }

    /* ---- 3. Final RMS norm on the LAST position only ---- */
    {
        TensorInfo *t_norm = w->tensors[TENSOR_OUTPUT_NORM];
        if (!t_norm) t_norm = w->layers[n_layer - 1].tensors[TENSOR_POST_ATTN_NORM];
        float *last_x = xs + (u64)(n_tokens - 1) * n_embd;
        rms_norm(ws->xb, last_x, t_norm,  n_embd, eps);
    }

    /* ---- 4. LM head ---- */
    {
        TensorInfo *t_out = w->tensors[TENSOR_OUTPUT];
        if (!t_out) t_out = w->tensors[TENSOR_TOKEN_EMBD];
        float *dst = logits ? logits : s->logits;
        if (!mat_vec_mul(dst, t_out,  ws->xb, c->n_vocab, n_embd, t_out->dim[0] == n_embd, s->pthreads)) goto fail;
    }

    /* ---- 5. Update session state ---- */
    for (u32 p = 0; p < n_tokens; p++)
        s->tokens[cache_start + p] = tokens[p];
    s->n_tokens = cache_start + n_tokens;

    sfree(xs); sfree(norm_buf); sfree(qkv_buf); sfree(qbuf);
    sfree(attn_buf); sfree(gate_buf); sfree(up_buf);
    return true;

fail:
    sfree(xs); sfree(norm_buf); sfree(qkv_buf); sfree(qbuf);
    sfree(attn_buf); sfree(gate_buf); sfree(up_buf);
    return false;
}

static void qwen35_reset(Session *s) {
    KvCache *kc = &s->cache;
    for (u32 i = 0; i < kc->n_layer; i++)
        kc->std[i].n = 0;
    s->n_tokens = 0;

    /* Zero SSM recurrent state and conv state. */
    Qwen35Workspace *ws = (Qwen35Workspace *)s->arch_data;
    if (ws && ws->ssm_state) {
        u64 state_sz = (u64)ws->ssm_groups * ws->ssm_dim * ws->ssm_dim;
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

/* Build a static execution graph for a batch of n_tokens tokens.
 * Mirrors the standard-attention pathway of qwen35_generate /
 * qwen35_prefill: token embedding, per-layer RMS norm, separate Q/K/V
 * projections (+bias, RoPE), causal GQA attention, attention output
 * gating and projection, residuals, SwiGLU FFN, final norm and LM head.
 * OP_ATTN writes the batch's K/V through to the session KV cache and
 * attends causally, so the graph is reusable for any n <= n_tokens.
 *
 * Configurations the graph executor cannot express are rejected up
 * front (build fails; the eager path must be used instead):
 *   - SSM (Gated DeltaNet) layers: causal Conv1d, L2 norms, the gated
 *     delta recurrence and the sigmoid output gate have no graph ops;
 *   - fused QKV: OP_MATMUL yields a single tensor and there is no split;
 *   - Q-norm / K-norm: per-head RMS norm over slices is not a graph op
 *     (and Q-norm implies the fused-QG sigmoid gate);
 *   - gated attention output weight (2*n_embd rows): SiLU-split gate;
 *   - partial RoPE: OP_ROPE_NEOX always rotates the full head, the
 *     eager path uses rope_partial (rope_dim != head_dim differs). */
static Graph *qwen35_graph_build(Session *s, u32 n_tokens) {
    Qwen35Workspace *ws   = (Qwen35Workspace *)s->arch_data;
    ArchConfig      *c    = &s->cfg;
    Weights         *w    = s->en->weights;

    if (!s || !w || !ws || n_tokens == 0 || n_tokens > s->ctx_size) return NULL;

    /* Dims for the matmul direction flags.  Each trans below must mirror
     * the expression used by qwen35_prefill / qwen35_generate for the
     * same weight, otherwise the executor dequantises the wrong axis. */
    u32 n_embd  = c->n_embd;
    u32 q_dim   = c->n_head * c->head_dim;
    u32 fh      = ws->ffn_hidden;
    float theta = ws->rope_theta;

    /* rope_partial degenerates to rope_neox only for a full-width rope. */
    if (c->rope_dim != c->head_dim) {
        slog(WARN, "graph_build: partial RoPE (rope_dim=%u of head_dim=%u) unsupported",
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

        if (lw->tensors[TENSOR_SSM_CONV1D]) {
            slog(WARN, "graph_build: layer %u is an SSM (Gated DeltaNet) layer, unsupported", l);
            goto fail;
        }

        TensorInfo *t_q   = lw->tensors[TENSOR_ATTN_Q];
        TensorInfo *t_k   = lw->tensors[TENSOR_ATTN_K];
        TensorInfo *t_v   = lw->tensors[TENSOR_ATTN_V];
        TensorInfo *t_o   = lw->tensors[TENSOR_ATTN_OUT];
        TensorInfo *t_ag  = lw->tensors[TENSOR_ATTN_GATE];
        TensorInfo *t_g   = lw->tensors[TENSOR_FFN_GATE];
        TensorInfo *t_u   = lw->tensors[TENSOR_FFN_UP];
        TensorInfo *t_d   = lw->tensors[TENSOR_FFN_DOWN];
        if (!t_q || !t_k || !t_v || !t_o || !t_g || !t_u || !t_d) {
            slog(WARN, "graph_build: layer %u missing tensors (fused QKV unsupported)", l);
            goto fail;
        }
        if (lw->tensors[TENSOR_ATTN_QKV]) {
            slog(WARN, "graph_build: layer %u uses fused QKV, unsupported", l);
            goto fail;
        }
        /* Q-norm implies the fused Q+gate projection (Qwen3.5) whose
         * per-head norm and sigmoid gate have no graph ops. */
        if (lw->tensors[TENSOR_ATTN_Q_NORM] || lw->tensors[TENSOR_ATTN_K_NORM]) {
            slog(WARN, "graph_build: layer %u uses per-head Q/K norm, unsupported", l);
            goto fail;
        }
        if (t_o->ndim >= 2 && t_o->dim[0] == 2 * (i64)n_embd && t_o->dim[1] == (i64)q_dim) {
            slog(WARN, "graph_build: layer %u uses gated output weight, unsupported", l);
            goto fail;
        }

        /* ---- Attention block ---- */
        u32 n = graph_rms_norm(g, cur, lw->tensors[TENSOR_ATTN_NORM]);
        if (n == GRAPH_NODE_NONE) goto fail;

        u32 q = graph_mul_mat(g, n, t_q, (q_dim != n_embd) && t_q->dim[0] == (i64)n_embd);
        u32 k = graph_mul_mat(g, n, t_k, t_k->dim[0] == (i64)n_embd);
        u32 v = graph_mul_mat(g, n, t_v, t_v->dim[0] == (i64)n_embd);
        if (q == GRAPH_NODE_NONE || k == GRAPH_NODE_NONE || v == GRAPH_NODE_NONE) goto fail;

        /* Qwen3.5 attention biases (bias=True); skip when absent. */
        if (lw->tensors[TENSOR_ATTN_Q_BIAS]) {
            q = graph_bias(g, q, lw->tensors[TENSOR_ATTN_Q_BIAS]);
            if (q == GRAPH_NODE_NONE) goto fail;
        }
        if (lw->tensors[TENSOR_ATTN_K_BIAS]) {
            k = graph_bias(g, k, lw->tensors[TENSOR_ATTN_K_BIAS]);
            if (k == GRAPH_NODE_NONE) goto fail;
        }
        if (lw->tensors[TENSOR_ATTN_V_BIAS]) {
            v = graph_bias(g, v, lw->tensors[TENSOR_ATTN_V_BIAS]);
            if (v == GRAPH_NODE_NONE) goto fail;
        }

        q = graph_rope(g, q, theta, c->n_head, c->head_dim);
        k = graph_rope(g, k, theta, c->n_kv_head, c->kv_head_dim);
        if (q == GRAPH_NODE_NONE || k == GRAPH_NODE_NONE) goto fail;

        u32 attn = graph_attn(g, q, k, v, l);
        if (attn == GRAPH_NODE_NONE) goto fail;

        /* Qwen3.5 attention output gating: silu(gate @ x) * attn. */
        if (t_ag) {
            u32 gate = graph_silu(g, graph_mul_mat(g, n, t_ag, t_ag->dim[0] == (i64)n_embd));
            if (gate == GRAPH_NODE_NONE) goto fail;
            attn = graph_binary(g, OP_MUL, attn, gate);
            if (attn == GRAPH_NODE_NONE) goto fail;
        }

        u32 o = graph_mul_mat(g, attn, t_o, (n_embd != q_dim) && t_o->dim[0] == (i64)q_dim);
        if (o == GRAPH_NODE_NONE) goto fail;

        u32 h = graph_binary(g, OP_ADD, cur, o);
        if (h == GRAPH_NODE_NONE) goto fail;

        /* ---- SwiGLU FFN ---- */
        u32 hn = graph_rms_norm(g, h, lw->tensors[TENSOR_POST_ATTN_NORM]);
        if (hn == GRAPH_NODE_NONE) goto fail;

        u32 gate = graph_silu(g, graph_mul_mat(g, hn, t_g, t_g->dim[0] == (i64)n_embd));
        u32 up   = graph_mul_mat(g, hn, t_u, t_u->dim[0] == (i64)n_embd);
        if (gate == GRAPH_NODE_NONE || up == GRAPH_NODE_NONE) goto fail;

        u32 mul = graph_binary(g, OP_MUL, gate, up);
        if (mul == GRAPH_NODE_NONE) goto fail;

        u32 down = graph_mul_mat(g, mul, t_d, t_d->dim[0] == (i64)fh);
        if (down == GRAPH_NODE_NONE) goto fail;

        cur = graph_binary(g, OP_ADD, h, down);
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
    .init        = qwen35_init,
    .free        = qwen35_free,
    .prefill     = qwen35_prefill,
    .generate    = qwen35_generate,
    .reset       = qwen35_reset,
    .decode      = qwen35_decode,
    .graph_build = qwen35_graph_build,
};
