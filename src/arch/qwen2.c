#include <math.h>
#include <stdio.h>
#include <string.h>
#include "arch.h"
#include "../core.h"
#include "../mm.h"
#include "../slog.h"
#include "../utils.h"


typedef struct {
    float *x;          /* [n_embd] hidden state                      */
    float *xb;         /* [n_embd] residual / scratch                */
    float *xb2;        /* [n_embd] second scratch                    */
    float *q;          /* [n_head * head_dim] query buffer           */
    float *k;          /* [n_kv_head * head_dim] key buffer          */
    float *v;          /* [n_kv_head * head_dim] value buffer        */
    float *qkv_fused;  /* [q_dim + 2*kv_dim] fused QKV buffer        */
    float *scores;     /* [ctx_size] attention scores (per head)     */
    float *hb;         /* [ffn_hidden] FFN hidden buffer             */
    float *hb2;        /* [ffn_hidden] FFN hidden buffer 2           */
    u32    ffn_hidden;

    /* SSM (Mamba) state — per-layer, persists across forward calls. */
    float *conv_state; /* [3 * conv_channels] last 3 inputs for conv1d */
    float *ssm_state;  /* [d_inner * d_state] SSM hidden state         */
    u32    conv_channels; /* total channels for conv1d (fused projection) */
    u32    d_inner;       /* SSM inner dimension                        */
    u32    d_state;       /* SSM state dimension                        */
    bool   is_ssm;        /* true if this layer has SSM tensors         */
    float  rope_theta;    /* RoPE frequency base                        */
} Qwen2Workspace;


/* Read a single f32/f16/bf16 weight from a GGUF tensor at index i.
 * Logs a warning and returns 0 for unsupported quantisation types. */
static inline float tensor_get_f32(TensorInfo *ti, const u8 *base, u64 i) {
    /* Bounds check: catch OOB access with diagnostic info. */
    if (i >= ti->n_element) {
        char name[128];
        snprintf(name, sizeof(name), "%.*s",
                 ti->key.len < 127 ? ti->key.len : 127, ti->key.content);
        slog(WARN, "tensor_get_f32 OOB: tensor='%s' n_element=%llu "
             "index=%llu type=%u ndim=%u",
             name, (unsigned long long)ti->n_element,
             (unsigned long long)i, ti->type, ti->ndim);
        if (ti->ndim > 0) {
            slog(WARN, "  dims: [%llu%s%s",
                 (unsigned long long)ti->dim[0],
                 ti->ndim > 1 ? "," : "]",
                 ti->ndim > 1 ? "" : "");
            for (u32 d = 1; d < ti->ndim && d < 8; d++)
                slog(WARN, "         %llu%s",
                     (unsigned long long)ti->dim[d],
                     d + 1 < ti->ndim ? "," : "]");
        }
        slog(ERROR, "Fatal: out-of-bounds tensor access: %ld >= %ld", i, ti->n_element);
    }
    switch (ti->type) {
        case GGUF_TYPE_F32: {
            const float *f = (const float *)(base + ti->offset);
            return f[i];
        }
        case GGUF_TYPE_F16: { /* IEEE 754 binary16 → float */
            const u16 *h = (const u16 *)(base + ti->offset);
            u16 bits = h[i];
            u32 sign = (u32)(bits >> 15) << 31;
            u32 exp  = (bits >> 10) & 0x1f;
            u32 mant = bits & 0x3ff;
            u32 f32;
            if (exp == 0) {
                if (mant == 0) { f32 = sign; }
                else {
                    /* subnormal normalisation */
                    int e = 1 - 15;
                    while ((mant & 0x400) == 0) { mant <<= 1; e--; }
                    mant &= 0x3ff;
                    f32 = sign | ((u32)(e + 127) << 23) | (mant << 13);
                }
            } else if (exp == 31) {
                f32 = sign | (0xff << 23) | (mant << 13); /* inf / nan */
            } else {
                f32 = sign | ((u32)(exp - 15 + 127) << 23) | (mant << 13);
            }
            float out;
            memcpy(&out, &f32, sizeof(out));
            return out;
        }
        case GGUF_TYPE_BF16: {
            const u16 *b = (const u16 *)(base + ti->offset);
            u32 f32 = (u32)b[i] << 16;
            float out;
            memcpy(&out, &f32, sizeof(out));
            return out;
        }
        default:
            slog(WARN, "Unsupported tensor type %u — use f32/f16/bf16 weights",
                 ti->type);
            return 0.0f;
    }
}

/* ---- Math primitives ------------------------------------------- */

/* RMS Normalisation: o = x / rms(x) * w  (in-place ok when o == x). */
static void rms_norm(float *o, const float *x,
                     TensorInfo *tw, const u8 *base,
                     int n, float eps) {
    u64 tn = tw->dim[0]; /* 1-D weight tensor */
    if ((u64)n > tn) {
        slog(WARN, "rms_norm: n=%d > tensor dim=%llu, clamping", n,
             (unsigned long long)tn);
        n = (int)tn;
    }
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float scale = 1.0f / sqrtf(ss / (float)n + eps);
    for (int i = 0; i < n; i++) {
        float w = tensor_get_f32(tw, base, (u64)i);
        o[i] = x[i] * scale * w;
    }
}

/* Matrix-vector multiply: y = W @ x  (W is [rows × cols], stored row-major).
 * cols is derived from the tensor's last dimension; rows must match dim[0].
 *
 * When trans=true, the weight is treated as transposed: y = W^T @ x.
 * In that case dim[0] is the input dimension and dim[ndim-1] is the output. */
static bool mat_vec_mul(float *y, TensorInfo *tw, const u8 *base,
                        const float *x, u64 rows, u64 cols, bool trans) {
    if (!tw || tw->ndim < 2) {
        slog(WARN, "mat_vec_mul: tensor missing or ndim < 2");
        return false;
    }
    u64 tc = tw->dim[tw->ndim - 1]; /* fastest-varying = column count */
    u64 tr = tw->dim[0];            /* slowest-varying = row count     */

    if (trans) {
        /* Transposed: W is stored as [cols × rows], we compute y = W^T @ x.
         * tr (=dim[0]) must equal cols (input dim), tc (=dim[1]) must equal rows (output dim). */
        if (tr != cols || tc != rows) {
            slog(WARN, "mat_vec_mul trans: dim mismatch cfg=[%lu,%lu] tensor^T=[%lu,%lu] for tensor=%s",
                 (unsigned long)rows, (unsigned long)cols,
                 (unsigned long)tc,   (unsigned long)tr,
                 get_key_name(tw->key));
            return false;
        }
        if (rows == 0 || cols == 0) {
            slog(WARN, "mat_vec_mul trans: zero-dim rows=%llu cols=%llu",
                 (unsigned long long)rows, (unsigned long long)cols);
            return false;
        }
        /* y[r] = sum_c W_stored[c][r] * x[c] = sum_c data[c * tc + r] * x[c] */
        for (u64 r = 0; r < rows; r++) {
            float sum = 0.0f;
            for (u64 c = 0; c < cols; c++)
                sum += tensor_get_f32(tw, base, c * tc + r) * x[c];
            y[r] = sum;
        }
        return true;
    }

    /* Standard: y = W @ x */
    if (tr != rows || tc != cols) {
        slog(WARN, "Bad GGUF file: dim mismatch cfg=[%lu,%lu] tensor=[%lu,%lu] for tensor=%s",
             (unsigned long)rows, (unsigned long)cols,
             (unsigned long)tr,   (unsigned long)tc,
             get_key_name(tw->key));
        return false;
    }
    /* Validate that the loop won't access beyond tensor bounds. */
    if (rows == 0 || cols == 0) {
        slog(WARN, "mat_vec_mul: zero-dim tensor rows=%llu cols=%llu",
             (unsigned long long)rows, (unsigned long long)cols);
        return false;
    }
    if (rows > 0 && cols > 0) {
        u64 last_idx = (rows - 1) * cols + (cols - 1);
        if (last_idx >= tw->n_element || cols * rows > tw->n_element) {
            char name[128];
            snprintf(name, sizeof(name), "%.*s",
                     tw->key.len < 127 ? tw->key.len : 127, tw->key.content);
            slog(WARN, "mat_vec_mul: OOB predict tensor='%s' n_el=%llu "
                 "rows=%llu cols=%llu last_idx=%llu",
                 name, (unsigned long long)tw->n_element,
                 (unsigned long long)rows, (unsigned long long)cols,
                 (unsigned long long)last_idx);
            slog(ERROR, "Fatal: mat_vec_mul would access tensor out of bounds");
        }
    }
    for (u64 r = 0; r < rows; r++) {
        float sum = 0.0f;
        for (u64 c = 0; c < cols; c++)
            sum += tensor_get_f32(tw, base, r * cols + c) * x[c];
        y[r] = sum;
    }
    return true;
}

/* RoPE: apply rotary position embedding in-place.
 * buf is [n_heads × head_dim], each head rotated independently. */
static void rope(float *buf, u32 n_heads, u32 head_dim,
                 u32 pos, float theta_base) {
    for (u32 h = 0; h < n_heads; h++) {
        float *bh = buf + h * head_dim;
        for (u32 d = 0; d + 1 < head_dim; d += 2) {
            float theta = 1.0f / powf(theta_base, (float)d / (float)head_dim);
            float c = cosf((float)pos * theta);
            float s = sinf((float)pos * theta);
            float a = bh[d], b = bh[d + 1];
            bh[d]     = a * c - b * s;
            bh[d + 1] = a * s + b * c;
        }
    }
}

/* SiLU activation in-place. */
static void silu(float *x, int n) {
    for (int i = 0; i < n; i++)
        x[i] = x[i] / (1.0f + expf(-x[i]));
}

/* ---- Arch ops --------------------------------------------------- */

static bool qwen2_init(Session *s) {
    ArchConfig *c = &s->cfg;

    /* Use separate head dims: Q may differ from K/V (e.g. Qwen3.5). */
    u32 q_head_dim  = c->head_dim;
    u32 kv_head_dim = c->kv_head_dim;
    u32 q_dim       = c->n_head * q_head_dim;
    u32 kv_dim      = c->n_kv_head * kv_head_dim;

    /* Allocate standard KV cache. */
    KvCache *kc   = &s->cache;
    kc->n_layer   = c->n_layer;
    kc->head_dim  = kv_head_dim;  /* cache stores full K/V per head */
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

    /* Detect SSM presence and dimensions from layer 0. */
    Weights *w = s->en->weights;
    bool has_ssm = (w->layers[0].tensors[TENSOR_SSM_CONV1D] != NULL);

    /* Compute max fused QKV / separate Q output dimensions from first layer. */
    u32 max_fused = q_dim + 2 * kv_dim; /* default for non-SSM */
    u32 ssm_channels = 0;
    u32 d_inner = 0;
    u32 d_state = 0;
    if (has_ssm) {
        TensorInfo *t_qkv = w->layers[0].tensors[TENSOR_ATTN_QKV];
        if (t_qkv && t_qkv->ndim >= 2) {
            u64 total = t_qkv->dim[0] > t_qkv->dim[1]
                        ? (u32)t_qkv->dim[0] : (u32)t_qkv->dim[1];
            max_fused = (u32)total;
            ssm_channels = (u32)total;
            /* d_inner = (total_channels - attention_channels) / 2
             * attention = q_dim + 2*kv_dim = n_head*q_head_dim + 2*n_kv_head*kv_head_dim */
            u32 attn_ch = q_dim + 2 * kv_dim;
            d_inner = (max_fused > attn_ch) ? (max_fused - attn_ch) / 2 : 0;
        }
        /* d_state from ssm_a tensor. */
        TensorInfo *t_ssm_a = w->layers[0].tensors[TENSOR_SSM_A];
        if (t_ssm_a && t_ssm_a->ndim >= 1)
            d_state = (u32)t_ssm_a->dim[0];
        if (d_state == 0) d_state = 16; /* fallback */
    }

    /* Allocate workspace buffers. */
    Qwen2Workspace *ws = scalloc(1, sizeof(Qwen2Workspace));
    ws->x      = scalloc((u64)c->n_embd, sizeof(float));
    ws->xb     = scalloc((u64)c->n_embd, sizeof(float));
    ws->xb2    = scalloc((u64)c->n_embd, sizeof(float));
    ws->q      = scalloc((u64)q_dim, sizeof(float));
    ws->k      = scalloc((u64)kv_dim, sizeof(float));
    ws->v      = scalloc((u64)kv_dim, sizeof(float));
    ws->qkv_fused = scalloc((u64)max_fused, sizeof(float));
    ws->scores = scalloc((u64)s->ctx_size, sizeof(float));

    /* Derive FFN hidden dim from layer 0 ffn_gate tensor. */
    TensorInfo *ffn_gate = w->layers[0].tensors[TENSOR_FFN_GATE];
    ws->ffn_hidden = (ffn_gate && ffn_gate->ndim >= 1)
                     ? (ffn_gate->dim[0] > ffn_gate->dim[1]
                        ? (u32)ffn_gate->dim[0]
                        : (u32)(ffn_gate->ndim >= 2 ? ffn_gate->dim[1] : ffn_gate->dim[0]))
                     : c->n_embd * 4;
    ws->hb  = scalloc((u64)ws->ffn_hidden, sizeof(float));
    ws->hb2 = scalloc((u64)ws->ffn_hidden, sizeof(float));

    /* Read rope theta from metadata (default 1e6 for Qwen2, 1e7 for Qwen3.5). */
    ws->rope_theta = 1000000.0f;
    {
        char k[96];
        const char *pfx = s->en->model->arch_name[0]
                          ? s->en->model->arch_name : "qwen2";
        if (snprintf(k, sizeof(k), "%s.rope.freq_base", pfx) > 0)
            model_get_f32(s->en->model, k, &ws->rope_theta);
    }

    /* SSM state allocation. */
    ws->is_ssm    = has_ssm;
    ws->d_inner   = d_inner;
    ws->d_state   = d_state;
    ws->conv_channels = ssm_channels;
    if (has_ssm && ssm_channels > 0 && d_state > 0) {
        /* conv_state: [n_layer * 3 * conv_channels] — last 3 inputs for depthwise conv1d */
        ws->conv_state = scalloc((u64)c->n_layer * 3 * (u64)ssm_channels, sizeof(float));
        /* ssm_state: [n_layer * d_inner * d_state] — SSM hidden state */
        ws->ssm_state  = scalloc((u64)c->n_layer * (u64)d_inner * (u64)d_state, sizeof(float));
    }

    /* Diagnostic. */
    slog(INFO, "Qwen2 init: n_embd=%u n_head=%u n_kv_head=%u head_dim=%u kv_head_dim=%u "
         "n_layer=%u n_vocab=%u ctx_size=%u",
         c->n_embd, c->n_head, c->n_kv_head, q_head_dim, kv_head_dim,
         c->n_layer, c->n_vocab, s->ctx_size);
    slog(INFO, "Qwen2 init: q_dim=%u kv_dim=%u ffn_hidden=%u max_fused=%u ssm=%d",
         q_dim, kv_dim, ws->ffn_hidden, max_fused, has_ssm);
    if (has_ssm) slog(INFO, "Qwen2 init SSM: channels=%u d_inner=%u d_state=%u",
                      ssm_channels, d_inner, d_state);
    {
        LayerWeights *lw0 = &w->layers[0];
        TensorInfo *t = lw0->tensors[TENSOR_ATTN_QKV];
        if (t) slog(INFO, "Layer 0: fused QKV dim[0]=%llu dim[1]=%llu ndim=%u",
                    (unsigned long long)t->dim[0],
                    (unsigned long long)(t->ndim >= 2 ? t->dim[1] : 0),
                    t->ndim);
        t = lw0->tensors[TENSOR_ATTN_Q];
        if (t) slog(INFO, "Layer 0: separate Q dim[0]=%llu dim[1]=%llu",
                    (unsigned long long)t->dim[0],
                    (unsigned long long)(t->ndim >= 2 ? t->dim[1] : 0));
        t = lw0->tensors[TENSOR_FFN_GATE];
        if (t) slog(INFO, "Layer 0: ffn_gate dim[0]=%llu dim[1]=%llu",
                    (unsigned long long)t->dim[0],
                    (unsigned long long)(t->ndim >= 2 ? t->dim[1] : 0));
    }

    s->arch_data = ws;
    return true;
}

static bool qwen2_forward(Session *s, u32 token, float *logits) {
    Qwen2Workspace *ws = (Qwen2Workspace *)s->arch_data;
    ArchConfig *c  = &s->cfg;
    Weights    *w  = s->en->weights;
    KvCache    *kc = &s->cache;
    const u8   *base = s->en->model->map;

    const u32 n_embd    = c->n_embd;
    const u32 n_head    = c->n_head;
    const u32 n_kv_head = c->n_kv_head;
    const u32 q_head_dim  = c->head_dim;
    const u32 kv_head_dim = c->kv_head_dim;
    const u32 n_vocab   = c->n_vocab;
    const u32 n_layer   = c->n_layer;
    const u32 kv_dim    = n_kv_head * kv_head_dim;
    const u32 q_dim     = n_head * q_head_dim;
    const u32 pos       = s->n_tokens;
    const u32 n_rep     = n_head / n_kv_head;

    /* Validate critical config values. */
    if (n_embd == 0 || n_head == 0 || n_kv_head == 0
        || q_head_dim == 0 || kv_head_dim == 0
        || n_vocab == 0 || n_layer == 0) {
        slog(WARN, "qwen2_forward: invalid config n_embd=%u n_head=%u "
             "n_kv_head=%u q_head_dim=%u kv_head_dim=%u n_vocab=%u n_layer=%u",
             n_embd, n_head, n_kv_head, q_head_dim, kv_head_dim, n_vocab, n_layer);
        return false;
    }

    const u32  qk_score_dim = q_head_dim < kv_head_dim ? q_head_dim : kv_head_dim;
    const float sqrt_d       = sqrtf((float)qk_score_dim);
    const float eps          = 1e-6f;
    const float rope_theta   = ws->rope_theta;

    float *x      = ws->x;
    float *xb     = ws->xb;
    float *xb2    = ws->xb2;
    float *q_buf  = ws->q;
    float *k_buf  = ws->k;
    float *v_buf  = ws->v;
    float *scores = ws->scores;
    u32    d_inner  = ws->d_inner;
    u32    d_state  = ws->d_state;
    u32    ssm_ch   = ws->conv_channels;

    /* ---- 1. Token embedding ---- */
    {
        TensorInfo *te = w->tensors[TENSOR_TOKEN_EMBD];
        if (!te) { slog(WARN, "token_embd tensor not found"); return false; }
        /* Handle both storage conventions: [n_vocab × n_embd] and [n_embd × n_vocab]. */
        if (te->ndim >= 2 && te->dim[0] == (u64)n_embd) {
            /* [n_embd, n_vocab]: x[i] = W[i][token] */
            for (u32 i = 0; i < n_embd; i++)
                x[i] = tensor_get_f32(te, base, (u64)i * (u64)n_vocab + (u64)token);
        } else {
            /* [n_vocab, n_embd]: x[i] = W[token][i] */
            for (u32 i = 0; i < n_embd; i++)
                x[i] = tensor_get_f32(te, base, (u64)token * (u64)n_embd + (u64)i);
        }
    }

    s->tokens[pos % (u32)s->ctx_size] = token;

    /* ---- 2. Transformer layers ---- */
    for (u32 l = 0; l < n_layer; l++) {
        LayerWeights *lw  = &w->layers[l];
        AttnKvCache  *akc = &kc->std[l];

        bool is_ssm        = (lw->tensors[TENSOR_SSM_CONV1D] != NULL);
        bool has_fused_qkv = (lw->tensors[TENSOR_ATTN_QKV] != NULL);

        /* -- 2a. RMS Norm (pre-attention) -- */
        memcpy(xb, x, n_embd * sizeof(float));
        {
            TensorInfo *tn = lw->tensors[TENSOR_ATTN_NORM];
            if (!tn) { slog(WARN, "attn_norm missing layer %u", l); return false; }
            rms_norm(xb2, x, tn, base, (int)n_embd, eps);
        }

        /* -- 2b. Q, K, V projections -- */
        if (has_fused_qkv) {
            /* ---- Fused QKV path (SSM layers or standard fused QKV) ---- */
            TensorInfo *t_qkv = lw->tensors[TENSOR_ATTN_QKV];
            bool trans = (t_qkv->dim[0] == (u64)n_embd);
            u32 total_out = (u32)(trans ? t_qkv->dim[1] : t_qkv->dim[0]);

            /* Compute fused projection. */
            if (!mat_vec_mul(ws->qkv_fused, t_qkv, base, xb2,
                             trans ? (u64)t_qkv->dim[1] : (u64)t_qkv->dim[0],
                             trans ? (u64)n_embd : (u64)(t_qkv->ndim >= 2 ? t_qkv->dim[1] : 0),
                             trans)) return false;

            if (is_ssm) {
                /* SSM hybrid layer: conv1d+SiLU on all channels, then split. */
                u32 conv_ch = ssm_ch > 0 ? ssm_ch : total_out;
                float *conv_state_l = ws->conv_state
                    + (u64)l * 3 * (u64)conv_ch;
                float *ssm_state_l  = ws->ssm_state
                    + (u64)l * (u64)d_inner * (u64)d_state;

                /* Depthwise conv1d + SiLU on the fused output (per-channel, in-place). */
                TensorInfo *t_conv = lw->tensors[TENSOR_SSM_CONV1D];
                if (t_conv) {
                    for (u32 ch = 0; ch < conv_ch; ch++) {
                        float cur = ws->qkv_fused[ch]; /* save before overwriting */
                        float sum = tensor_get_f32(t_conv, base,
                                                   (u64)0 * (u64)conv_ch + (u64)ch)
                                   * conv_state_l[0 * conv_ch + ch];
                        for (u32 k = 1; k < 3; k++)
                            sum += tensor_get_f32(t_conv, base,
                                                  (u64)k * (u64)conv_ch + (u64)ch)
                                   * conv_state_l[k * conv_ch + ch];
                        sum += tensor_get_f32(t_conv, base,
                                              (u64)3 * (u64)conv_ch + (u64)ch)
                               * cur;
                        ws->qkv_fused[ch] = sum / (1.0f + expf(-sum)); /* silu */
                        /* Shift conv state per channel. */
                        conv_state_l[0 * conv_ch + ch] = conv_state_l[1 * conv_ch + ch];
                        conv_state_l[1 * conv_ch + ch] = conv_state_l[2 * conv_ch + ch];
                        conv_state_l[2 * conv_ch + ch] = cur;
                    }
                }

                /* Split: Q[0:q_dim], K[q_dim:q_dim+kv_dim], V[q_dim+kv_dim:q_dim+2*kv_dim],
                 *       x[q_dim+2*kv_dim : q_dim+2*kv_dim+d_inner],
                 *       z[q_dim+2*kv_dim+d_inner : q_dim+2*kv_dim+2*d_inner] */
                u32 attn_total = q_dim + 2 * kv_dim;
                memcpy(q_buf, ws->qkv_fused, (u64)q_dim * sizeof(float));
                memcpy(k_buf, ws->qkv_fused + q_dim, (u64)kv_dim * sizeof(float));
                memcpy(v_buf, ws->qkv_fused + q_dim + kv_dim, (u64)kv_dim * sizeof(float));

                float *x_ssm = ws->qkv_fused + attn_total;
                float *z_ssm = ws->qkv_fused + attn_total + d_inner;

                /* -- RoPE -- */
                rope(q_buf, n_head, q_head_dim, pos, rope_theta);
                rope(k_buf, n_kv_head, kv_head_dim, pos, rope_theta);

                /* -- Store K/V into cache -- */
                {
                    float *kd = akc->k + (u64)pos * kv_dim;
                    float *vd = akc->v + (u64)pos * kv_dim;
                    memcpy(kd, k_buf, kv_dim * sizeof(float));
                    memcpy(vd, v_buf, kv_dim * sizeof(float));
                    akc->n = pos + 1;
                }

                /* -- Attention (GQA, causal) -- */
                u32 attn_out_dim = n_head * kv_head_dim;
                {
                    u32 n_kv = pos + 1;
                    float *attn_out = ws->hb; /* reuse hb (size ffn_hidden >= attn_out_dim) */
                    memset(attn_out, 0, (u64)attn_out_dim * sizeof(float));

                    for (u32 h = 0; h < n_head; h++) {
                        u32 kv_h  = h / n_rep;
                        float *qh = q_buf + h * q_head_dim;

                        float max_s = -INFINITY;
                        for (u32 t = 0; t < n_kv; t++) {
                            const float *kt = akc->k + (u64)t * kv_dim
                                              + kv_h * kv_head_dim;
                            float s = 0.0f;
                            for (u32 d = 0; d < qk_score_dim; d++)
                                s += qh[d] * kt[d];
                            s /= sqrt_d;
                            scores[t] = s;
                            if (s > max_s) max_s = s;
                        }
                        float sum = 0.0f;
                        for (u32 t = 0; t < n_kv; t++) {
                            scores[t] = expf(scores[t] - max_s);
                            sum += scores[t];
                        }
                        for (u32 t = 0; t < n_kv; t++)
                            scores[t] /= sum;

                        float *out_h = attn_out + h * kv_head_dim;
                        for (u32 t = 0; t < n_kv; t++) {
                            float a = scores[t];
                            const float *vt = akc->v + (u64)t * kv_dim
                                              + kv_h * kv_head_dim;
                            for (u32 d = 0; d < kv_head_dim; d++)
                                out_h[d] += a * vt[d];
                        }
                    }

                    /* -- Gate attention output -- */
                    TensorInfo *t_gate = lw->tensors[TENSOR_ATTN_GATE];
                    if (t_gate && d_inner > 0) {
                        bool gate_trans = (t_gate->dim[0] == (u64)n_embd);
                        float *gate = ws->hb2; /* reuse hb2 for gate [d_inner] */
                        if (!mat_vec_mul(gate, t_gate, base, xb2,
                                         gate_trans ? (u64)t_gate->dim[1] : (u64)t_gate->dim[0],
                                         gate_trans ? (u64)n_embd
                                                    : (u64)(t_gate->ndim >= 2 ? t_gate->dim[1] : 0),
                                         gate_trans)) return false;
                        for (u32 i = 0; i < d_inner && i < attn_out_dim; i++) {
                            gate[i] = gate[i] / (1.0f + expf(-gate[i])); /* silu */
                            attn_out[i] *= gate[i];
                        }
                    }

                    /* -- SSM forward -- */
                    float *ssm_hidden = ws->hb2; /* reuse hb2 after gate (size ffn_hidden >= d_inner) */
                    memset(ssm_hidden, 0, (u64)d_inner * sizeof(float));

                    if (d_inner > 0 && d_state > 0) {
                        /* dt = softplus(ssm_alpha @ xb2 + ssm_dt_bias) */
                        TensorInfo *t_alpha = lw->tensors[TENSOR_SSM_ALPHA];
                        TensorInfo *t_dt_bias = lw->tensors[TENSOR_SSM_DT_BIAS];
                        float dt_buf[32]; /* d_state <= 32 */
                        memset(dt_buf, 0, sizeof(dt_buf));
                        if (t_alpha) {
                            bool at = (t_alpha->dim[0] == (u64)n_embd);
                            u32 a_rows = (u32)(at ? t_alpha->dim[1] : t_alpha->dim[0]);
                            u32 a_cols = (u32)(at ? n_embd : (t_alpha->ndim >= 2 ? t_alpha->dim[1] : 0));
                            if (a_rows <= 32 && a_cols == n_embd) {
                                /* Compute manually — small projection. */
                                for (u32 r = 0; r < a_rows && r < d_state; r++) {
                                    float s_dt = 0.0f;
                                    for (u32 c_ = 0; c_ < n_embd; c_++)
                                        s_dt += tensor_get_f32(t_alpha, base,
                                            at ? (u64)c_ * (u64)t_alpha->dim[1] + (u64)r
                                               : (u64)r * (u64)a_cols + (u64)c_) * xb2[c_];
                                    if (t_dt_bias && r < (u32)t_dt_bias->n_element)
                                        s_dt += tensor_get_f32(t_dt_bias, base, (u64)r);
                                    if (s_dt < 20.0f)
                                        dt_buf[r] = logf(1.0f + expf(s_dt));
                                    else
                                        dt_buf[r] = s_dt;
                                }
                            }
                        }

                        /* B = ssm_beta @ xb2 */
                        TensorInfo *t_beta = lw->tensors[TENSOR_SSM_BETA];
                        float B_buf[32] = {0};
                        if (t_beta) {
                            bool bt = (t_beta->dim[0] == (u64)n_embd);
                            u32 b_rows = (u32)(bt ? t_beta->dim[1] : t_beta->dim[0]);
                            u32 b_cols = (u32)(bt ? n_embd : (t_beta->ndim >= 2 ? t_beta->dim[1] : 0));
                            if (b_rows <= 32) {
                                for (u32 r = 0; r < b_rows; r++) {
                                    float sb = 0.0f;
                                    for (u32 c_ = 0; c_ < n_embd && c_ < b_cols; c_++)
                                        sb += tensor_get_f32(t_beta, base,
                                            bt ? (u64)c_ * (u64)t_beta->dim[1] + (u64)r
                                               : (u64)r * (u64)b_cols + (u64)c_) * xb2[c_];
                                    B_buf[r] = sb;
                                }
                            }
                        }

                        /* Selective scan: h_new[i] = dA * h[i] + B * x_ssm[i].
                         * Results accumulated in ssm_hidden (hb2). */
                        for (u32 i = 0; i < d_inner; i++) {
                            float xi = x_ssm[i];
                            float *h_i = ssm_state_l + (u64)i * (u64)d_state;
                            float s = 0.0f;
                            for (u32 j = 0; j < d_state; j++) {
                                float A_j = -expf(tensor_get_f32(
                                    lw->tensors[TENSOR_SSM_A], base, (u64)j));
                                float dt_j = dt_buf[j];
                                float dA = expf(A_j * dt_j);
                                float dB = B_buf[j] * xi;
                                h_i[j] = dA * h_i[j] + dB;
                                s += h_i[j];
                            }
                            ssm_hidden[i] = s;
                        }

                        /* z gate + combine with attention output. */
                        for (u32 i = 0; i < d_inner; i++) {
                            ssm_hidden[i] *= z_ssm[i] / (1.0f + expf(-z_ssm[i]));
                            if (i < attn_out_dim)
                                ssm_hidden[i] += ws->hb[i]; /* add gated attn output */
                        }

                        /* SSM output projection writes directly to x (n_embd). */
                        TensorInfo *t_ssm_out = lw->tensors[TENSOR_SSM_OUT];
                        if (t_ssm_out) {
                            bool so_trans = (t_ssm_out->dim[0] == (u64)d_inner);
                            if (!mat_vec_mul(x, t_ssm_out, base, ssm_hidden,
                                             so_trans ? (u64)t_ssm_out->dim[1] : (u64)t_ssm_out->dim[0],
                                             so_trans ? (u64)d_inner
                                                      : (u64)(t_ssm_out->ndim >= 2 ? t_ssm_out->dim[1] : 0),
                                             so_trans)) return false;
                        }
                    }
                }

            } else {
                /* Standard fused QKV (non-SSM). */
                memcpy(q_buf, ws->qkv_fused, (u64)q_dim * sizeof(float));
                memcpy(k_buf, ws->qkv_fused + q_dim, (u64)kv_dim * sizeof(float));
                memcpy(v_buf, ws->qkv_fused + q_dim + kv_dim, (u64)kv_dim * sizeof(float));

                /* Optional Q/K LayerNorm (per-head). */
                {
                    TensorInfo *t_qn = lw->tensors[TENSOR_ATTN_Q_NORM];
                    if (t_qn) {
                        u32 norm_dim = (u32)t_qn->dim[0];
                        for (u32 h = 0; h < n_head; h++)
                            rms_norm(q_buf + h * q_head_dim,
                                     q_buf + h * q_head_dim,
                                     t_qn, base,
                                     (int)(norm_dim < q_head_dim ? norm_dim : q_head_dim), eps);
                    }
                }
                {
                    TensorInfo *t_kn = lw->tensors[TENSOR_ATTN_K_NORM];
                    if (t_kn) {
                        u32 norm_dim = (u32)t_kn->dim[0];
                        for (u32 h = 0; h < n_kv_head; h++)
                            rms_norm(k_buf + h * kv_head_dim,
                                     k_buf + h * kv_head_dim,
                                     t_kn, base,
                                     (int)(norm_dim < kv_head_dim ? norm_dim : kv_head_dim), eps);
                    }
                }

                /* RoPE */
                {
                    float *kd = akc->k + (u64)pos * kv_dim;
                    float *vd = akc->v + (u64)pos * kv_dim;
                    memcpy(kd, k_buf, kv_dim * sizeof(float));
                    memcpy(vd, v_buf, kv_dim * sizeof(float));
                    akc->n = pos + 1;
                }

                /* Attention (GQA) */
                u32 attn_out_dim = n_head * kv_head_dim;
                u32 n_kv = pos + 1;
                memset(xb2, 0, (u64)attn_out_dim * sizeof(float));
                for (u32 h = 0; h < n_head; h++) {
                    u32 kv_h  = h / n_rep;
                    float *qh = q_buf + h * q_head_dim;
                    float max_s = -INFINITY;
                    for (u32 t = 0; t < n_kv; t++) {
                        const float *kt = akc->k + (u64)t * kv_dim + kv_h * kv_head_dim;
                        float s = 0.0f;
                        for (u32 d = 0; d < qk_score_dim; d++) s += qh[d] * kt[d];
                        s /= sqrt_d;
                        scores[t] = s;
                        if (s > max_s) max_s = s;
                    }
                    float sum = 0.0f;
                    for (u32 t = 0; t < n_kv; t++) {
                        scores[t] = expf(scores[t] - max_s);
                        sum += scores[t];
                    }
                    for (u32 t = 0; t < n_kv; t++) scores[t] /= sum;
                    float *out_h = xb2 + h * kv_head_dim;
                    for (u32 t = 0; t < n_kv; t++) {
                        float a = scores[t];
                        const float *vt = akc->v + (u64)t * kv_dim + kv_h * kv_head_dim;
                        for (u32 d = 0; d < kv_head_dim; d++) out_h[d] += a * vt[d];
                    }
                }

                /* Output projection */
                TensorInfo *t_out = lw->tensors[TENSOR_ATTN_OUT];
                if (t_out) {
                    bool out_trans = (t_out->dim[0] == (u64)attn_out_dim);
                    if (!mat_vec_mul(x, t_out, base, xb2,
                                     out_trans ? (u64)t_out->dim[1] : (u64)t_out->dim[0],
                                     out_trans ? (u64)attn_out_dim
                                               : (u64)(t_out->ndim >= 2 ? t_out->dim[1] : 0),
                                     out_trans)) return false;
                } else {
                    /* No output projection — use residual as passthrough.
                     * x already holds residual in xb. */
                }

                /* Optional attention gate. */
                if (lw->tensors[TENSOR_ATTN_GATE]) {
                    u64 gate_n = lw->tensors[TENSOR_ATTN_GATE]->n_element;
                    for (u32 i = 0; i < n_embd && (u64)i < gate_n; i++)
                        x[i] *= tensor_get_f32(lw->tensors[TENSOR_ATTN_GATE], base, (u64)i);
                }
            }

        } else {
            /* ---- Separate Q/K/V path (full attention layers) ---- */
            TensorInfo *tq = lw->tensors[TENSOR_ATTN_Q];
            TensorInfo *tk = lw->tensors[TENSOR_ATTN_K];
            TensorInfo *tv = lw->tensors[TENSOR_ATTN_V];
            if (!tq || !tk || !tv) {
                slog(WARN, "attn Q/K/V missing layer %u", l);
                return false;
            }

            /* Detect convention: if dim[0] == n_embd, weights are transposed. */
            bool q_trans = (tq->dim[0] == (u64)n_embd);
            bool k_trans = (tk->dim[0] == (u64)n_embd);
            bool v_trans = (tv->dim[0] == (u64)n_embd);

            /* Use actual tensor output dims (may differ from config-derived q_dim/kv_dim). */
            u32 eff_q_dim  = (u32)(q_trans ? tq->dim[1] : tq->dim[0]);
            u32 eff_kv_dim = (u32)(k_trans ? tk->dim[1] : tk->dim[0]);
            u32 eff_q_head_dim  = eff_q_dim / n_head;
            u32 eff_kv_head_dim_local = eff_kv_dim / n_kv_head;

            if (!mat_vec_mul(q_buf, tq, base, xb2,
                             q_trans ? (u64)tq->dim[1] : (u64)tq->dim[0],
                             q_trans ? (u64)n_embd : (u64)(tq->ndim >= 2 ? tq->dim[1] : 0),
                             q_trans)) return false;
            if (!mat_vec_mul(k_buf, tk, base, xb2,
                             k_trans ? (u64)tk->dim[1] : (u64)tk->dim[0],
                             k_trans ? (u64)n_embd : (u64)(tk->ndim >= 2 ? tk->dim[1] : 0),
                             k_trans)) return false;
            if (!mat_vec_mul(v_buf, tv, base, xb2,
                             v_trans ? (u64)tv->dim[1] : (u64)tv->dim[0],
                             v_trans ? (u64)n_embd : (u64)(tv->ndim >= 2 ? tv->dim[1] : 0),
                             v_trans)) return false;

            /* Q/K LayerNorm (per-head). */
            {
                TensorInfo *t_qn = lw->tensors[TENSOR_ATTN_Q_NORM];
                if (t_qn) {
                    u32 norm_dim = (u32)t_qn->dim[0];
                    u32 n_heads_q = n_head;
                    u32 head_dim_q = eff_q_dim / n_heads_q;
                    for (u32 h = 0; h < n_heads_q; h++)
                        rms_norm(q_buf + h * head_dim_q,
                                 q_buf + h * head_dim_q,
                                 t_qn, base,
                                 (int)(norm_dim < head_dim_q ? norm_dim : head_dim_q), eps);
                }
            }
            {
                TensorInfo *t_kn = lw->tensors[TENSOR_ATTN_K_NORM];
                if (t_kn) {
                    u32 norm_dim = (u32)t_kn->dim[0];
                    u32 n_heads_k = n_kv_head;
                    u32 head_dim_k = eff_kv_dim / n_heads_k;
                    for (u32 h = 0; h < n_heads_k; h++)
                        rms_norm(k_buf + h * head_dim_k,
                                 k_buf + h * head_dim_k,
                                 t_kn, base,
                                 (int)(norm_dim < head_dim_k ? norm_dim : head_dim_k), eps);
                }
            }

            /* RoPE */
            rope(q_buf, n_head, eff_q_head_dim, pos, rope_theta);
            rope(k_buf, n_kv_head, eff_kv_head_dim_local, pos, rope_theta);

            /* Store K/V into cache (using full kv_dim from config for cache sizing) */
            {
                float *kd = akc->k + (u64)pos * kv_dim;
                float *vd = akc->v + (u64)pos * kv_dim;
                memcpy(kd, k_buf, kv_dim * sizeof(float));
                memcpy(vd, v_buf, kv_dim * sizeof(float));
                akc->n = pos + 1;
            }

            /* Attention (GQA) — output into hb (size ffn_hidden >= attn_out_dim). */
            u32 attn_out_dim = n_head * kv_head_dim;
            u32 n_kv = pos + 1;
            u32 score_dim = eff_q_head_dim < eff_kv_head_dim_local
                            ? eff_q_head_dim : eff_kv_head_dim_local;
            u32 eff_n_rep = n_head / n_kv_head;

            float *attn_buf = ws->hb; /* borrow hb for attn output */
            memset(attn_buf, 0, (u64)attn_out_dim * sizeof(float));
            for (u32 h = 0; h < n_head; h++) {
                u32 kv_h  = h / eff_n_rep;
                float *qh = q_buf + h * eff_q_head_dim;
                float max_s = -INFINITY;
                for (u32 t = 0; t < n_kv; t++) {
                    const float *kt = akc->k + (u64)t * kv_dim + kv_h * kv_head_dim;
                    float s = 0.0f;
                    for (u32 d = 0; d < score_dim; d++) s += qh[d] * kt[d];
                    s /= sqrtf((float)score_dim);
                    scores[t] = s;
                    if (s > max_s) max_s = s;
                }
                float sum = 0.0f;
                for (u32 t = 0; t < n_kv; t++) {
                    scores[t] = expf(scores[t] - max_s);
                    sum += scores[t];
                }
                for (u32 t = 0; t < n_kv; t++) scores[t] /= sum;
                float *out_h = attn_buf + h * kv_head_dim;
                for (u32 t = 0; t < n_kv; t++) {
                    float a = scores[t];
                    const float *vt = akc->v + (u64)t * kv_dim + kv_h * kv_head_dim;
                    for (u32 d = 0; d < kv_head_dim; d++) out_h[d] += a * vt[d];
                }
            }

            /* Attention output projection. */
            {
                TensorInfo *t_out = lw->tensors[TENSOR_ATTN_OUT];
                if (t_out) {
                    bool out_trans = (t_out->dim[0] == (u64)attn_out_dim);
                    if (!mat_vec_mul(x, t_out, base, attn_buf,
                                     out_trans ? (u64)t_out->dim[1] : (u64)t_out->dim[0],
                                     out_trans ? (u64)attn_out_dim
                                               : (u64)(t_out->ndim >= 2 ? t_out->dim[1] : 0),
                                     out_trans)) return false;
                }
            }

            /* Optional attention gate. */
            if (lw->tensors[TENSOR_ATTN_GATE]) {
                u64 gate_n = lw->tensors[TENSOR_ATTN_GATE]->n_element;
                for (u32 i = 0; i < n_embd && (u64)i < gate_n; i++)
                    x[i] *= tensor_get_f32(lw->tensors[TENSOR_ATTN_GATE], base, (u64)i);
            }
        }

        /* -- Residual connection -- */
        for (u32 i = 0; i < n_embd; i++)
            x[i] += xb[i];

        /* -- 2h. RMS Norm (pre-FFN) -- */
        memcpy(xb, x, n_embd * sizeof(float));
        {
            TensorInfo *tn = lw->tensors[TENSOR_POST_ATTN_NORM];
            if (!tn) {
                slog(WARN, "post_attn_norm missing layer %u", l);
                return false;
            }
            rms_norm(xb2, x, tn, base, (int)n_embd, eps);
        }

        /* -- 2i. SwiGLU FFN -- */
        {
            TensorInfo *t_gate = lw->tensors[TENSOR_FFN_GATE];
            TensorInfo *t_up   = lw->tensors[TENSOR_FFN_UP];
            TensorInfo *t_down = lw->tensors[TENSOR_FFN_DOWN];
            if (!t_gate || !t_up || !t_down) {
                slog(WARN, "FFN tensors missing layer %u", l);
                return false;
            }

            bool gate_trans = (t_gate->dim[0] == (u64)n_embd);
            bool up_trans   = (t_up->dim[0]   == (u64)n_embd);
            bool down_trans = (t_down->dim[0] == (u64)ws->ffn_hidden);

            u32 ffn_h = ws->ffn_hidden;
            if (!mat_vec_mul(ws->hb,  t_gate, base, xb2,
                             gate_trans ? (u64)t_gate->dim[1] : (u64)t_gate->dim[0],
                             gate_trans ? (u64)n_embd
                                        : (u64)(t_gate->ndim >= 2 ? t_gate->dim[1] : 0),
                             gate_trans)) return false;
            if (!mat_vec_mul(ws->hb2, t_up,   base, xb2,
                             up_trans ? (u64)t_up->dim[1] : (u64)t_up->dim[0],
                             up_trans ? (u64)n_embd
                                      : (u64)(t_up->ndim >= 2 ? t_up->dim[1] : 0),
                             up_trans)) return false;

            silu(ws->hb, (int)ffn_h);
            for (u32 i = 0; i < ffn_h; i++)
                ws->hb[i] *= ws->hb2[i];

            if (!mat_vec_mul(x, t_down, base, ws->hb,
                             down_trans ? (u64)t_down->dim[1] : (u64)t_down->dim[0],
                             down_trans ? (u64)ffn_h
                                        : (u64)(t_down->ndim >= 2 ? t_down->dim[1] : 0),
                             down_trans)) return false;
            for (u32 i = 0; i < n_embd; i++)
                x[i] += xb[i];
        }
    }

    /* ---- 3. Final RMS Norm ---- */
    {
        TensorInfo *out_norm = w->tensors[TENSOR_OUTPUT_NORM];
        if (out_norm)
            rms_norm(xb, x, out_norm, base, (int)n_embd, eps);
        else
            memcpy(xb, x, n_embd * sizeof(float));
    }

    /* ---- 4. LM head ---- */
    {
        TensorInfo *out = w->tensors[TENSOR_OUTPUT];
        if (!out) out = w->tensors[TENSOR_TOKEN_EMBD]; /* tied embeddings */
        if (!out) {
            slog(WARN, "output tensor not found");
            return false;
        }
        bool out_trans = (out->dim[0] == (u64)n_embd);
        if (!mat_vec_mul(logits, out, base, xb,
                         out_trans ? (u64)out->dim[1] : (u64)out->dim[0],
                         out_trans ? (u64)n_embd
                                   : (u64)(out->ndim >= 2 ? out->dim[1] : 0),
                         out_trans)) return false;
    }

    s->n_tokens++;
    return true;
}

static void qwen2_reset(Session *s) {
    KvCache *kc = &s->cache;
    for (u32 i = 0; i < kc->n_layer; i++)
        kc->std[i].n = 0;
    s->n_tokens = 0;

    /* Reset SSM state. */
    Qwen2Workspace *ws = (Qwen2Workspace *)s->arch_data;
    if (ws && ws->is_ssm) {
        u32 n_layer = kc->n_layer;
        if (ws->conv_state)
            memset(ws->conv_state, 0,
                   (u64)n_layer * 3 * (u64)ws->conv_channels * sizeof(float));
        if (ws->ssm_state)
            memset(ws->ssm_state, 0,
                   (u64)n_layer * (u64)ws->d_inner * (u64)ws->d_state * sizeof(float));
    }
}

static void qwen2_free(Session *s) {
    /* Free KV cache */
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

    /* Free workspace */
    Qwen2Workspace *ws = (Qwen2Workspace *)s->arch_data;
    if (ws) {
        sfree(ws->x);
        sfree(ws->xb);
        sfree(ws->xb2);
        sfree(ws->q);
        sfree(ws->k);
        sfree(ws->v);
        sfree(ws->qkv_fused);
        sfree(ws->scores);
        sfree(ws->hb);
        sfree(ws->hb2);
        sfree(ws->conv_state);
        sfree(ws->ssm_state);
        sfree(ws);
        s->arch_data = NULL;
    }
}

const ArchOps qwen2_ops = {
    .init    = qwen2_init,
    .free    = qwen2_free,
    .forward = qwen2_forward,
    .reset   = qwen2_reset,
};
