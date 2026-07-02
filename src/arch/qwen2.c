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
 * cols is derived from the tensor's last dimension; rows must match dim[0]. */
static bool mat_vec_mul(float *y, TensorInfo *tw, const u8 *base,
                        const float *x, u64 rows, u64 cols) {
    if (!tw || tw->ndim < 2) {
        slog(WARN, "mat_vec_mul: tensor missing or ndim < 2");
        return false;
    }
    u64 tc = tw->dim[tw->ndim - 1]; /* fastest-varying = column count */
    u64 tr = tw->dim[0];            /* slowest-varying = row count */
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

    /* Allocate standard KV cache. */
    KvCache *kc   = &s->cache;
    kc->n_layer   = c->n_layer;
    kc->head_dim  = c->head_dim;
    kc->n_kv_head = c->n_kv_head;
    kc->std       = scalloc((u64)c->n_layer, sizeof(AttnKvCache));

    u64 per_layer = (u64)s->ctx_size * (u64)c->n_kv_head * (u64)c->head_dim;
    for (u32 i = 0; i < c->n_layer; i++) {
        kc->std[i].k   = scalloc(per_layer, sizeof(float));
        kc->std[i].v   = scalloc(per_layer, sizeof(float));
        kc->std[i].cap = (u32)s->ctx_size;
    }

    s->tokens = scalloc((u64)s->ctx_size, sizeof(u32));
    s->logits = scalloc((u64)c->n_vocab, sizeof(float));

    /* Allocate workspace buffers. */
    Qwen2Workspace *ws = scalloc(1, sizeof(Qwen2Workspace));
    ws->x      = scalloc((u64)c->n_embd, sizeof(float));
    ws->xb     = scalloc((u64)c->n_embd, sizeof(float));
    ws->xb2    = scalloc((u64)c->n_embd, sizeof(float));
    ws->q      = scalloc((u64)c->n_head * (u64)c->head_dim, sizeof(float));
    ws->k      = scalloc((u64)c->n_kv_head * (u64)c->head_dim, sizeof(float));
    ws->v      = scalloc((u64)c->n_kv_head * (u64)c->head_dim, sizeof(float));
    ws->qkv_fused = scalloc((u64)c->n_head * (u64)c->head_dim
                           + 2 * (u64)c->n_kv_head * (u64)c->head_dim,
                           sizeof(float));
    ws->scores = scalloc((u64)s->ctx_size, sizeof(float));

    /* Derive FFN hidden dim from layer 0 ffn_gate tensor. */
    Weights *w = s->en->weights;
    TensorInfo *ffn_gate = w->layers[0].tensors[TENSOR_FFN_GATE];
    ws->ffn_hidden = (ffn_gate && ffn_gate->ndim >= 1)
                     ? (u32)ffn_gate->dim[0]
                     : c->n_embd * 4;
    ws->hb  = scalloc((u64)ws->ffn_hidden, sizeof(float));
    ws->hb2 = scalloc((u64)ws->ffn_hidden, sizeof(float));

    /* Diagnostic: log config and first-layer tensor dims. */
    slog(INFO, "Qwen2 init: n_embd=%u n_head=%u n_kv_head=%u head_dim=%u "
         "n_layer=%u n_vocab=%u ctx_size=%u",
         c->n_embd, c->n_head, c->n_kv_head, c->head_dim,
         c->n_layer, c->n_vocab, s->ctx_size);
    slog(INFO, "Qwen2 init: ffn_hidden=%u q_dim=%u kv_dim=%u",
         ws->ffn_hidden, (u32)(c->n_head * c->head_dim),
         (u32)(c->n_kv_head * c->head_dim));
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
    const u32 head_dim  = c->head_dim;
    const u32 n_vocab   = c->n_vocab;
    const u32 n_layer   = c->n_layer;
    const u32 kv_dim    = n_kv_head * head_dim;
    const u32 q_dim     = n_head * head_dim;
    const u32 pos       = s->n_tokens;

    /* Validate critical config values — crash-avoidance. */
    if (n_embd == 0 || n_head == 0 || n_kv_head == 0 || head_dim == 0
        || n_vocab == 0 || n_layer == 0) {
        slog(WARN, "qwen2_forward: invalid config n_embd=%u n_head=%u "
             "n_kv_head=%u head_dim=%u n_vocab=%u n_layer=%u",
             n_embd, n_head, n_kv_head, head_dim, n_vocab, n_layer);
        return false;
    }

    const u32 n_rep     = n_head / n_kv_head;
    const float sqrt_d  = sqrtf((float)head_dim);
    const float eps     = 1e-6f;
    const float rope_theta = 1000000.0f;

    float *x      = ws->x;
    float *xb     = ws->xb;
    float *xb2    = ws->xb2;
    float *q_buf  = ws->q;
    float *k_buf  = ws->k;
    float *v_buf  = ws->v;
    float *scores = ws->scores;

    /* ---- 1. Token embedding ---- */
    {
        TensorInfo *te = w->tensors[TENSOR_TOKEN_EMBD];
        if (!te) { slog(WARN, "token_embd tensor not found"); return false; }
        for (u32 i = 0; i < n_embd; i++)
            x[i] = tensor_get_f32(te, base, (u64)token * (u64)n_embd + (u64)i);
    }

    s->tokens[pos % (u32)s->ctx_size] = token;

    /* ---- 2. Transformer layers ---- */
    for (u32 l = 0; l < n_layer; l++) {
        LayerWeights *lw  = &w->layers[l];
        AttnKvCache  *akc = &kc->std[l];

        /* -- 2a. RMS Norm (pre-attention) -- */
        memcpy(xb, x, n_embd * sizeof(float));
        {
            TensorInfo *tn = lw->tensors[TENSOR_ATTN_NORM];
            if (!tn) { slog(WARN, "attn_norm missing layer %u", l); return false; }
            rms_norm(xb2, x, tn, base, (int)n_embd, eps);
        }

        /* -- 2b. Q, K, V projections (fused or separate) -- */
        {
            TensorInfo *t_qkv = lw->tensors[TENSOR_ATTN_QKV];
            if (t_qkv) {
                /* Fused QKV.  Derive per-head dims from the tensor:
                 * col dim (dim[1]) n_embd, row dim (dim[0]) =
                 * q_dim + 2*kv_dim = (n_head + 2*n_kv_head) * head_dim. */
                u64 tr = t_qkv->dim[0];
                u64 tc = t_qkv->ndim >= 2 ? t_qkv->dim[1] : 0;
                u64 total_heads = n_head + 2 * n_kv_head;
                if (total_heads == 0) {
                    slog(WARN, "fused QKV: n_head+n_kv_head is 0 — "
                         "metadata missing?");
                    return false;
                }
                u64 hd = n_head;  /* head_dim from tensor */
                u64 q_dim_t  = (u64)n_head * hd;
                u64 kv_dim_t = (u64)n_kv_head * hd;

                if (tc != n_embd || q_dim_t + 2 * kv_dim_t != tr) {
                    slog(WARN, "QKV fused dim anomaly: tr=%llu tc=%llu "
                         "hd=%llu q=%llu kv=%llu (cfg q=%u kv=%u)",
                         (unsigned long long)tr, (unsigned long long)tc,
                         (unsigned long long)hd,
                         (unsigned long long)q_dim_t,
                         (unsigned long long)kv_dim_t,
                         q_dim, kv_dim);
                    return false;
                }
                if (!mat_vec_mul(ws->qkv_fused, t_qkv, base, xb2, tr, tc)) return false;
                memcpy(q_buf, ws->qkv_fused, (size_t)q_dim_t * sizeof(float));
                memcpy(k_buf, ws->qkv_fused + q_dim_t, (size_t)kv_dim_t * sizeof(float));
                memcpy(v_buf, ws->qkv_fused + q_dim_t + kv_dim_t, (size_t)kv_dim_t * sizeof(float));
            } else {
                TensorInfo *tq = lw->tensors[TENSOR_ATTN_Q];
                TensorInfo *tk = lw->tensors[TENSOR_ATTN_K];
                TensorInfo *tv = lw->tensors[TENSOR_ATTN_V];
                if (!tq || !tk || !tv) {
                    slog(WARN, "attn Q/K/V missing layer %u", l);
                    return false;
                }
                if (!mat_vec_mul(q_buf, tq, base, xb2, q_dim, n_embd)) return false;
                if (!mat_vec_mul(k_buf, tk, base, xb2, kv_dim, n_embd)) return false;
                if (!mat_vec_mul(v_buf, tv, base, xb2, kv_dim, n_embd)) return false;
            }
        }

        /* -- 2c. Optional Q/K LayerNorm -- */
        if (lw->tensors[TENSOR_ATTN_Q_NORM])
            rms_norm(q_buf, q_buf, lw->tensors[TENSOR_ATTN_Q_NORM], base,
                     (int)q_dim, eps);
        if (lw->tensors[TENSOR_ATTN_K_NORM])
            rms_norm(k_buf, k_buf, lw->tensors[TENSOR_ATTN_K_NORM], base,
                     (int)kv_dim, eps);

        /* -- 2d. RoPE -- */
        rope(q_buf, n_head, head_dim, pos, rope_theta);
        rope(k_buf, n_kv_head, head_dim, pos, rope_theta);

        /* -- 2e. Store K/V into cache -- */
        {
            float *kd = akc->k + (u64)pos * kv_dim;
            float *vd = akc->v + (u64)pos * kv_dim;
            memcpy(kd, k_buf, kv_dim * sizeof(float));
            memcpy(vd, v_buf, kv_dim * sizeof(float));
            akc->n = pos + 1;
        }

        /* -- 2f. Attention (GQA, causal) -- */
        {
            u32 n_kv = pos + 1;
            memset(xb2, 0, q_dim * sizeof(float));

            for (u32 h = 0; h < n_head; h++) {
                u32 kv_h  = h / n_rep;
                float *qh = q_buf + h * head_dim;

                /* Scores against all cached positions */
                float max_s = -INFINITY;
                for (u32 t = 0; t < n_kv; t++) {
                    const float *kt = akc->k + (u64)t * kv_dim + kv_h * head_dim;
                    float s = 0.0f;
                    for (u32 d = 0; d < head_dim; d++)
                        s += qh[d] * kt[d];
                    s /= sqrt_d;
                    scores[t] = s;
                    if (s > max_s) max_s = s;
                }

                /* Softmax */
                float sum = 0.0f;
                for (u32 t = 0; t < n_kv; t++) {
                    scores[t] = expf(scores[t] - max_s);
                    sum += scores[t];
                }
                for (u32 t = 0; t < n_kv; t++)
                    scores[t] /= sum;

                /* Weighted sum of values */
                float *out_h = xb2 + h * head_dim;
                for (u32 t = 0; t < n_kv; t++) {
                    float a = scores[t];
                    const float *vt = akc->v + (u64)t * kv_dim + kv_h * head_dim;
                    for (u32 d = 0; d < head_dim; d++)
                        out_h[d] += a * vt[d];
                }
            }
        }

        /* -- 2g. Attention output projection + residual -- */
        {
            TensorInfo *t_out = lw->tensors[TENSOR_ATTN_OUT];
            if (t_out) {
                if (!mat_vec_mul(x, t_out, base, xb2, n_embd, q_dim)) return false;
            }
            /* else: no output projection (uses identity, unlikely) */

            /* Optional attention gate (QA-LoRA). */
            if (lw->tensors[TENSOR_ATTN_GATE]) {
                u64 gate_n = lw->tensors[TENSOR_ATTN_GATE]->n_element;
                for (u32 i = 0; i < n_embd && (u64)i < gate_n; i++)
                    x[i] *= tensor_get_f32(lw->tensors[TENSOR_ATTN_GATE],
                                           base, (u64)i);
            }

            /* Residual */
            for (u32 i = 0; i < n_embd; i++)
                x[i] += xb[i];
        }

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

            u32 ffn_h = ws->ffn_hidden;
            if (!mat_vec_mul(ws->hb,  t_gate, base, xb2, ffn_h, n_embd)) return false;
            if (!mat_vec_mul(ws->hb2, t_up,   base, xb2, ffn_h, n_embd)) return false;

            silu(ws->hb, (int)ffn_h);
            for (u32 i = 0; i < ffn_h; i++)
                ws->hb[i] *= ws->hb2[i];

            /* Down projection → add to residual */
            if (!mat_vec_mul(x, t_down, base, ws->hb, n_embd, ffn_h)) return false;
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
        if (!out) {
            slog(WARN, "output tensor not found");
            return false;
        }
        if (!mat_vec_mul(logits, out, base, xb, n_vocab, n_embd)) return false;
    }

    s->n_tokens++;
    return true;
}

static void qwen2_reset(Session *s) {
    KvCache *kc = &s->cache;
    for (u32 i = 0; i < kc->n_layer; i++)
        kc->std[i].n = 0;
    s->n_tokens = 0;
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
