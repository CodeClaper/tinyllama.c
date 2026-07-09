#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "arch.h"
#include "../core.h"
#include "../mm.h"
#include "../slog.h"
#include "../utils.h"


typedef struct {
    float *x;               /* [n_embd] hidden state                        */
    float *xb;              /* [n_embd] residual / scratch                  */
    float *xb2;             /* [n_embd] second scratch                      */
    float *q;               /* [n_head * head_dim] query buffer             */
    float *k;               /* [n_kv_head * head_dim] key buffer            */
    float *v;               /* [n_kv_head * head_dim] value buffer          */
    float *qkv_fused;       /* [q_dim + 2*kv_dim] fused QKV buffer          */
    float *scores;          /* [ctx_size] attention scores (per head)       */
    float *hb;              /* [ffn_hidden] FFN hidden buffer               */
    float *hb2;             /* [ffn_hidden] FFN hidden buffer 2             */
    u32    ffn_hidden;

    /* SSM (Mamba) state — per-layer, persists across forward calls. */
    float *conv_state;      /* [3 * conv_channels] last 3 inputs for conv1d */
    float *ssm_state;       /* [d_inner * d_state] SSM hidden state         */
    u32    conv_channels;   /* total channels for conv1d (fused projection) */
    u32    d_inner;         /* SSM inner dimension                          */
    u32    d_state;         /* SSM state dimension                          */
    u32    conv_kernel;     /* conv1d kernel size (from KV metadata)        */
    u32    group_count;     /* SSM group count (from KV metadata)           */
    u32    time_step_rank;  /* SSM time step rank (from KV metadata)        */
    bool   is_ssm;          /* true if this layer has SSM tensors           */
    float  rope_theta;      /* RoPE frequency base                          */
} Qwen2Workspace;

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

    /* Determine arch prefix for KV metadata lookups. */
    const char *pfx = s->en->model->arch_name[0]
                      ? s->en->model->arch_name : "qwen2";

    /* Detect SSM presence and dimensions from layer 0. */
    Weights *w = s->en->weights;
    bool has_ssm = (w->layers[0].tensors[TENSOR_SSM_CONV1D] != NULL);

    /* Compute max fused QKV dimension from first fused QKV tensor. */
    u32 max_fused = q_dim + 2 * kv_dim; /* default for non-SSM */
    u32 ssm_channels = 0;
    u32 d_inner = 0;
    u32 d_state = 0;
    u32 conv_kernel = 4;
    u32 group_count = 1;
    u32 time_step_rank = 16;
    if (has_ssm) {
        TensorInfo *t_qkv = w->layers[0].tensors[TENSOR_ATTN_QKV];
        if (t_qkv && t_qkv->ndim >= 2) {
            u64 total = t_qkv->dim[0] > t_qkv->dim[1]
                        ? (u32)t_qkv->dim[0] : (u32)t_qkv->dim[1];
            max_fused = (u32)total;
            ssm_channels = (u32)total;
            u32 attn_ch = q_dim + 2 * kv_dim;
            d_inner = (max_fused > attn_ch) ? (max_fused - attn_ch) / 2 : 0;
        }
        /* d_state from ssm_a tensor (fallback). */
        TensorInfo *t_ssm_a = w->layers[0].tensors[TENSOR_SSM_A];
        if (t_ssm_a && t_ssm_a->ndim >= 1)
            d_state = (u32)t_ssm_a->dim[0];

        /* Read SSM params from KV metadata (override derived values). */
        { i32 v; char k[96];
          if (snprintf(k, sizeof(k), "%s.ssm.inner_size", pfx) > 0
              && model_get_i32(s->en->model, k, &v)) d_inner = (u32)v;
          if (snprintf(k, sizeof(k), "%s.ssm.state_size", pfx) > 0
              && model_get_i32(s->en->model, k, &v)) d_state = (u32)v;
          if (snprintf(k, sizeof(k), "%s.ssm.conv_kernel", pfx) > 0
              && model_get_i32(s->en->model, k, &v)) conv_kernel = (u32)v;
          if (snprintf(k, sizeof(k), "%s.ssm.group_count", pfx) > 0
              && model_get_i32(s->en->model, k, &v)) group_count = (u32)v;
          if (snprintf(k, sizeof(k), "%s.ssm.time_step_rank", pfx) > 0
              && model_get_i32(s->en->model, k, &v)) time_step_rank = (u32)v;
        }
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
        if (snprintf(k, sizeof(k), "%s.rope.freq_base", pfx) > 0)
            model_get_f32(s->en->model, k, &ws->rope_theta);
    }

    /* SSM state allocation. */
    ws->is_ssm        = has_ssm;
    ws->d_inner       = d_inner;
    ws->d_state       = d_state;
    ws->conv_channels = ssm_channels;
    ws->conv_kernel   = conv_kernel;
    ws->group_count   = group_count;
    ws->time_step_rank = time_step_rank;
    if (has_ssm && ssm_channels > 0 && d_state > 0) {
        /* conv_state: [n_layer * (conv_kernel-1) * conv_channels] */
        ws->conv_state = scalloc((u64)c->n_layer * (u64)(conv_kernel - 1) * (u64)ssm_channels, sizeof(float));
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
    if (has_ssm) slog(INFO, "Qwen2 init SSM: channels=%u d_inner=%u d_state=%u "
                      "conv_kernel=%u group_count=%u time_step_rank=%u",
                      ssm_channels, d_inner, d_state,
                      conv_kernel, group_count, time_step_rank);
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

/* Run the SSM (Mamba-style) block for one layer at one position.
 * x_ssm and z are [d_inner] each, read from fused QKV output.
 * conv_state and ssm_state are updated in-place. */
static void ssm_block(Qwen2Workspace *ws, const u8 *base, LayerWeights *lw,
                      float *x_ssm, float *z, u32 layer_idx,
                      u32 d_inner, u32 d_state) {
    TensorInfo *t_conv1d  = lw->tensors[TENSOR_SSM_CONV1D];
    TensorInfo *t_a       = lw->tensors[TENSOR_SSM_A];
    TensorInfo *t_dt_bias = lw->tensors[TENSOR_SSM_DT_BIAS];

    /* 1. SiLU on input. */
    silu(x_ssm, d_inner);

    /* 2. Depthwise conv1d. */
    u32 ksize = ws->conv_kernel;
    float *cstate = ws->conv_state + (u64)layer_idx * (u64)(ksize - 1) * (u64)d_inner;
    for (u32 i = 0; i < d_inner; i++) {
        /* Current input * last weight (default 1.0). */
        float y = (t_conv1d
                   ? tensor_get_f32(t_conv1d, base, (u64)i * ksize + (ksize - 1))
                   : 1.0f) * x_ssm[i];
        /* Historical states. */
        for (u32 k = 0; k < ksize - 1; k++) {
            float wk = t_conv1d
                       ? tensor_get_f32(t_conv1d, base, (u64)i * ksize + k)
                       : 0.0f;
            y += wk * cstate[(ksize - 2 - k) * d_inner + i];
        }
        /* Shift state window. */
        for (u32 k = ksize - 2; k > 0; k--)
            cstate[k * d_inner + i] = cstate[(k - 1) * d_inner + i];
        cstate[0 * d_inner + i] = x_ssm[i];
        x_ssm[i] = y;
    }

    /* 3. SiLU on conv output. */
    silu(x_ssm, d_inner);

    /* 4. Selective SSM scan. */
    float *sm = ws->ssm_state + (u64)layer_idx * (u64)d_inner * (u64)d_state;
    for (u32 i = 0; i < d_inner; i++) {
        float dt_bias = (t_dt_bias && i < t_dt_bias->n_element) 
                           ? tensor_get_f32(t_dt_bias, base, i) 
                           : 0.0f;
        float dt = softplus(x_ssm[i] + dt_bias);

        float y = 0.0f;
        for (u32 j = 0; j < d_state; j++) {
            float A_j = (t_a && j < t_a->n_element) 
                          ? tensor_get_f32(t_a, base, j) 
                          : -(float)(j + 1);
            float A_bar = expf(dt * A_j);
            float h = sm[(u64)i * d_state + j];
            h = A_bar * h + dt * x_ssm[i];  /* B = 1 (simplified) */
            sm[(u64)i * d_state + j] = h;
            y += h;  /* C = 1 */
        }
        x_ssm[i] = y;
    }

    /* 5. Gate with z. */
    silu(z, d_inner);
    for (u32 i = 0; i < d_inner; i++)
        x_ssm[i] *= z[i];
}

static bool qwen2_forward(Session *s, u32 token, float *logits) {
    Qwen2Workspace *ws = (Qwen2Workspace *)s->arch_data;
    ArchConfig     *c  = &s->cfg;
    Weights        *w  = s->en->weights;
    const u8       *base = s->en->model->map;

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
    float eps       = 1e-6f;

    /* ---- 1. Token embedding ---- */
    {
        TensorInfo *te = w->tensors[TENSOR_TOKEN_EMBD];
        if (te->dim[0] == c->n_vocab) {
            /* [n_vocab, n_embd] */
            for (u32 i = 0; i < n_embd; i++)
                ws->x[i] = tensor_get_f32(te, base, (u64)token * n_embd + i);
        } else {
            /* [n_embd, n_vocab] */
            for (u32 i = 0; i < n_embd; i++)
                ws->x[i] = tensor_get_f32(te, base, (u64)i * c->n_vocab + token);
        }
    }

    /* ---- 2. Per-layer ---- */
    for (u32 l = 0; l < n_layer; l++) {
        slog(INFO, "Layer ===> %d", l);
        LayerWeights *lw  = &w->layers[l];
        AttnKvCache  *akc = &s->cache.std[l];
        bool is_ssm_layer = (lw->tensors[TENSOR_SSM_CONV1D] != NULL);

        /* ---- 2a. Attention norm ---- */
        rms_norm(ws->xb, ws->x, lw->tensors[TENSOR_ATTN_NORM], base, n_embd, eps);

        /* ---- 2b. Q / K / V projections ---- */
        TensorInfo *t_qkv = lw->tensors[TENSOR_ATTN_QKV];
        TensorInfo *t_q   = lw->tensors[TENSOR_ATTN_Q];
        TensorInfo *t_k = lw->tensors[TENSOR_ATTN_K];
        TensorInfo *t_v = lw->tensors[TENSOR_ATTN_V];
        u32 fused_total   = 0;  /* total output dim of fused QKV */

        if (t_qkv) {
            bool qkv_trans = (t_qkv->dim[0] == n_embd);
            fused_total = (u32)(qkv_trans ? t_qkv->dim[1] : t_qkv->dim[0]);
            if (!mat_vec_mul(ws->qkv_fused, t_qkv, base, ws->xb, fused_total, n_embd, qkv_trans)) return false;
            /* Split into q / k / v. */
            memcpy(ws->q, ws->qkv_fused, q_dim * sizeof(float));
            memcpy(ws->k, ws->qkv_fused + q_dim, kv_dim * sizeof(float));
            memcpy(ws->v, ws->qkv_fused + q_dim + kv_dim, kv_dim * sizeof(float));
        } else if (t_q && t_k && t_v) {
            /* Separate Q / K / V tensors. */
            if (!mat_vec_mul(ws->q, t_q, base, ws->xb, q_dim,  n_embd, t_q->dim[0] == n_embd)) return false;
            if (!mat_vec_mul(ws->k, t_k, base, ws->xb, kv_dim, n_embd, t_k->dim[0] == n_embd)) return false;
            if (!mat_vec_mul(ws->v, t_v, base, ws->xb, kv_dim, n_embd, t_v->dim[0] == n_embd)) return false;
        } else {
            slog(WARN, "Layer %u: missing QKV / Q,K,V tensors", l);
            return false;
        }

        /* Extract SSM channels when present. */
        u32 ssm_d_inner = 0;
        float *ssm_x = NULL, *ssm_z = NULL;
        if (is_ssm_layer && t_qkv && fused_total > q_dim + 2 * kv_dim) {
            u32 attn_ch = q_dim + 2 * kv_dim;
            ssm_d_inner = ws->d_inner > 0 ? ws->d_inner : (fused_total - attn_ch) / 2;
            ssm_x = ws->qkv_fused + attn_ch;
            ssm_z = ws->qkv_fused + attn_ch + ssm_d_inner;
        }

        /* ---- 2c. Q / K norms (Qwen3.5) ---- */
        if (lw->tensors[TENSOR_ATTN_Q_NORM])
            rms_norm(ws->q, ws->q, lw->tensors[TENSOR_ATTN_Q_NORM],
                     base, q_dim, eps);
        if (lw->tensors[TENSOR_ATTN_K_NORM])
            rms_norm(ws->k, ws->k, lw->tensors[TENSOR_ATTN_K_NORM],
                     base, kv_dim, eps);

        /* ---- 2d. RoPE ---- */
        rope(ws->q, n_head, q_head_dim, pos, ws->rope_theta);
        rope(ws->k, n_kv_head, kv_head_dim, pos, ws->rope_theta);

        /* ---- 2e. KV cache write ---- */
        memcpy(akc->k + (u64)pos * kv_dim, ws->k, kv_dim * sizeof(float));
        memcpy(akc->v + (u64)pos * kv_dim, ws->v, kv_dim * sizeof(float));
        akc->n = pos + 1;

        /* ---- 2f. Attention ---- */
        u32 n_cached = akc->n;
        float *attn_out = ws->q;  /* reuse Q buffer for per-head output */
        memset(attn_out, 0, q_dim * sizeof(float));

        for (u32 h = 0; h < n_head; h++) {
            u32 kv_h   = h / gqa_ratio;
            float *qh  = ws->q + (u64)h * q_head_dim;
            float *kh_base = akc->k + (u64)kv_h * kv_head_dim;
            float *vh_base = akc->v + (u64)kv_h * kv_head_dim;

            /* Q·K^T scores. */
            for (u32 t = 0; t < n_cached; t++) {
                float *kt = kh_base + (u64)t * kv_dim;
                float s = 0.0f;
                for (u32 d = 0; d < kv_head_dim; d++)
                    s += qh[d] * kt[d];
                ws->scores[t] = s * scale;
            }
            softmax(ws->scores, n_cached);

            /* Weighted V sum. */
            float *oh = attn_out + (u64)h * q_head_dim;
            for (u32 t = 0; t < n_cached; t++) {
                float *vt = vh_base + (u64)t * kv_dim;
                float st  = ws->scores[t];
                for (u32 d = 0; d < kv_head_dim; d++)
                    oh[d] += st * vt[d];
            }
        }

        /* ---- 2g. Attention output projection ---- */
        {
            TensorInfo *t_out = lw->tensors[TENSOR_ATTN_OUT];
            if (t_out) {
                if (!mat_vec_mul(ws->xb2, t_out, base, attn_out, n_embd, q_dim, t_out->dim[0] == q_dim)) return false;
            }
        }

        /* ---- 2h. Attention gate (Qwen3.5) ---- */
        {
            TensorInfo *t_gate = lw->tensors[TENSOR_ATTN_GATE];
            if (t_gate) {
                /* Gate: ws->xb2 *= sigmoid(W_gate @ ws->xb).
                 * W_gate shape: [n_embd, n_embd] or [1, n_embd]. */
                u32 gate_out = t_gate->dim[0] > t_gate->dim[1]
                               ? (u32)t_gate->dim[0] : (u32)(t_gate->ndim >= 2
                                   ? t_gate->dim[1] : t_gate->dim[0]);
                /* gate_out == 1 means scalar gate; otherwise per-channel. */
                float *gate_buf = ws->hb;
                if (!mat_vec_mul(gate_buf, t_gate, base, ws->xb, gate_out, n_embd, t_gate->dim[0] == n_embd)) return false;
                if (gate_out == 1) {
                    float g = 1.0f / (1.0f + expf(-gate_buf[0]));  /* sigmoid */
                    for (u32 i = 0; i < n_embd; i++)
                        ws->xb2[i] *= g;
                } else {
                    for (u32 i = 0; i < n_embd && i < gate_out; i++) {
                        float g = 1.0f / (1.0f + expf(-gate_buf[i]));
                        ws->xb2[i] *= g;
                    }
                }
            }
        }

        /* ---- 2i. Attention residual ---- */
        for (u32 i = 0; i < n_embd; i++)
            ws->x[i] += ws->xb2[i];

        /* ---- 2j. SSM block (Qwen3.5 hybrid layers) ---- */
        if (is_ssm_layer && ssm_d_inner > 0) {
            /* Copy SSM channels into hb / hb2 for in-place processing. */
            float *x_ssm = ws->hb;   /* [d_inner] */
            float *z     = ws->hb2;  /* [d_inner] */
            memcpy(x_ssm, ssm_x, ssm_d_inner * sizeof(float));
            memcpy(z,     ssm_z, ssm_d_inner * sizeof(float));

            ssm_block(ws, base, lw, x_ssm, z, l, ssm_d_inner, ws->d_state);

            /* Project back: ssm_out @ x_ssm -> [n_embd]. */
            TensorInfo *t_ssm_out = lw->tensors[TENSOR_SSM_OUT];
            if (t_ssm_out) {
                if (!mat_vec_mul(ws->xb2, t_ssm_out, base, x_ssm, n_embd, ssm_d_inner, t_ssm_out->dim[0] == ssm_d_inner)) return false;
            } else {
                /* No output projection: x_ssm is already [n_embd]. */
                memcpy(ws->xb2, x_ssm, n_embd * sizeof(float));
            }
            for (u32 i = 0; i < n_embd; i++)
                ws->x[i] += ws->xb2[i];
        }

        /* ---- 2k. Post-attention norm (pre-FFN) ---- */
        {
            TensorInfo *ti_att = lw->tensors[TENSOR_POST_ATTN_NORM];
            if (ti_att) rms_norm(ws->xb, ws->x, ti_att, base, n_embd, eps);
        }

        /* ---- 2l. SwiGLU FFN ---- */
        {
            TensorInfo *t_gate = lw->tensors[TENSOR_FFN_GATE];
            TensorInfo *t_up   = lw->tensors[TENSOR_FFN_UP];
            TensorInfo *t_down = lw->tensors[TENSOR_FFN_DOWN];
            u32 fh = ws->ffn_hidden;

            /* Gate projection: hb = W_gate @ xb. */
            if (!mat_vec_mul(ws->hb, t_gate, base, ws->xb, fh, n_embd, t_gate->dim[0] == n_embd)) return false;
            /* Up projection: hb2 = W_up @ xb. */
            if (!mat_vec_mul(ws->hb2, t_up, base, ws->xb, fh, n_embd, t_up->dim[0] == n_embd)) return false;

            /* SiLU(gate) * up. */
            silu(ws->hb, fh);
            for (u32 i = 0; i < fh; i++)
                ws->hb[i] *= ws->hb2[i];

            /* Down projection: xb = W_down @ hb. */
            if (!mat_vec_mul(ws->xb, t_down, base, ws->hb, n_embd, fh, t_down->dim[0] == fh)) return false;

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
        rms_norm(ws->xb, ws->x, t_norm, base, n_embd, eps);
    }

    /* ---- 4. LM head ---- */
    {
        TensorInfo *t_out = w->tensors[TENSOR_OUTPUT];
        if (!t_out) t_out = w->tensors[TENSOR_TOKEN_EMBD];  /* tied weights */
        float *dst = logits ? logits : s->logits;
        if (!mat_vec_mul(dst, t_out, base, ws->xb, c->n_vocab, n_embd, t_out->dim[0] == n_embd)) return false;
    }

    /* ---- 5. Update session state ---- */
    s->tokens[pos] = token;
    s->n_tokens    = pos + 1;
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
                   (u64)n_layer * (u64)(ws->conv_kernel - 1) * (u64)ws->conv_channels * sizeof(float));
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
