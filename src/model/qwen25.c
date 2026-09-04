#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "model.h"
#include "../core.h"
#include "../graph.h"
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
    float  rope_theta;      /* RoPE frequency base                          */
} Qwen25Workspace;

/* ---- Arch ops --------------------------------------------------- */

static bool qwen25_init(Session *s) {
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

    /* Also support fused QKV (e.g. Qwen2.5); check layer 0 tensor size. */
    Weights *w = s->en->weights;
    TensorInfo *t_qkv = w->layers[0].tensors[TENSOR_ATTN_QKV];
    if (t_qkv && t_qkv->ndim >= 2) {
        u64 total = t_qkv->dim[0] > t_qkv->dim[1]
                  ? (u32)t_qkv->dim[0] : (u32)t_qkv->dim[1];
        if (total > max_fused) max_fused = (u32)total;
    }

    /* Allocate workspace buffers. */
    Qwen25Workspace *ws = scalloc(1, sizeof(Qwen25Workspace));
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

    /* Read rope theta from metadata (default 1e6 for Qwen2). */
    ws->rope_theta = 1000000.0f;
    {
        const char *pfx = s->en->model->arch_name[0]
                          ? s->en->model->arch_name : "qwen2";
        char k[96];
        if (snprintf(k, sizeof(k), "%s.rope.freq_base", pfx) > 0)
            model_get_f32(s->en->model, k, &ws->rope_theta);
    }

    slog(INFO, "Qwen2 init: n_embd=%u n_head=%u n_kv_head=%u head_dim=%u kv_head_dim=%u "
         "n_layer=%u n_vocab=%u ctx_size=%u",
         c->n_embd, c->n_head, c->n_kv_head, q_head_dim, kv_head_dim,
         c->n_layer, c->n_vocab, s->ctx_size);
    slog(INFO, "Qwen2 init: rope_theta=%.6f", ws->rope_theta);

    s->arch_data = ws;
    return true;
}

/* Build the input graph for tokens[0..n_tokens) and run it.
 * Falls back silently to the manual forward path if the graph
 * cannot be built (e.g. fused-QKV models, missing tensors). */
static Graph *qwen25_graph_build(Session *s, u32 *token, u32 n_tokens);
static void qwen25_graph_run(Session *s, u32 *tokens, u32 n_tokens) {
    Graph *g = qwen25_graph_build(s, tokens, n_tokens);
    if (!g) return;
    GraphPlan plan = graph_plan(g);
    graph_compute(g, &plan, tokens, s);
    graph_free(g);
}

static bool qwen25_generate(Session *s, u32 token, float *logits) {
    Qwen25Workspace *ws = (Qwen25Workspace *)s->arch_data;
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

    /* Enter: build the input graph (single token) before the manual path. */
    qwen25_graph_run(s, &token, 1);

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
        u32 fused_total   = 0;

        if (t_qkv) {
            bool qkv_trans = (t_qkv->dim[0] == n_embd);
            fused_total = (u32)(qkv_trans ? t_qkv->dim[1] : t_qkv->dim[0]);
            if (!mat_vec_mul(ws->qkv_fused, t_qkv,  ws->xb, fused_total, n_embd, qkv_trans, s->pthreads)) return false;
            memcpy(ws->q, ws->qkv_fused, q_dim * sizeof(float));
            memcpy(ws->k, ws->qkv_fused + q_dim, kv_dim * sizeof(float));
            memcpy(ws->v, ws->qkv_fused + q_dim + kv_dim, kv_dim * sizeof(float));
        } else if (t_q && t_k && t_v) {
            if (!mat_vec_mul(ws->q, t_q,  ws->xb, q_dim,  n_embd, (q_dim != n_embd) && t_q->dim[0] == n_embd, s->pthreads)) return false;
            if (!mat_vec_mul(ws->k, t_k,  ws->xb, kv_dim, n_embd, t_k->dim[0] == n_embd, s->pthreads)) return false;
            if (!mat_vec_mul(ws->v, t_v,  ws->xb, kv_dim, n_embd, t_v->dim[0] == n_embd, s->pthreads)) return false;
        } else {
            slog(WARN, "Layer %u: missing QKV / Q,K,V tensors", l);
            return false;
        }

        /* Add biases when present (Qwen2.5 uses bias=True in attention). */
        {
            TensorInfo *tb_q = lw->tensors[TENSOR_ATTN_Q_BIAS];
            TensorInfo *tb_k = lw->tensors[TENSOR_ATTN_K_BIAS];
            TensorInfo *tb_v = lw->tensors[TENSOR_ATTN_V_BIAS];
            if (tb_q) {
                u32 n = tb_q->n_element < (u64)q_dim ? (u32)tb_q->n_element : q_dim;
                bias_add(ws->q, tb_q,  n);
            }
            if (tb_k) {
                u32 n = tb_k->n_element < (u64)kv_dim ? (u32)tb_k->n_element : kv_dim;
                bias_add(ws->k, tb_k,  n);
            }
            if (tb_v) {
                u32 n = tb_v->n_element < (u64)kv_dim ? (u32)tb_v->n_element : kv_dim;
                bias_add(ws->v, tb_v,  n);
            }
        }

        /* ---- 2c. RoPE ---- */
        rope_neox(ws->q, n_head, q_head_dim, pos, ws->rope_theta);
        rope_neox(ws->k, n_kv_head, kv_head_dim, pos, ws->rope_theta);

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
            TensorInfo *t_out = lw->tensors[TENSOR_ATTN_OUT];
            if (t_out) {
                if (!mat_vec_mul(ws->xb2, t_out,  attn_out, n_embd, q_dim, (n_embd != q_dim) && t_out->dim[0] == q_dim, s->pthreads)) return false;
            }
        }

        /* ---- 2g. Attention residual ---- */
        for (u32 i = 0; i < n_embd; i++)
            ws->x[i] += ws->xb2[i];

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

/* Unified forward: handles single token (n_tokens==1) via GEMV and
 * batched prefill (n_tokens>1) via GEMM for Q/K/V and FFN projections.
 * For batched paths, weight matrices are read once and reused across
 * all batch elements, converting memory-bandwidth-bound GEMV into
 * compute-bound GEMM.                                                */
static bool qwen25_prefill(Session *s, u32 *tokens, u32 n_tokens, float *logits) {
    if (n_tokens == 0) return true;
    if (n_tokens == 1) return qwen25_generate(s, tokens[0], logits);

    /* Enter: build the input graph (tokens + n_tokens) before the manual path. */
    qwen25_graph_run(s, tokens, n_tokens);

    Qwen25Workspace *ws  = (Qwen25Workspace *)s->arch_data;
    ArchConfig     *c    = &s->cfg;
    Weights        *w    = s->en->weights;
    
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

    /* Determine max fused QKV output width across layers. */
    u32 max_fused = q_dim + 2 * kv_dim;
    for (u32 l = 0; l < n_layer; l++) {
        TensorInfo *t_qkv = w->layers[l].tensors[TENSOR_ATTN_QKV];
        if (t_qkv && t_qkv->ndim >= 2) {
            u32 sz = (t_qkv->dim[0] > t_qkv->dim[1])
                   ? (u32)t_qkv->dim[0] : (u32)t_qkv->dim[1];
            if (sz > max_fused) max_fused = sz;
        }
    }

    u64 row_x  = (u64)n_tokens * n_embd;
    u64 row_q  = (u64)n_tokens * q_dim;
    u64 row_kv = (u64)n_tokens * max_fused;
    u64 row_fh = (u64)n_tokens * ws->ffn_hidden;

    /* ---- Temporary buffers ---- */
    float *xs       = smalloc(row_x * sizeof(float));   /* hidden states          */
    float *norm_buf = smalloc(row_x * sizeof(float));   /* RMS-norm scratch       */
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

            /* Dequantise bias tensors once outside the per-token loop.
             * Biases are identical across all tokens — dequantising them
             * once and reusing the f32 buffer saves (n_tokens-1) redundant
             * dequant calls per bias per layer. */
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

                    rope_neox(qbuf + (u64)p * q_dim, n_head, q_head_dim, pos, ws->rope_theta);
                    rope_neox(ws->k, n_kv_head, kv_head_dim, pos, ws->rope_theta);

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
                /* Separate Q/K/V: batch Q via mat_mat_mul, K/V via mat_mat_mul
                 * into temporary buffers within qkv_buf's allocation. */
                fused_total = q_dim + 2 * kv_dim;

                /* Q → qbuf (stride q_dim, ready for attention + RoPE in-place). */
                if (!mat_mat_mul(qbuf, t_q,  norm_buf, n_tokens, q_dim, n_embd, (q_dim != n_embd) && t_q->dim[0] == n_embd, s->pthreads)) goto fail;

                /* K → after Q in qkv_buf, V → after K. */
                float *kbuf = qkv_buf;
                float *vbuf = qkv_buf + (u64)n_tokens * kv_dim;
                if (!mat_mat_mul(kbuf, t_k,  norm_buf, n_tokens, kv_dim, n_embd, t_k->dim[0] == n_embd, s->pthreads)) goto fail;
                if (!mat_mat_mul(vbuf, t_v,  norm_buf, n_tokens, kv_dim, n_embd, t_v->dim[0] == n_embd, s->pthreads)) goto fail;

                for (u32 p = 0; p < n_tokens; p++) {
                    u32 pos = cache_start + p;

                    memcpy(ws->k, kbuf + (u64)p * kv_dim, kv_dim * sizeof(float));
                    memcpy(ws->v, vbuf + (u64)p * kv_dim, kv_dim * sizeof(float));

                    /* Apply pre-dequantised bias (f32 addition). */
                    if (tb_k) {
                        for (u32 i = 0; i < n_k; i++) ws->k[i] += k_bias[i];
                    }
                    if (tb_v) {
                        for (u32 i = 0; i < n_v; i++) ws->v[i] += v_bias[i];
                    }
                    if (tb_q) {
                        float *qp = qbuf + (u64)p * q_dim;
                        for (u32 i = 0; i < n_q; i++) qp[i] += q_bias[i];
                    }

                    rope_neox(qbuf + (u64)p * q_dim, n_head, q_head_dim, pos, ws->rope_theta);
                    rope_neox(ws->k, n_kv_head, kv_head_dim, pos, ws->rope_theta);

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
        if (t_out) {
            if (!mat_mat_mul(norm_buf, t_out,  attn_buf, n_tokens, n_embd, q_dim, (n_embd != q_dim) && t_out->dim[0] == q_dim, s->pthreads)) { goto fail; }
        } else {
            /* No output projection: copy attn_buf → norm_buf (if dims match). */
            memcpy(norm_buf, attn_buf, row_q * sizeof(float));
        }

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
            /* Gate projection: norm_buf → gate_buf */
            if (!mat_mat_mul(gate_buf, t_gate,  norm_buf, n_tokens, fh, n_embd, t_gate->dim[0] == n_embd, s->pthreads)) goto fail;

            /* Up projection: norm_buf → up_buf */
            if (!mat_mat_mul(up_buf, t_up,  norm_buf, n_tokens, fh, n_embd, t_up->dim[0] == n_embd, s->pthreads)) goto fail;

            /* Element-wise: gate = silu(gate) * up  (per token, cheap) */
            for (u32 p = 0; p < n_tokens; p++) {
                float *gp = gate_buf + (u64)p * fh;
                float *up = up_buf   + (u64)p * fh;
                silu(gp, fh);
                for (u32 i = 0; i < fh; i++)
                    gp[i] *= up[i];
            }

            /* Down projection: gate_buf → norm_buf (reuse norm_buf for output) */
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

static void qwen25_reset(Session *s) {
    KvCache *kc = &s->cache;
    for (u32 i = 0; i < kc->n_layer; i++)
        kc->std[i].n = 0;
    s->n_tokens = 0;
}

static void qwen25_free(Session *s) {
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

    Qwen25Workspace *ws = (Qwen25Workspace *)s->arch_data;
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

static int qwen25_decode(const u8 *raw, int raw_len, char *out, int max_len) {
    int w = 0;
    for (int b = 0; b < raw_len && w < max_len; b++) {
        unsigned char c = raw[b];
        /* GPT-2 byte-level decode.
         * printable ASCII 0x21-0x7E → itself.
         * Two-byte sequences 0xC2-0xC5 + 0x80-0xBF are the UTF-8 encoding
         * of codepoints from the standard bytes_to_unicode() mapping. */
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


/* Build a static execution graph for a fresh batch of n_tokens tokens.
 * The graph mirrors qwen25_prefill: token embedding, per-layer RMS norm,
 * Q/K/V projections (+bias, RoPE), causal GQA attention, output projection
 * and residuals, SwiGLU FFN, final norm and LM head.  Positions are the
 * batch row indices (0 .. n_tokens-1), so the graph performs a standalone
 * forward pass that does not consult the session KV cache. */
static Graph *qwen25_graph_build(Session *s, u32 *token, u32 n_tokens) {
    ArchConfig *c = &s->cfg;
    Weights    *w = s->en->weights;

    if (!s || !w || !token || n_tokens == 0 || n_tokens > s->ctx_size) return NULL;

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

        TensorInfo *t_q = lw->tensors[TENSOR_ATTN_Q];
        TensorInfo *t_k = lw->tensors[TENSOR_ATTN_K];
        TensorInfo *t_v = lw->tensors[TENSOR_ATTN_V];
        TensorInfo *t_o = lw->tensors[TENSOR_ATTN_OUT];
        TensorInfo *t_g = lw->tensors[TENSOR_FFN_GATE];
        TensorInfo *t_u = lw->tensors[TENSOR_FFN_UP];
        TensorInfo *t_d = lw->tensors[TENSOR_FFN_DOWN];
        if (!t_q || !t_k || !t_v || !t_o || !t_g || !t_u || !t_d) {
            slog(WARN, "graph_build: layer %u missing tensors (fused QKV unsupported)", l);
            goto fail;
        }

        /* ---- Attention block ---- */
        u32 n = graph_rms_norm(g, cur, lw->tensors[TENSOR_ATTN_NORM]);
        if (n == GRAPH_NODE_NONE) goto fail;

        u32 q = graph_mul_mat2(g, n, t_q);
        u32 k = graph_mul_mat2(g, n, t_k);
        u32 v = graph_mul_mat2(g, n, t_v);
        if (q == GRAPH_NODE_NONE || k == GRAPH_NODE_NONE || v == GRAPH_NODE_NONE) goto fail;

        /* Qwen2.5 attention biases (bias=True); skip when absent. */
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

        q = graph_rope(g, q);
        k = graph_rope(g, k);
        if (q == GRAPH_NODE_NONE || k == GRAPH_NODE_NONE) goto fail;

        u32 attn = graph_attn(g, q, k, v);
        if (attn == GRAPH_NODE_NONE) goto fail;

        u32 o = graph_mul_mat2(g, attn, t_o);
        if (o == GRAPH_NODE_NONE) goto fail;

        u32 h = graph_binary(g, OP_ADD, cur, o);
        if (h == GRAPH_NODE_NONE) goto fail;

        /* ---- SwiGLU FFN ---- */
        u32 hn = graph_rms_norm(g, h, lw->tensors[TENSOR_POST_ATTN_NORM]);
        if (hn == GRAPH_NODE_NONE) goto fail;

        u32 gate = graph_silu(g, graph_mul_mat2(g, hn, t_g));
        u32 up   = graph_mul_mat2(g, hn, t_u);
        if (gate == GRAPH_NODE_NONE || up == GRAPH_NODE_NONE) goto fail;

        u32 mul = graph_binary(g, OP_MUL, gate, up);
        if (mul == GRAPH_NODE_NONE) goto fail;

        u32 down = graph_mul_mat2(g, mul, t_d);
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
    if (graph_mul_mat2(g, fn, t_out) == GRAPH_NODE_NONE) goto fail;

    return g;
fail:
    graph_free(g);
    return NULL;
}

const ArchOps qwen25_ops = {
    .init        = qwen25_init,
    .free        = qwen25_free,
    .prefill     = qwen25_prefill,
    .generate    = qwen25_generate,
    .reset       = qwen25_reset,
    .decode      = qwen25_decode,
    .graph_build = qwen25_graph_build,
};
