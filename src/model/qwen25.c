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


/* Build a static execution graph for a fresh batch of n_tokens tokens:
 * token embedding, per-layer RMS norm, Q/K/V projections (+bias, RoPE),
 * causal GQA attention, output projection and residuals, SwiGLU FFN,
 * final norm and LM head.  Built once at ctx_size capacity and reused for
 * every batch; OP_ATTN extends the session KV cache, so decoding one row
 * at a time never recomputes the history. */
static Graph *qwen25_graph_build(Session *s, u32 n_tokens) {
    ArchConfig *c       = &s->cfg;
    Weights    *w       = s->en->weights;
    Qwen25Workspace *ws = (Qwen25Workspace *)s->arch_data;

    if (!s || !w || !ws|| n_tokens == 0 || n_tokens > s->ctx_size) return NULL;

    /* Dims for the matmul direction flags.  Each trans below must mirror
     * the expression used by qwen25_prefill / qwen25_generate for the
     * same weight, otherwise the executor dequantises the wrong axis. */
    u32 n_embd = c->n_embd;
    u32 q_dim  = c->n_head  * c->head_dim;
    u32 fh     = ws->ffn_hidden;
    float theta = ws->rope_theta;

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

        u32 q = graph_mul_mat(g, n, t_q, (q_dim != n_embd) && t_q->dim[0] == (i64)n_embd);
        u32 k = graph_mul_mat(g, n, t_k, t_k->dim[0] == (i64)n_embd);
        u32 v = graph_mul_mat(g, n, t_v, t_v->dim[0] == (i64)n_embd);
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

        /* qwen25 uses full-width neox RoPE, so rope_dim == head_dim. */
        q = graph_rope(g, q, theta, c->n_head, c->head_dim, c->head_dim);
        k = graph_rope(g, k, theta, c->n_kv_head, c->kv_head_dim, c->kv_head_dim);
        if (q == GRAPH_NODE_NONE || k == GRAPH_NODE_NONE) goto fail;

        u32 attn = graph_attn(g, q, k, v, l);
        if (attn == GRAPH_NODE_NONE) goto fail;

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

const ArchOps qwen25_ops = {
    .init          = qwen25_init,
    .free          = qwen25_free,
    .reset         = qwen25_reset,
    .decode        = qwen25_decode,
    .graph_build   = qwen25_graph_build,
    .graph_execute = graph_execute,
};
