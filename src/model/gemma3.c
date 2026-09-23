#include <math.h>
#include <stdio.h>
#include "model.h"
#include "../utils.h"
#include "../core.h"
#include "../mm.h"
#include "../graph.h"
#include "../slog.h"
#include "../tokenizer.h"

typedef struct {
    float *x;               /* [n_embd] hidden state / residual stream      */
    float *xb;              /* [n_embd] normed input / attn output scratch  */
    float *xb2;             /* [n_embd] second scratch                      */
    float *q;               /* [n_head * head_dim] query buffer             */
    float *k;               /* [n_kv_head * head_dim] key buffer            */
    float *v;               /* [n_kv_head * head_dim] value buffer          */
    float *scores;          /* [ctx_size] attention scores (per head)       */
    float *hb;              /* [ffn_hidden] FFN gate buffer                 */
    float *hb2;             /* [ffn_hidden] FFN up buffer                   */
    u32    ffn_hidden;      /* feed_forward_length (6912)                   */
    float  rope_theta;      /* RoPE frequency base (1e6)                    */
    float  attn_scale;      /* Q scale on top of OP_ATTN's 1/sqrt(kv_dim)   */
} Gemma3WorkSpace;

static bool gemma3_init(Session *s) {
    ArchConfig *c = &s->cfg;

    u32 q_dim       = c->n_head * c->head_dim;
    u32 kv_head_dim = c->kv_head_dim ? c->kv_head_dim : c->head_dim;
    u32 kv_dim      = c->n_kv_head * kv_head_dim;

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

    /* Allocate workspace buffers. */
    Gemma3WorkSpace *ws = scalloc(1, sizeof(Gemma3WorkSpace));
    ws->x      = scalloc((u64)c->n_embd, sizeof(float));
    ws->xb     = scalloc((u64)c->n_embd, sizeof(float));
    ws->xb2    = scalloc((u64)c->n_embd, sizeof(float));
    ws->q      = scalloc((u64)q_dim, sizeof(float));
    ws->k      = scalloc((u64)kv_dim, sizeof(float));
    ws->v      = scalloc((u64)kv_dim, sizeof(float));
    ws->scores = scalloc((u64)s->ctx_size, sizeof(float));

    /* Derive the FFN hidden dim from layer 0 ffn_gate [ffn_hidden, n_embd]. */
    Weights *w = s->en->weights;
    TensorInfo *ffn_gate = w->layers[0].tensors[TENSOR_FFN_GATE];
    ws->ffn_hidden = (ffn_gate && ffn_gate->ndim >= 1)
                     ? (ffn_gate->dim[0] > ffn_gate->dim[1]
                        ? (u32)ffn_gate->dim[0]
                        : (u32)(ffn_gate->ndim >= 2 ? ffn_gate->dim[1] : ffn_gate->dim[0]))
                     : c->n_embd * 4;
    ws->hb  = scalloc((u64)ws->ffn_hidden, sizeof(float));
    ws->hb2 = scalloc((u64)ws->ffn_hidden, sizeof(float));

    /* RoPE frequency base (gemma3.rope.freq_base, default 1e6). */
    ws->rope_theta = 1000000.0f;
    {
        const char *pfx = s->en->model->arch_name[0]
                          ? s->en->model->arch_name : "gemma3";
        char k[96];
        if (snprintf(k, sizeof(k), "%s.rope.freq_base", pfx) > 0)
            model_get_f32(s->en->model, k, &ws->rope_theta);
    }

    /* Attention scale.  OP_ATTN already applies 1/sqrt(kv_head_dim) to the
     * scores, so this factor is the Gemma3 correction on top of it:
     * Gemma3 scales by 1/sqrt(query_pre_attn_scalar), which equals
     * 1/sqrt(head_dim) for every size except 27B (n_layer == 62), where
     * query_pre_attn_scalar is n_embd / n_head instead.  (Same rule as
     * llama.cpp, which detects the 27B variant from the layer count.) */
    if (c->n_layer == 62 && c->n_head > 0) {
        float qk_scalar = (float)c->n_embd / (float)c->n_head;
        ws->attn_scale  = sqrtf((float)kv_head_dim / qk_scalar);
    } else {
        ws->attn_scale  = 1.0f;
    }

    slog(INFO, "Gemma3 init: n_embd=%u n_head=%u n_kv_head=%u head_dim=%u kv_head_dim=%u "
         "n_layer=%u n_vocab=%u ctx_size=%u",
         c->n_embd, c->n_head, c->n_kv_head, c->head_dim, kv_head_dim,
         c->n_layer, c->n_vocab, s->ctx_size);
    slog(INFO, "Gemma3 init: ffn_hidden=%u rope_theta=%.6f attn_scale=%.6f",
         ws->ffn_hidden, ws->rope_theta, ws->attn_scale);

    s->arch_data = ws;
    return true;
}

static void gemma3_reset(Session *s) {
    KvCache *kc = &s->cache;
    for (u32 i = 0; i < kc->n_layer; i++)
        kc->std[i].n = 0;
    s->n_tokens = 0;
}

static void gemma3_free(Session *s) {
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

    Gemma3WorkSpace *ws = (Gemma3WorkSpace *)s->arch_data;
    if (ws) {
        sfree(ws->x);
        sfree(ws->xb);
        sfree(ws->xb2);
        sfree(ws->q);
        sfree(ws->k);
        sfree(ws->v);
        sfree(ws->scores);
        sfree(ws->hb);
        sfree(ws->hb2);
        sfree(ws);
        s->arch_data = NULL;
    }
}


/* Hex digit value, or -1 when c is not a hex digit. */
static int gemma3_hex_digit(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

/* Gemma uses a SentencePiece vocab: pieces carry the U+2581 word-boundary
 * space and arbitrary bytes are represented by "<0xHH>" fallback tokens. */
static int gemma3_decode(const u8 *raw, int raw_len, char *out, int max_len) {
    int w = 0;

    if (raw_len == 6 && raw[0] == '<' && raw[1] == '0' && raw[2] == 'x' && raw[5] == '>') {
        int hi = gemma3_hex_digit((char)raw[3]);
        int lo = gemma3_hex_digit((char)raw[4]);
        if (hi >= 0 && lo >= 0) {
            if (w < max_len) out[w++] = (char)(unsigned char)(hi * 16 + lo);
            if (w < max_len) out[w] = '\0';
            return w;
        }
    }

    for (int i = 0; i < raw_len && w < max_len; ) {
        /* U+2581 (E2 96 81) is the SentencePiece space marker. */
        if (raw[i] == 0xE2 && i + 2 < raw_len && raw[i + 1] == 0x96 && raw[i + 2] == 0x81) {
            out[w++] = ' ';
            i += 3;
        } else {
            out[w++] = (char)raw[i++];
        }
    }
    if (w < max_len) out[w] = '\0';
    return w;
}

static Graph *gemma3_graph_build(Session *s, u32 n_tokens) {
    ArchConfig *c       = &s->cfg;
    Weights    *w       = s->en->weights;
    Gemma3WorkSpace *ws = (Gemma3WorkSpace *)s->arch_data;

    if (!s || !w || !ws|| n_tokens == 0 || n_tokens > s->ctx_size) return NULL;

    u32 n_embd  = c->n_embd;
    u32 q_dim   = c->n_head * c->head_dim;
    u32 fh      = ws->ffn_hidden;
    float theta = ws->rope_theta;

    TensorInfo *te = w->tensors[TENSOR_TOKEN_EMBD];
    if (!te || te->ndim < 2) {
        slog(WARN, "graph_build: missing token embedding");
        return NULL;
    }

    Graph *g = graph_new();
    if (!g) return NULL;

    /* Input tokens → OP_INPUT leaf (shape only; values bound at compute). */
    GraphNode *in = graph_input(g, n_tokens);
    if (in == GRAPH_NODE_NONE) goto fail;

    GraphNode *cur = graph_embed(g, in, te, n_tokens);
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
        GraphNode *n = graph_rms_norm(g, cur, lw->tensors[TENSOR_ATTN_NORM]);
        if (n == GRAPH_NODE_NONE) goto fail;

        GraphNode *q = graph_mul_mat(g, n, t_q, (q_dim != n_embd) && t_q->dim[0] == (i64)n_embd);
        GraphNode *k = graph_mul_mat(g, n, t_k, t_k->dim[0] == (i64)n_embd);
        GraphNode *v = graph_mul_mat(g, n, t_v, t_v->dim[0] == (i64)n_embd);
        if (q == GRAPH_NODE_NONE || k == GRAPH_NODE_NONE || v == GRAPH_NODE_NONE) goto fail;
        
        q = graph_rms_norm_heads(g, q, lw->tensors[TENSOR_ATTN_Q_NORM], c->n_head, c->head_dim, c->head_dim, c->head_dim);
        k = graph_rms_norm_heads(g, k, lw->tensors[TENSOR_ATTN_K_NORM], c->n_kv_head, c->kv_head_dim, c->kv_head_dim, c->kv_head_dim);
        if (q == GRAPH_NODE_NONE || k == GRAPH_NODE_NONE) goto fail;

        q = graph_rope(g, q, theta, c->n_head, c->head_dim, c->head_dim);
        k = graph_rope(g, k, theta, c->n_kv_head, c->kv_head_dim, c->kv_head_dim);
        if (q == GRAPH_NODE_NONE || k == GRAPH_NODE_NONE) goto fail;
        
        q = graph_scale(g, q, ws->attn_scale);
        if (q == GRAPH_NODE_NONE) goto fail;

        GraphNode *attn = graph_attn(g, q, k, v, l);
        if (attn == GRAPH_NODE_NONE) goto fail;

        GraphNode *o = graph_mul_mat(g, attn, t_o, (n_embd != q_dim) && t_o->dim[0] == (i64)q_dim);
        if (o == GRAPH_NODE_NONE) goto fail;

        o = graph_rms_norm(g, o, lw->tensors[TENSOR_POST_ATTN_NORM]);
        if (o == GRAPH_NODE_NONE) goto fail;

        GraphNode *h = graph_binary(g, OP_ADD, cur, o);
        if (h == GRAPH_NODE_NONE) goto fail;

        /* ---- GeGLU FFN ---- */
        GraphNode *hn = graph_rms_norm(g, h, lw->tensors[TENSOR_FFN_NORM]);
        if (hn == GRAPH_NODE_NONE) goto fail;

        GraphNode *gate = graph_mul_mat(g, hn, t_g, t_g->dim[0] == (i64)n_embd);
        gate = graph_gelu(g, gate);
        GraphNode *up = graph_mul_mat(g, hn, t_u, t_u->dim[0] == (i64)n_embd);
        if (gate == GRAPH_NODE_NONE || up == GRAPH_NODE_NONE) goto fail;

        GraphNode *mul = graph_binary(g, OP_MUL, gate, up);
        if (mul == GRAPH_NODE_NONE) goto fail;

        GraphNode *down = graph_mul_mat(g, mul, t_d, t_d->dim[0] == (i64)fh);
        if (down == GRAPH_NODE_NONE) goto fail;

        /* post_feedforward_layernorm, then the residual add. */
        down = graph_rms_norm(g, down, lw->tensors[TENSOR_FFN_POST_NORM]);
        if (down == GRAPH_NODE_NONE) goto fail;

        cur = graph_binary(g, OP_ADD, h, down);
        if (cur == GRAPH_NODE_NONE) goto fail;
    }

    /* ---- Final norm ---- */
    GraphNode *fn = graph_rms_norm(g, cur, w->tensors[TENSOR_OUTPUT_NORM]);
    if (fn == GRAPH_NODE_NONE) goto fail;

    /* ---- LM head (tied to token embeddings when absent) ---- */
    TensorInfo *t_out = w->tensors[TENSOR_OUTPUT];
    if (!t_out) t_out = te;
    cur = graph_mul_mat(g, fn, t_out, t_out->dim[0] == (i64)n_embd);
    if (cur == GRAPH_NODE_NONE) goto fail;
    return g;
fail:
    graph_free(g);
    return NULL;
}

const ArchOps gemma3_ops = {
    .init              = gemma3_init,
    .free              = gemma3_free,
    .reset             = gemma3_reset,
    .decode            = gemma3_decode,
    .graph_build       = gemma3_graph_build,
    .chat_prompt       = gemma_chat_prompt,
    .chat_continuation = gemma_chat_continuation,
};
