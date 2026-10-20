#include <math.h>
#include <stdio.h>
#include <string.h>
#include "model.h"
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
    u32    ffn_hidden;      /* feed_forward_length (8192)                   */
    float  rope_theta;      /* RoPE frequency base (500000 for Llama 3)     */
} LlamaWorkSpace;

static bool llama_init(Session *s) {
    ArchConfig *c = &s->cfg;

    u32 q_dim       = c->n_head * c->head_dim;
    u32 kv_head_dim = c->kv_head_dim ? c->kv_head_dim : c->head_dim;
    u32 kv_dim      = c->n_kv_head * kv_head_dim;

    s->tokens = scalloc((u64)s->ctx_size, sizeof(u32));
    s->logits = scalloc((u64)c->n_vocab, sizeof(float));

    LlamaWorkSpace *ws = scalloc(1, sizeof(LlamaWorkSpace));
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

    /* Llama 3 uses rope.freq_base = 500000 (Llama 2 used 10000). */
    const char *pfx = s->en->model->arch_name[0]
                      ? s->en->model->arch_name : "llama";
    char k[96];
    ws->rope_theta = 10000.0f;
    if (snprintf(k, sizeof(k), "%s.rope.freq_base", pfx) > 0)
        model_get_f32(s->en->model, k, &ws->rope_theta);

    slog(INFO, "Llama init: n_embd=%u n_head=%u n_kv_head=%u head_dim=%u kv_head_dim=%u "
         "n_layer=%u n_vocab=%u ctx_size=%u",
         c->n_embd, c->n_head, c->n_kv_head, c->head_dim, kv_head_dim,
         c->n_layer, c->n_vocab, s->ctx_size);
    slog(INFO, "Llama init: ffn_hidden=%u rope_theta=%.1f rope_dim=%u",
         ws->ffn_hidden, ws->rope_theta, c->rope_dim);

    s->arch_data = ws;
    return true;
}

static void llama_reset(Session *s) {
    s->n_tokens = 0;
}

static void llama_free(Session *s) {
    sfree(s->tokens);
    sfree(s->logits);
    s->tokens = NULL;
    s->logits = NULL;

    LlamaWorkSpace *ws = (LlamaWorkSpace *)s->arch_data;
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

/* GPT-2 byte-level decode.  Llama 3 uses the same byte⇄unicode
 * mapping as Qwen/GPT-2, so token bytes are mapped back to raw bytes. */
static int llama_decode(const u8 *raw, int raw_len, char *out, int max_len) {
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

/* Llama graph: token embedding, per-layer RMS norm, Q/K/V projections,
 * consecutive-pair (NORM) RoPE, causal GQA attention, output projection
 * and residuals, SwiGLU FFN, final norm and LM head.  Llama 3 ties the
 * output projection to the token embeddings when output.weight is absent. */
static Graph *llama_graph_build(Session *s, u32 n_tokens) {
    ArchConfig *c       = &s->cfg;
    Weights    *w       = s->en->weights;
    LlamaWorkSpace *ws  = (LlamaWorkSpace *)s->arch_data;

    if (!s || !w || !ws || n_tokens == 0 || n_tokens > s->ctx_size) return NULL;

    u32 n_embd = c->n_embd;
    u32 q_dim  = c->n_head * c->head_dim;
    u32 fh     = ws->ffn_hidden;
    float theta = ws->rope_theta;
    u32 rope_dim = c->rope_dim ? c->rope_dim : c->head_dim;

    TensorInfo *te = w->tensors[TENSOR_TOKEN_EMBD];
    if (!te || te->ndim < 2) {
        slog(WARN, "graph_build: missing token embedding");
        return NULL;
    }

    Graph *g = graph_new();
    if (!g) return NULL;

    GraphNode *in = graph_input(g, n_tokens);
    if (in == GRAPH_NODE_NONE) goto fail;

    GraphNode *cur = graph_embed(g, in, te, n_tokens);
    if (cur == GRAPH_NODE_NONE) goto fail;

    for (u32 l = 0; l < c->n_layer; l++) {
        graph_set_layer(g, l);
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

        /* Optional biases (some Llama-family models carry them). */
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

        /* Llama GGUF weights use GGML_ROPE_TYPE_NORM: consecutive pairs. */
        q = graph_rope_norm(g, q, theta, c->n_head, c->head_dim, rope_dim);
        k = graph_rope_norm(g, k, theta, c->n_kv_head,
                            c->kv_head_dim ? c->kv_head_dim : c->head_dim,
                            rope_dim <= (c->kv_head_dim ? c->kv_head_dim : c->head_dim)
                                ? rope_dim : (c->kv_head_dim ? c->kv_head_dim : c->head_dim));
        if (q == GRAPH_NODE_NONE || k == GRAPH_NODE_NONE) goto fail;

        GraphNode *attn = graph_attn(g, q, k, v, l, 0);
        if (attn == GRAPH_NODE_NONE) goto fail;

        GraphNode *o = graph_mul_mat(g, attn, t_o, (n_embd != q_dim) && t_o->dim[0] == (i64)q_dim);
        if (o == GRAPH_NODE_NONE) goto fail;

        GraphNode *h = graph_binary(g, OP_ADD, cur, o);
        if (h == GRAPH_NODE_NONE) goto fail;

        /* ---- SwiGLU FFN ---- */
        GraphNode *hn = graph_rms_norm(g, h, lw->tensors[TENSOR_POST_ATTN_NORM]);
        if (hn == GRAPH_NODE_NONE) goto fail;

        GraphNode *gate = graph_silu(g, graph_mul_mat(g, hn, t_g, t_g->dim[0] == (i64)n_embd));
        GraphNode *up   = graph_mul_mat(g, hn, t_u, t_u->dim[0] == (i64)n_embd);
        if (gate == GRAPH_NODE_NONE || up == GRAPH_NODE_NONE) goto fail;

        GraphNode *mul = graph_binary(g, OP_MUL, gate, up);
        if (mul == GRAPH_NODE_NONE) goto fail;

        GraphNode *down = graph_mul_mat(g, mul, t_d, t_d->dim[0] == (i64)fh);
        if (down == GRAPH_NODE_NONE) goto fail;

        cur = graph_binary(g, OP_ADD, h, down);
        if (cur == GRAPH_NODE_NONE) goto fail;

        cur = graph_layer_boundary(g, cur, s, l);
        if (cur == GRAPH_NODE_NONE) goto fail;
    }

    /* ---- Final norm ---- */
    graph_set_layer(g, c->n_layer);
    GraphNode *fn = graph_rms_norm(g, cur, w->tensors[TENSOR_OUTPUT_NORM]);
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

const ArchOps llama_ops = {
    .init              = llama_init,
    .free              = llama_free,
    .reset             = llama_reset,
    .decode            = llama_decode,
    .graph_build       = llama_graph_build,
    .chat_prompt       = llama3_chat_prompt,
    .chat_continuation = llama3_chat_continuation,
};
