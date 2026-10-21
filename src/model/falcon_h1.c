#include <math.h>
#include <stdio.h>
#include <string.h>
#include "model.h"
#include "../core.h"
#include "../mm.h"
#include "../graph.h"
#include "../slog.h"
#include "../tokenizer.h"
#include "../utils.h"

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

    /* Falcon-H1 mixes dense attention with a Mamba2 SSM in every layer. */
    u32    ssm_d_inner;     /* ssm.inner_size                               */
    u32    ssm_d_state;     /* ssm.state_size                               */
    u32    ssm_n_group;     /* ssm.group_count                              */
    u32    ssm_n_head;      /* ssm.time_step_rank                           */
    u32    ssm_head_dim;    /* ssm_d_inner / ssm_n_head                     */
    u32    ssm_conv_kernel; /* ssm.conv_kernel                              */
    u32    ssm_conv_dim;    /* d_inner + 2 * n_group * d_state              */
    float  rope_theta;
    float **ssm_state;      /* [n_layer][n_head * head_dim * d_state]       */
    float **conv_state;     /* [n_layer][(conv_kernel - 1) * conv_dim]      */
} FalconH1WorkSpace;

static bool fh1_get_i32(Model *m, const char *suffix, i32 *out) {
    char key[96];
    const char *pfx = m->arch_name[0] ? m->arch_name : "falcon-h1";
    int n = snprintf(key, sizeof(key), "%s.%s", pfx, suffix);
    if (n > 0 && (size_t)n < sizeof(key) && model_get_i32(m, key, out))
        return true;
    if (strcmp(pfx, "falcon-h1") != 0) {
        n = snprintf(key, sizeof(key), "falcon-h1.%s", suffix);
        if (n > 0 && (size_t)n < sizeof(key) && model_get_i32(m, key, out))
            return true;
    }
    return false;
}

static bool fh1_get_f32(Model *m, const char *suffix, float *out) {
    char key[96];
    const char *pfx = m->arch_name[0] ? m->arch_name : "falcon-h1";
    int n = snprintf(key, sizeof(key), "%s.%s", pfx, suffix);
    if (n > 0 && (size_t)n < sizeof(key) && model_get_f32(m, key, out))
        return true;
    if (strcmp(pfx, "falcon-h1") != 0) {
        n = snprintf(key, sizeof(key), "falcon-h1.%s", suffix);
        if (n > 0 && (size_t)n < sizeof(key) && model_get_f32(m, key, out))
            return true;
    }
    return false;
}

/* Decide whether a 2-D weight has to be read transposed for y = W @ x
 * producing `out` rows from `in` columns (see qwen35.c). */
static bool fh1_mm_trans(TensorInfo *t, u64 out, u64 in) {
    if (!t || t->ndim < 2) return false;
    u64 d0 = (u64)t->dim[0], d1 = (u64)t->dim[1];
    if (d0 == out && d1 == in) return false;   /* stored [out, in] */
    if (d0 == in  && d1 == out) return true;   /* stored [in, out] */
    return d0 == in;
}

static bool falcon_h1_init(Session *s) {
    ArchConfig *c = &s->cfg;
    Model      *m = s->en->model;

    /* The Mamba2 graph ops currently have no CUDA/Metal twin.  Keep the
     * whole model on the CPU so a GPU build still runs it (instead of
     * failing on an unsupported op mid-graph). */
    if (s->n_gpu_layers > 0)
        slog(WARN, "Falcon-H1: Mamba2 is CPU-only; ignoring -ngl and using the CPU");
    s->n_gpu_layers = 0;

    /* Falcon-H1 Instruct is ChatML: <|im_end|> terminates an assistant
     * reply even though the GGUF's generic eos id is <|endoftext|>.
     * Without this the model runs past its answer and starts writing the
     * next user turn, which corrupts every following chat round. */
    if (s->en->vocab && s->en->vocab->im_end_id != VOCAB_ID_NONE)
        s->en->vocab->eos_id = s->en->vocab->im_end_id;

    u32 q_dim       = c->n_head * c->head_dim;
    u32 kv_head_dim = c->kv_head_dim ? c->kv_head_dim : c->head_dim;
    u32 kv_dim      = c->n_kv_head * kv_head_dim;

    s->tokens = scalloc((u64)s->ctx_size, sizeof(u32));
    s->logits = scalloc((u64)c->n_vocab, sizeof(float));

    FalconH1WorkSpace *ws = scalloc(1, sizeof(FalconH1WorkSpace));
    ws->x      = scalloc((u64)c->n_embd, sizeof(float));
    ws->xb     = scalloc((u64)c->n_embd, sizeof(float));
    ws->xb2    = scalloc((u64)c->n_embd, sizeof(float));
    ws->q      = scalloc((u64)q_dim, sizeof(float));
    ws->k      = scalloc((u64)kv_dim, sizeof(float));
    ws->v      = scalloc((u64)kv_dim, sizeof(float));
    ws->scores = scalloc((u64)s->ctx_size, sizeof(float));

    /* ---- SSM (Mamba2) geometry from GGUF metadata ---- */
    i32 v32;
    ws->ssm_d_inner     = 2 * c->n_embd;
    ws->ssm_d_state     = 128;
    ws->ssm_n_group     = 1;
    ws->ssm_n_head      = c->n_embd / 64 ? c->n_embd / 64 : 1;
    ws->ssm_conv_kernel = 4;
    if (fh1_get_i32(m, "ssm.inner_size",     &v32)) ws->ssm_d_inner     = (u32)v32;
    if (fh1_get_i32(m, "ssm.state_size",     &v32)) ws->ssm_d_state     = (u32)v32;
    if (fh1_get_i32(m, "ssm.group_count",    &v32)) ws->ssm_n_group     = (u32)v32;
    if (fh1_get_i32(m, "ssm.time_step_rank", &v32)) ws->ssm_n_head      = (u32)v32;
    if (fh1_get_i32(m, "ssm.conv_kernel",    &v32)) ws->ssm_conv_kernel = (u32)v32;
    if (ws->ssm_n_group == 0) ws->ssm_n_group = 1;
    if (ws->ssm_n_head == 0)  ws->ssm_n_head  = 1;
    if (ws->ssm_conv_kernel == 0) ws->ssm_conv_kernel = 4;
    ws->ssm_head_dim = ws->ssm_d_inner ? ws->ssm_d_inner / ws->ssm_n_head : 0;
    ws->ssm_conv_dim = ws->ssm_d_inner + 2 * ws->ssm_n_group * ws->ssm_d_state;

    ws->rope_theta = 10000.0f;
    float f32;
    if (fh1_get_f32(m, "rope.freq_base", &f32)) ws->rope_theta = f32;

    /* Derived FFN scratch (the graph itself allocates its own slots). */
    TensorInfo *t_g = s->en->weights->layers[0].tensors[TENSOR_FFN_GATE];
    u32 ffn_hidden = (t_g && t_g->ndim >= 1) ? (u32)t_g->dim[0] : c->n_embd * 4;
    ws->hb  = scalloc((u64)ffn_hidden, sizeof(float));
    ws->hb2 = scalloc((u64)ffn_hidden, sizeof(float));

    /* Per-layer recurrent state: conv ring + Mamba2 SSM matrix. */
    ws->conv_state = scalloc((u64)c->n_layer, sizeof(float *));
    ws->ssm_state  = scalloc((u64)c->n_layer, sizeof(float *));
    u64 conv_sz  = (u64)(ws->ssm_conv_kernel - 1) * ws->ssm_conv_dim;
    u64 state_sz = (u64)ws->ssm_n_head * ws->ssm_head_dim * ws->ssm_d_state;
    for (u32 i = 0; i < c->n_layer; i++) {
        BackendType b = graph_layer_backend(s, i);
        ws->conv_state[i] = graph_state_alloc(conv_sz * sizeof(float), b);
        ws->ssm_state[i]  = graph_state_alloc(state_sz * sizeof(float), b);
        if (!ws->conv_state[i] || !ws->ssm_state[i])
            slog(ERROR, "Falcon-H1: state allocation failed for layer %u", i);
    }

    slog(INFO, "Falcon-H1 init: n_embd=%u n_head=%u n_kv_head=%u head_dim=%u "
         "n_layer=%u n_vocab=%u ctx_size=%u",
         c->n_embd, c->n_head, c->n_kv_head, c->head_dim,
         c->n_layer, c->n_vocab, s->ctx_size);
    slog(INFO, "Falcon-H1 SSM: d_inner=%u d_state=%u n_group=%u n_head=%u "
         "head_dim=%u conv_kernel=%u conv_dim=%u rope_theta=%.1f",
         ws->ssm_d_inner, ws->ssm_d_state, ws->ssm_n_group, ws->ssm_n_head,
         ws->ssm_head_dim, ws->ssm_conv_kernel, ws->ssm_conv_dim,
         (double)ws->rope_theta);

    s->arch_data = ws;
    return true;
}

static void falcon_h1_reset(Session *s) {
    s->n_tokens = 0;

    FalconH1WorkSpace *ws = (FalconH1WorkSpace *)s->arch_data;
    if (!ws) return;
    u64 conv_sz  = (u64)(ws->ssm_conv_kernel - 1) * ws->ssm_conv_dim;
    u64 state_sz = (u64)ws->ssm_n_head * ws->ssm_head_dim * ws->ssm_d_state;
    for (u32 i = 0; i < s->cfg.n_layer; i++) {
        BackendType b = graph_layer_backend(s, i);
        if (ws->conv_state && ws->conv_state[i])
            graph_state_zero(ws->conv_state[i], conv_sz * sizeof(float), b);
        if (ws->ssm_state && ws->ssm_state[i])
            graph_state_zero(ws->ssm_state[i], state_sz * sizeof(float), b);
    }
}

static void falcon_h1_free(Session *s) {
    sfree(s->tokens);
    sfree(s->logits);
    s->tokens = NULL;
    s->logits = NULL;

    FalconH1WorkSpace *ws = (FalconH1WorkSpace *)s->arch_data;
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
        if (ws->conv_state) {
            for (u32 i = 0; i < s->cfg.n_layer; i++) {
                BackendType b = graph_layer_backend(s, i);
                graph_state_free(ws->conv_state[i], b);
                graph_state_free(ws->ssm_state[i], b);
            }
            sfree(ws->conv_state);
            sfree(ws->ssm_state);
        }
        sfree(ws);
        s->arch_data = NULL;
    }
}

static int falcon_h1_decode(const u8 *raw, int raw_len, char *out, int max_len) {
    int w = 0;
    for (int b = 0; b < raw_len && w < max_len; b++) {
        unsigned char ch = raw[b];
        if (ch >= 0x21 && ch <= 0x7E) {
            out[w++] = (char)ch;
        } else if (ch >= 0xC2 && ch <= 0xC5 && b + 1 < raw_len) {
            unsigned char nc = raw[b + 1];
            int byte = -1;
            if (nc >= 0x80 && nc <= 0xBF) {
                if      (ch == 0xC2) byte = (int)nc;
                else if (ch == 0xC3) byte = (int)nc + 0x40;
                else if (ch == 0xC4) {
                    if      (nc <= 0xA0) byte = nc - 0x80;
                    else if (nc == 0xA1) byte = 0x7F;
                    else                 byte = nc - 0x22;
                } else if (ch == 0xC5) {
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
                out[w++] = (char)ch;
            }
        } else {
            out[w++] = (char)ch;
        }
    }
    if (w < max_len) out[w] = '\0';
    return w;
}

/* Build the static graph.  Every Falcon-H1 layer runs dense attention
 * and a Mamba2 mixer in parallel on the same normed input:
 *
 *   x = x + attn(norm(x)) + ssm(norm(x))
 *   x = x + ffn(norm(x))
 *
 * The Mamba2 branch projects with ssm_in, convolves/silu's the xBC
 * slice, runs the selective scan (using the z slice as gate and dt as
 * decay), then projects back with ssm_out. */
static Graph *falcon_h1_graph_build(Session *s, u32 n_tokens) {
    ArchConfig *c         = &s->cfg;
    Weights    *w         = s->en->weights;
    FalconH1WorkSpace *ws = (FalconH1WorkSpace *)s->arch_data;

    if (!s || !w || !ws || n_tokens == 0 || n_tokens > s->ctx_size) return NULL;

    u32 n_embd   = c->n_embd;
    u32 q_dim    = c->n_head * c->head_dim;
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

        TensorInfo *t_q    = lw->tensors[TENSOR_ATTN_Q];
        TensorInfo *t_k    = lw->tensors[TENSOR_ATTN_K];
        TensorInfo *t_v    = lw->tensors[TENSOR_ATTN_V];
        TensorInfo *t_o    = lw->tensors[TENSOR_ATTN_OUT];
        TensorInfo *t_g    = lw->tensors[TENSOR_FFN_GATE];
        TensorInfo *t_u    = lw->tensors[TENSOR_FFN_UP];
        TensorInfo *t_d    = lw->tensors[TENSOR_FFN_DOWN];
        TensorInfo *t_in   = lw->tensors[TENSOR_SSM_IN];
        TensorInfo *t_conv = lw->tensors[TENSOR_SSM_CONV1D];
        TensorInfo *t_cb   = lw->tensors[TENSOR_SSM_CONV1D_BIAS];
        TensorInfo *t_a    = lw->tensors[TENSOR_SSM_A];
        TensorInfo *t_dd   = lw->tensors[TENSOR_SSM_D];
        TensorInfo *t_dt   = lw->tensors[TENSOR_SSM_DT_BIAS];
        TensorInfo *t_sn   = lw->tensors[TENSOR_SSM_NORM];
        TensorInfo *t_so   = lw->tensors[TENSOR_SSM_OUT];
        if (!t_q || !t_k || !t_v || !t_o || !t_g || !t_u || !t_d ||
            !t_in || !t_conv || !t_a || !t_dd || !t_dt || !t_so) {
            slog(WARN, "graph_build: layer %u missing tensors", l);
            goto fail;
        }
        if (!ws->conv_state[l] || !ws->ssm_state[l]) {
            slog(WARN, "graph_build: layer %u has no SSM state", l);
            goto fail;
        }

        GraphNode *n = graph_rms_norm(g, cur, lw->tensors[TENSOR_ATTN_NORM]);
        if (n == GRAPH_NODE_NONE) goto fail;

        /* ---- Dense attention branch ---- */
        GraphNode *q = graph_mul_mat(g, n, t_q, fh1_mm_trans(t_q, q_dim, n_embd));
        GraphNode *k = graph_mul_mat(g, n, t_k, fh1_mm_trans(t_k, c->n_kv_head * c->kv_head_dim, n_embd));
        GraphNode *v = graph_mul_mat(g, n, t_v, fh1_mm_trans(t_v, c->n_kv_head * c->kv_head_dim, n_embd));
        if (q == GRAPH_NODE_NONE || k == GRAPH_NODE_NONE || v == GRAPH_NODE_NONE) goto fail;
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

        q = graph_rope(g, q, ws->rope_theta, c->n_head, c->head_dim, rope_dim);
        k = graph_rope(g, k, ws->rope_theta, c->n_kv_head, c->kv_head_dim, rope_dim);
        if (q == GRAPH_NODE_NONE || k == GRAPH_NODE_NONE) goto fail;

        GraphNode *attn = graph_attn(g, q, k, v, l, 0);
        if (attn == GRAPH_NODE_NONE) goto fail;
        attn = graph_mul_mat(g, attn, t_o, fh1_mm_trans(t_o, n_embd, q_dim));
        if (attn == GRAPH_NODE_NONE) goto fail;

        /* ---- Mamba2 SSM branch ---- */
        u32 proj_dim = ws->ssm_d_inner + ws->ssm_conv_dim + ws->ssm_n_head;
        GraphNode *proj = graph_mul_mat(g, n, t_in, fh1_mm_trans(t_in, proj_dim, n_embd));
        if (proj == GRAPH_NODE_NONE) goto fail;

        u32 conv_st = graph_state(g, ws->conv_state[l]);
        u32 ssm_st  = graph_state(g, ws->ssm_state[l]);
        if (conv_st == GRAPH_STATE_NONE || ssm_st == GRAPH_STATE_NONE) goto fail;

        GraphNode *cv = graph_mamba2_conv(g, proj, t_conv, t_cb, conv_st,
                                          ws->ssm_conv_kernel, ws->ssm_d_inner,
                                          ws->ssm_n_group, ws->ssm_d_state,
                                          ws->ssm_n_head);
        if (cv == GRAPH_NODE_NONE) goto fail;

        GraphNode *scan = graph_mamba2_scan(g, cv, proj, t_a, t_dd, t_dt, t_sn,
                                            ssm_st, ws->ssm_n_head, ws->ssm_head_dim,
                                            ws->ssm_d_state, ws->ssm_n_group,
                                            ws->ssm_d_inner);
        if (scan == GRAPH_NODE_NONE) goto fail;

        GraphNode *ssm = graph_mul_mat(g, scan, t_so,
                                       fh1_mm_trans(t_so, n_embd, ws->ssm_d_inner));
        if (ssm == GRAPH_NODE_NONE) goto fail;

        /* ---- Parallel residual + FFN ---- */
        cur = graph_binary(g, OP_ADD, cur, attn);
        if (cur == GRAPH_NODE_NONE) goto fail;
        cur = graph_binary(g, OP_ADD, cur, ssm);
        if (cur == GRAPH_NODE_NONE) goto fail;

        TensorInfo *t_fn = lw->tensors[TENSOR_FFN_NORM];
        if (!t_fn) t_fn = lw->tensors[TENSOR_POST_ATTN_NORM];
        GraphNode *hn = graph_rms_norm(g, cur, t_fn);
        if (hn == GRAPH_NODE_NONE) goto fail;

        u32 ffn_hidden = (u32)t_g->dim[0];
        GraphNode *gate = graph_silu(g, graph_mul_mat(g, hn, t_g,
                                       fh1_mm_trans(t_g, ffn_hidden, n_embd)));
        GraphNode *up   = graph_mul_mat(g, hn, t_u,
                                       fh1_mm_trans(t_u, ffn_hidden, n_embd));
        if (gate == GRAPH_NODE_NONE || up == GRAPH_NODE_NONE) goto fail;

        GraphNode *mul = graph_binary(g, OP_MUL, gate, up);
        if (mul == GRAPH_NODE_NONE) goto fail;
        GraphNode *down = graph_mul_mat(g, mul, t_d,
                                       fh1_mm_trans(t_d, n_embd, ffn_hidden));
        if (down == GRAPH_NODE_NONE) goto fail;

        cur = graph_binary(g, OP_ADD, cur, down);
        if (cur == GRAPH_NODE_NONE) goto fail;

        cur = graph_layer_boundary(g, cur, s, l);
        if (cur == GRAPH_NODE_NONE) goto fail;
    }

    graph_set_layer(g, c->n_layer);
    TensorInfo *t_norm = w->tensors[TENSOR_OUTPUT_NORM];
    if (!t_norm) t_norm = w->layers[c->n_layer - 1].tensors[TENSOR_FFN_NORM];
    GraphNode *fn = graph_rms_norm(g, cur, t_norm);
    if (fn == GRAPH_NODE_NONE) goto fail;

    TensorInfo *t_out = w->tensors[TENSOR_OUTPUT];
    if (!t_out) t_out = te;
    if (graph_mul_mat(g, fn, t_out, t_out->dim[0] == (i64)n_embd) == GRAPH_NODE_NONE)
        goto fail;

    return g;
fail:
    graph_free(g);
    return NULL;
}

const ArchOps falcon_h1_ops = {
    .init              = falcon_h1_init,
    .free              = falcon_h1_free,
    .reset             = falcon_h1_reset,
    .decode            = falcon_h1_decode,
    .graph_build       = falcon_h1_graph_build,
    .chat_prompt       = chatml_prompt,
    .chat_continuation = chatml_continuation,
};
