#include <stdio.h>
#include "model.h"
#include "../utils.h"
#include "../core.h"
#include "../mm.h"
#include "../graph.h"
#include "../slog.h"

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

    slog(INFO, "Gemma3 init: n_embd=%u n_head=%u n_kv_head=%u head_dim=%u kv_head_dim=%u "
         "n_layer=%u n_vocab=%u ctx_size=%u",
         c->n_embd, c->n_head, c->n_kv_head, c->head_dim, kv_head_dim,
         c->n_layer, c->n_vocab, s->ctx_size);
    slog(INFO, "Gemma3 init: ffn_hidden=%u rope_theta=%.6f",
         ws->ffn_hidden, ws->rope_theta);

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


static int gemma3_decode(const u8 *raw, int raw_len, char *out, int max_len) {

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
    
    return g;
fail:
    graph_free(g);
    return NULL;
}

const ArchOps gemma3_ops = {
    .init          = gemma3_init,
    .free          = gemma3_free,
    .reset         = gemma3_reset,
    .decode        = gemma3_decode,
    .graph_build   = gemma3_graph_build,
};
