#include "model.h"
#include "../core.h"
#include "../mm.h"
#include "../graph.h"
#include "../slog.h"

static bool llama_init(Session *s) {
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

    return true;
}

static void llama_reset(Session *s) {
    KvCache *kc = &s->cache;
    for (u32 i = 0; i < kc->n_layer; i++) {
        kc->std[i].n = 0;
    }
    s->n_tokens = 0;
}

static void llama_free(Session *s) {
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
}

/* ---- Ops table ------------------------------------------------ */

const ArchOps llama_ops = {
    .init          = llama_init,
    .free          = llama_free,
    .reset         = llama_reset,
    .graph_execute = graph_execute,
};
