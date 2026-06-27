#include "arch.h"
#include "../core.h"
#include "../mm.h"
#include "../slog.h"
#include "../utils.h"

static bool qwen2_init(Session *s) {
    ArchConfig *c = &s->cfg;

    KvCache *kc = &s->cache;
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

static bool qwen2_forward(Session *s, u32 token, float *logits) {
    UNUSED(s);
    UNUSED(token);
    UNUSED(logits);
    /* TODO: implement Qwen2 forward pass (self-attention + FFN,
     *       with optional SSM layers and Q/K norms). */
    slog(WARN, "qwen2_forward is not yet implemented");
    return false;
}

static void qwen2_reset(Session *s) {
    KvCache *kc = &s->cache;
    for (u32 i = 0; i < kc->n_layer; i++) {
        kc->std[i].n = 0;
    }
    s->n_tokens = 0;
}

static void qwen2_free(Session *s) {
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

const ArchOps qwen2_ops = {
    .init    = qwen2_init,
    .free    = qwen2_free,
    .forward = qwen2_forward,
    .reset   = qwen2_reset,
};
