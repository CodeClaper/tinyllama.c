#include "model.h"
#include "../utils.h"
#include "../core.h"
#include "../mm.h"
#include "../graph.h"
#include "../slog.h"

static bool deepseek_init(Session *s) {
    ArchConfig *c = &s->cfg;

    KvCache *kc = &s->cache;
    kc->n_layer   = c->n_layer;
    kc->head_dim  = c->head_dim;
    kc->n_kv_head = c->n_kv_head;
    kc->mla       = scalloc((u64)c->n_layer, sizeof(MLAKvCache));

    for (u32 i = 0; i < c->n_layer; i++) {
        MLAKvCache *mc = &kc->mla[i];

        /* Raw KV buffer — capacity = ctx_size. */
        mc->cap_raw = (u32)s->ctx_size;
        mc->raw_kv  = scalloc((u64)mc->cap_raw * (u64)c->head_dim,
                              sizeof(float));

        /* Compressed KV (MLA latent space). */
        mc->compress_ratio = c->kv_lora_rank > 0 ? c->head_dim / c->kv_lora_rank : 2;
        mc->comp_cap = (u32)s->ctx_size;
        u64 comp_elems = (u64)mc->comp_cap * (u64)c->kv_lora_rank;

        mc->attn_comp_kv      = scalloc(comp_elems, sizeof(float));
        mc->attn_state_kv     = scalloc(comp_elems, sizeof(float));
        mc->attn_state_score  = scalloc((u64)mc->comp_cap, sizeof(float));

        mc->index_comp_kv     = scalloc(comp_elems, sizeof(float));
        mc->index_state_kv    = scalloc(comp_elems, sizeof(float));
        mc->index_state_score = scalloc((u64)mc->comp_cap, sizeof(float));
    }

    s->tokens = scalloc((u64)s->ctx_size, sizeof(u32));
    s->logits = scalloc((u64)c->n_vocab, sizeof(float));

    return true;
}

static void deepseek_reset(Session *s) {
    KvCache *kc = &s->cache;
    for (u32 i = 0; i < kc->n_layer; i++) {
        kc->mla[i].n_raw  = 0;
        kc->mla[i].n_comp = 0;
    }
    s->n_tokens = 0;
}

static void deepseek_free(Session *s) {
    KvCache *kc = &s->cache;
    if (kc->mla) {
        for (u32 i = 0; i < kc->n_layer; i++) {
            MLAKvCache *mc = &kc->mla[i];
            sfree(mc->raw_kv);
            sfree(mc->attn_comp_kv);
            sfree(mc->attn_state_kv);
            sfree(mc->attn_state_score);
            sfree(mc->index_comp_kv);
            sfree(mc->index_state_kv);
            sfree(mc->index_state_score);
        }
        sfree(kc->mla);
        kc->mla = NULL;
    }
    sfree(s->tokens);
    sfree(s->logits);
    s->tokens = NULL;
    s->logits = NULL;
}

const ArchOps deepseek_ops = {
    .init          = deepseek_init,
    .free          = deepseek_free,
    .reset         = deepseek_reset,
    .graph_execute = graph_execute,
};
