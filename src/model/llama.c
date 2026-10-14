#include "model.h"
#include "../core.h"
#include "../mm.h"
#include "../graph.h"
#include "../slog.h"

static bool llama_init(Session *s) {
    ArchConfig *c = &s->cfg;

    s->tokens = scalloc((u64)s->ctx_size, sizeof(u32));
    s->logits = scalloc((u64)c->n_vocab, sizeof(float));

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
}

/* ---- Ops table ------------------------------------------------ */

const ArchOps llama_ops = {
    .init          = llama_init,
    .free          = llama_free,
    .reset         = llama_reset,
};
