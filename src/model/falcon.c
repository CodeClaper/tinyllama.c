#include "model.h"
#include "../core.h"
#include "../mm.h"
#include "../graph.h"
#include "../slog.h"
#include "../utils.h"

static bool falcon_init(Session *s) {
    ArchConfig *c = &s->cfg;

    s->tokens = scalloc((u64)s->ctx_size, sizeof(u32));
    s->logits = scalloc((u64)c->n_vocab, sizeof(float));

    return true;
}

static void falcon_reset(Session *s) {
    s->n_tokens = 0;
}

static void falcon_free(Session *s) {
    sfree(s->tokens);
    sfree(s->logits);
    s->tokens = NULL;
    s->logits = NULL;
}

const ArchOps falcon_ops = {
    .init          = falcon_init,
    .free          = falcon_free,
    .reset         = falcon_reset,
};
