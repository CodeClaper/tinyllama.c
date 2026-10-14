#include "model.h"
#include "../utils.h"
#include "../core.h"
#include "../mm.h"
#include "../graph.h"
#include "../slog.h"

static bool deepseek_init(Session *s) {
    ArchConfig *c = &s->cfg;

    s->tokens = scalloc((u64)s->ctx_size, sizeof(u32));
    s->logits = scalloc((u64)c->n_vocab, sizeof(float));

    return true;
}

static void deepseek_reset(Session *s) {
    s->n_tokens = 0;
}

static void deepseek_free(Session *s) {
    sfree(s->tokens);
    sfree(s->logits);
    s->tokens = NULL;
    s->logits = NULL;
}

const ArchOps deepseek_ops = {
    .init          = deepseek_init,
    .free          = deepseek_free,
    .reset         = deepseek_reset,
};
