#include <stdio.h>
#include <string.h>
#include "arch.h"
#include "../core.h"
#include "../mm.h"
#include "../slog.h"
#include "../utils.h"

void arch_config_init(Engine *en, ArchConfig *cfg) {
    memset(cfg, 0, sizeof(*cfg));
    Weights *w = en->weights;
    Vocab   *v = en->vocab;

    /* Common defaults derived from weights / vocab. */
    cfg->n_layer = w->n_layer;
    cfg->n_vocab = v ? v->n_vocab : 0;

    /* Derive n_embd from token_embd tensor dims. */
    TensorInfo *te = w->tensors[TENSOR_TOKEN_EMBD];
    if (te && te->ndim >= 1) cfg->n_embd = (u32)te->dim[te->ndim - 1];

    const char *pfx = arch_key_prefix(w->arch);

    /* Helper: try reading an i32 metadata key for this arch. */
    i32 v32;
    #define TRY_I32(suffix, field) do {                       \
        char k[96];                                           \
        int  n = snprintf(k, sizeof(k), "%s.%s", pfx, suffix);\
        if (n > 0 && (size_t)n < sizeof(k))                   \
            if (model_get_i32(en->model, k, &v32))            \
                cfg->field = (u32)v32;                        \
    } while(0)

    /* Attention head count. */
    TRY_I32("attention.head_count",     n_head);

    /* KV head count (GQA support). */
    TRY_I32("attention.head_count_kv",  n_kv_head);
    if (cfg->n_kv_head == 0) cfg->n_kv_head = cfg->n_head; /* MHA fallback */

    /* Head dimension. */
    TRY_I32("attention.head_dim",       head_dim);
    if (cfg->head_dim == 0 && cfg->n_head > 0)
        cfg->head_dim = cfg->n_embd / cfg->n_head;

    /* ---- DeepSeek MLA ---- */
    TRY_I32("attention.kv_lora_rank",     kv_lora_rank);
    TRY_I32("attention.qk_nope_head_dim", qk_nope_head_dim);
    TRY_I32("attention.qk_rope_head_dim", qk_rope_head_dim);

    #undef TRY_I32

    slog(INFO, "ArchConfig: arch=%s n_embd=%u n_head=%u n_kv_head=%u head_dim=%u n_layer=%u n_vocab=%u",
         arch_key_prefix(w->arch), cfg->n_embd, cfg->n_head,
         cfg->n_kv_head, cfg->head_dim, cfg->n_layer, cfg->n_vocab);

    if (cfg->kv_lora_rank)
        slog(INFO, "ArchConfig MLA: kv_lora_rank=%u qk_nope_head_dim=%u qk_rope_head_dim=%u",
             cfg->kv_lora_rank, cfg->qk_nope_head_dim, cfg->qk_rope_head_dim);
}

const char *arch_key_prefix(ModelArch arch) {
    switch (arch) {
        case ARCH_LLAMA:    return "llama";
        case ARCH_QWEN2:    return "qwen2";
        case ARCH_DEEPSEEK: return "deepseek2";
        case ARCH_FALCON:   return "falcon";
        default:            return "llama";
    }
}
