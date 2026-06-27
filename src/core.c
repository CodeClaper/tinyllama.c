#include <errno.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <inttypes.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>
#include <sys/file.h>
#include "core.h"
#include "mm.h"
#include "slog.h"
#include "utils.h"
#include "arch/arch.h"

#define GGUF_MAGIC              0x46554747u /* "GGUF", little endian. */
#define GGUF_VALID_VERSION      3           /* GGUF valid version, only support 3. */
#define GGUF_MAX_DIMS           8           /* GGUF max dims. */
#define GGUF_DEFAULT_ALIGNMENT  32          /* GGUF default alignment. */

typedef enum {
    GGUF_VALUE_UINT8   = 0,
    GGUF_VALUE_INT8    = 1,
    GGUF_VALUE_UINT16  = 2,
    GGUF_VALUE_INT16   = 3,
    GGUF_VALUE_UINT32  = 4,
    GGUF_VALUE_INT32   = 5,
    GGUF_VALUE_FLOAT32 = 6,
    GGUF_VALUE_BOOL    = 7,
    GGUF_VALUE_STRING  = 8,
    GGUF_VALUE_ARRAY   = 9,
    GGUF_VALUE_UINT64  = 10,
    GGUF_VALUE_INT64   = 11,
    GGUF_VALUE_FLOAT64 = 12,
} GGUFValueType;

typedef struct {
    const char *name;
    u32 block_elems;
    u32 block_bytes;
} GGUFTypeInfo;

static const GGUFTypeInfo gguf_types[] = {
    [0]  = {"f32",      1,   4},
    [1]  = {"f16",      1,   2},
    [2]  = {"q4_0",    32,  18},
    [3]  = {"q4_1",    32,  20},
    [6]  = {"q5_0",    32,  22},
    [7]  = {"q5_1",    32,  24},
    [8]  = {"q8_0",    32,  34},
    [9]  = {"q8_1",    32,  40},
    [10] = {"q2_k",   256,  84},
    [11] = {"q3_k",   256, 110},
    [12] = {"q4_k",   256, 144},
    [13] = {"q5_k",   256, 176},
    [14] = {"q6_k",   256, 210},
    [15] = {"q8_k",   256, 292},
    [16] = {"iq2_xxs",256,  66},
    [17] = {"iq2_xs", 256,  74},
    [18] = {"iq3_xxs",256,  98},
    [19] = {"iq1_s",  256, 110},
    [20] = {"iq4_nl", 256,  50},
    [21] = {"iq3_s",  256, 110},
    [22] = {"iq2_s",  256,  82},
    [23] = {"iq4_xs", 256, 136},
    [24] = {"i8",       1,   1},
    [25] = {"i16",      1,   2},
    [26] = {"i32",      1,   4},
    [27] = {"i64",      1,   8},
    [28] = {"f64",      1,   8},
    [29] = {"iq1_m",  256,  56},
    [30] = {"bf16",     1,   2},
};

typedef struct {
    u8 *base;
    u64 size;
    u64 post;
    char error[125];
} Cursor;

typedef enum {
    TOKENIZER_TYPE_NONE,
    TOKENIZER_TYPE_SPM,
    TOKENIZER_TYPE_BPE,
    TOKENIZER_TYPE_WPM,
    TOKENIZER_TYPE_UGM,
    TOKENIZER_TYPE_RWKV,
    TOKENIZER_TYPE_WHISPER
} TokenizerType;


static int global_lock_fd = -1;

static Cursor cursor_at(u8 *base, u64 size, u64 post) {
    Cursor c = {
        .base = base,
        .size = size,
        .post = post,
        .error = {0}
    };
    return c;
}

static void cursor_error(Cursor *c, const char *msg) {
    if (c->error[0] == '\0') {
        snprintf(c->error,  sizeof(c->error), "%s at byte %" PRIu64, msg, c->post);
    }
}

static bool cursor_has(Cursor *c, u64 n) {
    if (n > c->size || c->post + n > c->size) {
        cursor_error(c, "Truncated GGUF file");
        return false;
    }
    return true;
}

static bool cursor_skip(Cursor *c, u64 n) {
    if (!cursor_has(c, n)) return false;
    c->post += n;
    return true;
}

static bool cursor_read(Cursor *c, void *dest, u64 n) {
    if (!cursor_has(c, n)) return false;
    memcpy(dest, c->base + c->post, (size_t)n);
    c->post += n;
    return true;
}

static bool cursor_i32(Cursor *c, i32 *v) {
    return cursor_read(c, v, sizeof(*v));
}

static bool cursor_u32(Cursor *c, u32 *v) {
    return cursor_read(c, v, sizeof(*v));
}

static bool cursor_float(Cursor *c, float *v) {
    return cursor_read(c, v, sizeof(*v));
}

static bool cursor_u64(Cursor *c, u64 *v) {
    return cursor_read(c, v, sizeof(*v));
}

static bool cursor_key(Cursor *c, Key *k) {
    u64 len;
    if (!cursor_u64(c, &len)) return false;
    if (!cursor_has(c, len)) return false;
    k->content = (char *)(c->base + c->post);
    k->len = len;
    c->post += len;
    return true;
}

static bool cursor_skip_value(Cursor *c, GGUFValueType type, int depth) {
    if (depth > GGUF_MAX_DIMS) {
        cursor_error(c, "Metadata array nesting is too deep");
        return false;
    }

    switch (type) {
        case GGUF_VALUE_UINT8: case GGUF_VALUE_INT8: case GGUF_VALUE_BOOL:
            return cursor_skip(c, 1);
        case GGUF_VALUE_UINT16: case GGUF_VALUE_INT16:
            return cursor_skip(c, 2);
        case GGUF_VALUE_UINT32: case GGUF_VALUE_INT32: case GGUF_VALUE_FLOAT32:
            return cursor_skip(c, 4);
        case GGUF_VALUE_UINT64: case GGUF_VALUE_INT64: case GGUF_VALUE_FLOAT64:
            return cursor_skip(c, 8);
        case GGUF_VALUE_STRING: {
            Key ignore;
            return cursor_key(c, &ignore);
        }
        case GGUF_VALUE_ARRAY: {
            u32 item_type;
            u64 len;
            
            if (!cursor_u32(c, &item_type)) return false;
            if (!cursor_u64(c, &len)) return false;
            for (u64 i = 0; i < len; i++) {
                if (!cursor_skip_value(c, item_type, depth + 1)) return false;
            }

            return true;
        }
        default:
            cursor_error(c, "Unknown GGUF metadata type");
            return false;
    }
}

static void tokenizer_table_init(TokenizerTable *table, u64 expected) {
    table->cap = next_pow2(expected * 2 + 16);
    table->used = 0;
    table->entry = scalloc(table->cap, sizeof(table->entry[0]));
}

static void tokenizer_table_free(TokenizerTable *table) {
    if (table) {
        sfree(table->entry);
        memset(table, 0, sizeof(*table));
    }
}

static void tokenizer_table_put(TokenizerTable *table, Key key, i32 value) {
    u64 mask = table->cap - 1;
    u64 hash = hash_bytes(key.content, key.len);
    u64 i = hash & mask;
    
    while (table->entry[i].used) {
        if (key_eq(table->entry[i].key, key)) {
            table->entry[i].value = value;
            return;
        }
        i = (i + 1) & mask;
    }
    
    table->entry[i].used = true;
    table->entry[i].key = key;
    table->entry[i].value = value;
    table->used++;
}

static bool tokenizer_table_get(TokenizerTable *table, Key key, i32 *value) {
    if (table->cap == 0) return false;

    u64 mask = table->cap - 1;
    u64 hash = hash_bytes(key.content, key.len);
    u64 i = hash & mask;

    while (table->entry[i].used) {
        if (key_eq(table->entry[i].key, key)) {
            *value = table->entry[i].value;
            return true;
        }
        i = (i + 1) & mask;
    }
    return false;
}


/* Sumary model. */
static const char *gguf_value_type_name(u32 type) {
    switch (type) {
        case GGUF_VALUE_UINT8:   return "u8";
        case GGUF_VALUE_INT8:    return "i8";
        case GGUF_VALUE_UINT16:  return "u16";
        case GGUF_VALUE_INT16:   return "i16";
        case GGUF_VALUE_UINT32:  return "u32";
        case GGUF_VALUE_INT32:   return "i32";
        case GGUF_VALUE_FLOAT32: return "f32";
        case GGUF_VALUE_BOOL:    return "bool";
        case GGUF_VALUE_STRING:  return "string";
        case GGUF_VALUE_UINT64:  return "u64";
        case GGUF_VALUE_INT64:   return "i64";
        case GGUF_VALUE_FLOAT64: return "f64";
        default:                 return "unknown";
    }
}

static const GGUFTypeInfo *tensor_type(u32 type) {
    u32 n = sizeof(gguf_types) / sizeof(gguf_types[0]);
    if (type >= n || gguf_types[type].name == NULL) return NULL;
    else return &gguf_types[type];
}

static bool tensor_bytes(u32 type, u64 n_element, u64 *bytes) {
    const GGUFTypeInfo *info = tensor_type(type);
    if (info == NULL || info->block_elems == 0) return false;
    u64 blocks = (n_element + info->block_elems - 1) / info->block_elems;
    if (blocks > UINT64_MAX / info->block_bytes) return false;
    *bytes = blocks * info->block_bytes;
    return true;
}

static void release_instance_lock(void) {
    if (global_lock_fd >= 0) {
        close(global_lock_fd);
        global_lock_fd = -1;
    }
}

static void acquire_instance_lock(void) {
    const char *path = getenv("TINY_LLAMA_LOCK");
    if (!path || !path[0]) path = "/tmp/tiny_llama.lock";

    const int fd = open(path, O_RDWR | O_CREAT, 0600);
    if (fd < 0) slog_errno("Fail to open lock file");
    (void)fcntl(fd, F_SETFD, FD_CLOEXEC);
    
    if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
        if (errno == EWOULDBLOCK) {
            char buf[64];
            ssize_t n = pread(fd, buf, sizeof(buf) - 1, 0);
            long owner = -1;
            if (n > 0) {
                buf[n] = '\0';
                owner = parse_long(buf);
            }
            close(fd);
            if (owner > 0) {
                slog(ERROR, "Found another tiny_llama process is already running (pid %ld); refusing to start.", owner);
            } else {
                slog(ERROR, "Found another tiny_llama process is already running; refusing to start.");
            }
        }

        close(fd);
        slog_errno("Failed to lock %s", path);
    }

    if (ftruncate(fd, 0) != 0) {
        close(fd);
        slog_errno("Failed to truncated lock file:", path);
    }

    (void)dprintf(fd, "%ld\n", (long)getpid());
    global_lock_fd = fd;
    atexit(release_instance_lock);
}

/* Model find kv. */
static KV *model_find_kv(Model *m, char *s) {
    for (u64 i = 0; i < m->n_kv; i++) {
        if (key_streq(m->kv[i].key, s)) return &m->kv[i];
    }
    return NULL;
}

static bool model_get_key(Model *m, char *s, Key *out) {
    KV *kv = model_find_kv(m, s);
    if (!kv || kv->type != GGUF_VALUE_STRING) return false;
    Cursor c = cursor_at(m->map, m->size, kv->value_pos);
    return cursor_key(&c, out);
}

static bool model_get_array(Model *m, char *s, ArrayRef *out) {
    KV *kv = model_find_kv(m, s);
    if (!kv || kv->type != GGUF_VALUE_ARRAY) return false;
    Cursor c = cursor_at(m->map, m->size, kv->value_pos);
    if (!cursor_u32(&c, &out->type)) return false;
    if (!cursor_u64(&c, &out->len)) return false;
    out->data_pos = c.post;
    return true;
}

bool model_get_i32(Model *m, const char *key, i32 *out) {
    KV *kv = model_find_kv(m, (char *)key);
    if (!kv || kv->type != GGUF_VALUE_UINT32) return false;
    Cursor c = cursor_at(m->map, m->size, kv->value_pos);
    return cursor_i32(&c, out);
}

static ModelArch model_detect_arch(Model *m) {
    Key arch_name;
    if (!model_get_key(m, "general.architecture", &arch_name)) return ARCH_UNKNOWN;
    if (key_streq(arch_name, "llama"))   return ARCH_LLAMA;
    if (key_streq(arch_name, "qwen2"))   return ARCH_QWEN2;
    if (key_streq(arch_name, "qwen35"))  return ARCH_QWEN2;
    if (key_streq(arch_name, "deepseek2")) return ARCH_DEEPSEEK;
    if (key_streq(arch_name, "falcon"))  return ARCH_FALCON;
    slog(WARN, "Unknown architecture: %s, loading as generic.", get_key_name(arch_name));
    return ARCH_UNKNOWN;
}

static TensorInfo *model_find_tensor(Model *m, char *name) {
    for (u64 i = 0; i < m->n_tensor; i++) {
        if (key_streq(m->tensor[i].key, name))
            return &m->tensor[i];
    }
    return NULL;
}

static u32 model_count_layers(Model *m) {
    u32 max_n = 0;
    for (u64 i = 0; i < m->n_tensor; i++) {
        Key *k = &m->tensor[i].key;
        if (k->len < 5 || memcmp(k->content, "blk.", 4) != 0) continue;
        u32 n = 0;
        u32 j;
        for (j = 4; j < k->len && k->content[j] >= '0' && k->content[j] <= '9'; j++)
            n = n * 10 + (u32)(k->content[j] - '0');
        if (j > 4 && j < k->len && k->content[j] == '.' && n >= max_n)
            max_n = n;
    }
    return max_n + 1;
}

/* Load KVs.. */
KV *kv_load(Model *m, Cursor *c) {
    KV *kv = scalloc(m->n_kv, sizeof(m->kv[0]));
    m->alignment = GGUF_DEFAULT_ALIGNMENT;

    for (u64 i = 0; i < m->n_kv; i++) {
        KV *v = &kv[i];

        if (!cursor_key(c, &v->key)) slog(ERROR, c->error);
        if (!cursor_u32(c, &v->type)) slog(ERROR, c->error);
        v->value_pos = c->post;

        if (key_streq(v->key, "general.alignment") && 
            kv->type == GGUF_VALUE_UINT32
        ) {
            u32 alignment;
            Cursor tmp = cursor_at(m->map, m->size, kv->value_pos);
            if (cursor_u32(&tmp, &alignment) && alignment != 0) m->alignment = alignment;
        }
        
        if (!cursor_skip_value(c, v->type, 0)) slog(ERROR, c->error);
    }

    return kv;
}

/* Load tensor. */
TensorInfo *tensor_load(Model *m, Cursor *c) {
    TensorInfo *tensor = scalloc(m->n_tensor, sizeof(m->tensor[0]));
    
    for (u64 i = 0; i < m->n_tensor; i++) {
        TensorInfo *t = &tensor[i];

        if (!cursor_key(c, &t->key)) slog(ERROR, c->error);
        if (!cursor_u32(c, &t->ndim)) slog(ERROR, c->error);
        if (t->ndim == 0 || t->ndim > GGUF_MAX_DIMS) slog(ERROR, "Tensor has an unsupported number of dimensions: %d", t->ndim);

        t->n_element = 1;
        for (u32 d = 0; d < t->ndim; d++) {
            if (!cursor_u64(c, &t->dim[d])) slog(ERROR, c->error);
            if (t->dim[d] != 0 && t->n_element > UINT64_MAX / t->dim[d]) slog(ERROR, "Tensor element count overflow");
            t->n_element *= t->dim[d];
        }

        if (!cursor_u32(c, &t->type)) slog(ERROR, c->error);
        if (!cursor_u64(c, &t->offset)) slog(ERROR, c->error); 
        if (!tensor_bytes(t->type, t->n_element, &t->bytes)) slog(WARN, "Tensor %s has unsupported GGUF type %u", get_key_name(t->key), t->type);
    }

    return tensor;
}

static TokenizerType tokenizer_type(Model *m) {
    Key model_name;

    if (!model_get_key(m, "tokenizer.ggml.model", &model_name)) slog(ERROR, "GGUF tokenizer model type is missing");
    if (key_streq(model_name, "llama") || key_streq(model_name, "sentencepiece")) return TOKENIZER_TYPE_SPM;
    else if (key_streq(model_name, "gpt2")) return TOKENIZER_TYPE_BPE;
    else if (key_streq(model_name, "bert")) return TOKENIZER_TYPE_WPM;
    else if (key_streq(model_name, "unigram")) return TOKENIZER_TYPE_UGM;
    else if (key_streq(model_name, "rwkv")) return TOKENIZER_TYPE_RWKV;
    else if (key_streq(model_name, "whisper")) return TOKENIZER_TYPE_WHISPER;

    slog(ERROR, "Unknown tokenizer model type");
    return TOKENIZER_TYPE_NONE;
}

static bool vocab_try_lookup(Vocab *v, const char *text, i32 *id) {
    Key key = {.content = (char *)text, .len = strlen(text)};
    return tokenizer_table_get(&v->tokens, key, id);
}

/* Load vocab for BPE. */
static Vocab *vocab_load_for_bpe(Model *m) {
    Vocab *v;
    ArrayRef tokens, merges;
    
    v = smalloc(sizeof(Vocab));
    if (!model_get_array(m, "tokenizer.ggml.tokens", &tokens) ||
        tokens.type != GGUF_VALUE_STRING ||
        tokens.len > INT32_MAX
    ) slog(ERROR, "GGUF tokenizer token table is missing or invalid");
    if (!model_get_array(m, "tokenizer.ggml.merges", &merges) ||
        merges.type != GGUF_VALUE_STRING 
    ) slog(ERROR, "GGUF tokenizer merge table is missing or invalid");

    v->n_vocab = (int)tokens.len;
    v->token = scalloc((size_t)v->n_vocab, sizeof(v->token[0]));

    tokenizer_table_init(&v->tokens, tokens.len);
    Cursor c = cursor_at(m->map, m->size, tokens.data_pos);
    for (u32 i = 0; i < v->n_vocab; i++) {
        if (!cursor_key(&c, &v->token[i])) slog(ERROR, c.error);
        tokenizer_table_put(&v->tokens, v->token[i], i);
    }

    tokenizer_table_init(&v->merges, merges.len);
    c = cursor_at(m->map, m->size, merges.data_pos);
    for (u32 i = 0; i < merges.len; i++) {
        Key merge;
        if (!cursor_key(&c, &merge)) slog(ERROR, c.error);
        tokenizer_table_put(&v->merges, merge, i);
    }

    v->bos_id = VOCAB_ID_NONE;
    v->eos_id = VOCAB_ID_NONE;

    if (!model_get_i32(m, "tokenizer.ggml.bos_token_id", &v->bos_id)) {
        static const char *bos_cands[] = {
            "<|begin_of_text|>",
            "<｜begin▁of▁sentence｜>",
            "<|im_start|>",
            "<bos>",
            "<s>",
        };
        for (int i = 0; i < (int)(sizeof(bos_cands) / sizeof(bos_cands[0])); i++) {
            if (vocab_try_lookup(v, bos_cands[i], &v->bos_id)) break;
        }
    }

    if (!model_get_i32(m, "tokenizer.ggml.eos_token_id", &v->eos_id)) {
        static const char *eos_cands[] = {
            "<|end_of_text|>",
            "<｜end▁of▁sentence｜>",
            "<|im_end|>",
            "<eos>",
            "</s>",
        };
        for (int i = 0; i < (int)(sizeof(eos_cands) / sizeof(eos_cands[0])); i++) {
            if (vocab_try_lookup(v, eos_cands[i], &v->eos_id)) break;
        }
    }

    if (v->bos_id == VOCAB_ID_NONE)
        slog(ERROR, "Cannot find BOS token in vocabulary");
    if (v->eos_id == VOCAB_ID_NONE)
        slog(ERROR, "Cannot find EOS token in vocabulary");

    v->user_id        = VOCAB_ID_NONE;
    v->assistant_id   = VOCAB_ID_NONE;
    v->think_start_id = VOCAB_ID_NONE;
    v->think_end_id   = VOCAB_ID_NONE;
    v->dsml_id        = VOCAB_ID_NONE;

    (void)vocab_try_lookup(v, "<｜User｜>",      &v->user_id);
    (void)vocab_try_lookup(v, "<｜Assistant｜>", &v->assistant_id);
    (void)vocab_try_lookup(v, "<think>",         &v->think_start_id);
    (void)vocab_try_lookup(v, "</think>",        &v->think_end_id);
    (void)vocab_try_lookup(v, "｜DSML｜",        &v->dsml_id);

    return v;
}

/* Load vocab. */
static Vocab *vocab_load(Model *m) {
    TokenizerType type = tokenizer_type(m);
    switch (type) {
        case TOKENIZER_TYPE_BPE: return vocab_load_for_bpe(m);
        case TOKENIZER_TYPE_NONE: ERRRET(NULL, "Unknown tokenizer model type");
        default: ERRRET(NULL, "Not support tokenizer type");
    }
}

/* Free vocab. */
static void vocab_free(Vocab *v) {
    if (!v) return;
    sfree(v->token);
    tokenizer_table_free(&v->tokens);
    tokenizer_table_free(&v->merges);
    memset(v, 0, sizeof(*v));
}

/* Tensor name map. */
typedef struct {
    TensorRole  role;
    const char *name;
    bool        required;
} TensorMapEntry;

#define ARR_LEN(a) ((int)(sizeof(a) / sizeof((a)[0])))

/* Architecture tensor maps (non-layer tensors only). */

static const TensorMapEntry llama_tensor_map[] = {
    {TENSOR_TOKEN_EMBD,  "token_embd.weight",   true},
    {TENSOR_OUTPUT,      "output.weight",       false},
    {TENSOR_OUTPUT_NORM, "output_norm.weight",  true},
};

static const TensorMapEntry qwen2_tensor_map[] = {
    {TENSOR_TOKEN_EMBD,  "token_embd.weight",   true},
    {TENSOR_OUTPUT,      "output.weight",       false},
    {TENSOR_OUTPUT_NORM, "output_norm.weight",  true},
};

static const TensorMapEntry deepseek_tensor_map[] = {
    {TENSOR_TOKEN_EMBD,       "token_embd.weight",         true},
    {TENSOR_OUTPUT,           "output.weight",             true},
    {TENSOR_OUTPUT_NORM,      "output_norm.weight",        true},
    {TENSOR_OUTPUT_HC_BASE,   "output_hc_base.weight",     true},
    {TENSOR_OUTPUT_HC_FN,     "output_hc_fn.weight",       true},
    {TENSOR_OUTPUT_HC_SCALE,  "output_hc_scale.weight",    true},
};

static const TensorMapEntry unknown_tensor_map[] = {
    {TENSOR_TOKEN_EMBD,  "token_embd.weight",   true},
    {TENSOR_OUTPUT,      "output.weight",       false},
    {TENSOR_OUTPUT_NORM, "output_norm.weight",  false},
};

/* Layer tensor suffix maps (blk.{N}.{suffix}.weight). */

typedef struct {
    TensorRole  role;
    const char *suffix;
    bool        required;
} LayerTensorMap;

static const LayerTensorMap llama_layer_map[] = {
    {TENSOR_ATTN_NORM,  "attn_norm",    true},
    {TENSOR_ATTN_Q,     "attn_q",       true},
    {TENSOR_ATTN_K,     "attn_k",       true},
    {TENSOR_ATTN_V,     "attn_v",       true},
    {TENSOR_ATTN_OUT,   "attn_output",  true},
    {TENSOR_FFN_GATE,   "ffn_gate",     true},
    {TENSOR_FFN_DOWN,   "ffn_down",     true},
    {TENSOR_FFN_UP,     "ffn_up",       true},
};

static const LayerTensorMap qwen2_layer_map[] = {
    {TENSOR_ATTN_NORM,        "attn_norm",             true},
    {TENSOR_ATTN_QKV,         "attn_qkv",              false},
    {TENSOR_ATTN_Q,           "attn_q",                false},
    {TENSOR_ATTN_K,           "attn_k",                false},
    {TENSOR_ATTN_V,           "attn_v",                false},
    {TENSOR_ATTN_Q_NORM,      "attn_q_norm",           false},
    {TENSOR_ATTN_K_NORM,      "attn_k_norm",           false},
    {TENSOR_ATTN_OUT,         "attn_output",           false},
    {TENSOR_ATTN_GATE,        "attn_gate",             false},
    {TENSOR_POST_ATTN_NORM,   "post_attention_norm",   true},
    {TENSOR_SSM_CONV1D,       "ssm_conv1d",            false},
    {TENSOR_SSM_ALPHA,        "ssm_alpha",             false},
    {TENSOR_SSM_BETA,         "ssm_beta",              false},
    {TENSOR_SSM_NORM,         "ssm_norm",              false},
    {TENSOR_SSM_OUT,          "ssm_out",               false},
    {TENSOR_FFN_GATE,         "ffn_gate",              true},
    {TENSOR_FFN_DOWN,         "ffn_down",              true},
    {TENSOR_FFN_UP,           "ffn_up",                true},
};

static const LayerTensorMap deepseek_layer_map[] = {
    {TENSOR_ATTN_NORM,  "attn_norm",    true},
    {TENSOR_ATTN_Q,     "attn_q",       true},
    {TENSOR_ATTN_K,     "attn_k",       true},
    {TENSOR_ATTN_V,     "attn_v",       true},
    {TENSOR_ATTN_OUT,   "attn_output",  true},
    {TENSOR_FFN_GATE,   "ffn_gate",     true},
    {TENSOR_FFN_DOWN,   "ffn_down",     true},
    {TENSOR_FFN_UP,     "ffn_up",       true},
};

static const LayerTensorMap unknown_layer_map[] = {
    {TENSOR_ATTN_NORM,  "attn_norm",    false},
    {TENSOR_ATTN_Q,     "attn_q",       false},
    {TENSOR_ATTN_K,     "attn_k",       false},
    {TENSOR_ATTN_V,     "attn_v",       false},
    {TENSOR_ATTN_QKV,   "attn_qkv",     false},
    {TENSOR_ATTN_OUT,   "attn_output",  false},
    {TENSOR_FFN_GATE,   "ffn_gate",     false},
    {TENSOR_FFN_DOWN,   "ffn_down",     false},
    {TENSOR_FFN_UP,     "ffn_up",       false},
};

static LayerWeights *layers_weights_load(Model *m, ModelArch arch, u32 n_layer) {
    LayerWeights *layers = scalloc(n_layer, sizeof(LayerWeights));

    static const struct {
        ModelArch arch;
        const LayerTensorMap *entries;
        int count;
    } arch_layer_maps[] = {
        {ARCH_LLAMA,    llama_layer_map,     ARR_LEN(llama_layer_map)},
        {ARCH_QWEN2,    qwen2_layer_map,     ARR_LEN(qwen2_layer_map)},
        {ARCH_DEEPSEEK, deepseek_layer_map,  ARR_LEN(deepseek_layer_map)},
        {ARCH_UNKNOWN,  unknown_layer_map,   ARR_LEN(unknown_layer_map)},
    };

    const LayerTensorMap *map = NULL;
    int map_count = 0;
    for (int i = 0; i < ARR_LEN(arch_layer_maps); i++) {
        if (arch_layer_maps[i].arch == arch) {
            map = arch_layer_maps[i].entries;
            map_count = arch_layer_maps[i].count;
            break;
        }
    }
    if (!map) {
        map = unknown_layer_map;
        map_count = ARR_LEN(unknown_layer_map);
    }

    for (u64 i = 0; i < m->n_tensor; i++) {
        Key *k = &m->tensor[i].key;
        if (k->len < 7) continue;
        if (memcmp(k->content, "blk.", 4) != 0) continue;

        u32 n = 0;
        u32 j;
        for (j = 4; j < k->len && k->content[j] >= '0' && k->content[j] <= '9'; j++)
            n = n * 10 + (u32)(k->content[j] - '0');
        if (j >= k->len || k->content[j] != '.' || n >= n_layer) continue;

        const char *suffix_start = (const char *)k->content + j + 1;
        u32 suffix_len = k->len - j - 1;

        /* Strip trailing ".weight" (GGUF tensors always end with it). */
        if (suffix_len <= 7 || memcmp(suffix_start + suffix_len - 7, ".weight", 7) != 0)
            continue;
        suffix_len -= 7;

        for (int e = 0; e < map_count; e++) {
            const char *s = map[e].suffix;
            u32 slen = (u32)strlen(s);
            if (slen == suffix_len && memcmp(suffix_start, s, slen) == 0) {
                layers[n].tensors[map[e].role] = &m->tensor[i];
                break;
            }
        }
    }

    return layers;
}

/* Load weights. */
static Weights *weights_load(Model *m) {
    Weights *w = smalloc(sizeof(*w));

    static const struct {
        ModelArch arch;
        const TensorMapEntry *entries;
        int count;
    } arch_maps[] = {
        {ARCH_LLAMA,    llama_tensor_map,    ARR_LEN(llama_tensor_map)},
        {ARCH_QWEN2,    qwen2_tensor_map,    ARR_LEN(qwen2_tensor_map)},
        {ARCH_DEEPSEEK, deepseek_tensor_map, ARR_LEN(deepseek_tensor_map)},
        {ARCH_UNKNOWN,  unknown_tensor_map,  ARR_LEN(unknown_tensor_map)},
    };

    w->arch = model_detect_arch(m);

    const TensorMapEntry *map = NULL;
    int map_count = 0;
    for (int i = 0; i < ARR_LEN(arch_maps); i++) {
        if (arch_maps[i].arch == w->arch) {
            map = arch_maps[i].entries;
            map_count = arch_maps[i].count;
            break;
        }
    }
    if (!map) {
        map = unknown_tensor_map;
        map_count = ARR_LEN(unknown_tensor_map);
    }

    for (int i = 0; i < map_count; i++) {
        TensorInfo *t = model_find_tensor(m, (char *)map[i].name);
        if (!t && map[i].required) 
            slog(ERROR, "Required tensor is missing: %s", map[i].name);
        w->tensors[map[i].role] = t;
    }

    if (!w->tensors[TENSOR_OUTPUT] && w->tensors[TENSOR_TOKEN_EMBD]) {
        w->tensors[TENSOR_OUTPUT] = w->tensors[TENSOR_TOKEN_EMBD];
        slog(WARN, "No output.weight found, using tied embeddings (output = token_embd)");
    }

    w->n_layer = model_count_layers(m);
    w->layers  = layers_weights_load(m, w->arch, w->n_layer);
    return w;
}

/* Load model. */
Model *model_load(const char *path) {
    Model *m;
    int fd;
    struct stat st;
    void *map;

    m = smalloc(sizeof(Model));
    fd = open(path, O_RDONLY);
    if (fd == -1) slog_errno("Cannot open model, which path: %s", path);
    if (fstat(fd, &st) == -1) slog_errno("Cannot stat model");
    if (st.st_size < 32) slog(ERROR, "Model file is too small to be GGUF.");
    map = mmap(NULL, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) slog_errno("Cannot mmap model");

    m->fd = fd;
    m->size = (u64)st.st_size;
    m->map = map;

    Cursor c = cursor_at(m->map, m->size, 0);
    u32 magic;
    if (!cursor_u32(&c, &magic)) slog(ERROR, c.error);
    if (magic != GGUF_MAGIC) slog(ERROR, "Model is not a GGUF file");
    if (!cursor_u32(&c, &m->version)) slog(ERROR, c.error);
    if (m->version != GGUF_VALID_VERSION) slog(ERROR, "Only GGUF v3 is supported.");
    if (!cursor_u64(&c, &m->n_tensor)) slog(ERROR, c.error);
    if (!cursor_u64(&c, &m->n_kv)) slog(ERROR, c.error);

    m->kv = kv_load(m, &c);
    m->tensor = tensor_load(m, &c); 

    return m;
}

static void model_summary(Model *m) {
    slog(INFO, "Metadata:");
    for (u64 i = 0; i < m->n_kv; i++) {
        KV *kv = &m->kv[i];
        printf("-|%-45s", get_key_name(kv->key));
        Cursor c = cursor_at(m->map, m->size, kv->value_pos);
        switch (kv->type) {
            case GGUF_VALUE_UINT32: {
                u32 v = 0;
                cursor_u32(&c, &v);
                printf("\t\t= %d\n", v);
                break;
            }
            case GGUF_VALUE_UINT64: {
                u64 v = 0;
                cursor_u64(&c, &v);
                printf("\t\t= %"PRIu64 "\n", v);
                break;
            }
            case GGUF_VALUE_FLOAT32: {
                float v = 0;
                cursor_float(&c, &v);
                printf("\t\t= %.2f\n", v);
                break;
            }
            case GGUF_VALUE_STRING: {
                Key v = {0};
                cursor_key(&c, &v);
                printf("\t\t= %s\n", get_key_name(v));
                break;
            }
            case GGUF_VALUE_ARRAY: {
                u32 item_type;
                u64 len;
                cursor_u32(&c, &item_type);
                cursor_u64(&c, &len);
                printf("\t\t= [%s * %"PRIu64"]\n",
                       gguf_value_type_name(item_type), len);
                break;
            }
            default: {
                printf("\t\t--\n");
                break;
            }
        }
    }
}


void model_close(Model *m) {

}

/* Open engine. */
Engine *engine_open(EngineOptons *opts) {
    Engine *en = smalloc(sizeof(*en));
    acquire_instance_lock();
    en->model = model_load(opts->model_path);
    if (!opts->inspect) en->vocab = vocab_load(en->model);
    en->weights = weights_load(en->model);
    return en;
}

/* Summary engine. */
void engine_summary(Engine *en) {
    model_summary(en->model);
}

void engine_close(Engine *en) {
    if (!en) return;
    vocab_free(en->vocab);
    model_close(en->model);
}

Session *session_create(Engine *en, u32 ctx_size) {
    if (!en || ctx_size == 0) return NULL;

    Session *s = smalloc(sizeof(*s));
    s->en       = en;
    s->ctx_size = ctx_size;
    s->temperature = 0.0f;
    s->top_p       = 1.0f;
    s->top_k       = 1;
    s->max_tokens  = 0;

    /* Fill architecture config from model metadata. */
    arch_config_init(en, &s->cfg);

    /* Bind the right ops table. */
    switch (en->weights->arch) {
        case ARCH_LLAMA:    s->ops = llama_ops;    break;
        case ARCH_QWEN2:    s->ops = qwen2_ops;    break;
        case ARCH_DEEPSEEK: s->ops = deepseek_ops; break;
        case ARCH_FALCON:   s->ops = falcon_ops;   break;
        default:
            slog(WARN, "Unknown architecture, falling back to llama.");
            s->ops = llama_ops;
            break;
    }

    /* Architecture-specific initialisation (allocates cache,
     * token buffer, logits). */
    if (!s->ops.init(s)) {
        slog(WARN, "Session init failed — cleaning up.");
        sfree(s);
        return NULL;
    }

    slog(INFO, "Session created: ctx_size=%u n_layer=%u n_embd=%u "
         "n_head=%u head_dim=%u",
         s->ctx_size, s->cfg.n_layer, s->cfg.n_embd,
         s->cfg.n_head, s->cfg.head_dim);

    return s;
}

void session_free(Session *s) {
    if (!s) return;
    if (s->ops.free) s->ops.free(s);
    sfree(s);
}

