#include <fcntl.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <inttypes.h>
#include <stdint.h>
#include <time.h>
#include "core.h"
#include "mm.h"
#include "slog.h"
#include "utils.h"

#define GGUF_MAGIC              0x46554747u /* "GGUF", little endian. */
#define GGUF_VALID_VERSION      3           /* GGUF valid version, only support 3. */
#define GGUF_MAX_DIMS           8           /* GGUF max dims. */
#define GGUF_DEFAULT_ALIGNMENT  32          /* GGUF default alignment. */

typedef struct {
    u8 *base;
    u64 size;
    u64 post;
    char error[125];
} Cursor;

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

static bool cursor_u32(Cursor *c, u32 *v) {
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

static const GGUFTypeInfo *tensor_type(u32 type) {
    u32 n = sizeof(gguf_types) / sizeof(gguf_types[0]);
    if (type >= n || gguf_types[type].name == NULL) return NULL;
    else return &gguf_types[type];
}

static const char *tensor_name(u32 type) {
    const GGUFTypeInfo *info = tensor_type(type);
    return info == NULL ? "unknow" : info->name;
}

static bool tensor_bytes(u32 type, u64 n_element, u64 *bytes) {
    const GGUFTypeInfo *info = tensor_type(type);
    if (info == NULL || info->block_elems == 0) return false;
    u64 blocks = (n_element + info->block_elems - 1) / info->block_elems;
    if (blocks > UINT64_MAX / info->block_bytes) return false;
    *bytes = blocks * info->block_bytes;
    return true;
}

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
    map = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
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

/* Load engine. */
Engine *engine_load(EngineOptons *opts) {
    Engine *en = smalloc(sizeof(*en));
    en->model = model_load(opts->model_path);
    return en;
}
