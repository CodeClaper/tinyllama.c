#include <stdbool.h>
#include <stdint.h>

#ifndef __DEF_H__
#define __DEF_H__

#define MAX_DIMS 8
#define VOCAB_ID_NONE ((u32)-1)
#define DEFAULT_EPS (1e-6f)

typedef uint8_t     u8;
typedef uint16_t    u16;
typedef uint32_t    u32;
typedef uint64_t    u64;
typedef int8_t      i8;
typedef int16_t     i16;
typedef int32_t     i32;
typedef int64_t     i64;

/* GGUF tensor types — format-level constants, shared by all arches. */
typedef enum {
    GGUF_TYPE_F32      = 0,
    GGUF_TYPE_F16      = 1,
    GGUF_TYPE_Q4_0     = 2,
    GGUF_TYPE_Q4_1     = 3,
    GGUF_TYPE_Q5_0     = 6,
    GGUF_TYPE_Q5_1     = 7,
    GGUF_TYPE_Q8_0     = 8,
    GGUF_TYPE_Q8_1     = 9,
    GGUF_TYPE_Q2_K     = 10,
    GGUF_TYPE_Q3_K     = 11,
    GGUF_TYPE_Q4_K     = 12,
    GGUF_TYPE_Q5_K     = 13,
    GGUF_TYPE_Q6_K     = 14,
    GGUF_TYPE_Q8_K     = 15,
    GGUF_TYPE_IQ2_XXS  = 16,
    GGUF_TYPE_IQ2_XS   = 17,
    GGUF_TYPE_IQ3_XXS  = 18,
    GGUF_TYPE_IQ1_S    = 19,
    GGUF_TYPE_IQ4_NL   = 20,
    GGUF_TYPE_IQ3_S    = 21,
    GGUF_TYPE_IQ2_S    = 22,
    GGUF_TYPE_IQ4_XS   = 23,
    GGUF_TYPE_I8       = 24,
    GGUF_TYPE_I16      = 25,
    GGUF_TYPE_I32      = 26,
    GGUF_TYPE_I64      = 27,
    GGUF_TYPE_F64      = 28,
    GGUF_TYPE_IQ1_M    = 29,
    GGUF_TYPE_BF16     = 30,
} GGUFType;

typedef enum {
    TOKENIZER_TYPE_NONE,
    TOKENIZER_TYPE_SPM,
    TOKENIZER_TYPE_BPE,
    TOKENIZER_TYPE_WPM,
    TOKENIZER_TYPE_UGM,
    TOKENIZER_TYPE_RWKV,
    TOKENIZER_TYPE_WHISPER
} TokenizerType;

typedef struct {
    const char *model_path;
    bool inspect;
} EngineOptons;

typedef struct {
    EngineOptons engine;
    char *host;
    int port;
    int ctx_size;
    int default_tokens;
    int nthread;
    float temperature;
    u32 top_k;
    float top_p;
    float min_p;
    float repeat_penalty;
    u32 repeat_last_n;
    bool inspect;
} ServerOptions;

typedef struct {
    u8 len;
    char *content;
} Key;

typedef struct {
    u32 type;
    u64 len;
    u64 data_pos;
} ArrayRef;

typedef struct {
    Key key;
    u32 type;
    u64 value_pos;
} KV;

typedef struct {
    Key key;
    u32 ndim;
    u64 dim[MAX_DIMS];
    u32 type;
    u64 offset;
    u64 n_element;
    u64 bytes;
    u64 padded_dim; /* padded last-dimension stride for quantised types */
} TensorInfo;

typedef struct {
    Key key;
    i32 value;
    bool used;
} TokenizerEntry;

typedef struct {
    TokenizerEntry *entry;
    u64 cap;
    u64 used;
} TokenizerTable;

typedef struct {
    Key *token;
    u32 n_vocab;
    i32 bos_id;
    i32 eos_id;
    i32 user_id;
    i32 assistant_id;
    i32 think_start_id;
    i32 think_end_id;
    i32 dsml_id;
    i32 im_start_id;
    i32 im_end_id;
    i32 byte_token_ids[256];
    TokenizerType  tokenizer_type;
    TokenizerTable tokens;
    TokenizerTable merges;
} Vocab;

typedef enum {
    ARCH_UNKNOWN,
    ARCH_LLAMA,
    ARCH_QWEN2,
    ARCH_DEEPSEEK,
    ARCH_FALCON,
} ModelArch;

typedef enum {
    TENSOR_TOKEN_EMBD,
    TENSOR_OUTPUT,
    TENSOR_OUTPUT_NORM,
    TENSOR_ATTN_NORM,
    TENSOR_ATTN_Q,
    TENSOR_ATTN_K,
    TENSOR_ATTN_V,
    TENSOR_ATTN_Q_BIAS,
    TENSOR_ATTN_K_BIAS,
    TENSOR_ATTN_V_BIAS,
    TENSOR_ATTN_QKV,
    TENSOR_ATTN_OUT,
    TENSOR_ATTN_GATE,
    TENSOR_POST_ATTN_NORM,
    TENSOR_FFN_GATE,
    TENSOR_FFN_DOWN,
    TENSOR_FFN_UP,
    TENSOR_SSM_IN,
    TENSOR_SSM_CONV1D,
    TENSOR_SSM_ALPHA,
    TENSOR_SSM_BETA,
    TENSOR_SSM_A,
    TENSOR_SSM_DT_BIAS,
    TENSOR_SSM_OUT,
    TENSOR_SSM_NORM,
    TENSOR_ATTN_Q_NORM,
    TENSOR_ATTN_K_NORM,
    TENSOR_OUTPUT_HC_BASE,
    TENSOR_OUTPUT_HC_FN,
    TENSOR_OUTPUT_HC_SCALE,
    TENSOR_COUNT,
} TensorRole;

typedef struct {
    TensorInfo *tensors[TENSOR_COUNT];
} LayerWeights;

typedef struct {
    u32 n_layer;
    TensorInfo *tensors[TENSOR_COUNT];
    LayerWeights *layers;
} Weights;

typedef struct {
    int fd;
    u64 size;
    u8  *map;
    u32 version;
    u64 n_kv;
    u64 n_tensor;
    u64 alignment;
    ModelArch arch;
    char  arch_name[32];
    KV  *kv;
    TensorInfo *tensor;
    u64 bytes;
} Model;

typedef struct {
    Model *model;
    Vocab *vocab;
    Weights *weights;
} Engine;

/* ================================================================
 * Architecture Config
 * ================================================================
 * Hyper-parameters extracted from GGUF metadata, used by all
 * architecture backends to size buffers / configure computation. */

typedef struct {
    u32 n_head;           /* attention.head_count                  */
    u32 n_kv_head;        /* attention.head_count_kv (1 = MHA)     */
    u32 head_dim;         /* attention head dimension (Q)          */
    u32 kv_head_dim;      /* K/V per-head dimension                */
    u32 n_embd;           /* embedding_length                     */
    u32 n_layer;          /* block_count                          */
    u32 n_vocab;          /* vocabulary size                      */

    /* ---- DeepSeek MLA ---- */
    u32 kv_lora_rank;     /* attention.kv_lora_rank               */
    u32 qk_nope_head_dim; /* attention.qk_nope_head_dim           */
    u32 qk_rope_head_dim; /* attention.qk_rope_head_dim           */
} ArchConfig;

/* ================================================================
 * KV Cache
 * ================================================================
 * Two cache shapes: standard attention (Llama-style) and MLA
 * (DeepSeek).  The KvCache union picks the right one at init
 * time based on ModelArch. */

/* Standard per-layer KV cache — Llama / Qwen2 / Falcon. */
typedef struct {
    float *k;       /* [cap * n_kv_head * head_dim]               */
    float *v;       /* [cap * n_kv_head * head_dim]               */
    u32    n;       /* cached token count                         */
    u32    cap;     /* capacity (= ctx_size)                      */
} AttnKvCache;

/* DeepSeek MLA per-layer cache — compressed latent KV. */
typedef struct {
    float *raw_kv;
    u32    n_raw;
    u32    cap_raw;

    u32    compress_ratio;
    u32    comp_cap;
    u32    n_comp;
    float *attn_comp_kv;
    float *attn_state_kv;
    float *attn_state_score;

    u32    n_index_comp;
    float *index_comp_kv;
    float *index_state_kv;
    float *index_state_score;
} MLAKvCache;

typedef struct {
    u32 n_layer;
    u32 head_dim;
    u32 n_kv_head;
    union {
        AttnKvCache *std;   /* [n_layer] — non-MLA architectures   */
        MLAKvCache  *mla;   /* [n_layer] — DeepSeek MLA            */
    };
} KvCache;

/* ================================================================
 * Session
 * ================================================================
 * Session owns the inference loop state and delegates per-
 * architecture computation to an ArchOps vtable.  Server code
 * only sees the generic Session type. */

typedef struct Session Session;

typedef struct {
    bool (*init)    (Session *s);
    void (*free)    (Session *s);
    bool (*forward) (Session *s, u32 *tokens, u32 n_tokens, float *logits);
    void (*reset)   (Session *s);
    int  (*decode)  (const u8 *raw, int raw_len, char *out, int max_len);
} ArchOps;

struct Session {
    Engine    *en;
    ArchConfig cfg;
    ArchOps    ops;

    /* ---- Inference state ---- */
    u32       *tokens;       /* [ctx_size] ring buffer             */
    u32        n_tokens;     /* current position in ring buffer    */
    float     *logits;       /* [n_vocab] output buffer            */
    u32        ctx_size;

    /* ---- KV cache ---- */
    KvCache    cache;

    /* ---- Arch-specific workspace ---- */
    void       *arch_data;

    /* ---- Thread pool ---- */
    struct pthreads *pthreads;

    /* ---- Sampling ---- */
    float      temperature;
    u32        top_k;
    float      top_p;
    float      min_p;
    float      repeat_penalty;
    u32        repeat_last_n;
    u32        max_tokens;
};

#endif
