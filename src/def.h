#include <stdbool.h>
#include <stdint.h>

#ifndef __DEF_H__
#define __DEF_H__

#define MAX_DIMS 8
#define VOCAB_ID_NONE ((u32)-1)

typedef uint8_t     u8;
typedef uint16_t    u16;
typedef uint32_t    u32;
typedef uint64_t    u64;
typedef int8_t      i8;
typedef int16_t     i16;
typedef int32_t     i32;
typedef int64_t     i64;

typedef struct {
    const char *model_path;
    bool inspect;
} EngineOptons; 

typedef struct {
    EngineOptons engine;
    const char *host;
    int port;
    int ctx_size;
    int default_tokens;
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
    ModelArch arch;
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
    KV  *kv;
    TensorInfo *tensor;
    u64 bytes;
} Model;

typedef struct {
    Model *model;
    Vocab *vocab;
    Weights *weights;
} Engine;

typedef struct {
    Engine *en;
} Session;

#endif 
