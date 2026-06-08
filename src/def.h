#include <stdbool.h>
#include <stdint.h>

#ifndef __DEF_H__
#define __DEF_H__

#define MAX_DIMS 8

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
} Engine;

#endif 
