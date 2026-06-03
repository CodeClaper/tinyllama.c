#ifndef __DEF_H__
#define __DEF_H__


typedef struct {
    const char *model_path;
} EngineOptons; 

typedef struct {
    EngineOptons engine;
    const char *host;
    int port;
    int ctx_size;
    int default_tokens;
} ServerOptions;

#endif 
