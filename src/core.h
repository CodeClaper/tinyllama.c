#include "def.h"

bool model_get_i32(Model *m, const char *key, i32 *out);
Model *model_load(const char *path);
Engine *engine_open(EngineOptons *opts);
void engine_close(Engine *en);
void engine_summary(Engine *en);
Session *session_create(Engine *en, u32 ctx_size);
void session_free(Session *s);
