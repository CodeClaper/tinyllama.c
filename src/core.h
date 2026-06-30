#include "def.h"

#define DEFAULT_TEMPERATURE 1.0f
#define DEFAULT_TOP_P 1.0f
#define DEFAULT_MIN_P 0.05f

bool model_get_i32(Model *m, const char *key, i32 *out);
Model *model_load(const char *path);
Engine *engine_open(EngineOptons *opts);
void engine_close(Engine *en);
void engine_summary(Engine *en);
Session *session_create(Engine *en, u32 ctx_size);
void session_free(Session *s);
i32 vocab_lookup(Vocab *v, const char *text);
