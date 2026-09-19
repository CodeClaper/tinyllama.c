#include "def.h"
#include "pthreads.h"
#include "quants.h"


#define DEFAULT_TEMPERATURE 1.0f
#define DEFAULT_TOP_K 40
#define DEFAULT_TOP_P 0.9f
#define DEFAULT_MIN_P 0.05f
#define DEFAULT_REPEAT_PENALTY 1.1f
#define DEFAULT_REPEAT_LAST_N 64
#define DEFAULT_FREQUENCY_PENALTY 0.0f
#define DEFAULT_PRESENCE_PENALTY 0.0f
#define DEFAULT_N_THREAD -1

/* ---- Tensor dequant accessors ---- */
float tensor_get_f32(TensorInfo *ti, u64 i);
void  tensor_get_f32_batch(TensorInfo *ti, u64 i0, u64 nb, float *out);

/* ---- Model metadata access & loading ---- */
bool model_get_i32(Model *m, const char *key, i32 *out);
bool model_get_f32(Model *m, const char *key, float *out);
Model *model_load(const char *path);

/* ---- Engine lifecycle ---- */
Engine *engine_open(EngineOptons *opts);
void engine_close(Engine *en);
void engine_summary(Engine *en);

/* ---- Session (inference context) lifecycle ---- */
Session *session_create(Engine *en, u32 ctx_size, int nthreads);
void session_free(Session *s);

/* ---- Vocab / BPE lookups ---- */
i32 vocab_lookup(Vocab *v, const char *text);
bool vocab_lookup_len(Vocab *v, const char *text, int len, i32 *id);
i32 vocab_merge_rank(Vocab *v, i32 id1, i32 id2);
i32 vocab_merge_result(Vocab *v, i32 id1, i32 id2);
