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
#define DEFAULT_N_UBATCH 512
#define DEFAULT_CTX_SIZE 4096

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
Session *session_create(Engine *en, u32 ctx_size, u32 n_ubatch,
                        int nthreads, int n_gpu_layers, int n_parallel);
void session_free(Session *s);

/* Resolve the context size when the user did not pass -c.  Returns the
 * model's GGUF context_length when it fits in available device memory,
 * otherwise the largest size that does (rounded down to 256 tokens).
 * With auto offload (n_gpu_layers < 0) the size is chosen so every layer
 * fits on the device, making the resolved -ngl L/L.  n_gpu_layers uses
 * the same convention as session_create().  When no device is in use the
 * model's full context length is returned. */
u32 engine_auto_ctx_size(Engine *en, u32 n_ubatch, int n_gpu_layers,
                         int n_parallel);

/* ---- Vocab / BPE lookups ---- */
i32 vocab_lookup(Vocab *v, const char *text);
bool vocab_lookup_len(Vocab *v, const char *text, int len, i32 *id);
i32 vocab_merge_rank(Vocab *v, i32 id1, i32 id2);
i32 vocab_merge_result(Vocab *v, i32 id1, i32 id2);
