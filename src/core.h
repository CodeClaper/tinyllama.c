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

/* Stack buffer threshold for bias dequant (see bias_add). */
#define BIAS_BUF_STACK 4096

/* ---- Operators (tensor compute kernels; implementations in core.c) ---- */
void rms_norm(float *o, const float *x, TensorInfo *tw, const u8 *base, int n, float eps);
void bias_add(float *dst, TensorInfo *tb, const u8 *base, u32 n);
bool mat_vec_mul(float *y, TensorInfo *tw, const u8 *base, const float *x, u64 rows, u64 cols, bool trans, pthreads_t *pool);
bool mat_mat_mul(float *Y, TensorInfo *tw, const u8 *base, const float *X, u64 batch, u64 rows, u64 cols, bool trans, pthreads_t *pool);
void rope(float *buf, u32 n_heads, u32 head_dim, u32 pos, float theta_base);
void rope_neox(float *buf, u32 n_heads, u32 head_dim, u32 pos, float theta_base);
void rope_partial(float *buf, u32 n_heads, u32 head_dim, u32 rope_dim, u32 pos, float theta_base);
void silu(float *x, int n);
void softmax(float *x, u32 n);
float softplus(float x);
float sigmoid(float x);

/* ---- Tensor dequant accessors ---- */
float tensor_get_f32(TensorInfo *ti, const u8 *base, u64 i);
void  tensor_get_f32_batch(TensorInfo *ti, const u8 *base, u64 i0, u64 nb, float *out);

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
