#include "def.h"

#define DEFAULT_TEMPERATURE 1.0f
#define DEFAULT_TOP_P 1.0f
#define DEFAULT_MIN_P 0.05f

float tensor_get_f32(TensorInfo *ti, const u8 *base, u64 i);
void rms_norm(float *o, const float *x, TensorInfo *tw, const u8 *base, int n, float eps);
bool mat_vec_mul(float *y, TensorInfo *tw, const u8 *base, const float *x, u64 rows, u64 cols, bool trans);
void rope(float *buf, u32 n_heads, u32 head_dim, u32 pos, float theta_base);
void silu(float *x, int n);
void softmax(float *x, u32 n);
float softplus(float x);
bool model_get_i32(Model *m, const char *key, i32 *out);
bool model_get_f32(Model *m, const char *key, float *out);
Model *model_load(const char *path);
Engine *engine_open(EngineOptons *opts);
void engine_close(Engine *en);
void engine_summary(Engine *en);
Session *session_create(Engine *en, u32 ctx_size);
void session_free(Session *s);
i32 vocab_lookup(Vocab *v, const char *text);
bool vocab_lookup_len(Vocab *v, const char *text, int len, i32 *id);
i32 vocab_merge_rank(Vocab *v, i32 id1, i32 id2);
i32 vocab_merge_result(Vocab *v, i32 id1, i32 id2);
