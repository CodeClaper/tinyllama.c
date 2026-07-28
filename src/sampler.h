#ifndef __SAMPLER_H__
#define __SAMPLER_H__

#include "def.h"

u32 sample_greedy(float *logits, u32 n_vocab);
u32 sample_token(float *logits, u32 n_vocab,
                 float temperature, u32 top_k, float top_p, float min_p,
                 float repeat_penalty, u32 repeat_last_n,
                 float frequency_penalty, float presence_penalty,
                 u32 *history_tokens, u32 n_history);

#endif
