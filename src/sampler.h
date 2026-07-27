#ifndef __SAMPLER_H__
#define __SAMPLER_H__

#include "def.h"

u32 sample_greedy(float *logits, u32 n_vocab);
u32 sample_token(float *logits, u32 n_vocab, float temperature, u32 top_k, float top_p, float min_p);

#endif
