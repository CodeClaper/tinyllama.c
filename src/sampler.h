#ifndef __SAMPLER_H__
#define __SAMPLER_H__

#include "def.h"

u32 sample_token(float *logits, u32 n_vocab, const SampleParam *p,
                 u32 *history_tokens, u32 n_history);

#endif
