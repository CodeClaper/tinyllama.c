#ifndef __QUANTS_H__
#define __QUANTS_H__

#include "def.h"

float gguf_dequant(TensorInfo *ti, const u8 *base, u64 i);

#endif
