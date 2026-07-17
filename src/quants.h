#ifndef __QUANTS_H__
#define __QUANTS_H__

#include "def.h"

/* ---- public API ---- */
float gguf_dequant(TensorInfo *ti, const u8 *base, u64 i);

/* ---- shared by CPU-specific dequant kernels ---- */
float f16_to_f32(u16 bits);
i8   k_scale_6bit(const u8 *scales, u32 sb);
u8   k_scale_6bit_u(const u8 *scales, u32 sb);
void q4k_scale_min(const u8 *s, u32 sb, u8 *scale, u8 *min);

extern const float iq2_xxs_grid[];
extern const float iq2_xs_grid[];
extern const float iq3_xxs_grid[];
extern const float iq1_s_grid[];
extern const float iq4_nl_values[];
extern const float iq3_s_grid[];

#endif
