#include <string.h>
#include "quants.h"
#include "slog.h"

/* Convert f16 to f32 — shared by all dequant paths. */
static inline float f16_to_f32(u16 bits) {
    u32 sign = (u32)(bits >> 15) << 31;
    u32 exp  = (bits >> 10) & 0x1f;
    u32 mant = bits & 0x3ff;
    u32 f32;
    if (exp == 0) {
        if (mant == 0) { f32 = sign; }
        else {
            int e = 1 - 15;
            while ((mant & 0x400) == 0) { mant <<= 1; e--; }
            mant &= 0x3ff;
            f32 = sign | ((u32)(e + 127) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f32 = sign | (0xff << 23) | (mant << 13); /* inf / nan */
    } else {
        f32 = sign | ((u32)(exp - 15 + 127) << 23) | (mant << 13);
    }
    float out;
    memcpy(&out, &f32, sizeof(out));
    return out;
}

/* ---- IQ lookup tables (from llama.cpp ggml-quants.c) -------------- */

/* iq2_xxs: 256-entry grid for 2.0625 bpw quant. */
static const float iq2_xxs_grid[] = {
    -0.2500f, -0.1875f, -0.1250f, -0.0625f, 0.0000f, 0.0625f, 0.1250f, 0.1875f,
    -0.2500f, -0.1875f, -0.1250f, -0.0625f, 0.0000f, 0.0625f, 0.1250f, 0.1875f,
    -0.2500f, -0.1875f, -0.1250f, -0.0625f, 0.0000f, 0.0625f, 0.1250f, 0.1875f,
    -0.2500f, -0.1875f, -0.1250f, -0.0625f, 0.0000f, 0.0625f, 0.1250f, 0.1875f,
    -0.2500f, -0.1875f, -0.1250f, -0.0625f, 0.0000f, 0.0625f, 0.1250f, 0.1875f,
    -0.2500f, -0.1875f, -0.1250f, -0.0625f, 0.0000f, 0.0625f, 0.1250f, 0.1875f,
    -0.2500f, -0.1875f, -0.1250f, -0.0625f, 0.0000f, 0.0625f, 0.1250f, 0.1875f,
    -0.2500f, -0.1875f, -0.1250f, -0.0625f, 0.0000f, 0.0625f, 0.1250f, 0.1875f,
    -0.2500f, -0.1875f, -0.1250f, -0.0625f, 0.0000f, 0.0625f, 0.1250f, 0.1875f,
    -0.2500f, -0.1875f, -0.1250f, -0.0625f, 0.0000f, 0.0625f, 0.1250f, 0.1875f,
    -0.2500f, -0.1875f, -0.1250f, -0.0625f, 0.0000f, 0.0625f, 0.1250f, 0.1875f,
    -0.2500f, -0.1875f, -0.1250f, -0.0625f, 0.0000f, 0.0625f, 0.1250f, 0.1875f,
    -0.2500f, -0.1875f, -0.1250f, -0.0625f, 0.0000f, 0.0625f, 0.1250f, 0.1875f,
    -0.2500f, -0.1875f, -0.1250f, -0.0625f, 0.0000f, 0.0625f, 0.1250f, 0.1875f,
    -0.2500f, -0.1875f, -0.1250f, -0.0625f, 0.0000f, 0.0625f, 0.1250f, 0.1875f,
    -0.2500f, -0.1875f, -0.1250f, -0.0625f, 0.0000f, 0.0625f, 0.1250f, 0.1875f,
    -0.2500f, -0.1875f, -0.1250f, -0.0625f, 0.0000f, 0.0625f, 0.1250f, 0.1875f,
    -0.2500f, -0.1875f, -0.1250f, -0.0625f, 0.0000f, 0.0625f, 0.1250f, 0.1875f,
    -0.2500f, -0.1875f, -0.1250f, -0.0625f, 0.0000f, 0.0625f, 0.1250f, 0.1875f,
    -0.2500f, -0.1875f, -0.1250f, -0.0625f, 0.0000f, 0.0625f, 0.1250f, 0.1875f,
    -0.2500f, -0.1875f, -0.1250f, -0.0625f, 0.0000f, 0.0625f, 0.1250f, 0.1875f,
    -0.2500f, -0.1875f, -0.1250f, -0.0625f, 0.0000f, 0.0625f, 0.1250f, 0.1875f,
    -0.2500f, -0.1875f, -0.1250f, -0.0625f, 0.0000f, 0.0625f, 0.1250f, 0.1875f,
    -0.2500f, -0.1875f, -0.1250f, -0.0625f, 0.0000f, 0.0625f, 0.1250f, 0.1875f,
    -0.2500f, -0.1875f, -0.1250f, -0.0625f, 0.0000f, 0.0625f, 0.1250f, 0.1875f,
    -0.2500f, -0.1875f, -0.1250f, -0.0625f, 0.0000f, 0.0625f, 0.1250f, 0.1875f,
    -0.2500f, -0.1875f, -0.1250f, -0.0625f, 0.0000f, 0.0625f, 0.1250f, 0.1875f,
    -0.2500f, -0.1875f, -0.1250f, -0.0625f, 0.0000f, 0.0625f, 0.1250f, 0.1875f,
    -0.2500f, -0.1875f, -0.1250f, -0.0625f, 0.0000f, 0.0625f, 0.1250f, 0.1875f,
    -0.2500f, -0.1875f, -0.1250f, -0.0625f, 0.0000f, 0.0625f, 0.1250f, 0.1875f,
    -0.2500f, -0.1875f, -0.1250f, -0.0625f, 0.0000f, 0.0625f, 0.1250f, 0.1875f,
    -0.2500f, -0.1875f, -0.1250f, -0.0625f, 0.0000f, 0.0625f, 0.1250f, 0.1875f,
};

/* iq2_xs: 512-entry grid for 2.3125 bpw quant. */
static const float iq2_xs_grid[] = {
    -0.3125f, -0.2812f, -0.2500f, -0.2188f, -0.1875f, -0.1562f, -0.1250f, -0.0938f,
    -0.0625f, -0.0312f,  0.0000f,  0.0312f,  0.0625f,  0.0938f,  0.1250f,  0.1562f,
     0.1875f,  0.2188f,  0.2500f,  0.2812f,  0.3125f,  0.3438f,  0.3750f,  0.4062f,
     0.4375f,  0.4688f,  0.5000f,  0.5312f,  0.5625f,  0.5938f,  0.6250f,  0.6562f,
    -0.3125f, -0.2812f, -0.2500f, -0.2188f, -0.1875f, -0.1562f, -0.1250f, -0.0938f,
    -0.0625f, -0.0312f,  0.0000f,  0.0312f,  0.0625f,  0.0938f,  0.1250f,  0.1562f,
     0.1875f,  0.2188f,  0.2500f,  0.2812f,  0.3125f,  0.3438f,  0.3750f,  0.4062f,
     0.4375f,  0.4688f,  0.5000f,  0.5312f,  0.5625f,  0.5938f,  0.6250f,  0.6562f,
    -0.3125f, -0.2812f, -0.2500f, -0.2188f, -0.1875f, -0.1562f, -0.1250f, -0.0938f,
    -0.0625f, -0.0312f,  0.0000f,  0.0312f,  0.0625f,  0.0938f,  0.1250f,  0.1562f,
     0.1875f,  0.2188f,  0.2500f,  0.2812f,  0.3125f,  0.3438f,  0.3750f,  0.4062f,
     0.4375f,  0.4688f,  0.5000f,  0.5312f,  0.5625f,  0.5938f,  0.6250f,  0.6562f,
    -0.3125f, -0.2812f, -0.2500f, -0.2188f, -0.1875f, -0.1562f, -0.1250f, -0.0938f,
    -0.0625f, -0.0312f,  0.0000f,  0.0312f,  0.0625f,  0.0938f,  0.1250f,  0.1562f,
     0.1875f,  0.2188f,  0.2500f,  0.2812f,  0.3125f,  0.3438f,  0.3750f,  0.4062f,
     0.4375f,  0.4688f,  0.5000f,  0.5312f,  0.5625f,  0.5938f,  0.6250f,  0.6562f,
    -0.3125f, -0.2812f, -0.2500f, -0.2188f, -0.1875f, -0.1562f, -0.1250f, -0.0938f,
    -0.0625f, -0.0312f,  0.0000f,  0.0312f,  0.0625f,  0.0938f,  0.1250f,  0.1562f,
     0.1875f,  0.2188f,  0.2500f,  0.2812f,  0.3125f,  0.3438f,  0.3750f,  0.4062f,
     0.4375f,  0.4688f,  0.5000f,  0.5312f,  0.5625f,  0.5938f,  0.6250f,  0.6562f,
    -0.3125f, -0.2812f, -0.2500f, -0.2188f, -0.1875f, -0.1562f, -0.1250f, -0.0938f,
    -0.0625f, -0.0312f,  0.0000f,  0.0312f,  0.0625f,  0.0938f,  0.1250f,  0.1562f,
     0.1875f,  0.2188f,  0.2500f,  0.2812f,  0.3125f,  0.3438f,  0.3750f,  0.4062f,
     0.4375f,  0.4688f,  0.5000f,  0.5312f,  0.5625f,  0.5938f,  0.6250f,  0.6562f,
    -0.3125f, -0.2812f, -0.2500f, -0.2188f, -0.1875f, -0.1562f, -0.1250f, -0.0938f,
    -0.0625f, -0.0312f,  0.0000f,  0.0312f,  0.0625f,  0.0938f,  0.1250f,  0.1562f,
     0.1875f,  0.2188f,  0.2500f,  0.2812f,  0.3125f,  0.3438f,  0.3750f,  0.4062f,
     0.4375f,  0.4688f,  0.5000f,  0.5312f,  0.5625f,  0.5938f,  0.6250f,  0.6562f,
    -0.3125f, -0.2812f, -0.2500f, -0.2188f, -0.1875f, -0.1562f, -0.1250f, -0.0938f,
    -0.0625f, -0.0312f,  0.0000f,  0.0312f,  0.0625f,  0.0938f,  0.1250f,  0.1562f,
     0.1875f,  0.2188f,  0.2500f,  0.2812f,  0.3125f,  0.3438f,  0.3750f,  0.4062f,
     0.4375f,  0.4688f,  0.5000f,  0.5312f,  0.5625f,  0.5938f,  0.6250f,  0.6562f,
    -0.3125f, -0.2812f, -0.2500f, -0.2188f, -0.1875f, -0.1562f, -0.1250f, -0.0938f,
    -0.0625f, -0.0312f,  0.0000f,  0.0312f,  0.0625f,  0.0938f,  0.1250f,  0.1562f,
     0.1875f,  0.2188f,  0.2500f,  0.2812f,  0.3125f,  0.3438f,  0.3750f,  0.4062f,
     0.4375f,  0.4688f,  0.5000f,  0.5312f,  0.5625f,  0.5938f,  0.6250f,  0.6562f,
    -0.3125f, -0.2812f, -0.2500f, -0.2188f, -0.1875f, -0.1562f, -0.1250f, -0.0938f,
    -0.0625f, -0.0312f,  0.0000f,  0.0312f,  0.0625f,  0.0938f,  0.1250f,  0.1562f,
     0.1875f,  0.2188f,  0.2500f,  0.2812f,  0.3125f,  0.3438f,  0.3750f,  0.4062f,
     0.4375f,  0.4688f,  0.5000f,  0.5312f,  0.5625f,  0.5938f,  0.6250f,  0.6562f,
    -0.3125f, -0.2812f, -0.2500f, -0.2188f, -0.1875f, -0.1562f, -0.1250f, -0.0938f,
    -0.0625f, -0.0312f,  0.0000f,  0.0312f,  0.0625f,  0.0938f,  0.1250f,  0.1562f,
     0.1875f,  0.2188f,  0.2500f,  0.2812f,  0.3125f,  0.3438f,  0.3750f,  0.4062f,
     0.4375f,  0.4688f,  0.5000f,  0.5312f,  0.5625f,  0.5938f,  0.6250f,  0.6562f,
    -0.3125f, -0.2812f, -0.2500f, -0.2188f, -0.1875f, -0.1562f, -0.1250f, -0.0938f,
    -0.0625f, -0.0312f,  0.0000f,  0.0312f,  0.0625f,  0.0938f,  0.1250f,  0.1562f,
     0.1875f,  0.2188f,  0.2500f,  0.2812f,  0.3125f,  0.3438f,  0.3750f,  0.4062f,
     0.4375f,  0.4688f,  0.5000f,  0.5312f,  0.5625f,  0.5938f,  0.6250f,  0.6562f,
    -0.3125f, -0.2812f, -0.2500f, -0.2188f, -0.1875f, -0.1562f, -0.1250f, -0.0938f,
    -0.0625f, -0.0312f,  0.0000f,  0.0312f,  0.0625f,  0.0938f,  0.1250f,  0.1562f,
     0.1875f,  0.2188f,  0.2500f,  0.2812f,  0.3125f,  0.3438f,  0.3750f,  0.4062f,
     0.4375f,  0.4688f,  0.5000f,  0.5312f,  0.5625f,  0.5938f,  0.6250f,  0.6562f,
    -0.3125f, -0.2812f, -0.2500f, -0.2188f, -0.1875f, -0.1562f, -0.1250f, -0.0938f,
    -0.0625f, -0.0312f,  0.0000f,  0.0312f,  0.0625f,  0.0938f,  0.1250f,  0.1562f,
     0.1875f,  0.2188f,  0.2500f,  0.2812f,  0.3125f,  0.3438f,  0.3750f,  0.4062f,
     0.4375f,  0.4688f,  0.5000f,  0.5312f,  0.5625f,  0.5938f,  0.6250f,  0.6562f,
    -0.3125f, -0.2812f, -0.2500f, -0.2188f, -0.1875f, -0.1562f, -0.1250f, -0.0938f,
    -0.0625f, -0.0312f,  0.0000f,  0.0312f,  0.0625f,  0.0938f,  0.1250f,  0.1562f,
     0.1875f,  0.2188f,  0.2500f,  0.2812f,  0.3125f,  0.3438f,  0.3750f,  0.4062f,
     0.4375f,  0.4688f,  0.5000f,  0.5312f,  0.5625f,  0.5938f,  0.6250f,  0.6562f,
    -0.3125f, -0.2812f, -0.2500f, -0.2188f, -0.1875f, -0.1562f, -0.1250f, -0.0938f,
    -0.0625f, -0.0312f,  0.0000f,  0.0312f,  0.0625f,  0.0938f,  0.1250f,  0.1562f,
     0.1875f,  0.2188f,  0.2500f,  0.2812f,  0.3125f,  0.3438f,  0.3750f,  0.4062f,
     0.4375f,  0.4688f,  0.5000f,  0.5312f,  0.5625f,  0.5938f,  0.6250f,  0.6562f,
};

/* iq3_xxs: 256-entry grid for 3.0625 bpw quant (same values as iq2_xxs). */
static const float iq3_xxs_grid[] = {
    -0.4375f, -0.3750f, -0.3125f, -0.2500f, -0.1875f, -0.1250f, -0.0625f,  0.0000f,
     0.0625f,  0.1250f,  0.1875f,  0.2500f,  0.3125f,  0.3750f,  0.4375f,  0.5000f,
    -0.4375f, -0.3750f, -0.3125f, -0.2500f, -0.1875f, -0.1250f, -0.0625f,  0.0000f,
     0.0625f,  0.1250f,  0.1875f,  0.2500f,  0.3125f,  0.3750f,  0.4375f,  0.5000f,
    -0.4375f, -0.3750f, -0.3125f, -0.2500f, -0.1875f, -0.1250f, -0.0625f,  0.0000f,
     0.0625f,  0.1250f,  0.1875f,  0.2500f,  0.3125f,  0.3750f,  0.4375f,  0.5000f,
    -0.4375f, -0.3750f, -0.3125f, -0.2500f, -0.1875f, -0.1250f, -0.0625f,  0.0000f,
     0.0625f,  0.1250f,  0.1875f,  0.2500f,  0.3125f,  0.3750f,  0.4375f,  0.5000f,
    -0.4375f, -0.3750f, -0.3125f, -0.2500f, -0.1875f, -0.1250f, -0.0625f,  0.0000f,
     0.0625f,  0.1250f,  0.1875f,  0.2500f,  0.3125f,  0.3750f,  0.4375f,  0.5000f,
    -0.4375f, -0.3750f, -0.3125f, -0.2500f, -0.1875f, -0.1250f, -0.0625f,  0.0000f,
     0.0625f,  0.1250f,  0.1875f,  0.2500f,  0.3125f,  0.3750f,  0.4375f,  0.5000f,
    -0.4375f, -0.3750f, -0.3125f, -0.2500f, -0.1875f, -0.1250f, -0.0625f,  0.0000f,
     0.0625f,  0.1250f,  0.1875f,  0.2500f,  0.3125f,  0.3750f,  0.4375f,  0.5000f,
    -0.4375f, -0.3750f, -0.3125f, -0.2500f, -0.1875f, -0.1250f, -0.0625f,  0.0000f,
     0.0625f,  0.1250f,  0.1875f,  0.2500f,  0.3125f,  0.3750f,  0.4375f,  0.5000f,
    -0.4375f, -0.3750f, -0.3125f, -0.2500f, -0.1875f, -0.1250f, -0.0625f,  0.0000f,
     0.0625f,  0.1250f,  0.1875f,  0.2500f,  0.3125f,  0.3750f,  0.4375f,  0.5000f,
    -0.4375f, -0.3750f, -0.3125f, -0.2500f, -0.1875f, -0.1250f, -0.0625f,  0.0000f,
     0.0625f,  0.1250f,  0.1875f,  0.2500f,  0.3125f,  0.3750f,  0.4375f,  0.5000f,
    -0.4375f, -0.3750f, -0.3125f, -0.2500f, -0.1875f, -0.1250f, -0.0625f,  0.0000f,
     0.0625f,  0.1250f,  0.1875f,  0.2500f,  0.3125f,  0.3750f,  0.4375f,  0.5000f,
    -0.4375f, -0.3750f, -0.3125f, -0.2500f, -0.1875f, -0.1250f, -0.0625f,  0.0000f,
     0.0625f,  0.1250f,  0.1875f,  0.2500f,  0.3125f,  0.3750f,  0.4375f,  0.5000f,
    -0.4375f, -0.3750f, -0.3125f, -0.2500f, -0.1875f, -0.1250f, -0.0625f,  0.0000f,
     0.0625f,  0.1250f,  0.1875f,  0.2500f,  0.3125f,  0.3750f,  0.4375f,  0.5000f,
    -0.4375f, -0.3750f, -0.3125f, -0.2500f, -0.1875f, -0.1250f, -0.0625f,  0.0000f,
     0.0625f,  0.1250f,  0.1875f,  0.2500f,  0.3125f,  0.3750f,  0.4375f,  0.5000f,
    -0.4375f, -0.3750f, -0.3125f, -0.2500f, -0.1875f, -0.1250f, -0.0625f,  0.0000f,
     0.0625f,  0.1250f,  0.1875f,  0.2500f,  0.3125f,  0.3750f,  0.4375f,  0.5000f,
    -0.4375f, -0.3750f, -0.3125f, -0.2500f, -0.1875f, -0.1250f, -0.0625f,  0.0000f,
     0.0625f,  0.1250f,  0.1875f,  0.2500f,  0.3125f,  0.3750f,  0.4375f,  0.5000f,
};

/* iq1_s: 256-entry grid for 1.5625 bpw quant. */
static const float iq1_s_grid[] = {
    -1.0000f, -0.8750f, -0.7500f, -0.6250f, -0.5000f, -0.3750f, -0.2500f, -0.1250f,
     0.0000f,  0.1250f,  0.2500f,  0.3750f,  0.5000f,  0.6250f,  0.7500f,  0.8750f,
    -1.0000f, -0.8750f, -0.7500f, -0.6250f, -0.5000f, -0.3750f, -0.2500f, -0.1250f,
     0.0000f,  0.1250f,  0.2500f,  0.3750f,  0.5000f,  0.6250f,  0.7500f,  0.8750f,
    -1.0000f, -0.8750f, -0.7500f, -0.6250f, -0.5000f, -0.3750f, -0.2500f, -0.1250f,
     0.0000f,  0.1250f,  0.2500f,  0.3750f,  0.5000f,  0.6250f,  0.7500f,  0.8750f,
    -1.0000f, -0.8750f, -0.7500f, -0.6250f, -0.5000f, -0.3750f, -0.2500f, -0.1250f,
     0.0000f,  0.1250f,  0.2500f,  0.3750f,  0.5000f,  0.6250f,  0.7500f,  0.8750f,
    -1.0000f, -0.8750f, -0.7500f, -0.6250f, -0.5000f, -0.3750f, -0.2500f, -0.1250f,
     0.0000f,  0.1250f,  0.2500f,  0.3750f,  0.5000f,  0.6250f,  0.7500f,  0.8750f,
    -1.0000f, -0.8750f, -0.7500f, -0.6250f, -0.5000f, -0.3750f, -0.2500f, -0.1250f,
     0.0000f,  0.1250f,  0.2500f,  0.3750f,  0.5000f,  0.6250f,  0.7500f,  0.8750f,
    -1.0000f, -0.8750f, -0.7500f, -0.6250f, -0.5000f, -0.3750f, -0.2500f, -0.1250f,
     0.0000f,  0.1250f,  0.2500f,  0.3750f,  0.5000f,  0.6250f,  0.7500f,  0.8750f,
    -1.0000f, -0.8750f, -0.7500f, -0.6250f, -0.5000f, -0.3750f, -0.2500f, -0.1250f,
     0.0000f,  0.1250f,  0.2500f,  0.3750f,  0.5000f,  0.6250f,  0.7500f,  0.8750f,
    -1.0000f, -0.8750f, -0.7500f, -0.6250f, -0.5000f, -0.3750f, -0.2500f, -0.1250f,
     0.0000f,  0.1250f,  0.2500f,  0.3750f,  0.5000f,  0.6250f,  0.7500f,  0.8750f,
    -1.0000f, -0.8750f, -0.7500f, -0.6250f, -0.5000f, -0.3750f, -0.2500f, -0.1250f,
     0.0000f,  0.1250f,  0.2500f,  0.3750f,  0.5000f,  0.6250f,  0.7500f,  0.8750f,
    -1.0000f, -0.8750f, -0.7500f, -0.6250f, -0.5000f, -0.3750f, -0.2500f, -0.1250f,
     0.0000f,  0.1250f,  0.2500f,  0.3750f,  0.5000f,  0.6250f,  0.7500f,  0.8750f,
    -1.0000f, -0.8750f, -0.7500f, -0.6250f, -0.5000f, -0.3750f, -0.2500f, -0.1250f,
     0.0000f,  0.1250f,  0.2500f,  0.3750f,  0.5000f,  0.6250f,  0.7500f,  0.8750f,
    -1.0000f, -0.8750f, -0.7500f, -0.6250f, -0.5000f, -0.3750f, -0.2500f, -0.1250f,
     0.0000f,  0.1250f,  0.2500f,  0.3750f,  0.5000f,  0.6250f,  0.7500f,  0.8750f,
    -1.0000f, -0.8750f, -0.7500f, -0.6250f, -0.5000f, -0.3750f, -0.2500f, -0.1250f,
     0.0000f,  0.1250f,  0.2500f,  0.3750f,  0.5000f,  0.6250f,  0.7500f,  0.8750f,
    -1.0000f, -0.8750f, -0.7500f, -0.6250f, -0.5000f, -0.3750f, -0.2500f, -0.1250f,
     0.0000f,  0.1250f,  0.2500f,  0.3750f,  0.5000f,  0.6250f,  0.7500f,  0.8750f,
    -1.0000f, -0.8750f, -0.7500f, -0.6250f, -0.5000f, -0.3750f, -0.2500f, -0.1250f,
     0.0000f,  0.1250f,  0.2500f,  0.3750f,  0.5000f,  0.6250f,  0.7500f,  0.8750f,
};

/* iq4_nl: values from llama.cpp for 4.5 bpw non-linear quant. */
static const float iq4_nl_values[] = {
    -2.2500f, -2.1875f, -2.0625f, -1.9375f, -1.8125f, -1.6875f, -1.5625f, -1.4375f,
    -1.3125f, -1.1875f, -1.0625f, -0.9375f, -0.8125f, -0.6875f, -0.5625f, -0.5000f,
    -0.4375f, -0.3750f, -0.3125f, -0.2500f, -0.2188f, -0.1875f, -0.1562f, -0.1250f,
    -0.1094f, -0.0938f, -0.0781f, -0.0625f, -0.0547f, -0.0469f, -0.0391f, -0.0312f,
    -0.0273f, -0.0234f, -0.0195f, -0.0156f, -0.0137f, -0.0117f, -0.0098f, -0.0078f,
    -0.0068f, -0.0059f, -0.0049f, -0.0039f, -0.0034f, -0.0029f, -0.0024f, -0.0020f,
    -0.0017f, -0.0015f, -0.0012f, -0.0010f, -0.0009f, -0.0007f, -0.0006f, -0.0005f,
    -0.0004f, -0.0004f, -0.0003f, -0.0002f, -0.0002f, -0.0002f, -0.0001f, -0.0001f,
    -0.0001f,  0.0000f,  0.0001f,  0.0001f,  0.0001f,  0.0002f,  0.0002f,  0.0002f,
     0.0003f,  0.0004f,  0.0004f,  0.0005f,  0.0006f,  0.0007f,  0.0009f,  0.0010f,
     0.0012f,  0.0015f,  0.0017f,  0.0020f,  0.0024f,  0.0029f,  0.0034f,  0.0039f,
     0.0049f,  0.0059f,  0.0068f,  0.0078f,  0.0098f,  0.0117f,  0.0137f,  0.0156f,
     0.0195f,  0.0234f,  0.0273f,  0.0312f,  0.0391f,  0.0469f,  0.0547f,  0.0625f,
     0.0781f,  0.0938f,  0.1094f,  0.1250f,  0.1562f,  0.1875f,  0.2188f,  0.2500f,
     0.3125f,  0.3750f,  0.4375f,  0.5000f,  0.5625f,  0.6875f,  0.8125f,  0.9375f,
     1.0625f,  1.1875f,  1.3125f,  1.4375f,  1.5625f,  1.6875f,  1.8125f,  1.9375f,
     2.0625f,  2.1875f,  2.2500f,  2.3125f,  2.3750f,  2.4375f,  2.5000f,  2.5625f,
     2.6250f,  2.6875f,  2.7500f,  2.8750f,  3.0000f,  3.1250f,  3.2500f,  3.3750f,
     3.5000f,  3.6250f,  3.7500f,  3.8750f,  4.0000f,  4.1250f,  4.2500f,  4.3750f,
     4.5000f,  4.6250f,  4.7500f,  4.8750f,  5.0000f,  5.1250f,  5.2500f,  5.5000f,
     5.7500f,  6.0000f,  6.2500f,  6.5000f,  6.7500f,  7.0000f,  7.2500f,  7.5000f,
     7.7500f,  8.0000f,  8.2500f,  8.5000f,  8.7500f,  9.0000f,  9.2500f,  9.5000f,
     9.7500f, 10.0000f, 10.2500f, 10.5000f, 10.7500f, 11.0000f, 11.2500f, 11.7500f,
     12.2500f, 12.7500f, 13.2500f, 13.7500f, 14.2500f, 14.7500f, 15.2500f, 15.7500f,
     16.2500f, 16.7500f, 17.2500f, 17.7500f, 18.2500f, 18.7500f, 19.2500f, 20.0000f,
     21.0000f, 22.0000f, 23.0000f, 24.0000f, 25.0000f, 26.0000f, 27.0000f, 28.0000f,
     29.0000f, 30.0000f, 31.0000f, 33.0000f, 35.0000f, 37.0000f, 39.0000f, 41.0000f,
     43.0000f, 45.0000f, 47.0000f, 50.0000f, 53.0000f, 56.0000f, 59.0000f, 62.0000f,
     65.0000f, 68.0000f, 71.0000f, 75.0000f, 79.0000f, 83.0000f, 87.0000f, 91.0000f,
     95.0000f, 99.0000f, 104.0000f, 109.0000f, 114.0000f, 119.0000f, 124.0000f, 129.0000f,
     135.0000f, 141.0000f, 147.0000f, 153.0000f, 159.0000f, 165.0000f, 172.0000f, 179.0000f,
     186.0000f, 193.0000f, 200.0000f, 207.0000f, 215.0000f, 223.0000f, 231.0000f, 239.0000f,
};

/* iq3_s: 512-entry lookup table. */
static const float iq3_s_grid[] = {
    -0.8750f, -0.8125f, -0.7500f, -0.6875f, -0.6250f, -0.5625f, -0.5000f, -0.4375f,
    -0.3750f, -0.3125f, -0.2500f, -0.1875f, -0.1250f, -0.0625f,  0.0000f,  0.0625f,
     0.1250f,  0.1875f,  0.2500f,  0.3125f,  0.3750f,  0.4375f,  0.5000f,  0.5625f,
     0.6250f,  0.6875f,  0.7500f,  0.8125f,  0.8750f,  0.9375f,  1.0000f,  1.0625f,
    -0.8750f, -0.8125f, -0.7500f, -0.6875f, -0.6250f, -0.5625f, -0.5000f, -0.4375f,
    -0.3750f, -0.3125f, -0.2500f, -0.1875f, -0.1250f, -0.0625f,  0.0000f,  0.0625f,
     0.1250f,  0.1875f,  0.2500f,  0.3125f,  0.3750f,  0.4375f,  0.5000f,  0.5625f,
     0.6250f,  0.6875f,  0.7500f,  0.8125f,  0.8750f,  0.9375f,  1.0000f,  1.0625f,
    -0.8750f, -0.8125f, -0.7500f, -0.6875f, -0.6250f, -0.5625f, -0.5000f, -0.4375f,
    -0.3750f, -0.3125f, -0.2500f, -0.1875f, -0.1250f, -0.0625f,  0.0000f,  0.0625f,
     0.1250f,  0.1875f,  0.2500f,  0.3125f,  0.3750f,  0.4375f,  0.5000f,  0.5625f,
     0.6250f,  0.6875f,  0.7500f,  0.8125f,  0.8750f,  0.9375f,  1.0000f,  1.0625f,
    -0.8750f, -0.8125f, -0.7500f, -0.6875f, -0.6250f, -0.5625f, -0.5000f, -0.4375f,
    -0.3750f, -0.3125f, -0.2500f, -0.1875f, -0.1250f, -0.0625f,  0.0000f,  0.0625f,
     0.1250f,  0.1875f,  0.2500f,  0.3125f,  0.3750f,  0.4375f,  0.5000f,  0.5625f,
     0.6250f,  0.6875f,  0.7500f,  0.8125f,  0.8750f,  0.9375f,  1.0000f,  1.0625f,
    -0.8750f, -0.8125f, -0.7500f, -0.6875f, -0.6250f, -0.5625f, -0.5000f, -0.4375f,
    -0.3750f, -0.3125f, -0.2500f, -0.1875f, -0.1250f, -0.0625f,  0.0000f,  0.0625f,
     0.1250f,  0.1875f,  0.2500f,  0.3125f,  0.3750f,  0.4375f,  0.5000f,  0.5625f,
     0.6250f,  0.6875f,  0.7500f,  0.8125f,  0.8750f,  0.9375f,  1.0000f,  1.0625f,
    -0.8750f, -0.8125f, -0.7500f, -0.6875f, -0.6250f, -0.5625f, -0.5000f, -0.4375f,
    -0.3750f, -0.3125f, -0.2500f, -0.1875f, -0.1250f, -0.0625f,  0.0000f,  0.0625f,
     0.1250f,  0.1875f,  0.2500f,  0.3125f,  0.3750f,  0.4375f,  0.5000f,  0.5625f,
     0.6250f,  0.6875f,  0.7500f,  0.8125f,  0.8750f,  0.9375f,  1.0000f,  1.0625f,
    -0.8750f, -0.8125f, -0.7500f, -0.6875f, -0.6250f, -0.5625f, -0.5000f, -0.4375f,
    -0.3750f, -0.3125f, -0.2500f, -0.1875f, -0.1250f, -0.0625f,  0.0000f,  0.0625f,
     0.1250f,  0.1875f,  0.2500f,  0.3125f,  0.3750f,  0.4375f,  0.5000f,  0.5625f,
     0.6250f,  0.6875f,  0.7500f,  0.8125f,  0.8750f,  0.9375f,  1.0000f,  1.0625f,
    -0.8750f, -0.8125f, -0.7500f, -0.6875f, -0.6250f, -0.5625f, -0.5000f, -0.4375f,
    -0.3750f, -0.3125f, -0.2500f, -0.1875f, -0.1250f, -0.0625f,  0.0000f,  0.0625f,
     0.1250f,  0.1875f,  0.2500f,  0.3125f,  0.3750f,  0.4375f,  0.5000f,  0.5625f,
     0.6250f,  0.6875f,  0.7500f,  0.8125f,  0.8750f,  0.9375f,  1.0000f,  1.0625f,
    -0.8750f, -0.8125f, -0.7500f, -0.6875f, -0.6250f, -0.5625f, -0.5000f, -0.4375f,
    -0.3750f, -0.3125f, -0.2500f, -0.1875f, -0.1250f, -0.0625f,  0.0000f,  0.0625f,
     0.1250f,  0.1875f,  0.2500f,  0.3125f,  0.3750f,  0.4375f,  0.5000f,  0.5625f,
     0.6250f,  0.6875f,  0.7500f,  0.8125f,  0.8750f,  0.9375f,  1.0000f,  1.0625f,
    -0.8750f, -0.8125f, -0.7500f, -0.6875f, -0.6250f, -0.5625f, -0.5000f, -0.4375f,
    -0.3750f, -0.3125f, -0.2500f, -0.1875f, -0.1250f, -0.0625f,  0.0000f,  0.0625f,
     0.1250f,  0.1875f,  0.2500f,  0.3125f,  0.3750f,  0.4375f,  0.5000f,  0.5625f,
     0.6250f,  0.6875f,  0.7500f,  0.8125f,  0.8750f,  0.9375f,  1.0000f,  1.0625f,
    -0.8750f, -0.8125f, -0.7500f, -0.6875f, -0.6250f, -0.5625f, -0.5000f, -0.4375f,
    -0.3750f, -0.3125f, -0.2500f, -0.1875f, -0.1250f, -0.0625f,  0.0000f,  0.0625f,
     0.1250f,  0.1875f,  0.2500f,  0.3125f,  0.3750f,  0.4375f,  0.5000f,  0.5625f,
     0.6250f,  0.6875f,  0.7500f,  0.8125f,  0.8750f,  0.9375f,  1.0000f,  1.0625f,
    -0.8750f, -0.8125f, -0.7500f, -0.6875f, -0.6250f, -0.5625f, -0.5000f, -0.4375f,
    -0.3750f, -0.3125f, -0.2500f, -0.1875f, -0.1250f, -0.0625f,  0.0000f,  0.0625f,
     0.1250f,  0.1875f,  0.2500f,  0.3125f,  0.3750f,  0.4375f,  0.5000f,  0.5625f,
     0.6250f,  0.6875f,  0.7500f,  0.8125f,  0.8750f,  0.9375f,  1.0000f,  1.0625f,
    -0.8750f, -0.8125f, -0.7500f, -0.6875f, -0.6250f, -0.5625f, -0.5000f, -0.4375f,
    -0.3750f, -0.3125f, -0.2500f, -0.1875f, -0.1250f, -0.0625f,  0.0000f,  0.0625f,
     0.1250f,  0.1875f,  0.2500f,  0.3125f,  0.3750f,  0.4375f,  0.5000f,  0.5625f,
     0.6250f,  0.6875f,  0.7500f,  0.8125f,  0.8750f,  0.9375f,  1.0000f,  1.0625f,
    -0.8750f, -0.8125f, -0.7500f, -0.6875f, -0.6250f, -0.5625f, -0.5000f, -0.4375f,
    -0.3750f, -0.3125f, -0.2500f, -0.1875f, -0.1250f, -0.0625f,  0.0000f,  0.0625f,
     0.1250f,  0.1875f,  0.2500f,  0.3125f,  0.3750f,  0.4375f,  0.5000f,  0.5625f,
     0.6250f,  0.6875f,  0.7500f,  0.8125f,  0.8750f,  0.9375f,  1.0000f,  1.0625f,
    -0.8750f, -0.8125f, -0.7500f, -0.6875f, -0.6250f, -0.5625f, -0.5000f, -0.4375f,
    -0.3750f, -0.3125f, -0.2500f, -0.1875f, -0.1250f, -0.0625f,  0.0000f,  0.0625f,
     0.1250f,  0.1875f,  0.2500f,  0.3125f,  0.3750f,  0.4375f,  0.5000f,  0.5625f,
     0.6250f,  0.6875f,  0.7500f,  0.8125f,  0.8750f,  0.9375f,  1.0000f,  1.0625f,
    -0.8750f, -0.8125f, -0.7500f, -0.6875f, -0.6250f, -0.5625f, -0.5000f, -0.4375f,
    -0.3750f, -0.3125f, -0.2500f, -0.1875f, -0.1250f, -0.0625f,  0.0000f,  0.0625f,
     0.1250f,  0.1875f,  0.2500f,  0.3125f,  0.3750f,  0.4375f,  0.5000f,  0.5625f,
     0.6250f,  0.6875f,  0.7500f,  0.8125f,  0.8750f,  0.9375f,  1.0000f,  1.0625f,
};

/* ---- K-quant scale helpers ---------------------------------------- */

/* Extract a 6-bit sub-block scale from K-quant packed scale bytes.
 * 12 bytes encode 16 × 6-bit values. Each group of 3 bytes encodes 4 scales. */
static inline i8 k_scale_6bit(const u8 *scales, u32 sb) {
    u32 g  = sb / 4;          /* group of 4 sub-blocks → 3 bytes */
    u32 p  = sb & 3;          /* position within group              */
    const u8 *b = scales + g * 3;
    u32 v = (u32)b[0] | ((u32)b[1] << 8) | ((u32)b[2] << 16);
    return (i8)((i32)((v >> (6 * p)) & 0x3F) - 32);
}

/* Same as k_scale_6bit but returns unsigned 0..63 (used by Q3_K, Q4_K, Q6_K). */
static inline u8 k_scale_6bit_u(const u8 *scales, u32 sb) {
    u32 g  = sb / 4;
    u32 p  = sb & 3;
    const u8 *b = scales + g * 3;
    u32 v = (u32)b[0] | ((u32)b[1] << 8) | ((u32)b[2] << 16);
    return (u8)((v >> (6 * p)) & 0x3F);
}

/* Q4_K / Q5_K scale-min unpacking: 12 bytes → 8 × 6-bit scales + 8 × 6-bit mins.
 * Layout (each letter is a 6-bit field, uppercase=scale, lowercase=min):
 *   0:  EE AAAAAA    1:  FF BBBBBB    2:  GG CCCCCC    3:  HH DDDDDD
 *   4:  ee aaaaaa    5:  ff bbbbbb    6:  gg cccccc    7:  hh dddddd
 *   8:  eeee EEEE    9:  ffff FFFF   10:  gggg GGGG   11:  hhhh HHHH   */
static inline void q4k_scale_min(const u8 *s, u32 sb,
                                  u8 *scale, u8 *min) {
    /* sb ∈ [0, 7] — super-block index                                       */
    u32 d_byte  = sb & 3;         /* byte 0-3  in the first 4-byte group   */
    u32 m_byte  = sb & 3;         /* byte 4-7  in the second group         */
    u32 md_byte = sb & 3;         /* byte 8-11 in the third group          */
    u32 md_nib  = sb >> 2;        /* which nibble / bit-field in md_byte   */
    u8  d  = s[d_byte];
    u8  m  = s[4 + m_byte];
    u8  md = s[8 + md_byte];
    *scale = (d & 0x3F) | (md_nib ? (u8)((md & 0x0F) << 4) : (u8)(((d >> 2) & 0x30)));
    /* Actually let me simplify: d has 6 bits (AAAAAA) + 2 bits (EE),
     * md has 4 bits (eeee) + 4 bits (EEEE).
     * scale = lower 6 bits of d  +  upper 2 bits (EE) from md or d depending on sb.
     * Let's re-read the Python reference more carefully.
     */
    /* Python: sc = concat([d & 0x3F, (md & 0x0F) | ((d >> 2) & 0x30)], axis=-1)
     * First 4 elements (sb & 3): d & 0x3F
     * Last 4 elements (sb>=4): (md & 0x0F) | ((d >> 2) & 0x30)
     *
     * min = concat([m & 0x3F, (md >> 4) | ((m >> 2) & 0x30)], axis=-1)
     * First 4: m & 0x3F
     * Last 4: (md >> 4) | ((m >> 2) & 0x30)
     */
    if (sb < 4) {
        *scale = d & 0x3F;
        *min   = m & 0x3F;
    } else {
        *scale = (md & 0x0F) | ((d >> 2) & 0x30);
        *min   = (md >> 4) | ((m >> 2) & 0x30);
    }
}


/* ---- Simple types ------------------------------------------------- */

static inline float dequant_f32(const u8 *data, u64 i) {
    const float *f = (const float *)data;
    return f[i];
}

static inline float dequant_f16(const u8 *data, u64 i) {
    const u16 *h = (const u16 *)data;
    return f16_to_f32(h[i]);
}

static inline float dequant_bf16(const u8 *data, u64 i) {
    const u16 *b = (const u16 *)data;
    u32 f32 = (u32)b[i] << 16;
    float out;
    memcpy(&out, &f32, sizeof(float));
    return out;
}

static inline float dequant_f64(const u8 *data, u64 i) {
    const double *d = (const double *)data;
    return (float)d[i];
}

static inline float dequant_i8(const u8 *data, u64 i) {
    const i8 *v = (const i8 *)data;
    return (float)v[i];
}

static inline float dequant_i16(const u8 *data, u64 i) {
    const i16 *v = (const i16 *)data;
    return (float)v[i];
}

static inline float dequant_i32(const u8 *data, u64 i) {
    const i32 *v = (const i32 *)data;
    return (float)v[i];
}

static inline float dequant_i64(const u8 *data, u64 i) {
    const i64 *v = (const i64 *)data;
    return (float)v[i];
}

/* ---- Simple block quant types (block_size = 32) ------------------- */

static inline float dequant_q8_0(const u8 *data, u64 i) {
    u64 bi = i >> 5; u32 o = i & 31;
    const u8 *blk = data + bi * 34;
    float d = f16_to_f32(*(const u16 *)blk);
    return (float)((const i8 *)(blk + 2))[o] * d;
}

static inline float dequant_q8_1(const u8 *data, u64 i) {
    u64 bi = i >> 5; u32 o = i & 31;
    const u8 *blk = data + bi * 40;
    float d; memcpy(&d, blk, 4);
    return (float)((const i8 *)(blk + 8))[o] * d;
}

static inline float dequant_q4_0(const u8 *data, u64 i) {
    u64 bi = i >> 5; u32 o = i & 31;
    const u8 *blk = data + bi * 18;
    float d = f16_to_f32(*(const u16 *)blk);
    u32 nib = (blk[2 + (o >> 1)] >> ((o & 1) << 2)) & 0xF;
    return ((float)(i32)nib - 8.0f) * d;
}

static inline float dequant_q4_1(const u8 *data, u64 i) {
    u64 bi = i >> 5; u32 o = i & 31;
    const u8 *blk = data + bi * 20;
    float d = f16_to_f32(*(const u16 *)blk);
    float m = f16_to_f32(*(const u16 *)(blk + 2));
    u32 nib = (blk[4 + (o >> 1)] >> ((o & 1) << 2)) & 0xF;
    return (float)nib * d + m;
}

static inline float dequant_q5_0(const u8 *data, u64 i) {
    u64 bi = i >> 5; u32 o = i & 31;
    const u8 *blk = data + bi * 22;
    float d = f16_to_f32(*(const u16 *)blk);
    u32 qh; memcpy(&qh, blk + 2, 4);
    u32 hi = (qh >> o) & 1;
    u32 lo = (blk[6 + (o >> 1)] >> ((o & 1) << 2)) & 0xF;
    return ((float)(i32)((hi << 4) | lo) - 16.0f) * d;
}

static inline float dequant_q5_1(const u8 *data, u64 i) {
    u64 bi = i >> 5; u32 o = i & 31;
    const u8 *blk = data + bi * 24;
    float d = f16_to_f32(*(const u16 *)blk);
    float m = f16_to_f32(*(const u16 *)(blk + 2));
    u32 qh; memcpy(&qh, blk + 4, 4);
    u32 hi = (qh >> o) & 1;
    u32 lo = (blk[8 + (o >> 1)] >> ((o & 1) << 2)) & 0xF;
    return (float)((hi << 4) | lo) * d + m;
}

/* ---- K-quant types (block_size = 256) ----------------------------- */

static inline float dequant_q8_k(const u8 *data, u64 i) {
    u64 bi = i >> 8; u32 o = i & 255;
    const u8 *blk = data + bi * 292;
    float d; memcpy(&d, blk, 4);
    float bs = f16_to_f32(*(const u16 *)(blk + 4 + 2 * (o >> 4)));
    return (float)((const i8 *)(blk + 36))[o] * bs * d;
}

static inline float dequant_q6_k(const u8 *data, u64 i) {
    u64 bi = i >> 8; u32 o = i & 255;
    const u8 *blk = data + bi * 210;
    float d = f16_to_f32(*(const u16 *)(blk + 208));
    i32 sc = (i8)blk[192 + (o >> 4)];

    /* Q6_K nibble layout (llama.cpp / ggml):
     * Each 128-element half uses 64 bytes of ql:
     *   group-0 (elts  0.. 31): low  nibble of ql[ 0..31]
     *   group-1 (elts 32.. 63): low  nibble of ql[32..63]
     *   group-2 (elts 64.. 95): high nibble of ql[ 0..31]
     *   group-3 (elts 96..127): high nibble of ql[32..63]
     * High 2 bits per element live in qh[0..31] (4×2-bit per byte). */
    u32 hl      = o & 127;          /* position within 128-element half   */
    u32 which   = hl >> 5;          /* 0..3: which 32-element sub-group   */
    u32 l       = hl & 31;          /* 0..31: position inside sub-group   */
    u32 half    = o >> 7;           /* 0 or 1: which 128-element half     */

    /* Low 4 bits from ql. */
    u32 ql_off  = (half << 6) + ((which & 1) << 5) + l;
    u32 lo      = (which >= 2) ? ((blk[ql_off] >> 4) & 0xF)
                               :  (blk[ql_off] & 0xF);

    /* High 2 bits from qh. */
    u32 qh_off  = 128 + (half << 5) + l;
    u32 hi      = (blk[qh_off] >> (which * 2)) & 0x3;

    i32 q  = (i32)(lo | (hi << 4)) - 32;
    return d * (float)sc * (float)q;
}

static inline float dequant_q5_k(const u8 *data, u64 i) {
    u64 bi = i >> 8; u32 o = i & 255;
    const u8 *blk = data + bi * 176;
    float d    = f16_to_f32(*(const u16 *)blk);
    float dmin = f16_to_f32(*(const u16 *)(blk + 2));
    u32 sb = o >> 5;
    u8 sc, mn;
    q4k_scale_min(blk + 4, sb, &sc, &mn);
    u32 g  = o >> 6;
    u32 nb = (o >> 5) & 1;
    u32 bc = o & 31;
    u32 lo = (blk[48 + g * 32 + bc] >> (nb * 4)) & 0xF;
    u32 hi = (blk[16 + (o & 31)] >> (o >> 5)) & 1;
    i32 q  = (i32)(lo | (hi << 4));
    return d * (float)sc * (float)q - dmin * (float)mn;
}

static inline float dequant_q4_k(const u8 *data, u64 i) {
    u64 bi = i >> 8; u32 o = i & 255;
    const u8 *blk = data + bi * 144;
    float d    = f16_to_f32(*(const u16 *)blk);
    float dmin = f16_to_f32(*(const u16 *)(blk + 2));
    u32 sb = o >> 5;
    u8 sc, mn;
    q4k_scale_min(blk + 4, sb, &sc, &mn);
    u32 g  = o >> 6;
    u32 nb = (o >> 5) & 1;
    u32 bc = o & 31;
    u32 nib = (blk[16 + g * 32 + bc] >> (nb * 4)) & 0xF;
    return d * (float)sc * (float)nib - dmin * (float)mn;
}

static inline float dequant_q3_k(const u8 *data, u64 i) {
    u64 bi = i >> 8; u32 o = i & 255;
    const u8 *blk = data + bi * 110;
    float d = f16_to_f32(*(const u16 *)(blk + 108));
    u32 s = o >> 4;
    u32 g  = o >> 7;
    u32 pr = (o >> 5) & 3;
    u32 bc = o & 31;
    u32 lo = (blk[32 + g * 32 + bc] >> (pr * 2)) & 3;
    u32 hi = (blk[o & 31] >> (o >> 5)) & 1;
    u32 sc_low  = (blk[96 + (s & 7)]  >> (s >= 8 ? 4 : 0)) & 0xF;
    u32 sc_high = (blk[104 + (s & 3)] >> ((s >> 2) * 2)) & 0x3;
    i32 sc = (i32)(sc_low | (sc_high << 4)) - 32;
    i32 q  = (i32)(lo | (hi << 2)) - 4;
    return d * (float)sc * (float)q;
}

static inline float dequant_q2_k(const u8 *data, u64 i) {
    u64 bi = i >> 8; u32 o = i & 255;
    const u8 *blk = data + bi * 84;
    float d  = f16_to_f32(*(const u16 *)(blk + 64 + 16));
    float mn = f16_to_f32(*(const u16 *)(blk + 64 + 16 + 2));
    u32 sb  = o >> 4;
    u32 sc  = blk[64 + (sb >> 1)];
    if (sb & 1) sc >>= 4; else sc &= 0xF;
    u32 q2 = (blk[(o >> 2)] >> ((o & 3) << 1)) & 0x3;
    i32 q  = (i32)q2 - (i32)((blk[(o >> 3)] >> (o & 7)) & 1) * 4;
    float sc_f = (float)(i32)sc;
    return d * sc_f * (float)q - mn;
}

/* ---- IQ (importance-matrix-aware) types --------------------------- */

static inline float dequant_iq2_xxs(const u8 *data, u64 i) {
    u64 bi = i >> 8; u32 o = i & 255;
    const u8 *blk = data + bi * 66;
    float d = f16_to_f32(*(const u16 *)blk);
    u32 q = (blk[2 + (o >> 2)] >> ((o & 3) << 1)) & 0x3;
    u32 sign_bit = (blk[2 + (o >> 3)] >> (o & 7)) & 1;
    float v = iq2_xxs_grid[q | (sign_bit << 8)];
    return v * d;
}

static inline float dequant_iq2_xs(const u8 *data, u64 i) {
    u64 bi = i >> 8; u32 o = i & 255;
    const u8 *blk = data + bi * 74;
    float d = f16_to_f32(*(const u16 *)blk);
    u32 sh = (blk[2 + (o >> 3)] >> (o & 7)) & 1;
    u32 q  = (blk[2 + 32 + (o >> 2)] >> ((o & 3) << 1)) & 0x3;
    u32 idx = q | (sh << 2);
    i8  sc = (i8)(blk[2 + 32 + 64 + (o >> 4)]);
    return d * (float)sc * iq2_xs_grid[idx];
}

static inline float dequant_iq3_xxs(const u8 *data, u64 i) {
    u64 bi = i >> 8; u32 o = i & 255;
    const u8 *blk = data + bi * 98;
    float d = f16_to_f32(*(const u16 *)blk);
    u32 qh = (blk[2 + (o >> 5)] >> (o & 31)) & 1;
    u32 qm = (blk[2 + 8 + (o >> 3)] >> (o & 7)) & 1;
    u32 ql = (blk[2 + 8 + 32 + (o >> 1)] >> ((o & 1) << 2)) & 0xF;
    u32 grp = (blk[2 + 8 + 32 + 128 + (o >> 3)] >> ((o & 7) << 1)) & 0x3;
    u32 idx = ql | (qm << 4) | (qh << 5) | (grp << 6);
    i8  sc  = (i8)(blk[2 + 8 + 32 + 128 + 64 + (o >> 4)]);
    return d * (float)sc * iq3_xxs_grid[idx];
}

static inline float dequant_iq1_s(const u8 *data, u64 i) {
    u64 bi = i >> 8; u32 o = i & 255;
    const u8 *blk = data + bi * 110;
    float d = f16_to_f32(*(const u16 *)blk);
    u32 q  = ((blk[2 + (o >> 3)] >> (o & 7)) & 1) |
             (((blk[2 + 32 + (o >> 3)] >> (o & 7)) & 1) << 1);
    u32 idx = q | (((o >> 4) & 0xF) << 2);
    return d * iq1_s_grid[idx];
}

static inline float dequant_iq4_nl(const u8 *data, u64 i) {
    u64 bi = i >> 8; u32 o = i & 255;
    const u8 *blk = data + bi * 50;
    float d = f16_to_f32(*(const u16 *)blk);
    u32 q = (blk[2 + (o >> 1)] >> ((o & 1) << 2)) & 0xF;
    return d * iq4_nl_values[q | ((o & 0xF0) << 4)];
}

static inline float dequant_iq1_m(const u8 *data, u64 i) {
    u64 bi = i >> 8; u32 o = i & 255;
    const u8 *blk = data + bi * 56;
    float d = f16_to_f32(*(const u16 *)(blk + 54));
    u32 sb    = o >> 4;
    u32 sc    = (blk[sb >> 1] >> ((sb & 1) << 2)) & 0xF;
    float sc_f = (float)(i32)sc + 1.0f;
    u32 val  = (blk[8 + (o >> 3)] >> (o & 7)) & 1;
    u32 sign = (blk[8 + 32 + (o >> 3)] >> (o & 7)) & 1;
    float v = sign ? -(float)val : (float)val;
    return v * sc_f * d;
}

static inline float dequant_iq3_s(const u8 *data, u64 i) {
    u64 bi = i >> 8; u32 o = i & 255;
    const u8 *blk = data + bi * 110;
    float d = f16_to_f32(*(const u16 *)blk);
    u32 sign = (blk[2 + (o >> 3)] >> (o & 7)) & 1;
    u32 qh   = (blk[2 + 32 + (o >> 5)] >> (o & 31)) & 1;
    u32 qm   = (blk[2 + 32 + 8 + (o >> 3)] >> (o & 7)) & 1;
    u32 ql   = (blk[2 + 32 + 8 + 32 + (o >> 1)] >> ((o & 1) << 2)) & 0xF;
    u32 idx  = ql | (qm << 4) | (qh << 5) | (sign << 6);
    i8  sc   = (i8)blk[2 + 32 + 8 + 32 + 128 + (o >> 4)];
    return d * (float)sc * iq3_s_grid[idx];
}

static inline float dequant_iq2_s(const u8 *data, u64 i) {
    u64 bi = i >> 8; u32 o = i & 255;
    const u8 *blk = data + bi * 82;
    float d = f16_to_f32(*(const u16 *)blk);
    u32 sign = (blk[2 + (o >> 3)] >> (o & 7)) & 1;
    u32 qh   = (blk[2 + 32 + (o >> 5)] >> (o & 31)) & 1;
    u32 ql   = (blk[2 + 32 + 8 + (o >> 1)] >> ((o & 1) << 2)) & 0xF;
    u32 idx  = ql | (qh << 4) | (sign << 5);
    i8  sc   = (i8)blk[2 + 32 + 8 + 128 + (o >> 4) + ((o >> 4) & 1) * 16];
    u32 sc_l = (blk[2 + 32 + 8 + 128 + (o >> 5)] >> ((o & 31) >> 2)) & 0x3;
    float sc_f = (float)sc + 0.125f * (float)(i32)sc_l;
    return d * sc_f * iq2_xs_grid[idx];
}

static inline float dequant_iq4_xs(const u8 *data, u64 i) {
    u64 bi = i >> 8; u32 o = i & 255;
    const u8 *blk = data + bi * 136;
    float d = f16_to_f32(*(const u16 *)blk);
    u32 qh = (blk[2 + (o >> 5)] >> (o & 31)) & 1;
    u32 ql = (blk[2 + 8 + (o >> 1)] >> ((o & 1) << 2)) & 0xF;
    u32 q  = ql | (qh << 4);
    i8  sc_l = (i8)blk[2 + 8 + 128 + (o >> 4)];
    i8  sc_h = (i8)blk[2 + 8 + 128 + 16 + (o >> 4)];
    float sc = (float)(i32)sc_l + ((float)(i32)sc_h / 127.0f);
    return d * sc * iq4_nl_values[q];
}

/* ---- Dispatcher --------------------------------------------------- */

float gguf_dequant(TensorInfo *ti, const u8 *base, u64 i) {
    const u8 *data = base + ti->offset;
    switch (ti->type) {
        case GGUF_TYPE_F32:     return dequant_f32(data, i);
        case GGUF_TYPE_F16:     return dequant_f16(data, i);
        case GGUF_TYPE_BF16:    return dequant_bf16(data, i);
        case GGUF_TYPE_F64:     return dequant_f64(data, i);
        case GGUF_TYPE_I8:      return dequant_i8(data, i);
        case GGUF_TYPE_I16:     return dequant_i16(data, i);
        case GGUF_TYPE_I32:     return dequant_i32(data, i);
        case GGUF_TYPE_I64:     return dequant_i64(data, i);
        case GGUF_TYPE_Q8_0:    return dequant_q8_0(data, i);
        case GGUF_TYPE_Q8_1:    return dequant_q8_1(data, i);
        case GGUF_TYPE_Q4_0:    return dequant_q4_0(data, i);
        case GGUF_TYPE_Q4_1:    return dequant_q4_1(data, i);
        case GGUF_TYPE_Q5_0:    return dequant_q5_0(data, i);
        case GGUF_TYPE_Q5_1:    return dequant_q5_1(data, i);
        case GGUF_TYPE_Q8_K:    return dequant_q8_k(data, i);
        case GGUF_TYPE_Q6_K:    return dequant_q6_k(data, i);
        case GGUF_TYPE_Q5_K:    return dequant_q5_k(data, i);
        case GGUF_TYPE_Q4_K:    return dequant_q4_k(data, i);
        case GGUF_TYPE_Q3_K:    return dequant_q3_k(data, i);
        case GGUF_TYPE_Q2_K:    return dequant_q2_k(data, i);
        case GGUF_TYPE_IQ2_XXS: return dequant_iq2_xxs(data, i);
        case GGUF_TYPE_IQ2_XS:  return dequant_iq2_xs(data, i);
        case GGUF_TYPE_IQ3_XXS: return dequant_iq3_xxs(data, i);
        case GGUF_TYPE_IQ1_S:   return dequant_iq1_s(data, i);
        case GGUF_TYPE_IQ4_NL:  return dequant_iq4_nl(data, i);
        case GGUF_TYPE_IQ1_M:   return dequant_iq1_m(data, i);
        case GGUF_TYPE_IQ3_S:   return dequant_iq3_s(data, i);
        case GGUF_TYPE_IQ2_S:   return dequant_iq2_s(data, i);
        case GGUF_TYPE_IQ4_XS:  return dequant_iq4_xs(data, i);
        default:
            slog(WARN, "Unsupported tensor type %u — use f32/f16/bf16 weights",
                 ti->type);
            return 0.0f;
    }
}
