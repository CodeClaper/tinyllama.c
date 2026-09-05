/*
 * test_gpu.c — CUDA matmul backend unit tests (built only with GPU=1).
 *
 * For every quant type and several tensor shapes (including a
 * non-block-aligned one, forcing quant blocks to span row boundaries):
 *
 *   Part A — exact decode:  gpu_dequant_batch() must reproduce
 *            gguf_dequant() bit-for-bit on the same raw bytes.
 *   Part B — matmul:  gpu_matvec()/gpu_matmat() (both trans
 *            orientations) vs a double-precision reference; tolerance
 *            covers f32-vs-double accumulation only, not decode bugs
 *            (those are Part A's job, bit-exact).
 *
 * Random bytes are masked to 0x3F so every f16/f32/bf16 header bit
 * pattern is finite (no inf/nan) and magnitudes stay bounded.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <stdbool.h>
#include "minunit.h"
#include "gpu/quants_gpu.h"
#include "quants.h"

/* Short name for diagnostics (no such helper exists in the engine). */
static const char *type_name(u32 type) {
    switch (type) {
        case GGUF_TYPE_F32:     return "F32";    
        case GGUF_TYPE_F16:     return "F16";
        case GGUF_TYPE_BF16:    return "BF16";  
        case GGUF_TYPE_F64:     return "F64";
        case GGUF_TYPE_I8:      return "I8";      
        case GGUF_TYPE_I16:     return "I16";
        case GGUF_TYPE_I32:     return "I32";    
        case GGUF_TYPE_I64:     return "I64";
        case GGUF_TYPE_Q4_0:    return "Q4_0";  
        case GGUF_TYPE_Q4_1:    return "Q4_1";
        case GGUF_TYPE_Q5_0:    return "Q5_0";  
        case GGUF_TYPE_Q5_1:    return "Q5_1";
        case GGUF_TYPE_Q8_0:    return "Q8_0";  
        case GGUF_TYPE_Q8_1:    return "Q8_1";
        case GGUF_TYPE_Q2_K:    return "Q2_K";  
        case GGUF_TYPE_Q3_K:    return "Q3_K";
        case GGUF_TYPE_Q4_K:    return "Q4_K";  
        case GGUF_TYPE_Q5_K:    return "Q5_K";
        case GGUF_TYPE_Q6_K:    return "Q6_K";  
        case GGUF_TYPE_Q8_K:    return "Q8_K";
        case GGUF_TYPE_IQ2_XXS: return "IQ2_XXS"; 
        case GGUF_TYPE_IQ2_XS:  return "IQ2_XS";
        case GGUF_TYPE_IQ3_XXS: return "IQ3_XXS"; 
        case GGUF_TYPE_IQ1_S:   return "IQ1_S";
        case GGUF_TYPE_IQ4_NL:  return "IQ4_NL"; 
        case GGUF_TYPE_IQ3_S:   return "IQ3_S";
        case GGUF_TYPE_IQ2_S:   return "IQ2_S"; 
        case GGUF_TYPE_IQ4_XS:  return "IQ4_XS";
        case GGUF_TYPE_IQ1_M:   return "IQ1_M";
        default:                return "?";
    }
}

/* ---- deterministic PRNG (xorshift64*) ---- */
static uint64_t rng_state = 0x9E3779B97F4A7C15ull;
static uint32_t rng_next(void) {
    uint64_t x = rng_state;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    rng_state = x;
    return (uint32_t)((x * 0x2545F4914F6CDD1Dull) >> 32);
}

/* Per-type block geometry — must match gpu_geom() in quants_gpu.cu
 * and gguf_types[] in core.c. */
typedef struct { u32 type; u64 be; u64 bb; } TypeGeom;

#define GEOM(T, BE, BB) { GGUF_TYPE_##T, BE, BB }
static const TypeGeom type_geoms[] = {
    GEOM(F32, 1, 4),    GEOM(F16, 1, 2),    GEOM(BF16, 1, 2),   GEOM(F64, 1, 8),
    GEOM(I8, 1, 1),     GEOM(I16, 1, 2),    GEOM(I32, 1, 4),    GEOM(I64, 1, 8),
    GEOM(Q4_0, 32, 18), GEOM(Q4_1, 32, 20), GEOM(Q5_0, 32, 22), GEOM(Q5_1, 32, 24),
    GEOM(Q8_0, 32, 34), GEOM(Q8_1, 32, 40),
    GEOM(Q2_K, 256, 84), GEOM(Q3_K, 256, 110), GEOM(Q4_K, 256, 144),
    GEOM(Q5_K, 256, 176), GEOM(Q6_K, 256, 210), GEOM(Q8_K, 256, 292),
    GEOM(IQ2_XXS, 256, 66), GEOM(IQ2_XS, 256, 74), GEOM(IQ3_XXS, 256, 98),
    GEOM(IQ1_S, 256, 110), GEOM(IQ4_NL, 256, 50), GEOM(IQ3_S, 256, 110),
    GEOM(IQ2_S, 256, 82), GEOM(IQ4_XS, 256, 136), GEOM(IQ1_M, 256, 56),
};
#define N_TYPES (sizeof(type_geoms) / sizeof(type_geoms[0]))

/* Tensor shapes: [rows, cols] (nontrans).  The first is not
 * block-aligned (forces cross-row blocks); the third exercises
 * trans rows > 256.  The last two give the K-quants more than one
 * block per row, which is the case the block-cooperative GEMV fast
 * path loops over (cols == 256 would only ever produce bpr == 1). */
static const u64 shapes[][2] = { {37, 120}, {64, 256}, {300, 256},
                                 {64, 768}, {37, 512} };
#define N_SHAPES (sizeof(shapes) / sizeof(shapes[0]))

/* Build a random weight tensor.  buf holds ceil(rows*cols/be)*bb
 * bytes plus a padding tail; ti describes it (offset 0).  Returns
 * n_element.
 *
 * The padding is included in ti->bytes on purpose: some IQ decoders
 * (both CPU and GPU, identically) read a few bytes past the last
 * quant block (block-relative offsets that spill beyond the block
 * end).  With a pad inside the declared tensor size, those reads land
 * in deterministic bytes on both sides instead of host-heap vs
 * device-allocator garbage. */
#define GPU_TEST_PAD 512u
static u64 make_tensor(u32 type, u64 be, u64 bb,
                       u64 rows, u64 cols, u8 **buf, TensorInfo *ti) {
    u64 n_elems = rows * cols;
    u64 n_blocks = (n_elems + be - 1) / be;
    u64 bytes = n_blocks * bb + GPU_TEST_PAD;
    *buf = malloc(bytes);
    for (u64 i = 0; i < bytes; i++)
        (*buf)[i] = (u8)(rng_next() & 0x3F);

    memset(ti, 0, sizeof(*ti));
    ti->key.len    = 0;
    ti->ndim       = 2;
    ti->dim[0]     = rows;
    ti->dim[1]     = cols;
    ti->type       = type;
    ti->offset     = 0;
    ti->n_element  = n_elems;
    ti->bytes      = bytes;
    ti->data       = *buf;
    return n_elems;
}

/* Part A: gpu_dequant_batch vs gguf_dequant, bit-exact.  Returns the
 * number of mismatching elements (prints the first few). */
static int check_dequant_exact(TensorInfo *ti) {
    u64 n = ti->n_element;
    float *out = malloc(n * sizeof(float));
    if (gpu_dequant_batch(ti, 0, n, out) != 0) {
        printf("    gpu_dequant_batch failed\n");
        free(out);
        return 1;
    }
    int bad = 0;
    for (u64 i = 0; i < n; i++) {
        float ref = gguf_dequant(ti, i);
        union { float f; u32 u; } a = { out[i] }, b = { ref };
        if (a.u != b.u) {
            if (bad < 5)
                printf("    elem %llu: gpu=0x%08x (%g) ref=0x%08x (%g)\n",
                       (unsigned long long)i, a.u, a.f, b.u, b.f);
            bad++;
        }
    }
    free(out);
    return bad;
}

/* Reference dot product in double: y[b*rows + r] = sum_c w[i]*x[b*cols+c]
 * with i = r*cols+c (nontrans) or c*rows+r (trans). */
static double ref_dot(TensorInfo *ti, u64 rows, u64 cols,
                      bool trans, u64 b, u64 r, u64 batch, const float *x) {
    double sum = 0.0;
    for (u64 c = 0; c < cols; c++) {
        u64 i = trans ? c * rows + r : r * cols + c;
        sum += (double)gguf_dequant(ti, i) * (double)x[b * cols + c];
    }
    return sum;
}

/* Part B: matvec (both orientations) and matmat (batch=3).  Tolerance
 * covers f32 accumulation drift (values bounded by the 0x3F mask);
 * any index/decode error is orders of magnitude larger. */
static int check_matmul(TensorInfo *ti, u64 rows, u64 cols,
                        bool trans, u64 batch) {
    u64 nx = batch * cols, ny = batch * rows;
    float *x = malloc(nx * sizeof(float));
    float *got = malloc(ny * sizeof(float));
    for (u64 i = 0; i < nx; i++)
        x[i] = (float)((int32_t)rng_next() % 2000001 - 1000000) / 1000000.0f;

    int bad = 0;
    int rc;
    if (batch == 1)
        rc = gpu_matvec(ti, x, got, rows, cols, trans);
    else
        rc = gpu_matmat(ti, x, got, batch, rows, cols, trans);
    if (rc != 0) {
        printf("    gpu_matvec/matmat returned %d\n", rc);
        bad++;
    } else {
        for (u64 b = 0; b < batch; b++) {
            for (u64 r = 0; r < rows; r++) {
                double ref = ref_dot(ti, rows, cols, trans, b, r, batch, x);
                double tol = 1e-2 * (fabs(ref) + 100.0);
                double err = fabs((double)got[b * rows + r] - ref);
                if (err > tol) {
                    if (bad < 5)
                        printf("    [b=%llu r=%llu] got=%g ref=%g err=%g tol=%g\n",
                               (unsigned long long)b, (unsigned long long)r,
                               got[b * rows + r], ref, err, tol);
                    bad++;
                }
            }
        }
    }
    free(x);
    free(got);
    return bad;
}

/* Part C — decode identity for the block-cooperative GEMV fast path.
 *
 * The matvec fast path (cols % block_elems == 0) decodes each quant
 * block's scale/min header once per lane instead of once per element.
 * A wrong per-lane element mapping would still land inside Part B's
 * loose tolerance, so compare against a reference built from the
 * per-element decode (which Part A proves bit-exact) and accumulated
 * in double.  Only f32 accumulation order — not decode — may then
 * separate the two, so the tolerance can be tight. */
static int check_decode_identity(TensorInfo *ti, u64 rows, u64 cols) {
    float *w = malloc(ti->n_element * sizeof(float));
    float *x = malloc(cols * sizeof(float));
    float *got = malloc(rows * sizeof(float));
    if (!w || !x || !got) { free(w); free(x); free(got); return 1; }

    if (gpu_dequant_batch(ti, 0, ti->n_element, w) != 0) {
        printf("    gpu_dequant_batch failed\n");
        free(w); free(x); free(got);
        return 1;
    }
    for (u64 i = 0; i < cols; i++)
        x[i] = (float)((int32_t)rng_next() % 2000001 - 1000000) / 1000000.0f;

    int bad = 0;
    if (gpu_matvec(ti, x, got, rows, cols, false) != 0) {
        printf("    gpu_matvec failed\n");
        bad++;
    } else {
        for (u64 r = 0; r < rows; r++) {
            double ref = 0.0;
            for (u64 c = 0; c < cols; c++)
                ref += (double)w[r * cols + c] * (double)x[c];
            double tol = 1e-3 * (fabs(ref) + 1.0) + 1e-4 * (double)cols;
            double err = fabs((double)got[r] - ref);
            if (err > tol) {
                if (bad < 5)
                    printf("    [r=%llu] got=%g ref=%g err=%g tol=%g\n",
                           (unsigned long long)r, got[r], ref, err, tol);
                bad++;
            }
        }
    }
    free(w); free(x); free(got);
    return bad;
}

/* Run every check for one type; returns total failures. */
static int run_type(u32 type, u64 be, u64 bb) {
    int fails = 0;
    for (u64 s = 0; s < N_SHAPES; s++) {
        u64 rows = shapes[s][0], cols = shapes[s][1];
        u8 *buf;
        TensorInfo ti;
        make_tensor(type, be, bb, rows, cols, &buf, &ti);

        int bad = check_dequant_exact(&ti);
        if (bad)
            printf("  [%s %llux%llu] dequant: %d mismatches\n",
                   type_name(type), (unsigned long long)rows,
                   (unsigned long long)cols, bad);

        bad = check_matmul(&ti, rows, cols, false, 1);
        if (bad)
            printf("  [%s %llux%llu] matvec: %d mismatches\n",
                   type_name(type), (unsigned long long)rows,
                   (unsigned long long)cols, bad);

        /* Fast path (block-aligned cols) must decode identically to the
         * per-element path. */
        if (cols % be == 0) {
            bad = check_decode_identity(&ti, rows, cols);
            if (bad)
                printf("  [%s %llux%llu] decode identity: %d mismatches\n",
                       type_name(type), (unsigned long long)rows,
                       (unsigned long long)cols, bad);
            fails += bad;
        }

        /* trans: tensor reinterpreted as [cols x rows] */
        bad = check_matmul(&ti, cols, rows, true, 1);
        if (bad)
            printf("  [%s %llux%llu] matvec trans: %d mismatches\n",
                   type_name(type), (unsigned long long)rows,
                   (unsigned long long)cols, bad);

        bad = check_matmul(&ti, rows, cols, false, 3);
        if (bad)
            printf("  [%s %llux%llu] matmat: %d mismatches\n",
                   type_name(type), (unsigned long long)rows,
                   (unsigned long long)cols, bad);

        fails += bad;
        free(buf);
    }
    return fails;
}

#define TYPE_TEST(T) MU_TEST(test_##T) { \
    int f = 0; \
    for (u64 i = 0; i < N_TYPES; i++) \
        if (type_geoms[i].type == GGUF_TYPE_##T) \
            f += run_type(type_geoms[i].type, type_geoms[i].be, type_geoms[i].bb); \
    mu_assert_int_eq(0, f); \
}

TYPE_TEST(F32)
TYPE_TEST(F16)
TYPE_TEST(BF16)
TYPE_TEST(F64)
TYPE_TEST(I8)
TYPE_TEST(I16)
TYPE_TEST(I32)
TYPE_TEST(I64)
TYPE_TEST(Q4_0)
TYPE_TEST(Q4_1)
TYPE_TEST(Q5_0)
TYPE_TEST(Q5_1)
TYPE_TEST(Q8_0)
TYPE_TEST(Q8_1)
TYPE_TEST(Q2_K)
TYPE_TEST(Q3_K)
TYPE_TEST(Q4_K)
TYPE_TEST(Q5_K)
TYPE_TEST(Q6_K)
TYPE_TEST(Q8_K)
TYPE_TEST(IQ2_XXS)
TYPE_TEST(IQ2_XS)
TYPE_TEST(IQ3_XXS)
TYPE_TEST(IQ1_S)
TYPE_TEST(IQ4_NL)
TYPE_TEST(IQ3_S)
TYPE_TEST(IQ2_S)
TYPE_TEST(IQ4_XS)
TYPE_TEST(IQ1_M)

/* Cache behaviour: a second call on the same tensor must reuse the
 * cached device copy and give identical results. */
MU_TEST(test_cache_hit) {
    u8 *buf;
    TensorInfo ti;
    make_tensor(GGUF_TYPE_Q4_K, 256, 144, 64, 256, &buf, &ti);
    u64 n = ti.n_element;
    float *x = malloc(n * sizeof(float));
    float *y1 = malloc(64 * sizeof(float));
    float *y2 = malloc(64 * sizeof(float));
    for (u64 i = 0; i < n; i++)
        x[i] = (float)((int32_t)rng_next() % 2000001 - 1000000) / 1000000.0f;

    mu_assert_int_eq(0, gpu_matvec(&ti, x, y1, 64, 256, false));
    mu_assert_int_eq(0, gpu_matvec(&ti, x, y2, 64, 256, false));
    mu_assert_int_eq(0, memcmp(y1, y2, 64 * sizeof(float)));

    free(x); free(y1); free(y2); free(buf);
}

/* Failure contract: bad dims must not be computed (nonzero return). */
MU_TEST(test_bad_dims) {
    u8 *buf;
    TensorInfo ti;
    make_tensor(GGUF_TYPE_Q4_K, 256, 144, 64, 256, &buf, &ti);
    float x[256], y[64];
    mu_assert(gpu_matvec(&ti, x, y, 64, 300, false) != 0,
              "oversized cols should fail");
    mu_assert(gpu_matvec(NULL, x, y, 64, 256, false) != 0,
              "NULL tensor should fail");
    free(buf);
}

static void all_tests(void) {
    if (!gpu_available()) {
        printf("test_gpu requires a CUDA device (build with GPU=1)\n");
        minunit_fail = 1;
        return;
    }
    MU_RUN_TEST(test_F32);   MU_RUN_TEST(test_F16);   MU_RUN_TEST(test_BF16);
    MU_RUN_TEST(test_F64);   MU_RUN_TEST(test_I8);    MU_RUN_TEST(test_I16);
    MU_RUN_TEST(test_I32);   MU_RUN_TEST(test_I64);
    MU_RUN_TEST(test_Q4_0);  MU_RUN_TEST(test_Q4_1);  MU_RUN_TEST(test_Q5_0);
    MU_RUN_TEST(test_Q5_1);  MU_RUN_TEST(test_Q8_0);  MU_RUN_TEST(test_Q8_1);
    MU_RUN_TEST(test_Q2_K);  MU_RUN_TEST(test_Q3_K);  MU_RUN_TEST(test_Q4_K);
    MU_RUN_TEST(test_Q5_K);  MU_RUN_TEST(test_Q6_K);  MU_RUN_TEST(test_Q8_K);
    MU_RUN_TEST(test_IQ2_XXS); MU_RUN_TEST(test_IQ2_XS);
    MU_RUN_TEST(test_IQ3_XXS); MU_RUN_TEST(test_IQ1_S);
    MU_RUN_TEST(test_IQ4_NL); MU_RUN_TEST(test_IQ3_S);
    MU_RUN_TEST(test_IQ2_S);  MU_RUN_TEST(test_IQ4_XS);
    MU_RUN_TEST(test_IQ1_M);
    MU_RUN_TEST(test_cache_hit);
    MU_RUN_TEST(test_bad_dims);
}

int main(void) {
    all_tests();
    gpu_shutdown();
    MU_REPORT();
    return MU_EXIT_CODE;
}
