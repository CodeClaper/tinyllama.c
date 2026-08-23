#include <errno.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <inttypes.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>
#include <sys/file.h>
#include "core.h"
#include "cpu/quants_cpu.h"
#ifdef GPU_BUILD
#include "gpu/quants_gpu.h"
#endif
#include "pthreads.h"
#include "mm.h"
#include "slog.h"
#include <math.h>
#include "utils.h"
#include "model/model.h"

#define GGUF_MAGIC              0x46554747u /* "GGUF", little endian. */
#define GGUF_VALID_VERSION      3           /* GGUF valid version, only support 3. */
#define GGUF_MAX_DIMS           8           /* GGUF max dims. */
#define GGUF_DEFAULT_ALIGNMENT  32          /* GGUF default alignment. */

typedef enum {
    GGUF_VALUE_UINT8   = 0,
    GGUF_VALUE_INT8    = 1,
    GGUF_VALUE_UINT16  = 2,
    GGUF_VALUE_INT16   = 3,
    GGUF_VALUE_UINT32  = 4,
    GGUF_VALUE_INT32   = 5,
    GGUF_VALUE_FLOAT32 = 6,
    GGUF_VALUE_BOOL    = 7,
    GGUF_VALUE_STRING  = 8,
    GGUF_VALUE_ARRAY   = 9,
    GGUF_VALUE_UINT64  = 10,
    GGUF_VALUE_INT64   = 11,
    GGUF_VALUE_FLOAT64 = 12,
} GGUFValueType;

typedef struct {
    const char *name;
    u32 block_elems;
    u32 block_bytes;
} GGUFTypeInfo;

static const GGUFTypeInfo gguf_types[] = {
    [0]  = {"f32",      1,   4},
    [1]  = {"f16",      1,   2},
    [2]  = {"q4_0",    32,  18},
    [3]  = {"q4_1",    32,  20},
    [6]  = {"q5_0",    32,  22},
    [7]  = {"q5_1",    32,  24},
    [8]  = {"q8_0",    32,  34},
    [9]  = {"q8_1",    32,  40},
    [10] = {"q2_k",   256,  84},
    [11] = {"q3_k",   256, 110},
    [12] = {"q4_k",   256, 144},
    [13] = {"q5_k",   256, 176},
    [14] = {"q6_k",   256, 210},
    [15] = {"q8_k",   256, 292},
    [16] = {"iq2_xxs",256,  66},
    [17] = {"iq2_xs", 256,  74},
    [18] = {"iq3_xxs",256,  98},
    [19] = {"iq1_s",  256, 110},
    [20] = {"iq4_nl", 256,  50},
    [21] = {"iq3_s",  256, 110},
    [22] = {"iq2_s",  256,  82},
    [23] = {"iq4_xs", 256, 136},
    [24] = {"i8",       1,   1},
    [25] = {"i16",      1,   2},
    [26] = {"i32",      1,   4},
    [27] = {"i64",      1,   8},
    [28] = {"f64",      1,   8},
    [29] = {"iq1_m",  256,  56},
    [30] = {"bf16",     1,   2},
};

/* Read a single f32/f16/bf16/quantised weight from a GGUF tensor at index i.
 * Supports all GGUF v3 types. Delegates to gguf_dequant() for per-type
 * unpacking. */
float tensor_get_f32(TensorInfo *ti, const u8 *base, u64 i) {
    if (i >= ti->n_element) {
        char name[128];
        snprintf(name, sizeof(name), "%.*s",
                 ti->key.len < 127 ? ti->key.len : 127, ti->key.content);
        slog(WARN, "tensor_get_f32 OOB: tensor='%s' n_element=%llu "
             "index=%llu type=%u ndim=%u",
             name, (unsigned long long)ti->n_element,
             (unsigned long long)i, ti->type, ti->ndim);
        if (ti->ndim > 0) {
            slog(WARN, "  dims: [%llu%s%s",
                 (unsigned long long)ti->dim[0],
                 ti->ndim > 1 ? "," : "]",
                 ti->ndim > 1 ? "" : "");
            for (u32 d = 1; d < ti->ndim && d < 8; d++)
                slog(WARN, "         %llu%s",
                     (unsigned long long)ti->dim[d],
                     d + 1 < ti->ndim ? "," : "]");
        }
        slog(ERROR, "Fatal: out-of-bounds tensor access");
    }
    return gguf_dequant(ti, base, i);
}

/* Batch version of tensor_get_f32: dequantises nb contiguous elements
 * starting at i0 into out[0..nb-1].  Uses NEON SIMD for full blocks. */
void tensor_get_f32_batch(TensorInfo *ti, const u8 *base, u64 i0, u64 nb, float *out) {
    if (i0 + nb > ti->n_element || nb == 0) {
        char name[128];
        snprintf(name, sizeof(name), "%.*s", ti->key.len < 127 ? ti->key.len : 127, ti->key.content);
        slog(WARN, "tensor_get_f32_batch OOB: tensor='%s' n_element=%llu "
             "range=[%llu,%llu) type=%u",
             name, (unsigned long long)ti->n_element,
             (unsigned long long)i0, (unsigned long long)(i0 + nb), ti->type);
        if (ti->ndim > 0) {
            slog(WARN, "  dims: [%llu%s%s",
                 (unsigned long long)ti->dim[0],
                 ti->ndim > 1 ? "," : "]",
                 ti->ndim > 1 ? "" : "");
            for (u32 d = 1; d < ti->ndim && d < 8; d++)
                slog(WARN, "         %llu%s",
                     (unsigned long long)ti->dim[d],
                     d + 1 < ti->ndim ? "," : "]");
        }
        slog(ERROR, "Fatal: out-of-bounds batch tensor access");
    }
    gguf_dequant_batch(ti, base, i0, nb, out);
}

/* ---- Math primitives ------------------------------------------- */

/* RMS Normalisation: o = x / rms(x) * w  (in-place ok when o == x). */
void rms_norm(float *o, const float *x, TensorInfo *tw, const u8 *base, int n, float eps) {
    u64 tn = tw->dim[0]; /* 1-D weight tensor */
    if ((u64)n > tn) {
        slog(WARN, "rms_norm: n=%d > tensor dim=%llu, clamping", n, (unsigned long long)tn);
        n = (int)tn;
    }
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float scale = 1.0f / sqrtf(ss / (float)n + eps);

    /* Batch-dequantise the 1-D weight into a contiguous buffer so
     * the hot element-wise multiply loop only touches f32 data. */
    #define RMS_STACK 4096
    float  rms_stack[RMS_STACK];
    float *w_buf = (u64)n <= RMS_STACK ? rms_stack : smalloc((u64)n * sizeof(float));
    tensor_get_f32_batch(tw, base, 0, (u64)n, w_buf);
    for (int i = 0; i < n; i++)
        o[i] = x[i] * scale * w_buf[i];
    if (w_buf != rms_stack) sfree(w_buf);
    #undef RMS_STACK
}

/* ---- Parallel mat-vec row workers ----------------------------------
 * Each invocation of the worker computes one output element y[i].
 * The context is shared read-only across threads (it is fully
 * initialised before pthreads_parallel_for is entered), and each
 * worker writes a disjoint index of y, so there is no data race. */
typedef struct {
    float       *y;       /* output row slice                      */
    TensorInfo  *tw;      /* weight tensor                         */
    const u8    *base;    /* mmap base                             */
    const float *x;       /* input vector                          */
    u64          cols;    /* row length / dot-product length       */
    const i8    *x_i8;    /* i8-quantised x (dotprod path only)    */
    float        x_scale; /* scale for x_i8 (dotprod path only)    */
} MatVecCtx;

/* Standard fused dequant + dot product row worker. */
static void matvec_worker(void *arg, int tid, int i) {
    (void)tid;
    MatVecCtx *c = (MatVecCtx *)arg;
    c->y[i] = gguf_dot_batch(c->tw, c->base, (u64)i * c->cols, c->cols, c->x);
}

/* i8 dot-product row worker (ARMv8.2+ dotprod, Q8_0 / Q8_K). */
#if defined(__ARM_FEATURE_DOTPROD)
static void matvec_i8_worker(void *arg, int tid, int i) {
    (void)tid;
    MatVecCtx *c = (MatVecCtx *)arg;
    c->y[i] = gguf_dot_i8_batch(c->tw, c->base, (u64)i * c->cols, c->cols,
                                c->x_i8, c->x_scale);
}
#endif

/* Matrix-vector multiply: y = W @ x  (W is [rows × cols], stored row-major).
 * cols is derived from the tensor's last dimension; rows must match dim[0].
 *
 * When trans=true, the weight is treated as transposed: y = W^T @ x.
 * In that case dim[0] is the input dimension and dim[ndim-1] is the output.
 *
 * When pool is non-NULL and has more than one thread, the row loop of the
 * standard (non-transposed) path is dispatched across the worker pool;
 * each row's fused dequant + dot product runs on a separate core.  The
 * transposed path stays serial (its strided gather needs per-row scratch). */
bool mat_vec_mul(float *y, TensorInfo *tw, const u8 *base, const float *x, u64 rows, u64 cols, bool trans, pthreads_t *pool) {
    if (!tw || tw->ndim < 2) {
        slog(WARN, "mat_vec_mul: tensor missing or ndim < 2");
        return false;
    }
    u64 tc = tw->dim[tw->ndim - 1]; /* fastest-varying = column count */
    u64 tr = tw->dim[0];            /* slowest-varying = row count     */

    /* Orientation validation, checked once for both paths. */
    if (trans ? (tr != cols || tc != rows)
              : (tr != rows || tc != cols)) {
        slog(WARN, trans ? "mat_vec_mul trans: dim mismatch cfg=[%lu,%lu] tensor^T=[%lu,%lu]"
                         : "mat_vec_mul: dim mismatch cfg=[%lu,%lu] tensor=[%lu,%lu]",
             (unsigned long)rows, (unsigned long)cols,
             (unsigned long)(trans ? tc : tr), (unsigned long)(trans ? tr : tc));
        return false;
    }

#ifdef GPU_BUILD
    /* GPU matvec: persistent weights + one kernel per call.  Falls
     * through to the CPU path whenever the GPU path is unavailable,
     * below the size threshold, or fails (y is untouched on failure). */
    if (rows * cols >= GPU_MAT_MIN_OPS && gpu_available()) {
        if (gpu_matvec(tw, base, x, y, rows, cols, trans) == 0)
            return true;
    }
#endif

    if (trans) {
        /* Transposed: W stored as [cols × rows], compute y = W^T @ x.
         * Strided access — need a gather buffer. */
        #define MATVEC_STACK 4096
        float  stack_buf[MATVEC_STACK];
        float *w_row = cols <= MATVEC_STACK ? stack_buf : smalloc(cols * sizeof(float));
        if (!w_row) return false;

        for (u64 r = 0; r < rows; r++) {
            for (u64 c = 0; c < cols; c++)
                w_row[c] = tensor_get_f32(tw, base, c * tc + r);
            float sum = 0.0f;
            for (u64 c = 0; c < cols; c++)
                sum += w_row[c] * x[c];
            y[r] = sum;
        }
        if (w_row != stack_buf) sfree(w_row);
        #undef MATVEC_STACK
    } else {
        /* Standard: W stored as [rows × cols], compute y = W @ x.
         * Contiguous rows — fused dequant + dot. */

        /* For Q8_0 / Q8_K: if dotprod available, quantise x to i8
         * once, then use vdotq_s32 (1 insn / 4 MACs). */
#if defined(__ARM_FEATURE_DOTPROD)
        if (tw->type == GGUF_TYPE_Q8_0 || tw->type == GGUF_TYPE_Q8_K) {
            i8 *x_i8 = smalloc(cols * sizeof(i8));
            if (!x_i8) return false;
            float x_scale = quantize_f32_to_i8(x, x_i8, cols);
            if (pool && pool->nthreads > 1 && rows > 1) {
                MatVecCtx ctx = { y, tw, base, x, cols, x_i8, x_scale };
                pthreads_parallel_for(pool, 0, (int)rows, matvec_i8_worker, &ctx);
            } else {
                for (u64 r = 0; r < rows; r++)
                    y[r] = gguf_dot_i8_batch(tw, base, r * cols, cols,
                                             x_i8, x_scale);
            }
            sfree(x_i8);
        } else
#endif
        {
            if (pool && pool->nthreads > 1 && rows > 1) {
                MatVecCtx ctx = { y, tw, base, x, cols, NULL, 0.0f };
                pthreads_parallel_for(pool, 0, (int)rows, matvec_worker, &ctx);
            } else {
                for (u64 r = 0; r < rows; r++)
                    y[r] = gguf_dot_batch(tw, base, r * cols, cols, x);
            }
        }
    }
    return true;
}

/* ---- Parallel mat-mat row workers -----------------------------------
 * Each invocation computes one output row: dequantise the weight row
 * into a per-thread buffer, then compute a dot product against every
 * batch element's input row.  The w_row_buf array has nthreads slices
 * of cols floats; thread tid writes to w_row_buf[tid * cols]. */
typedef struct {
    float       *Y;         /* output matrix [batch × rows]           */
    TensorInfo  *tw;        /* weight tensor                          */
    const u8    *base;      /* mmap base                              */
    const float *X;         /* input matrix [batch × cols]            */
    u64          batch;
    u64          rows;
    u64          cols;      /* dot-product length                     */
    u64          tc;        /* tensor column-stride (transposed only) */
    bool         trans;
    float       *w_row_buf; /* nthreads * cols float buffer           */
} MatMatCtx;

static void matmat_worker(void *arg, int tid, int i) {
    MatMatCtx *c = (MatMatCtx *)arg;
    float *w_row = c->w_row_buf + (u64)tid * c->cols;

    if (c->trans) {
        for (u64 ci = 0; ci < c->cols; ci++)
            w_row[ci] = tensor_get_f32(c->tw, c->base, ci * c->tc + (u64)i);
    } else {
        tensor_get_f32_batch(c->tw, c->base, (u64)i * c->cols, c->cols, w_row);
    }

    for (u64 b = 0; b < c->batch; b++) {
        const float *x_row = c->X + b * c->cols;
        float sum = 0.0f;
        for (u64 ci = 0; ci < c->cols; ci++)
            sum += w_row[ci] * x_row[ci];
        c->Y[b * c->rows + (u64)i] = sum;
    }
}

/* Batch matrix-matrix multiply:  Y[b] = W @ X[b]  for b ∈ [0, batch).
 *
 * Strategy: for each output row r, dequantise the entire weight
 * row into a contiguous buffer, then compute a dot product against
 * every batch element's input row.  The inner dot-product loop
 * runs over two contiguous arrays (trivially auto-vectorisable by
 * the compiler), and each weight is dequantised only once.
 *
 * When pool is non-NULL and has more than one thread, the outer row
 * loop is dispatched across the worker pool (one row per work item);
 * each thread gets its own w_row slice to avoid data races. */
bool mat_mat_mul(float *Y, TensorInfo *tw, const u8 *base, const float *X,
                 u64 batch, u64 rows, u64 cols, bool trans, pthreads_t *pool) {
    if (!tw || tw->ndim < 2) {
        slog(WARN, "mat_mat_mul: tensor missing or ndim < 2");
        return false;
    }
    u64 tc = tw->dim[tw->ndim - 1]; /* fastest-varying = column count */
    u64 tr = tw->dim[0];            /* slowest-varying = row count     */

    if (trans) {
        if (tr != cols || tc != rows) {
            slog(WARN, "mat_mat_mul trans: dim mismatch cfg=[%lu,%lu] tensor^T=[%lu,%lu]",
                 (unsigned long)rows, (unsigned long)cols,
                 (unsigned long)tc, (unsigned long)tr);
            return false;
        }
    } else {
        if (tr != rows || tc != cols) {
            slog(WARN, "mat_mat_mul: dim mismatch cfg=[%lu,%lu] tensor=[%lu,%lu]",
                 (unsigned long)rows, (unsigned long)cols,
                 (unsigned long)tr, (unsigned long)tc);
            return false;
        }
    }

#ifdef GPU_BUILD
    /* GPU matmat: same fall-through contract as mat_vec_mul. */
    if (batch * rows * cols >= GPU_MAT_MIN_OPS && gpu_available()) {
        if (gpu_matmat(tw, base, X, Y, batch, rows, cols, trans) == 0)
            return true;
    }
#endif

    /* Parallel path: dispatch rows across the thread pool. */
    if (pool && pool->nthreads > 1 && rows > 1) {
        int nt = pool->nthreads;
        float *w_row_buf = smalloc((u64)nt * cols * sizeof(float));
        if (!w_row_buf) {
            slog(WARN, "mat_mat_mul: smalloc failed for parallel row buffers");
            return false;
        }
        MatMatCtx ctx = { Y, tw, base, X, batch, rows, cols, tc, trans, w_row_buf };
        pthreads_parallel_for(pool, 0, (int)rows, matmat_worker, &ctx);
        sfree(w_row_buf);
        return true;
    }

    /* Serial path: single reusable row buffer. */
    float *w_row = smalloc(cols * sizeof(float));
    if (!w_row) {
        slog(WARN, "mat_mat_mul: smalloc failed for row buffer (%lu cols)",
             (unsigned long)cols);
        return false;
    }

    if (trans) {
        for (u64 r = 0; r < rows; r++) {
            for (u64 c = 0; c < cols; c++)
                w_row[c] = tensor_get_f32(tw, base, c * tc + r);

            for (u64 b = 0; b < batch; b++) {
                const float *x_row = X + b * cols;
                float sum = 0.0f;
                for (u64 c = 0; c < cols; c++)
                    sum += w_row[c] * x_row[c];
                Y[b * rows + r] = sum;
            }
        }
    } else {
        for (u64 r = 0; r < rows; r++) {
            tensor_get_f32_batch(tw, base, r * cols, cols, w_row);

            for (u64 b = 0; b < batch; b++) {
                const float *x_row = X + b * cols;
                float sum = 0.0f;
                for (u64 c = 0; c < cols; c++)
                    sum += w_row[c] * x_row[c];
                Y[b * rows + r] = sum;
            }
        }
    }

    sfree(w_row);
    return true;
}

/* RoPE: apply rotary position embedding in-place.
 * buf is [n_heads × head_dim], each head rotated independently. */
void rope(float *buf, u32 n_heads, u32 head_dim, u32 pos, float theta_base) {
    for (u32 h = 0; h < n_heads; h++) {
        float *bh = buf + h * head_dim;
        for (u32 d = 0; d + 1 < head_dim; d += 2) {
            float theta = 1.0f / powf(theta_base, (float)d / (float)head_dim);
            float c     = cosf((float)pos * theta);
            float s     = sinf((float)pos * theta);
            float a     = bh[d], b = bh[d + 1];
            bh[d]       = a * c - b * s;
            bh[d + 1]   = a * s + b * c;
        }
    }
}

/* RoPE (GPT-NeoX / Llama style): rotate-half pairing.
 * buf is [n_heads × head_dim], each head rotated independently.
 * Pairs element i with i + head_dim/2 (across halves) instead of the
 * interleaved (2i, 2i+1) pairing used by rope(). Required for Qwen2/Llama. */
void rope_neox(float *buf, u32 n_heads, u32 head_dim, u32 pos, float theta_base) {
    u32 half = head_dim / 2;
    for (u32 h = 0; h < n_heads; h++) {
        float *bh = buf + (u64)h * head_dim;
        for (u32 i = 0; i < half; i++) {
            float theta  = 1.0f / powf(theta_base, (float)(2 * i) / (float)head_dim);
            float c      = cosf((float)pos * theta);
            float s      = sinf((float)pos * theta);
            float a      = bh[i], b = bh[i + half];
            bh[i]        = a * c - b * s;
            bh[i + half] = a * s + b * c;
        }
    }
}

/* Partial RoPE: rotates only the first rope_dim dimensions of each head.
 * head_dim is the full head dimension (stride between heads),
 * rope_dim is how many dims to rotate (64 of 256 for Qwen3.5).
 * The remaining (head_dim - rope_dim) dimensions are left unchanged. */
void rope_partial(float *buf, u32 n_heads, u32 head_dim, u32 rope_dim, u32 pos, float theta_base) {
    u32 half    = head_dim / 2;       /* pair stride (full head)     */
    u32 n_pairs = rope_dim / 2;       /* how many pairs to rotate    */
    for (u32 h = 0; h < n_heads; h++) {
        float *bh = buf + (u64)h * head_dim;
        for (u32 i = 0; i < n_pairs; i++) {
            float theta  = 1.0f / powf(theta_base, (float)(2 * i) / (float)rope_dim);
            float c      = cosf((float)pos * theta);
            float s      = sinf((float)pos * theta);
            float a      = bh[i], b = bh[i + half];
            bh[i]        = a * c - b * s;
            bh[i + half] = a * s + b * c;
        }
    }
}

/* SiLU activation in-place. */
void silu(float *x, int n) {
    for (int i = 0; i < n; i++)
        x[i] = x[i] / (1.0f + expf(-x[i]));
}

/* Softmax in-place on a slice of length n. */
void softmax(float *x, u32 n) {
    float mx = x[0];
    for (u32 i = 1; i < n; i++)
        if (x[i] > mx) mx = x[i];
    float sum = 0.0f;
    for (u32 i = 0; i < n; i++) {
        x[i] = expf(x[i] - mx);
        sum += x[i];
    }
    for (u32 i = 0; i < n; i++)
        x[i] /= sum;
}

/* Softplus: y = log(1 + exp(x)), numerically stable.
 * For x > 20, log(1 + e^x) ≈ x, so return x directly to avoid
 * expf overflow (which would otherwise produce +inf). */
float softplus(float x) {
    return (x > 20.0f) ? x : log1pf(expf(x));
}

/* Sigmoid: y = 1 / (1 + exp(-x)).  Naturally stable: for x << -88,
 * expf(-x) overflows to +inf and the result saturates to 0. */
float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}


/* Byte cursor over the mmap'd model file: sequential, bounds-checked
 * reader.  base = mapped bytes, size = file size, post = current
 * read offset, error = first failure message (empty until an error). */
typedef struct {
    u8 *base;
    u64 size;
    u64 post;
    char error[125];
} Cursor;

/* fd of the held instance lock file (-1 when not locked). */
static int global_lock_fd = -1;

/* Create a cursor positioned at byte offset post. */
static Cursor cursor_at(u8 *base, u64 size, u64 post) {
    Cursor c = {
        .base = base,
        .size = size,
        .post = post,
        .error = {0}
    };
    return c;
}

/* Record the first error message (with the byte offset at failure).
 * Later calls are ignored so the earliest error is preserved. */
static void cursor_error(Cursor *c, const char *msg) {
    if (c->error[0] == '\0') {
        snprintf(c->error,  sizeof(c->error), "%s at byte %" PRIu64, msg, c->post);
    }
}

/* Bounds check: can n bytes be read from c->post without passing
 * the end of the mapped file? */
static bool cursor_has(Cursor *c, u64 n) {
    if (n > c->size || c->post + n > c->size) {
        cursor_error(c, "Truncated GGUF file");
        return false;
    }
    return true;
}

/* Advance the cursor by n bytes (bounds-checked). */
static bool cursor_skip(Cursor *c, u64 n) {
    if (!cursor_has(c, n)) return false;
    c->post += n;
    return true;
}

/* Read n bytes into dest and advance the cursor (bounds-checked). */
static bool cursor_read(Cursor *c, void *dest, u64 n) {
    if (!cursor_has(c, n)) return false;
    memcpy(dest, c->base + c->post, (size_t)n);
    c->post += n;
    return true;
}

/* Typed single-value reads (host byte order matches the little-endian
 * GGUF on-disk layout). */
static bool cursor_i32(Cursor *c, i32 *v) {
    return cursor_read(c, v, sizeof(*v));
}

static bool cursor_u32(Cursor *c, u32 *v) {
    return cursor_read(c, v, sizeof(*v));
}

static bool cursor_float(Cursor *c, float *v) {
    return cursor_read(c, v, sizeof(*v));
}

static bool cursor_u64(Cursor *c, u64 *v) {
    return cursor_read(c, v, sizeof(*v));
}

/* Read a length-prefixed GGUF string into a Key.  content points
 * directly into the mmap'd bytes (no copy), valid while mapped. */
static bool cursor_key(Cursor *c, Key *k) {
    u64 len;
    if (!cursor_u64(c, &len)) return false;
    if (!cursor_has(c, len)) return false;
    k->content = (char *)(c->base + c->post);
    k->len = len;
    c->post += len;
    return true;
}

/* Skip a metadata value of the given GGUF type without reading it,
 * recursing into arrays.  depth guards against pathological nesting. */
static bool cursor_skip_value(Cursor *c, GGUFValueType type, int depth) {
    if (depth > GGUF_MAX_DIMS) {
        cursor_error(c, "Metadata array nesting is too deep");
        return false;
    }

    switch (type) {
        case GGUF_VALUE_UINT8: case GGUF_VALUE_INT8: case GGUF_VALUE_BOOL:
            return cursor_skip(c, 1);
        case GGUF_VALUE_UINT16: case GGUF_VALUE_INT16:
            return cursor_skip(c, 2);
        case GGUF_VALUE_UINT32: case GGUF_VALUE_INT32: case GGUF_VALUE_FLOAT32:
            return cursor_skip(c, 4);
        case GGUF_VALUE_UINT64: case GGUF_VALUE_INT64: case GGUF_VALUE_FLOAT64:
            return cursor_skip(c, 8);
        case GGUF_VALUE_STRING: {
            Key ignore;
            return cursor_key(c, &ignore);
        }
        case GGUF_VALUE_ARRAY: {
            u32 item_type;
            u64 len;
            
            if (!cursor_u32(c, &item_type)) return false;
            if (!cursor_u64(c, &len)) return false;
            for (u64 i = 0; i < len; i++) {
                if (!cursor_skip_value(c, item_type, depth + 1)) return false;
            }

            return true;
        }
        default:
            cursor_error(c, "Unknown GGUF metadata type");
            return false;
    }
}

/* Open-addressing hash table (power-of-two capacity) mapping token
 * pieces / merge strings to their IDs / ranks. */
static void tokenizer_table_init(TokenizerTable *table, u64 expected) {
    table->cap = next_pow2(expected * 2 + 16);
    table->used = 0;
    table->entry = scalloc(table->cap, sizeof(table->entry[0]));
}

/* Free a tokenizer hash table. */
static void tokenizer_table_free(TokenizerTable *table) {
    if (table) {
        sfree(table->entry);
        memset(table, 0, sizeof(*table));
    }
}

/* Insert or update key -> value, probing linearly on hash conflict. */
static void tokenizer_table_put(TokenizerTable *table, Key key, i32 value) {
    u64 mask = table->cap - 1;
    u64 hash = hash_bytes(key.content, key.len);
    u64 i = hash & mask;
    
    while (table->entry[i].used) {
        if (key_eq(table->entry[i].key, key)) {
            table->entry[i].value = value;
            return;
        }
        i = (i + 1) & mask;
    }
    
    table->entry[i].used = true;
    table->entry[i].key = key;
    table->entry[i].value = value;
    table->used++;
}

/* Look up key; on a hit set *value and return true. */
static bool tokenizer_table_get(TokenizerTable *table, Key key, i32 *value) {
    if (table->cap == 0) return false;

    u64 mask = table->cap - 1;
    u64 hash = hash_bytes(key.content, key.len);
    u64 i = hash & mask;

    while (table->entry[i].used) {
        if (key_eq(table->entry[i].key, key)) {
            *value = table->entry[i].value;
            return true;
        }
        i = (i + 1) & mask;
    }
    return false;
}


/* Human-readable name of a GGUF metadata value type (model summary). */
static const char *gguf_value_type_name(u32 type) {
    switch (type) {
        case GGUF_VALUE_UINT8:   return "u8";
        case GGUF_VALUE_INT8:    return "i8";
        case GGUF_VALUE_UINT16:  return "u16";
        case GGUF_VALUE_INT16:   return "i16";
        case GGUF_VALUE_UINT32:  return "u32";
        case GGUF_VALUE_INT32:   return "i32";
        case GGUF_VALUE_FLOAT32: return "f32";
        case GGUF_VALUE_BOOL:    return "bool";
        case GGUF_VALUE_STRING:  return "string";
        case GGUF_VALUE_UINT64:  return "u64";
        case GGUF_VALUE_INT64:   return "i64";
        case GGUF_VALUE_FLOAT64: return "f64";
        default:                 return "unknown";
    }
}

/* Fallback metadata key prefix for an architecture.  When the model
 * stores its own arch name (e.g. "qwen35") that is preferred. */
static const char *arch_key_prefix(ModelArch arch) {
    switch (arch) {
        case ARCH_LLAMA:    return "llama";
        case ARCH_QWEN2:    return "qwen2";
        case ARCH_DEEPSEEK: return "deepseek2";
        case ARCH_FALCON:   return "falcon";
        default:            return "llama";
    }
}

/* GGUF type-info lookup by type ID; NULL for unknown types. */
static const GGUFTypeInfo *tensor_type(u32 type) {
    u32 n = sizeof(gguf_types) / sizeof(gguf_types[0]);
    if (type >= n || gguf_types[type].name == NULL) return NULL;
    else return &gguf_types[type];
}

/* Storage size in bytes for n_element elements of `type`, rounded up
 * to whole blocks (e.g. 256 elements per K-quant block). */
static bool tensor_bytes(u32 type, u64 n_element, u64 *bytes) {
    const GGUFTypeInfo *info = tensor_type(type);
    if (info == NULL || info->block_elems == 0) return false;
    u64 blocks = (n_element + info->block_elems - 1) / info->block_elems;
    if (blocks > UINT64_MAX / info->block_bytes) return false;
    *bytes = blocks * info->block_bytes;
    return true;
}

/* Release the instance lock file (registered via atexit). */
static void release_instance_lock(void) {
    if (global_lock_fd >= 0) {
        close(global_lock_fd);
        global_lock_fd = -1;
    }
}

/* Single-instance guard: take an exclusive flock on a lock file
 * (path from TINY_LLAMA_LOCK, default /tmp/tiny_llama.lock) and
 * write this process's PID into it.  Refuses to start if another
 * tiny_llama process already holds the lock. */
static void acquire_instance_lock(void) {
    const char *path = getenv("TINY_LLAMA_LOCK");
    if (!path || !path[0]) path = "/tmp/tiny_llama.lock";

    const int fd = open(path, O_RDWR | O_CREAT, 0600);
    if (fd < 0) slog_errno("Fail to open lock file");
    (void)fcntl(fd, F_SETFD, FD_CLOEXEC);
    
    if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
        if (errno == EWOULDBLOCK) {
            char buf[64];
            ssize_t n = pread(fd, buf, sizeof(buf) - 1, 0);
            long owner = -1;
            if (n > 0) {
                buf[n] = '\0';
                owner = parse_long(buf);
            }
            close(fd);
            if (owner > 0) {
                slog(ERROR, "Found another tiny_llama process is already running (pid %ld); refusing to start.", owner);
            } else {
                slog(ERROR, "Found another tiny_llama process is already running; refusing to start.");
            }
        }

        close(fd);
        slog_errno("Failed to lock %s", path);
    }

    if (ftruncate(fd, 0) != 0) {
        close(fd);
        slog_errno("Failed to truncated lock file:", path);
    }

    (void)dprintf(fd, "%ld", (long)getpid());
    global_lock_fd = fd;
    atexit(release_instance_lock);
}

/* Find a metadata KV by key name, or NULL. */
static KV *model_find_kv(Model *m, char *s) {
    for (u64 i = 0; i < m->n_kv; i++) {
        if (key_streq(m->kv[i].key, s)) return &m->kv[i];
    }
    return NULL;
}

/* Read a string-typed metadata value into a Key (mmap-backed). */
static bool model_get_key(Model *m, char *s, Key *out) {
    KV *kv = model_find_kv(m, s);
    if (!kv || kv->type != GGUF_VALUE_STRING) return false;
    Cursor c = cursor_at(m->map, m->size, kv->value_pos);
    return cursor_key(&c, out);
}

/* Read an array-typed metadata value header (item type + count);
 * out->data_pos points at the first element. */
static bool model_get_array(Model *m, char *s, ArrayRef *out) {
    KV *kv = model_find_kv(m, s);
    if (!kv || kv->type != GGUF_VALUE_ARRAY) return false;
    Cursor c = cursor_at(m->map, m->size, kv->value_pos);
    if (!cursor_u32(&c, &out->type)) return false;
    if (!cursor_u64(&c, &out->len)) return false;
    out->data_pos = c.post;
    return true;
}

/* Read a u32-typed metadata value (GGUF stores ints as u32). */
bool model_get_i32(Model *m, const char *key, i32 *out) {
    KV *kv = model_find_kv(m, (char *)key);
    if (!kv || kv->type != GGUF_VALUE_UINT32) return false;
    Cursor c = cursor_at(m->map, m->size, kv->value_pos);
    return cursor_i32(&c, out);
}

/* Read an f32-typed metadata value. */
bool model_get_f32(Model *m, const char *key, float *out) {
    KV *kv = model_find_kv(m, (char *)key);
    if (!kv || kv->type != GGUF_VALUE_FLOAT32) return false;
    Cursor c = cursor_at(m->map, m->size, kv->value_pos);
    return cursor_float(&c, out);
}

/* Detect the architecture from "general.architecture" and remember
 * the arch name string (e.g. "qwen35") for metadata key lookups. */
static ModelArch model_detect_arch(Model *m) {
    Key arch_name;
    if (!model_get_key(m, "general.architecture", &arch_name)) return ARCH_UNKNOWN;
    /* Store the architecture name for metadata key lookups. */
    u32 n = arch_name.len < sizeof(m->arch_name) - 1 ? arch_name.len
                                                      : (u32)sizeof(m->arch_name) - 1;
    memcpy(m->arch_name, arch_name.content, n);
    m->arch_name[n] = '\0';
    if (key_streq(arch_name, "llama"))   return ARCH_LLAMA;
    if (key_streq(arch_name, "qwen2"))   return ARCH_QWEN2;
    if (key_streq(arch_name, "qwen35"))  return ARCH_QWEN3;
    if (key_streq(arch_name, "deepseek2")) return ARCH_DEEPSEEK;
    if (key_streq(arch_name, "falcon"))  return ARCH_FALCON;
    slog(WARN, "Unknown architecture: %s, loading as generic.", get_key_name(arch_name));
    return ARCH_UNKNOWN;
}

/* Find a tensor by name, or NULL. */
static TensorInfo *model_find_tensor(Model *m, char *name) {
    for (u64 i = 0; i < m->n_tensor; i++) {
        if (key_streq(m->tensor[i].key, name))
            return &m->tensor[i];
    }
    return NULL;
}

/* Count layers: scan tensor names of the form "blk.{N}...." and
 * return the maximum N + 1. */
static u32 model_count_layers(Model *m) {
    u32 max_n = 0;
    for (u64 i = 0; i < m->n_tensor; i++) {
        Key *k = &m->tensor[i].key;
        if (k->len < 5 || memcmp(k->content, "blk.", 4) != 0) continue;
        u32 n = 0;
        u32 j;
        for (j = 4; j < k->len && k->content[j] >= '0' && k->content[j] <= '9'; j++)
            n = n * 10 + (u32)(k->content[j] - '0');
        if (j > 4 && j < k->len && k->content[j] == '.' && n >= max_n)
            max_n = n;
    }
    return max_n + 1;
}

/* Parse the metadata (KV) section into an array of KV entries; also
 * records the file alignment from "general.alignment" if present. */
KV *kv_load(Model *m, Cursor *c) {
    KV *kv = scalloc(m->n_kv, sizeof(m->kv[0]));
    m->alignment = GGUF_DEFAULT_ALIGNMENT;

    for (u64 i = 0; i < m->n_kv; i++) {
        KV *v = &kv[i];

        if (!cursor_key(c, &v->key)) slog(ERROR, c->error);
        if (!cursor_u32(c, &v->type)) slog(ERROR, c->error);
        v->value_pos = c->post;

        if (key_streq(v->key, "general.alignment") &&
            v->type == GGUF_VALUE_UINT32
        ) {
            u32 alignment;
            Cursor tmp = cursor_at(m->map, m->size, kv->value_pos);
            if (cursor_u32(&tmp, &alignment) && alignment != 0) m->alignment = alignment;
        }
        
        if (!cursor_skip_value(c, v->type, 0)) slog(ERROR, c->error);
    }

    return kv;
}

/* Parse the tensor-info section: name, dims, type and offset, plus
 * derived fields (element count, byte size, padded last-dim stride). */
TensorInfo *tensor_load(Model *m, Cursor *c) {
    TensorInfo *tensor = scalloc(m->n_tensor, sizeof(m->tensor[0]));
    
    for (u64 i = 0; i < m->n_tensor; i++) {
        TensorInfo *t = &tensor[i];

        if (!cursor_key(c, &t->key)) slog(ERROR, c->error);
        if (!cursor_u32(c, &t->ndim)) slog(ERROR, c->error);
        if (t->ndim == 0 || t->ndim > GGUF_MAX_DIMS) slog(ERROR, "Tensor has an unsupported number of dimensions: %d", t->ndim);

        t->n_element = 1;
        for (u32 d = 0; d < t->ndim; d++) {
            if (!cursor_u64(c, &t->dim[d])) slog(ERROR, c->error);
            if (t->dim[d] != 0 && t->n_element > UINT64_MAX / t->dim[d]) slog(ERROR, "Tensor element count overflow");
            t->n_element *= t->dim[d];
        }

        /* GGUF stores dimensions with dim[0] as the fastest-varying
          * (innermost) dimension.  Swapping dim[0]↔ dim[1] makes dim[0]
          * the row count and dim[1] the column count, matching the
          * row-major convention used by the inference code. */
        if (t->ndim == 2) {
            u64 tmp = t->dim[0];
            t->dim[0] = t->dim[1];
            t->dim[1] = tmp;
        }

        if (!cursor_u32(c, &t->type)) slog(ERROR, c->error);
        if (!cursor_u64(c, &t->offset)) slog(ERROR, c->error);
        if (!tensor_bytes(t->type, t->n_element, &t->bytes)) slog(WARN, "Tensor %s has unsupported GGUF type %u", get_key_name(t->key), t->type);

        /* Compute padded last-dimension stride for quantised multi-dim tensors.
         * K-quant types (Q2_K..Q6_K, etc.) have block_size=256; when the
         * innermost dimension is not a multiple of 256, GGUF pads it. */
        t->padded_dim = 0;
        if (t->ndim >= 2) {
            const GGUFTypeInfo *ti = tensor_type(t->type);
            if (ti && ti->block_elems > 1) {
                u64 last = t->dim[t->ndim - 1];
                u64 pad  = (last + ti->block_elems - 1) / ti->block_elems * ti->block_elems;
                if (pad != last) t->padded_dim = pad;
            }
        }
    }

    return tensor;
}

/* Determine the tokenizer flavour from "tokenizer.ggml.model". */
static TokenizerType tokenizer_type(Model *m) {
    Key model_name;

    if (!model_get_key(m, "tokenizer.ggml.model", &model_name)) slog(ERROR, "GGUF tokenizer model type is missing");
    if (key_streq(model_name, "llama") || key_streq(model_name, "sentencepiece")) return TOKENIZER_TYPE_SPM;
    else if (key_streq(model_name, "gpt2")) return TOKENIZER_TYPE_BPE;
    else if (key_streq(model_name, "bert")) return TOKENIZER_TYPE_WPM;
    else if (key_streq(model_name, "unigram")) return TOKENIZER_TYPE_UGM;
    else if (key_streq(model_name, "rwkv")) return TOKENIZER_TYPE_RWKV;
    else if (key_streq(model_name, "whisper")) return TOKENIZER_TYPE_WHISPER;

    slog(ERROR, "Unknown tokenizer model type");
    return TOKENIZER_TYPE_NONE;
}

/* Hash-table lookup of a C string in the token table. */
static bool vocab_try_lookup(Vocab *v, const char *text, i32 *id) {
    Key key = {.content = (char *)text, .len = strlen(text)};
    return tokenizer_table_get(&v->tokens, key, id);
}

/* Look up a token by C string; VOCAB_ID_NONE when absent. */
i32 vocab_lookup(Vocab *v, const char *text) {
    i32 id;
    if (vocab_try_lookup(v, text, &id)) return id;
    return (i32)VOCAB_ID_NONE;
}

/* Look up a token by explicit length — needed for byte tokens
 * (e.g. NUL byte) that strlen() cannot handle. */
bool vocab_lookup_len(Vocab *v, const char *text, int len, i32 *id) {
    Key key = {.content = (char *)text, .len = (u8)len};
    return tokenizer_table_get(&v->tokens, key, id);
}

/* Return merge rank for a pair of token IDs, or -1 if no merge exists. */
i32 vocab_merge_rank(Vocab *v, i32 id1, i32 id2) {
    if (id1 < 0 || id2 < 0 || (u32)id1 >= v->n_vocab || (u32)id2 >= v->n_vocab)
        return -1;
    Key *t1 = &v->token[id1];
    Key *t2 = &v->token[id2];
    int key_len = t1->len + 1 + t2->len;
    if (key_len > 255) return -1;

    char merge_key[256];
    memcpy(merge_key, t1->content, t1->len);
    merge_key[t1->len] = ' ';
    memcpy(merge_key + t1->len + 1, t2->content, t2->len);

    Key key = {.content = merge_key, .len = (u8)key_len};
    i32 rank;
    if (tokenizer_table_get(&v->merges, key, &rank)) return rank;
    return -1;
}

/* Return the merged token ID for a pair (id1+id2), or VOCAB_ID_NONE. */
i32 vocab_merge_result(Vocab *v, i32 id1, i32 id2) {
    if (id1 < 0 || id2 < 0 || (u32)id1 >= v->n_vocab || (u32)id2 >= v->n_vocab)
        return (i32)VOCAB_ID_NONE;
    Key *t1 = &v->token[id1];
    Key *t2 = &v->token[id2];
    int merged_len = t1->len + t2->len;
    if (merged_len > 255) return (i32)VOCAB_ID_NONE;

    char merged[256];
    memcpy(merged, t1->content, t1->len);
    memcpy(merged + t1->len, t2->content, t2->len);

    i32 id;
    if (vocab_lookup_len(v, merged, merged_len, &id)) return id;
    return (i32)VOCAB_ID_NONE;
}

/* Load a GPT-2 style BPE vocab: token & merge tables, special token
 * IDs, and the byte->token mapping (bytes_to_unicode). */
static Vocab *vocab_load_for_bpe(Model *m) {
    Vocab *v;
    ArrayRef tokens, merges;
    
    v = smalloc(sizeof(Vocab));
    v->tokenizer_type = TOKENIZER_TYPE_BPE;
    if (!model_get_array(m, "tokenizer.ggml.tokens", &tokens) ||
        tokens.type != GGUF_VALUE_STRING ||
        tokens.len > INT32_MAX
    ) slog(ERROR, "GGUF tokenizer token table is missing or invalid");
    if (!model_get_array(m, "tokenizer.ggml.merges", &merges) ||
        merges.type != GGUF_VALUE_STRING 
    ) slog(ERROR, "GGUF tokenizer merge table is missing or invalid");

    v->n_vocab = (int)tokens.len;
    v->token = scalloc((size_t)v->n_vocab, sizeof(v->token[0]));

    tokenizer_table_init(&v->tokens, tokens.len);
    Cursor c = cursor_at(m->map, m->size, tokens.data_pos);
    for (u32 i = 0; i < v->n_vocab; i++) {
        if (!cursor_key(&c, &v->token[i])) slog(ERROR, c.error);
        tokenizer_table_put(&v->tokens, v->token[i], i);
    }

    tokenizer_table_init(&v->merges, merges.len);
    c = cursor_at(m->map, m->size, merges.data_pos);
    for (u32 i = 0; i < merges.len; i++) {
        Key merge;
        if (!cursor_key(&c, &merge)) slog(ERROR, c.error);
        tokenizer_table_put(&v->merges, merge, i);
    }

    v->bos_id = VOCAB_ID_NONE;
    v->eos_id = VOCAB_ID_NONE;

    if (!model_get_i32(m, "tokenizer.ggml.bos_token_id", &v->bos_id)) {
        static const char *bos_cands[] = {
            "<|begin_of_text|>",
            "<｜begin▁of▁sentence｜>",
            "<|im_start|>",
            "<bos>",
            "<s>",
        };
        for (int i = 0; i < (int)(sizeof(bos_cands) / sizeof(bos_cands[0])); i++) {
            if (vocab_try_lookup(v, bos_cands[i], &v->bos_id)) break;
        }
    }

    if (!model_get_i32(m, "tokenizer.ggml.eos_token_id", &v->eos_id)) {
        static const char *eos_cands[] = {
            "<|end_of_text|>",
            "<｜end▁of▁sentence｜>",
            "<|im_end|>",
            "<eos>",
            "</s>",
        };
        for (int i = 0; i < (int)(sizeof(eos_cands) / sizeof(eos_cands[0])); i++) {
            if (vocab_try_lookup(v, eos_cands[i], &v->eos_id)) break;
        }
    }

    if (v->bos_id == VOCAB_ID_NONE)
        slog(ERROR, "Cannot find BOS token in vocabulary");
    if (v->eos_id == VOCAB_ID_NONE)
        slog(ERROR, "Cannot find EOS token in vocabulary");

    v->user_id        = VOCAB_ID_NONE;
    v->assistant_id   = VOCAB_ID_NONE;
    v->think_start_id = VOCAB_ID_NONE;
    v->think_end_id   = VOCAB_ID_NONE;
    v->dsml_id        = VOCAB_ID_NONE;
    v->im_start_id    = VOCAB_ID_NONE;
    v->im_end_id      = VOCAB_ID_NONE;

    (void)vocab_try_lookup(v, "<｜User｜>",      &v->user_id);
    (void)vocab_try_lookup(v, "<｜Assistant｜>", &v->assistant_id);
    (void)vocab_try_lookup(v, "<think>",         &v->think_start_id);
    (void)vocab_try_lookup(v, "</think>",        &v->think_end_id);
    (void)vocab_try_lookup(v, "｜DSML｜",        &v->dsml_id);
    (void)vocab_try_lookup(v, "<|im_start|>",    &v->im_start_id);
    (void)vocab_try_lookup(v, "<|im_end|>",      &v->im_end_id);

    /* Build byte-to-token-ID table for GPT-2 bytes_to_unicode mapping.
     * In GPT-2 BPE, control bytes (0-32,127-160,173) are mapped to
     * Unicode codepoints U+0100+ before being stored as token pieces. */
    for (int b = 0; b < 256; b++) {
        v->byte_token_ids[b] = VOCAB_ID_NONE;
        bool is_self = (b >= 33 && b <= 126)
                    || (b >= 161 && b <= 172)
                    || (b >= 174 && b <= 255);
        if (is_self) {
            char byte_char = (char)b;
            i32 id;
            if (vocab_lookup_len(v, &byte_char, 1, &id))
                v->byte_token_ids[b] = id;
        } else {
            /* Count non-self-mapping bytes < b to get the Unicode offset. */
            int off = 0;
            for (int i = 0; i < b; i++) {
                bool si = (i >= 33 && i <= 126)
                       || (i >= 161 && i <= 172)
                       || (i >= 174 && i <= 255);
                if (!si) off++;
            }
            int cp = 256 + off;
            /* Encode Unicode codepoint as UTF-8 (cp < 0x800 → 2 bytes). */
            char utf8[4];
            int ulen;
            if (cp < 0x80)      { utf8[0] = (char)cp; ulen = 1; }
            else if (cp < 0x800){ utf8[0] = (char)(0xC0 | (cp >> 6));
                                  utf8[1] = (char)(0x80 | (cp & 0x3F)); ulen = 2; }
            else                { utf8[0] = (char)(0xE0 | (cp >> 12));
                                  utf8[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
                                  utf8[2] = (char)(0x80 | (cp & 0x3F)); ulen = 3; }
            i32 id;
            if (vocab_lookup_len(v, utf8, ulen, &id))
                v->byte_token_ids[b] = id;
        }
    }

    return v;
}

/* Load vocab. */
static Vocab *vocab_load(Model *m) {
    TokenizerType type = tokenizer_type(m);
    switch (type) {
        case TOKENIZER_TYPE_BPE: return vocab_load_for_bpe(m);
        case TOKENIZER_TYPE_NONE: ERRRET(NULL, "Unknown tokenizer model type");
        default: ERRRET(NULL, "Not support tokenizer type");
    }
}

/* Free vocab. */
static void vocab_free(Vocab *v) {
    if (!v) return;
    sfree(v->token);
    tokenizer_table_free(&v->tokens);
    tokenizer_table_free(&v->merges);
    memset(v, 0, sizeof(*v));
}

/* Non-layer tensor map entry: role, GGUF tensor name, and whether a
 * missing tensor is fatal at load time. */
typedef struct {
    TensorRole  role;
    const char *name;
    bool        required;
} TensorMapEntry;

#define ARR_LEN(a) ((int)(sizeof(a) / sizeof((a)[0])))

/* Architecture tensor maps (non-layer tensors only). */

static const TensorMapEntry llama_tensor_map[] = {
    {TENSOR_TOKEN_EMBD,  "token_embd.weight",   true},
    {TENSOR_OUTPUT,      "output.weight",       false},
    {TENSOR_OUTPUT_NORM, "output_norm.weight",  true},
};

static const TensorMapEntry qwen2_tensor_map[] = {
    {TENSOR_TOKEN_EMBD,  "token_embd.weight",   true},
    {TENSOR_OUTPUT,      "output.weight",       false},
    {TENSOR_OUTPUT_NORM, "output_norm.weight",  true},
};

static const TensorMapEntry deepseek_tensor_map[] = {
    {TENSOR_TOKEN_EMBD,       "token_embd.weight",         true},
    {TENSOR_OUTPUT,           "output.weight",             true},
    {TENSOR_OUTPUT_NORM,      "output_norm.weight",        true},
    {TENSOR_OUTPUT_HC_BASE,   "output_hc_base.weight",     true},
    {TENSOR_OUTPUT_HC_FN,     "output_hc_fn.weight",       true},
    {TENSOR_OUTPUT_HC_SCALE,  "output_hc_scale.weight",    true},
};

static const TensorMapEntry unknown_tensor_map[] = {
    {TENSOR_TOKEN_EMBD,  "token_embd.weight",   true},
    {TENSOR_OUTPUT,      "output.weight",       false},
    {TENSOR_OUTPUT_NORM, "output_norm.weight",  false},
};

/* Layer tensor suffix maps (blk.{N}.{suffix}.weight). */

/* Per-layer tensor map entry: role, suffix matched against
 * "blk.{N}.{suffix}.weight", and whether a missing tensor is fatal. */
typedef struct {
    TensorRole  role;
    const char *suffix;
    bool        required;
} LayerTensorMap;

static const LayerTensorMap llama_layer_map[] = {
    {TENSOR_ATTN_NORM,  "attn_norm",    true},
    {TENSOR_ATTN_Q,     "attn_q",       true},
    {TENSOR_ATTN_K,     "attn_k",       true},
    {TENSOR_ATTN_V,     "attn_v",       true},
    {TENSOR_ATTN_Q_BIAS,"attn_q.bias",   false},
    {TENSOR_ATTN_K_BIAS,"attn_k.bias",   false},
    {TENSOR_ATTN_V_BIAS,"attn_v.bias",   false},
    {TENSOR_ATTN_OUT,   "attn_output",  true},
    {TENSOR_FFN_GATE,   "ffn_gate",     true},
    {TENSOR_FFN_DOWN,   "ffn_down",     true},
    {TENSOR_FFN_UP,     "ffn_up",       true},
};

static const LayerTensorMap qwen2_layer_map[] = {
    {TENSOR_ATTN_NORM,        "attn_norm",             true},
    {TENSOR_ATTN_QKV,         "attn_qkv",              false},
    {TENSOR_ATTN_Q,           "attn_q",                false},
    {TENSOR_ATTN_K,           "attn_k",                false},
    {TENSOR_ATTN_V,           "attn_v",                false},
    {TENSOR_ATTN_Q_BIAS,      "attn_q.bias",           false},
    {TENSOR_ATTN_K_BIAS,      "attn_k.bias",           false},
    {TENSOR_ATTN_V_BIAS,      "attn_v.bias",           false},
    {TENSOR_ATTN_OUT,         "attn_output",           false},
    {TENSOR_SSM_OUT,          "ssm_out",               false},
    {TENSOR_ATTN_GATE,        "attn_gate",             false},
    {TENSOR_ATTN_Q_NORM,      "attn_q_norm",           false},
    {TENSOR_ATTN_K_NORM,      "attn_k_norm",           false},
    {TENSOR_SSM_CONV1D,       "ssm_conv1d",            false},
    {TENSOR_SSM_ALPHA,        "ssm_alpha",             false},
    {TENSOR_SSM_BETA,         "ssm_beta",              false},
    {TENSOR_SSM_A,            "ssm_a",                 false},
    {TENSOR_SSM_DT_BIAS,      "ssm_dt.bias",           false},
    {TENSOR_SSM_NORM,         "ssm_norm",              false},
    {TENSOR_POST_ATTN_NORM,   "post_attention_norm",   false},
    {TENSOR_POST_ATTN_NORM,   "ffn_norm",              false},
    {TENSOR_FFN_GATE,         "ffn_gate",              true},
    {TENSOR_FFN_DOWN,         "ffn_down",              true},
    {TENSOR_FFN_UP,           "ffn_up",                true},
};

static const LayerTensorMap deepseek_layer_map[] = {
    {TENSOR_ATTN_NORM,  "attn_norm",    true},
    {TENSOR_ATTN_Q,     "attn_q",       true},
    {TENSOR_ATTN_K,     "attn_k",       true},
    {TENSOR_ATTN_V,     "attn_v",       true},
    {TENSOR_ATTN_Q_BIAS,"attn_q.bias",  false},
    {TENSOR_ATTN_K_BIAS,"attn_k.bias",  false},
    {TENSOR_ATTN_V_BIAS,"attn_v.bias",  false},
    {TENSOR_ATTN_OUT,   "attn_output",  true},
    {TENSOR_FFN_GATE,   "ffn_gate",     true},
    {TENSOR_FFN_DOWN,   "ffn_down",     true},
    {TENSOR_FFN_UP,     "ffn_up",       true},
};

static const LayerTensorMap unknown_layer_map[] = {
    {TENSOR_ATTN_NORM,  "attn_norm",    false},
    {TENSOR_ATTN_Q,     "attn_q",       false},
    {TENSOR_ATTN_K,     "attn_k",       false},
    {TENSOR_ATTN_V,     "attn_v",       false},
    {TENSOR_ATTN_QKV,   "attn_qkv",     false},
    {TENSOR_ATTN_OUT,   "attn_output",  false},
    {TENSOR_FFN_GATE,   "ffn_gate",     false},
    {TENSOR_FFN_DOWN,   "ffn_down",     false},
    {TENSOR_FFN_UP,     "ffn_up",       false},
};

/* Build the per-layer weight table: match each "blk.{N}.{suffix}.weight"
 * tensor against the architecture's layer map and record its role. */
static LayerWeights *layers_weights_load(Model *m, ModelArch arch, u32 n_layer) {
    LayerWeights *layers = scalloc(n_layer, sizeof(LayerWeights));

    static const struct {
        ModelArch arch;
        const LayerTensorMap *entries;
        int count;
    } arch_layer_maps[] = {
        {ARCH_LLAMA,    llama_layer_map,     ARR_LEN(llama_layer_map)},
        {ARCH_QWEN2,    qwen2_layer_map,     ARR_LEN(qwen2_layer_map)},
        {ARCH_QWEN3,    qwen2_layer_map,     ARR_LEN(qwen2_layer_map)},
        {ARCH_DEEPSEEK, deepseek_layer_map,  ARR_LEN(deepseek_layer_map)},
        {ARCH_UNKNOWN,  unknown_layer_map,   ARR_LEN(unknown_layer_map)},
    };

    const LayerTensorMap *map = NULL;
    int map_count = 0;
    for (int i = 0; i < ARR_LEN(arch_layer_maps); i++) {
        if (arch_layer_maps[i].arch == arch) {
            map = arch_layer_maps[i].entries;
            map_count = arch_layer_maps[i].count;
            break;
        }
    }
    if (!map) {
        map = unknown_layer_map;
        map_count = ARR_LEN(unknown_layer_map);
    }

    for (u64 i = 0; i < m->n_tensor; i++) {
        Key *k = &m->tensor[i].key;
        if (k->len < 7) continue;
        if (memcmp(k->content, "blk.", 4) != 0) continue;

        u32 n = 0;
        u32 j;
        for (j = 4; j < k->len && k->content[j] >= '0' && k->content[j] <= '9'; j++)
            n = n * 10 + (u32)(k->content[j] - '0');
        if (j >= k->len || k->content[j] != '.' || n >= n_layer) continue;

        const char *suffix_start = (const char *)k->content + j + 1;
        u32 suffix_len = k->len - j - 1;

        /* Strip trailing ".weight" if present (most GGUF tensors have it,
         * but some like ssm_a / ssm_dt.bias do not). */
        u32 match_len = suffix_len;
        if (suffix_len > 7 && memcmp(suffix_start + suffix_len - 7, ".weight", 7) == 0)
            match_len = suffix_len - 7;

        for (int e = 0; e < map_count; e++) {
            const char *s = map[e].suffix;
            u32 slen = (u32)strlen(s);
            if (slen == match_len && memcmp(suffix_start, s, slen) == 0) {
                layers[n].tensors[map[e].role] = &m->tensor[i];
                break;
            }
        }
    }

    return layers;
}

/* Load all weights: non-layer tensors (token_embd, output, norms)
 * plus the per-layer tensors.  Falls back to tied embeddings when
 * output.weight is absent. */
static Weights *weights_load(Model *m) {
    Weights *w = smalloc(sizeof(*w));

    static const struct {
        ModelArch arch;
        const TensorMapEntry *entries;
        int count;
    } arch_maps[] = {
        {ARCH_LLAMA,    llama_tensor_map,    ARR_LEN(llama_tensor_map)},
        {ARCH_QWEN2,    qwen2_tensor_map,    ARR_LEN(qwen2_tensor_map)},
        {ARCH_QWEN3,    qwen2_tensor_map,    ARR_LEN(qwen2_tensor_map)},
        {ARCH_DEEPSEEK, deepseek_tensor_map, ARR_LEN(deepseek_tensor_map)},
        {ARCH_UNKNOWN,  unknown_tensor_map,  ARR_LEN(unknown_tensor_map)},
    };

    const TensorMapEntry *map = NULL;
    int map_count = 0;
    for (int i = 0; i < ARR_LEN(arch_maps); i++) {
        if (arch_maps[i].arch == m->arch) {
            map = arch_maps[i].entries;
            map_count = arch_maps[i].count;
            break;
        }
    }
    if (!map) {
        map = unknown_tensor_map;
        map_count = ARR_LEN(unknown_tensor_map);
    }

    for (int i = 0; i < map_count; i++) {
        TensorInfo *t = model_find_tensor(m, (char *)map[i].name);
        if (!t && map[i].required) 
            slog(ERROR, "Required tensor is missing: %s", map[i].name);
        w->tensors[map[i].role] = t;
    }

    if (!w->tensors[TENSOR_OUTPUT] && w->tensors[TENSOR_TOKEN_EMBD]) {
        w->tensors[TENSOR_OUTPUT] = w->tensors[TENSOR_TOKEN_EMBD];
        slog(WARN, "No output.weight found, using tied embeddings (output = token_embd)");
    }

    w->n_layer = model_count_layers(m);
    w->layers  = layers_weights_load(m, m->arch, w->n_layer);
    return w;
}

/* Load a GGUF v3 model: open + mmap the file, then parse the header,
 * KV and tensor-info sections, and detect the architecture. */
Model *model_load(const char *path) {
    Model *m;
    int fd;
    struct stat st;
    void *map;

    m = smalloc(sizeof(Model));
    fd = open(path, O_RDONLY);
    if (fd == -1) slog_errno("Cannot open model, which path: %s", path);
    if (fstat(fd, &st) == -1) slog_errno("Cannot stat model");
    if (st.st_size < 32) slog(ERROR, "Model file is too small to be GGUF.");
    map = mmap(NULL, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) slog_errno("Cannot mmap model");

    m->fd = fd;
    m->size = (u64)st.st_size;
    m->map = map;

    Cursor c = cursor_at(m->map, m->size, 0);
    u32 magic;
    if (!cursor_u32(&c, &magic)) slog(ERROR, c.error);
    if (magic != GGUF_MAGIC) slog(ERROR, "Model is not a GGUF file");
    if (!cursor_u32(&c, &m->version)) slog(ERROR, c.error);
    if (m->version != GGUF_VALID_VERSION) slog(ERROR, "Only GGUF v3 is supported.");
    if (!cursor_u64(&c, &m->n_tensor)) slog(ERROR, c.error);
    if (!cursor_u64(&c, &m->n_kv)) slog(ERROR, c.error);

    m->kv = kv_load(m, &c);
    m->tensor = tensor_load(m, &c);

    /* GGUF stores tensor offsets relative to the start of the tensor
     * data section (not absolute file offsets).  The data section begins
     * at the next alignment boundary after the tensor-info entries. */
    u64 data_base = c.post;
    if (data_base % m->alignment)
        data_base += m->alignment - (data_base % m->alignment);
    for (u64 i = 0; i < m->n_tensor; i++)
        m->tensor[i].offset += data_base;

    m->arch = model_detect_arch(m);

    return m;
}

/* Print a model summary: header, metadata KVs and tensor list. */
static void model_summary(Model *m) {
    /* ---- Header ---- */
    const char *arch_name = m->arch_name[0] ? m->arch_name : "unknown";
    fprintf(stdout, "Header:\n\tversion=%u  tensors=%" PRIu64 "  kvs=%" PRIu64
         "  alignment=%" PRIu64 "  arch=%s",
         m->version, m->n_tensor, m->n_kv, m->alignment, arch_name);

    /* ---- Metadata ---- */
    fprintf(stdout, "\nMetadata:\n");
    for (u64 i = 0; i < m->n_kv; i++) {
        KV *kv = &m->kv[i];
        fprintf(stdout, "\t-|%-45s", get_key_name(kv->key));
        Cursor c = cursor_at(m->map, m->size, kv->value_pos);
        switch (kv->type) {
            case GGUF_VALUE_UINT32: {
                u32 v = 0;
                cursor_u32(&c, &v);
                fprintf(stdout, "\t\t= %d\n", v);
                break;
            }
            case GGUF_VALUE_UINT64: {
                u64 v = 0;
                cursor_u64(&c, &v);
                fprintf(stdout, "\t\t= %"PRIu64 "\n", v);
                break;
            }
            case GGUF_VALUE_FLOAT32: {
                float v = 0;
                cursor_float(&c, &v);
                fprintf(stdout, "\t\t= %.2f\n", v);
                break;
            }
            case GGUF_VALUE_STRING: {
                Key v = {0};
                cursor_key(&c, &v);
                fprintf(stdout, "\t\t= %s\n", get_key_name(v));
                break;
            }
            case GGUF_VALUE_ARRAY: {
                u32 item_type;
                u64 len;
                cursor_u32(&c, &item_type);
                cursor_u64(&c, &len);
                fprintf(stdout, "\t\t= [%s * %"PRIu64"]\n", gguf_value_type_name(item_type), len);
                break;
            }
            default: {
                printf("\t\t--\n");
                break;
            }
        }
    }

    /* ---- Tensor summary ---- */
    u64 total_bytes = 0;
    for (u64 i = 0; i < m->n_tensor; i++) {
        TensorInfo *t = &m->tensor[i];
        total_bytes += t->bytes;
    }
    fprintf(stdout, "Tensors:\n\tcount=%" PRIu64 "  total_bytes=%" PRIu64 " (%.2f GB)\n",
         m->n_tensor, total_bytes, size_convert(total_bytes, GB));
    for (u64 i = 0; i < m->n_tensor; i++) {
        TensorInfo *t = &m->tensor[i];
        fprintf(stdout, "\t-|%-45s", get_key_name(t->key));
        fprintf(stdout, "\t\t= (%s)", gguf_types[t->type].name);
        fprintf(stdout, "[");
        for (u32 j = 0; j < t->ndim; j++) {
            fprintf(stdout, "%ld", t->dim[j]);
            if (j < t->ndim - 1) fprintf(stdout, ",");
        }
        fprintf(stdout, "]\n");
    }
}


/* Release the model (munmap + close); currently an empty shell. */
void model_close(Model *m) {

}

/* Open the engine: take the instance lock, then load the model,
 * vocab (unless inspect-only) and weights. */
Engine *engine_open(EngineOptons *opts) {
    Engine *en = smalloc(sizeof(*en));
    acquire_instance_lock();
    en->model = model_load(opts->model_path);
    if (!opts->inspect) en->vocab = vocab_load(en->model);
    en->weights = weights_load(en->model);
    return en;
}

/* Print the summary of the loaded model. */
void engine_summary(Engine *en) {
    model_summary(en->model);
}

/* Close the engine: shut down the GPU path, free vocab and model. */
void engine_close(Engine *en) {
    if (!en) return;
#ifdef GPU_BUILD
    gpu_shutdown(); /* free cached device weight buffers */
#endif
    vocab_free(en->vocab);
    model_close(en->model);
}

/* Derive the architecture config (embedding size, head counts, RoPE,
 * MLA dims) from model metadata and weight tensor shapes. */
void arch_config_init(Engine *en, ArchConfig *cfg) {
    memset(cfg, 0, sizeof(*cfg));
    Weights *w = en->weights;
    Vocab   *v = en->vocab;

    /* Common defaults derived from weights / vocab. */
    cfg->n_layer = w->n_layer;
    cfg->n_vocab = v ? v->n_vocab : 0;

    /* Derive n_embd from token_embd tensor dims.
     * GGUF files may store the weight as [n_embd,n_vocab] or [n_vocab,n_embd].
     * Use n_vocab to disambiguate: the other dimension is n_embd. */
    TensorInfo *te = w->tensors[TENSOR_TOKEN_EMBD];
    if (te && te->ndim >= 2) {
        if (te->dim[0] == cfg->n_vocab) cfg->n_embd = (u32)te->dim[1];
        else cfg->n_embd = (u32)te->dim[0];
    } else if (te && te->ndim == 1)
        cfg->n_embd = (u32)te->dim[0];
    if (cfg->n_embd == 0 && cfg->n_vocab > 0)
        cfg->n_embd = cfg->n_vocab; /* ultimate fallback */

    /* Use the architecture name stored in the model (e.g. "qwen35")
     * as the metadata key prefix; fall back to the generic prefix
     * derived from the arch enum for older models. */
    const char *pfx = en->model->arch_name[0] ? en->model->arch_name
                                              : arch_key_prefix(en->model->arch);

    /* Helper: try reading an i32 metadata key for this arch.
     * Try the arch-name prefix first, then the generic prefix. */
    i32 v32;
    #define TRY_I32(suffix, field) do {                                 \
        char k[96];                                                     \
        int  n = snprintf(k, sizeof(k), "%s.%s", pfx, suffix);          \
        if (n > 0 && (size_t)n < sizeof(k))                             \
            if (model_get_i32(en->model, k, &v32))                      \
                cfg->field = (u32)v32;                                  \
        /* Also try the generic arch prefix if different from pfx */    \
        if (cfg->field == 0) {                                          \
            const char *gpfx = arch_key_prefix(en->model->arch);        \
            if (strcmp(gpfx, pfx) != 0) {                               \
                n = snprintf(k, sizeof(k), "%s.%s", gpfx, suffix);      \
                if (n > 0 && (size_t)n < sizeof(k))                     \
                    if (model_get_i32(en->model, k, &v32))              \
                        cfg->field = (u32)v32;                          \
            }                                                           \
        }                                                               \
    } while(0)

    /* Attention head count. */
    TRY_I32("attention.head_count",     n_head);

    /* KV head count (GQA support). */
    TRY_I32("attention.head_count_kv",  n_kv_head);
    if (cfg->n_kv_head == 0) cfg->n_kv_head = cfg->n_head; /* MHA fallback */

    /* Head dimension (Q). */
    TRY_I32("attention.head_dim",       head_dim);

    /* KV head dimension (may differ from Q head_dim in Qwen3.5 etc.). */
    TRY_I32("attention.key_length",      kv_head_dim);

    /* Fallback: prefer kv_head_dim if head_dim not explicitly given
     * (Qwen3.5 models store key_length but not head_dim in metadata). */
    if (cfg->head_dim == 0 && cfg->kv_head_dim > 0)
        cfg->head_dim = cfg->kv_head_dim;
    else if (cfg->head_dim == 0 && cfg->n_head > 0)
        cfg->head_dim = cfg->n_embd / cfg->n_head;

    if (cfg->kv_head_dim == 0) cfg->kv_head_dim = cfg->head_dim;

    /* Partial RoPE dimension (Qwen3.5 uses 64 of 256 head dims). */
    TRY_I32("rope.dimension_count",      rope_dim);
    /* Default to head_dim (full RoPE) when not specified. */
    if (cfg->rope_dim == 0) cfg->rope_dim = cfg->head_dim;

    /* ---- DeepSeek MLA ---- */
    TRY_I32("attention.kv_lora_rank",     kv_lora_rank);
    TRY_I32("attention.qk_nope_head_dim", qk_nope_head_dim);
    TRY_I32("attention.qk_rope_head_dim", qk_rope_head_dim);

    #undef TRY_I32

    slog(INFO, "ArchConfig: arch=%s n_embd=%u n_head=%u n_kv_head=%u head_dim=%u kv_head_dim=%u rope_dim=%u n_layer=%u n_vocab=%u",
         arch_key_prefix(en->model->arch), cfg->n_embd, cfg->n_head,
         cfg->n_kv_head, cfg->head_dim, cfg->kv_head_dim, cfg->rope_dim,
         cfg->n_layer, cfg->n_vocab);

    if (cfg->kv_lora_rank)
        slog(INFO, "ArchConfig MLA: kv_lora_rank=%u qk_nope_head_dim=%u qk_rope_head_dim=%u",
             cfg->kv_lora_rank, cfg->qk_nope_head_dim, cfg->qk_rope_head_dim);
}


/* Create a session: sampling parameters, thread pool, architecture
 * config, and the ops table for the detected architecture. */
Session *session_create(Engine *en, u32 ctx_size, int nthreads) {
    if (!en || ctx_size == 0) return NULL;

    Session *s = smalloc(sizeof(*s));
    s->en                = en;
    s->ctx_size          = ctx_size;
    s->temperature       = DEFAULT_TEMPERATURE;
    s->top_p             = DEFAULT_TOP_P ;
    s->top_k             = DEFAULT_TOP_K ;
    s->min_p             = DEFAULT_MIN_P;
    s->repeat_penalty    = DEFAULT_REPEAT_PENALTY;
    s->repeat_last_n     = DEFAULT_REPEAT_LAST_N;
    s->frequency_penalty = 0.0f;
    s->presence_penalty  = 0.0f;
    s->max_tokens        = 0;
    s->pthreads          = pthreads_create(nthreads);

    /* Fill architecture config from model metadata. */
    arch_config_init(en, &s->cfg);

    /* Bind the right ops table. */
    switch (en->model->arch) {
        case ARCH_LLAMA:    s->ops = llama_ops;    break;
        case ARCH_QWEN2:    s->ops = qwen25_ops;   break;
        case ARCH_QWEN3:    s->ops = qwen35_ops;   break;
        case ARCH_DEEPSEEK: s->ops = deepseek_ops; break;
        case ARCH_FALCON:   s->ops = falcon_ops;   break;
        default:
            slog(WARN, "Unknown architecture, falling back to llama.");
            s->ops = llama_ops;
            break;
    }

    /* Architecture-specific initialisation (allocates cache,
     * token buffer, logits). */
    if (!s->ops.init(s)) {
        slog(WARN, "Session init failed — cleaning up.");
        sfree(s);
        return NULL;
    }

    slog(INFO, "Session created: ctx_size=%u n_layer=%u n_embd=%u "
         "n_head=%u head_dim=%u",
         s->ctx_size, s->cfg.n_layer, s->cfg.n_embd,
         s->cfg.n_head, s->cfg.head_dim);

    return s;
}

/* Destroy a session: free the thread pool and arch-specific state. */
void session_free(Session *s) {
    if (!s) return;
    pthreads_destroy(s->pthreads);
    if (s->ops.free) s->ops.free(s);
    sfree(s);
}

