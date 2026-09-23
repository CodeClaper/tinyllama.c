/*
 * cpu_op.c — CPU graph operators (the op_table formerly in graph.c).
 *
 * Every op function mirrors its CUDA twin in backend/gpu/gpu_op.cu
 * op-for-op — same indexing, same math order, same params[] layout.
 * The executor calls cpu_graph_op() once per node with host-resident
 * pointers; returning false aborts the execution, and an operator must
 * free whatever scratch it allocated before returning.
 */

#include <stdbool.h>
#include <string.h>
#include <math.h>

#include "../../def.h"
#include "../../core.h"
#include "../../mm.h"
#include "../../slog.h"
#include "cpu.h"
#include "cpu_op.h"

/* Row-major source tensor of edge `k`. */
static float *op_src(const OpCtx *c, int k) {
    return (float *)c->node->src[k]->data;
}

/* Op-specific parameter `k` of the node being executed. */
static u32 op_param(const OpCtx *c, int k) {
    return c->node->params[k];
}

/* ---- CPU compute kernels (moved from core.c) -------------------
 * Used only by the op_* functions below, so they live in this
 * translation unit.  The CUDA/Metal twins are device kernels in
 * backend/gpu/gpu_op.cu and backend/metal/metal_op.m. */

/* Stack buffer threshold for bias dequant (see bias_add). */
#define BIAS_BUF_STACK 4096

/* ---- Math primitives ------------------------------------------- */

/* RMS Normalisation: o = x / rms(x) * w  (in-place ok when o == x). */
static void rms_norm(float *o, const float *x, TensorInfo *tw, int n, float eps) {
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
    tensor_get_f32_batch(tw, 0, (u64)n, w_buf);
    for (int i = 0; i < n; i++)
        o[i] = x[i] * scale * w_buf[i];
    if (w_buf != rms_stack) sfree(w_buf);
    #undef RMS_STACK
}

/* Batch-dequantise a 1-D bias tensor and add it element-wise to dst[0..n-1].
 * Uses a stack buffer when n <= BIAS_BUF_STACK; falls back to heap for
 * oversized tensors (unlikely for bias).  This trades memory for speed:
 * tensor_get_f32_batch uses SIMD block dequant, and the addition loop
 * is trivially auto-vectorisable with contiguous f32 data. */
static void bias_add(float *dst, TensorInfo *tb, u32 n) {
    float  stack_buf[BIAS_BUF_STACK];
    float *buf = n <= BIAS_BUF_STACK ? stack_buf : smalloc((u64)n * sizeof(float));
    tensor_get_f32_batch(tb, 0, n, buf);
    for (u32 i = 0; i < n; i++)
        dst[i] += buf[i];
    if (buf != stack_buf) sfree(buf);
}

/* ---- Parallel mat-vec row workers ----------------------------------
 * Each invocation of the worker computes one output element y[i].
 * The context is shared read-only across threads (it is fully
 * initialised before pthreads_parallel_for is entered), and each
 * worker writes a disjoint index of y, so there is no data race. */
typedef struct {
    float       *y;       /* output row slice                      */
    TensorInfo  *tw;      /* weight tensor                         */
    const float *x;       /* input vector                          */
    u64          cols;    /* row length / dot-product length       */
    const i8    *x_i8;    /* i8-quantised x (dotprod path only)    */
    float        x_scale; /* scale for x_i8 (dotprod path only)    */
} MatVecCtx;

/* Standard fused dequant + dot product row worker. */
static void matvec_worker(void *arg, int tid, int i) {
    (void)tid;
    MatVecCtx *c = (MatVecCtx *)arg;
    c->y[i] = gguf_dot_batch(c->tw, (u64)i * c->cols, c->cols, c->x);
}

/* i8 dot-product row worker (ARMv8.2+ dotprod, Q8_0 / Q8_K). */
#if defined(__ARM_FEATURE_DOTPROD)
static void matvec_i8_worker(void *arg, int tid, int i) {
    (void)tid;
    MatVecCtx *c = (MatVecCtx *)arg;
    c->y[i] = gguf_dot_i8_batch(c->tw, (u64)i * c->cols, c->cols,
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
static bool mat_vec_mul(float *y, TensorInfo *tw, const float *x, u64 rows, u64 cols, bool trans, pthreads_t *pool) {
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

    if (trans) {
        /* Transposed: W stored as [cols × rows], compute y = W^T @ x.
         * Strided access — need a gather buffer. */
        #define MATVEC_STACK 4096
        float  stack_buf[MATVEC_STACK];
        float *w_row = cols <= MATVEC_STACK ? stack_buf : smalloc(cols * sizeof(float));
        if (!w_row) return false;

        for (u64 r = 0; r < rows; r++) {
            for (u64 c = 0; c < cols; c++)
                w_row[c] = tensor_get_f32(tw, c * tc + r);
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
                MatVecCtx ctx = { y, tw, x, cols, x_i8, x_scale };
                pthreads_parallel_for(pool, 0, (int)rows, matvec_i8_worker, &ctx);
            } else {
                for (u64 r = 0; r < rows; r++)
                    y[r] = gguf_dot_i8_batch(tw, r * cols, cols,
                                             x_i8, x_scale);
            }
            sfree(x_i8);
        } else
#endif
        {
            if (pool && pool->nthreads > 1 && rows > 1) {
                MatVecCtx ctx = { y, tw, x, cols, NULL, 0.0f };
                pthreads_parallel_for(pool, 0, (int)rows, matvec_worker, &ctx);
            } else {
                for (u64 r = 0; r < rows; r++)
                    y[r] = gguf_dot_batch(tw, r * cols, cols, x);
            }
        }
    }
    return true;
}

/* ---- Parallel mat-mat chunk workers -----------------------------------
 * The product Y[b][r] = W[r] · X[b]  is the matrix multiply Y = X @ W^T.
 * X is transposed once into X^T [cols × batch]; each work item then owns
 * a chunk of output rows r0..r1, dequantises those weight rows into a
 * contiguous buffer and runs the SIMD mul_mat_mat() micro-kernel against
 * X^T.  Per-thread scratch keeps the buffers race-free. */
#define MATMAT_CHUNK 16
typedef struct {
    float       *Y;         /* output matrix [batch × rows]           */
    TensorInfo  *tw;        /* weight tensor                          */
    const float *X_T;       /* transposed input [cols × batch]        */
    u64          batch;
    u64          rows;
    u64          cols;
    u64          tc;        /* tensor column-stride (transposed only) */
    bool         trans;
    float       *row_buf;   /* nthreads × chunk × cols  dequant scratch */
    float       *tmp_buf;   /* nthreads × chunk × batch result scratch  */
} MatMatCtx;

/* Compute output rows [r0, r1) into Y through a single mul_mat_mat call. */
static void matmat_chunk(float *Y, TensorInfo *tw, const float *X_T,
                         u64 batch, u64 rows, u64 cols, u64 tc, bool trans,
                         u64 r0, u64 r1, float *w_row, float *tmp) {
    u64 nchunk = r1 - r0;
    if (trans) {
        for (u64 r = 0; r < nchunk; r++)
            for (u64 ci = 0; ci < cols; ci++)
                w_row[r * cols + ci] = tensor_get_f32(tw, ci * tc + (r0 + r));
    } else {
        for (u64 r = 0; r < nchunk; r++)
            tensor_get_f32_batch(tw, (r0 + r) * cols, cols, w_row + r * cols);
    }

    mul_mat_mat(tmp, w_row, X_T, nchunk, cols, batch);

    for (u64 r = 0; r < nchunk; r++)
        for (u64 b = 0; b < batch; b++)
            Y[b * rows + (r0 + r)] = tmp[r * batch + b];
}

static void matmat_worker(void *arg, int tid, int idx) {
    MatMatCtx *c = (MatMatCtx *)arg;
    u64 r0 = (u64)idx * MATMAT_CHUNK;
    u64 r1 = r0 + MATMAT_CHUNK;
    if (r1 > c->rows) r1 = c->rows;
    matmat_chunk(c->Y, c->tw, c->X_T, c->batch, c->rows, c->cols, c->tc,
                 c->trans, r0, r1,
                 c->row_buf + (u64)tid * MATMAT_CHUNK * c->cols,
                 c->tmp_buf + (u64)tid * MATMAT_CHUNK * c->batch);
}

/* Batch matrix-matrix multiply:  Y[b] = W @ X[b]  for b ∈ [0, batch).
 *
 * Reformulated as Y = X @ W^T: X is transposed once up front and the
 * row loop is chunked so every chunk runs the SIMD mul_mat_mat kernel
 * (4-row micro-block on NEON, 4×8 on AVX2/FMA) with X^T streamed once
 * per chunk, replacing the per-row scalar dot-products.
 *
 * When pool is non-NULL and has more than one thread, the chunks are
 * dispatched across the worker pool; each thread gets its own scratch
 * slices to avoid data races. */
static bool mat_mat_mul(float *Y, TensorInfo *tw, const float *X,
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

    if (rows == 0 || batch == 0 || cols == 0) return true;

    u64 n_chunks = (rows + MATMAT_CHUNK - 1) / MATMAT_CHUNK;

    /* Transpose the input once: [batch × cols] → [cols × batch]. */
    float *X_T = smalloc((size_t)cols * batch * sizeof(float));
    if (!X_T) {
        slog(WARN, "mat_mat_mul: smalloc failed for X^T [%lu × %lu]",
             (unsigned long)cols, (unsigned long)batch);
        return false;
    }
    for (u64 c = 0; c < cols; c++)
        for (u64 b = 0; b < batch; b++)
            X_T[c * batch + b] = X[b * cols + c];

    /* Parallel path: dispatch chunks across the thread pool. */
    if (pool && pool->nthreads > 1 && n_chunks > 1) {
        int nt = pool->nthreads;
        float *row_buf = smalloc((u64)nt * MATMAT_CHUNK * cols * sizeof(float));
        float *tmp_buf = smalloc((u64)nt * MATMAT_CHUNK * batch * sizeof(float));
        if (!row_buf || !tmp_buf) {
            slog(WARN, "mat_mat_mul: smalloc failed for parallel chunk buffers");
            sfree(X_T); sfree(row_buf); sfree(tmp_buf);
            return false;
        }
        MatMatCtx ctx = { Y, tw, X_T, batch, rows, cols, tc, trans, row_buf, tmp_buf };
        pthreads_parallel_for(pool, 0, (int)n_chunks, matmat_worker, &ctx);
        sfree(row_buf); sfree(tmp_buf);
        sfree(X_T);
        return true;
    }

    /* Serial path: single reusable chunk buffers. */
    float *row_buf = smalloc((size_t)MATMAT_CHUNK * cols * sizeof(float));
    float *tmp_buf = smalloc((size_t)MATMAT_CHUNK * batch * sizeof(float));
    if (!row_buf || !tmp_buf) {
        slog(WARN, "mat_mat_mul: smalloc failed for chunk buffers");
        sfree(X_T); sfree(row_buf); sfree(tmp_buf);
        return false;
    }
    for (u64 ch = 0; ch < n_chunks; ch++) {
        u64 r0 = ch * MATMAT_CHUNK;
        u64 r1 = r0 + MATMAT_CHUNK;
        if (r1 > rows) r1 = rows;
        matmat_chunk(Y, tw, X_T, batch, rows, cols, tc, trans, r0, r1,
                     row_buf, tmp_buf);
    }
    sfree(row_buf); sfree(tmp_buf);
    sfree(X_T);
    return true;
}


/* SiLU activation in-place. */
static void silu(float *x, int n) {
    for (int i = 0; i < n; i++)
        x[i] = x[i] / (1.0f + expf(-x[i]));
}

/* Softmax in-place on a slice of length n. */
static void softmax(float *x, u32 n) {
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
static float softplus(float x) {
    return (x > 20.0f) ? x : log1pf(expf(x));
}

/* Sigmoid: y = 1 / (1 + exp(-x)).  Naturally stable: for x << -88,
 * expf(-x) overflows to +inf and the result saturates to 0. */
static float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

/* ---- SSM (Gated DeltaNet) kernels ----
 * Shared by the eager arch implementations and the graph executor. */

/* In-place RMS normalization of one slice.
 * Equivalent to rms_norm(o, x, weight, eps) with o == x, but takes an
 * already-dequantised weight so callers can hoist the dequant out of
 * their per-row loops. */
static void rms_norm_inplace(float *x, const float *weight, u32 n, float eps) {
    float ss = 0.0f;
    for (u32 j = 0; j < n; j++)
        ss += x[j] * x[j];
    ss = 1.0f / sqrtf(ss / (float)n + eps);
    for (u32 j = 0; j < n; j++)
        x[j] = x[j] * ss * weight[j];
}

/* L2-normalize each row of a [n_rows, dim] matrix in place.
 * Used for Q and K in the Gated DeltaNet recurrence. */
static void l2_norm_rows(float *x, u32 n_rows, u32 dim, float eps) {
    for (u32 r = 0; r < n_rows; r++) {
        float *row = x + (u64)r * dim;
        float ss = 0.0f;
        for (u32 i = 0; i < dim; i++)
            ss += row[i] * row[i];
        float inv = 1.0f / sqrtf(ss + eps);
        for (u32 i = 0; i < dim; i++)
            row[i] *= inv;
    }
}

/* Depthwise causal Conv1d step for single-token inference.
 * Processes fused QKV output (channels) through a depthwise conv
 * with kernel_size taps, maintaining a ring buffer of previous inputs.
 * weight – [channels * kernel_size] dequantised weight (GGUF layout,
 *          PyTorch Conv1d convention), weight[c * kernel_size + k] for
 *          channel c, tap k.
 *          Tap k multiplies the input from k taps ago: tap 0 is the
 *          OLDEST input (x[t-(K-1)]), tap K-1 is the current input,
 *          matching ggml_ssm_conv and the reference implementation.
 * state  – [(kernel_size-1) * channels] ring buffer (oldest first).
 * On return, state is shifted and the new input is stored. */
static void causal_conv1d_step(float *output, const float *input,
                        const float *weight, float *state,
                        u32 channels, u32 kernel_size) {
    /* Callers run this in-place (output == input), which clobbers the
     * raw inputs before the state update below.  The state must hold
     * the RAW pre-conv inputs — storing the conv outputs there instead
     * corrupts every subsequent token's window and silently diverges
     * the whole recurrence.  Save a copy only when aliased. */
    float *saved = NULL;
    if (input == output) {
        saved = smalloc((u64)channels * sizeof(float));
        memcpy(saved, input, (u64)channels * sizeof(float));
    }
    const float *raw = saved ? saved : input;

    /* State is stored [oldest, ..., newest] = [x[t-(K-1)], ..., x[t-1]]
     * at positions [0, ..., K-2].  Weight tap k (GGUF convention)
     * multiplies the input at lag (K-1-k): state position k for
     * k < K-1, and the current input for k == K-1. */
    for (u32 c = 0; c < channels; c++) {
        float sum = 0.0f;
        for (u32 k = 0; k < kernel_size; k++) {
            float x = (k == kernel_size - 1) ? raw[c]
                                             : state[k * channels + c];
            sum += x * weight[c * kernel_size + k];
        }
        output[c] = sum;
    }
    /* Shift state: drop oldest (index 0), shift remaining left,
     * store current input at the end. */
    if (kernel_size > 1) {
        u32 shift_len = (kernel_size - 2) * channels;
        if (shift_len > 0)
            memmove(state, state + channels, shift_len * sizeof(float));
        memcpy(state + shift_len, raw, channels * sizeof(float));
    }
    sfree(saved);
}

/* Single-step Gated DeltaNet recurrence.
 * Q, K, V     – [n_k_heads * head_dim] for Q/K, [n_v_heads * head_dim] for V
 * state       – [n_v_heads * head_dim * head_dim] recurrent state (in-place)
 * g           – [n_v_heads] decay gate, pre-computed as exp(g_raw)
 * beta        – [n_v_heads] update gate, pre-computed as sigmoid(beta_raw)
 * kv_mem      – [n_v_heads * head_dim] scratch buffer
 * output      – [n_v_heads * head_dim] result
 *
 * Grouped-value attention (GVA): when n_v_heads > n_k_heads the Q/K heads
 * are reused, value head h reading Q/K head `h % n_k_heads`.  That is the
 * mapping ggml_repeat_4d produces in the reference implementation
 * (llama.cpp src/models/qwen35.cpp), which tiles the whole Q/K head block.
 * n_v_heads is always a multiple of n_k_heads. */
static void gated_delta_step(const float *Q, const float *K, const float *V,
                      float *state, const float *g, const float *beta,
                      float *kv_mem, float *output,
                      u32 n_v_heads, u32 n_k_heads, u32 head_dim) {
    u64 d2 = (u64)head_dim * head_dim;
    for (u32 grp = 0; grp < n_v_heads; grp++) {
        u32 kh = n_k_heads ? grp % n_k_heads : 0;   /* GVA head reuse */
        const float *Qg = Q + (u64)kh * head_dim;
        const float *Kg = K + (u64)kh * head_dim;
        const float *Vg = V + (u64)grp * head_dim;
        float *Sg       = state + (u64)grp * d2;
        float *mem      = kv_mem + (u64)grp * head_dim;

        float decay = g[grp];
        float upd   = beta[grp];

        /* 1. Decay: S *= decay */
        for (u64 i = 0; i < d2; i++)
            Sg[i] *= decay;

        /* 2. kv_mem = S^T @ K  →  mem[j] = sum_i Sg[i][j] * Kg[i] */
        memset(mem, 0, head_dim * sizeof(float));
        for (u32 i = 0; i < head_dim; i++) {
            float ki = Kg[i];
            float *row = Sg + (u64)i * head_dim; /* Sg[i][:] */
            for (u32 j = 0; j < head_dim; j++)
                mem[j] += row[j] * ki;
        }

        /* 3. Compute delta vector: delta[j] = (Vg[j] - mem[j]) * beta */
        float delta_j[head_dim];
        for (u32 j = 0; j < head_dim; j++)
            delta_j[j] = (Vg[j] - mem[j]) * upd;

        /* 4. S += K @ delta^T (outer product)
         * 5. output = S^T @ Q (readout) */
        float out_j[head_dim];
        memset(out_j, 0, head_dim * sizeof(float));
        for (u32 i = 0; i < head_dim; i++) {
            float qi = Qg[i];
            float ki = Kg[i];
            float *row = Sg + (u64)i * head_dim;
            for (u32 j = 0; j < head_dim; j++) {
                row[j] += ki * delta_j[j];   /* 4. S[i][j] += K[i] * delta[j] */
                out_j[j] += row[j] * qi;      /* 5. readout */
            }
        }
        memcpy(output + (u64)grp * head_dim, out_j, head_dim * sizeof(float));
    }
}

static bool op_embed(OpCtx *c) {
    TensorInfo *te = c->node->weights[0];
    bool te_trans  = (te->dim[0] == (i64)c->cfg->n_vocab);
    u32 *tok = (u32 *)c->node->src[0]->data;

    /* Always the full batch: the embedding of every input row is needed. */
    for (u32 p = 0; p < c->n; p++) {
        float *dp = c->dst + (u64)p * c->od;
        if (te_trans) {
            tensor_get_f32_batch(te, (u64)tok[p] * c->od, c->od, dp);
        } else {
            for (u32 j = 0; j < c->od; j++)
                dp[j] = tensor_get_f32(te, (u64)j * c->cfg->n_vocab + tok[p]);
        }
    }
    return true;
}

static bool op_rms_norm(OpCtx *c) {
    TensorInfo *tw = c->node->weights[0];
    float *src = op_src(c, 0);
    for (u32 p = 0; p < c->r; p++)
        rms_norm(c->dst + (u64)p * c->od, src + ((u64)c->base + p) * c->od,
                 tw, (int)c->od, DEFAULT_EPS);
    return true;
}

static bool op_matmul(OpCtx *c) {
    TensorInfo *w = c->node->weights[0];
    bool tr       = (c->node->op == OP_MATMUL_T);
    u32 in        = (u32)(tr ? w->dim[0] : w->dim[1]);
    float *x      = op_src(c, 0) + (u64)c->base * in;

    if (c->r == 1)
        return mat_vec_mul(c->dst, w, x, c->od, in, tr, c->s->pthreads);
    return mat_mat_mul(c->dst, w, x, c->r, c->od, in, tr, c->s->pthreads);
}

static bool op_bias(OpCtx *c) {
    TensorInfo *tb = c->node->weights[0];
    float *src = op_src(c, 0);
    u32 nb = tb->n_element < (u64)c->od ? (u32)tb->n_element : c->od;
    for (u32 p = 0; p < c->r; p++) {
        float *dp = c->dst + (u64)p * c->od;
        memcpy(dp, src + ((u64)c->base + p) * c->od, (size_t)c->od * sizeof(float));
        bias_add(dp, tb, nb);
    }
    return true;
}

static bool op_binary(OpCtx *c) {
    bool add = (c->node->op == OP_ADD);
    float *a = op_src(c, 0);
    float *b = op_src(c, 1);
    for (u32 p = 0; p < c->r; p++) {
        float *dp = c->dst + (u64)p * c->od;
        float *ap = a + ((u64)c->base + p) * c->od;
        float *bp = b + ((u64)c->base + p) * c->od;
        if (add)
            for (u32 j = 0; j < c->od; j++) dp[j] = ap[j] + bp[j];
        else
            for (u32 j = 0; j < c->od; j++) dp[j] = ap[j] * bp[j];
    }
    return true;
}

/* Element-wise unary: copy the row, then apply f in place. */
static bool op_unary(OpCtx *c, void (*f)(float *, int)) {
    float *src = op_src(c, 0);
    for (u32 p = 0; p < c->r; p++) {
        float *dp = c->dst + (u64)p * c->od;
        memcpy(dp, src + ((u64)c->base + p) * c->od, (size_t)c->od * sizeof(float));
        f(dp, (int)c->od);
    }
    return true;
}

/* softmax() takes a u32 count; adapt it to the unary signature. */
static void softmax_n(float *x, int n) { softmax(x, (u32)n); }

/* tanh-approximation GELU (== ggml_gelu / HF gelu_pytorch_tanh). */
static void gelu_n(float *x, int n) {
    const float GELU_COEF_A     = 0.044715f;
    const float SQRT_2_OVER_PI  = 0.7978845608028654f;
    for (int i = 0; i < n; i++) {
        float v = x[i];
        x[i] = 0.5f * v * (1.0f + tanhf(SQRT_2_OVER_PI * v * (1.0f + GELU_COEF_A * v * v)));
    }
}

static bool op_silu(OpCtx *c)    { return op_unary(c, silu); }
static bool op_gelu(OpCtx *c)    { return op_unary(c, gelu_n); }
static bool op_softmax(OpCtx *c) { return op_unary(c, softmax_n); }

static bool op_scale(OpCtx *c) {
    u32 bits = op_param(c, 0);
    float scale;
    memcpy(&scale, &bits, sizeof(scale));
    float *src = op_src(c, 0);
    for (u32 p = 0; p < c->r; p++) {
        float *dp = c->dst + (u64)p * c->od;
        const float *sp = src + ((u64)c->base + p) * c->od;
        for (u32 j = 0; j < c->od; j++)
            dp[j] = sp[j] * scale;
    }
    return true;
}

/* Gemma final-logit soft cap: out = tanh(x / cap) * cap. */
static bool op_softcap(OpCtx *c) {
    u32 bits = op_param(c, 0);
    float cap;
    memcpy(&cap, &bits, sizeof(cap));
    float *src = op_src(c, 0);
    for (u32 p = 0; p < c->r; p++) {
        float *dp = c->dst + (u64)p * c->od;
        const float *sp = src + ((u64)c->base + p) * c->od;
        for (u32 j = 0; j < c->od; j++)
            dp[j] = tanhf(sp[j] / cap) * cap;
    }
    return true;
}


/* Partial RoPE: rotates only the first rope_dim dimensions of each head.
 * head_dim is the full head dimension (stride between heads),
 * rope_dim is how many dims to rotate (64 of 256 for Qwen3.5).
 * The remaining (head_dim - rope_dim) dimensions are left unchanged. */
static void rope_partial(float *buf, u32 n_heads, u32 head_dim, u32 rope_dim, u32 pos, float theta_base) {
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

static bool op_rope_neox(OpCtx *c) {
    u32 bits = op_param(c, 0);
    float theta;
    memcpy(&theta, &bits, sizeof(theta));
    u32 heads = op_param(c, 1);
    u32 hdim  = op_param(c, 2);
    u32 rdim  = op_param(c, 3);
    float *src = op_src(c, 0);

    for (u32 p = 0; p < c->r; p++) {
        u32 abspos = c->pos + c->base + p;   /* absolute position */
        float *dp  = c->dst + (u64)p * c->od;
        memcpy(dp, src + (u64)(c->base + p) * c->od, (size_t)c->od * sizeof(float));
        /* rope_partial degenerates to rope_neox for rdim == hdim. */
        rope_partial(dp, heads, hdim, rdim, abspos, theta);
    }
    return true;
}

static bool op_rms_norm_heads(OpCtx *c) {
    /* Per-head RMS norm over [n_heads] slices of head_dim, read at
     * in_stride and written at out_stride.  Q-norm over a fused
     * [q, gate] projection uses in_stride == 2*head_dim, which also
     * drops the (unnormed) gate half. */
    u32 nh = op_param(c, 0), hd = op_param(c, 1);
    u32 is = op_param(c, 2), os = op_param(c, 3);
    float *src = op_src(c, 0);
    float *nw  = smalloc((u64)hd * sizeof(float));
    if (!nw) return false;
    tensor_get_f32_batch(c->node->weights[0], 0, hd, nw);

    for (u32 p = 0; p < c->r; p++) {
        const float *sp = src + ((u64)c->base + p) * (u64)nh * is;
        float *dp = c->dst + (u64)p * c->od;
        for (u32 h = 0; h < nh; h++) {
            float *o = dp + (u64)h * os;
            memcpy(o, sp + (u64)h * is, (size_t)hd * sizeof(float));
            rms_norm_inplace(o, nw, hd, DEFAULT_EPS);
        }
    }
    sfree(nw);
    return true;
}

static bool op_sigmoid_gate(OpCtx *c) {
    /* out = a * sigmoid(gate), gate taken from the second half of each
     * 2*head_dim block of the fused source. */
    u32 nh = op_param(c, 0), hd = op_param(c, 1);
    float *a  = op_src(c, 0);
    float *gt = op_src(c, 1);

    for (u32 p = 0; p < c->r; p++) {
        const float *ap = a  + ((u64)c->base + p) * c->od;
        const float *gp = gt + ((u64)c->base + p) * (u64)nh * 2 * hd;
        float *dp = c->dst + (u64)p * c->od;
        for (u32 h = 0; h < nh; h++)
            for (u32 d = 0; d < hd; d++)
                dp[(u64)h * hd + d] =
                    ap[(u64)h * hd + d] * sigmoid(gp[(u64)h * 2 * hd + hd + d]);
    }
    return true;
}

static bool op_ssm_conv(OpCtx *c) {
    /* Depthwise causal conv1d over the fused QKV projection, advancing
     * the per-layer ring buffer one row at a time. */
    u32 st = op_param(c, 0), ck = op_param(c, 1);
    if (st >= c->g->n_state) return false;
    float *state = (float *)c->g->state[st];
    float *src   = op_src(c, 0);
    float *cw    = smalloc((u64)ck * c->od * sizeof(float));
    if (!cw) return false;
    tensor_get_f32_batch(c->node->weights[0], 0, (u64)ck * c->od, cw);

    for (u32 p = 0; p < c->r; p++) {
        float *dp = c->dst + (u64)p * c->od;
        memcpy(dp, src + ((u64)c->base + p) * c->od, (size_t)c->od * sizeof(float));
        causal_conv1d_step(dp, dp, cw, state, c->od, ck);
    }
    sfree(cw);
    return true;
}

static bool op_ssm_delta(OpCtx *c) {
    /* Gated DeltaNet recurrence.  Sequential over rows by construction:
     * each row advances the recurrent state. */
    u32 st = op_param(c, 0), n_v = op_param(c, 1);
    u32 n_k = op_param(c, 2), hd = op_param(c, 3);
    if (st >= c->g->n_state) return false;
    u32 key_dim = n_k * hd;
    u64 val_dim = (u64)n_v * hd;
    if (c->od != (u32)val_dim) return false;
    float *state = (float *)c->g->state[st];
    float *fused = op_src(c, 0);
    float *alpha = op_src(c, 1);
    float *beta  = op_src(c, 2);

    /* Per-call scratch: gates, dequantised params, and a private
     * [q|k|v] copy (the fused source is shared with other consumers
     * and must not be mutated). */
    float *gv    = smalloc((u64)n_v * sizeof(float));
    float *bv    = smalloc((u64)n_v * sizeof(float));
    float *a_log = smalloc((u64)n_v * sizeof(float));
    float *dt_bi = smalloc((u64)n_v * sizeof(float));
    float *nw    = smalloc((u64)hd * sizeof(float));
    float *mem   = smalloc(val_dim * sizeof(float));
    float *qkv   = smalloc((2 * (u64)key_dim + val_dim) * sizeof(float));
    if (!gv || !bv || !a_log || !dt_bi || !nw || !mem || !qkv) {
        sfree(gv); sfree(bv); sfree(a_log); sfree(dt_bi);
        sfree(nw); sfree(mem); sfree(qkv);
        return false;
    }
    tensor_get_f32_batch(c->node->weights[0], 0, n_v, a_log);
    tensor_get_f32_batch(c->node->weights[1], 0, n_v, dt_bi);
    if (c->node->weights[2]) tensor_get_f32_batch(c->node->weights[2], 0, hd, nw);

    for (u32 p = 0; p < c->r; p++) {
        u64 fstride = 2 * (u64)key_dim + val_dim;
        const float *fp = fused + ((u64)c->base + p) * fstride;
        const float *ap = alpha + ((u64)c->base + p) * n_v;
        const float *bp = beta  + ((u64)c->base + p) * n_v;
        float *dp = c->dst + (u64)p * c->od;

        /* ssm.a already stores -exp(A_log) in the GGUF (converter:
         * A_log → -exp(A_log)), so the decay gate is
         * g = exp(ssm.a * softplus(alpha + dt_bias)). */
        for (u32 i = 0; i < n_v; i++) {
            gv[i] = expf(a_log[i] * softplus(ap[i] + dt_bi[i]));
            bv[i] = sigmoid(bp[i]);
        }

        memcpy(qkv, fp, fstride * sizeof(float));
        float *q_d = qkv, *k_d = qkv + key_dim, *v_d = qkv + 2 * key_dim;
        l2_norm_rows(q_d, n_k, hd, DEFAULT_EPS);
        l2_norm_rows(k_d, n_k, hd, DEFAULT_EPS);
        /* Q *= 1/sqrt(head_dim): the S^T q readout carries no 1/sqrt(d)
         * of its own, so the scale lives on q. */
        float q_scale = 1.0f / sqrtf((float)hd);
        for (u32 i = 0; i < key_dim; i++)
            q_d[i] *= q_scale;

        gated_delta_step(q_d, k_d, v_d, state, gv, bv, mem, dp, n_v, n_k, hd);

        /* Per-value-head RMS norm on the delta output. */
        if (c->node->weights[2])
            for (u32 grp = 0; grp < n_v; grp++)
                rms_norm_inplace(dp + (u64)grp * hd, nw, hd, DEFAULT_EPS);
    }
    sfree(gv); sfree(bv); sfree(a_log); sfree(dt_bi);
    sfree(nw); sfree(mem); sfree(qkv);
    return true;
}

static bool op_attn(OpCtx *c) {
    /* Write the batch's K/V into the session cache at positions
     * pos..pos+n-1, then attend causally over the cached keys (rows
     * 0..pos+qi) — history included. */
    if (!c->s->cache.std) {
        slog(WARN, "cpu_graph_op: OP_ATTN requires the std KV cache");
        return false;
    }
    u32 layer = op_param(c, 0);
    u32 win   = op_param(c, 1);   /* sliding window; 0 = full causal */
    AttnKvCache *akc = &c->s->cache.std[layer];
    if (!akc->k || !akc->v) return false;

    float *qd  = op_src(c, 0);
    float *kd  = op_src(c, 1);
    float *vd  = op_src(c, 2);
    u32 n_head = c->cfg->n_head;
    u32 n_kv   = c->cfg->n_kv_head;
    u32 hd     = c->cfg->head_dim;
    u32 khd    = c->cfg->kv_head_dim;
    u32 gqa    = n_head / n_kv;
    u32 hs     = akc->cap * khd;   /* stride between heads */

    /* K/V write-through (head-major: [head][pos][dim]). */
    for (u32 i = 0; i < c->n; i++) {
        const float *kr = kd + (u64)i * c->kv_dim;
        const float *vr = vd + (u64)i * c->kv_dim;
        for (u32 h = 0; h < n_kv; h++) {
            memcpy(akc->k + (u64)h * hs + (u64)(c->pos + i) * khd,
                   kr + (u64)h * khd, (size_t)khd * sizeof(float));
            memcpy(akc->v + (u64)h * hs + (u64)(c->pos + i) * khd,
                   vr + (u64)h * khd, (size_t)khd * sizeof(float));
        }
    }
    akc->n = c->pos + c->n;

    memset(c->dst, 0, (size_t)c->n * c->q_dim * sizeof(float));
    for (u32 h = 0; h < n_head; h++) {
        u32 kv_h = h / gqa;
        float *kh_base = akc->k + (u64)kv_h * hs;
        float *vh_base = akc->v + (u64)kv_h * hs;
        for (u32 qi = 0; qi < c->n; qi++) {
            float *qh = qd + (u64)qi * c->q_dim + (u64)h * hd;
            u32 n_keys = c->pos + qi + 1;   /* causal */

            /* Sliding window: only the last `win` keys (including the
             * current one) stay visible; win == 0 means full causal. */
            u32 start = (win && n_keys > win) ? n_keys - win : 0;
            u32 n_vis = n_keys - start;

            /* Scores = K-block [n_vis × khd] (contiguous per head) @ qh.
             * SIMD matrix-vector dot, then apply the 1/sqrt(kv_head_dim)
             * scale once to the whole score vector. */
            mul_mat_vec(c->scr, kh_base + (u64)start * khd, qh, n_vis, khd);
            for (u32 t = 0; t < n_vis; t++)
                c->scr[t] *= c->scale;
            softmax(c->scr, n_vis);

            float *oh = c->dst + (u64)qi * c->q_dim + (u64)h * hd;
            float *vt = vh_base + (u64)start * khd;
            for (u32 t = 0; t < n_vis; t++, vt += khd) {
                float st = c->scr[t];
                for (u32 d = 0; d < khd; d++)
                    oh[d] += st * vt[d];
            }
        }
    }
    return true;
}

/* ================================================================
 * Dispatch — the CPU twin of graph.c's former op_table
 * ================================================================ */

bool cpu_graph_op(OpCtx *c) {
    if (!c || !c->g || !c->s || !c->cfg || !c->node || !c->dst || c->od == 0)
        return false;
    switch (c->node->op) {
        case OP_EMBED:          return op_embed(c);
        case OP_RMS_NORM:       return op_rms_norm(c);
        case OP_RMS_NORM_HEADS: return op_rms_norm_heads(c);
        case OP_MATMUL:
        case OP_MATMULARRY:
        case OP_MATMUL_T:       return op_matmul(c);
        case OP_BIAS:           return op_bias(c);
        case OP_ADD:
        case OP_MUL:            return op_binary(c);
        case OP_SILU:           return op_silu(c);
        case OP_GELU:           return op_gelu(c);
        case OP_SCALE:          return op_scale(c);
        case OP_SOFTCAP:        return op_softcap(c);
        case OP_SOFTMAX:        return op_softmax(c);
        case OP_ROPE_NEOX:      return op_rope_neox(c);
        case OP_SIGMOID_GATE:   return op_sigmoid_gate(c);
        case OP_SSM_CONV:       return op_ssm_conv(c);
        case OP_SSM_DELTA:      return op_ssm_delta(c);
        case OP_ATTN:           return op_attn(c);
        /* OP_INPUT has nothing to compute; OP_H2D / OP_D2H stay with
         * the executor. */
        default:                return false;
    }
}
