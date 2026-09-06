/*
 * gpu_op.cu — CUDA graph operators (device twins of graph.c's op_table).
 *
 * The graph executor runs with the arena, the session KV cache, the
 * Gated DeltaNet state buffers and the token ids all resident in VRAM,
 * so activations flow device-to-device and no per-op H2D/D2H takes
 * place.  Weights are the only host-originated data: they are served
 * from gpu.cu's persistent device cache through the gpu_*_dev()
 * primitives in gpu.h (uploaded once on first use).
 *
 * Every op function mirrors its CPU twin in graph.c op-for-op — same
 * indexing, same math order, same params[] layout — with the loops
 * re-expressed as kernels.  CUDA errors are fatal via CHECK();
 * structural problems return false so the executor can abort.
 */

#include <cuda_runtime.h>
#include <math.h>
#include <string.h>

#include "../../def.h"
#include "gpu.h"
#include "gpu_op.h"

extern "C" {
#include "../../slog.h"
}

/* ================================================================
 * Launch plumbing + context accessors
 * ================================================================ */

#define GPU_OP_THREADS   256
#define GPU_OP_MAX_BLOCKS 65535u

static inline unsigned op_block_count(u64 n) {
    u64 want = (n + GPU_OP_THREADS - 1) / GPU_OP_THREADS;
    return (unsigned)(want < GPU_OP_MAX_BLOCKS ? want : GPU_OP_MAX_BLOCKS);
}

/* Mirror op_src()/op_param() in graph.c. */
static inline float *op_src(const GpuOpCtx *c, int k) {
    return (float *)c->g->node[(u32)c->node->src[k]].data;
}
static inline u32 op_param(const GpuOpCtx *c, int k) {
    return c->node->params[k];
}

__device__ static inline float op_neg_inf(void) {
    return __int_as_float((int)0xff800000u);
}

/* Whole-block sum / max reduction.  blockDim must be a multiple of 32;
 * callers keep every thread uniform through the call.  Threads outside
 * the active range contribute the identity element. */
__device__ static inline float block_sum(float v, float *red, u32 tid, u32 nthreads) {
    for (int o = 16; o > 0; o >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, o);
    u32 nw = (nthreads + 31) >> 5;
    if ((tid & 31) == 0) red[tid >> 5] = v;
    __syncthreads();
    if (tid == 0) {
        float s = red[0];
        for (u32 w = 1; w < nw; w++) s += red[w];
        red[0] = s;
    }
    __syncthreads();
    return red[0];
}

__device__ static inline float block_max(float v, float *red, u32 tid, u32 nthreads) {
    for (int o = 16; o > 0; o >>= 1)
        v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, o));
    u32 nw = (nthreads + 31) >> 5;
    if ((tid & 31) == 0) red[tid >> 5] = v;
    __syncthreads();
    if (tid == 0) {
        float m = red[0];
        for (u32 w = 1; w < nw; w++) m = fmaxf(m, red[w]);
        red[0] = m;
    }
    __syncthreads();
    return red[0];
}

/* ================================================================
 * Dequantized-weight cache (small 1-D tensors)
 * ================================================================
 * Norms, biases and SSM parameters are read on every call; each
 * tensor is dequantized once into a device f32 buffer and kept for
 * the process lifetime.  Identity + snapshot guards mirror the raw
 * byte cache in gpu.cu. */

#define GPU_OP_WCACHE_MAX 1024
static TensorInfo *g_wc_ti[GPU_OP_WCACHE_MAX];
static float      *g_wc_dev[GPU_OP_WCACHE_MAX];
static u64         g_wc_elems[GPU_OP_WCACHE_MAX];
static u32         g_wc_type[GPU_OP_WCACHE_MAX];
static u64         g_wc_off[GPU_OP_WCACHE_MAX];
static int         g_wc_n = 0;

static float *wc_upload(TensorInfo *ti) {
    const u8 *raw = gpu_weight_dev(ti);
    if (!raw) return NULL;
    float *d = NULL;
    CHECK(cudaMalloc(&d, (size_t)ti->n_element * sizeof(float)));
    if (gpu_dequant_dev(ti->type, raw, 0, ti->n_element, d) != 0) {
        cudaFree(d);
        return NULL;
    }
    return d;
}

static const float *gpu_op_weight_f32(TensorInfo *ti) {
    if (!ti || !ti->data || ti->n_element == 0) return NULL;
    for (int i = 0; i < g_wc_n; i++) {
        if (g_wc_ti[i] != ti) continue;
        if (g_wc_elems[i] == ti->n_element && g_wc_type[i] == ti->type &&
            g_wc_off[i] == ti->offset)
            return g_wc_dev[i];   /* same identity, same content */
        /* Same pointers, different content: replace the dead copy. */
        float *d = wc_upload(ti);
        if (!d) return NULL;
        cudaFree(g_wc_dev[i]);
        g_wc_dev[i]   = d;
        g_wc_elems[i] = ti->n_element;
        g_wc_type[i]  = ti->type;
        g_wc_off[i]   = ti->offset;
        return d;
    }
    if (g_wc_n >= GPU_OP_WCACHE_MAX) return NULL;
    float *d = wc_upload(ti);
    if (!d) return NULL;
    g_wc_ti[g_wc_n]   = ti;
    g_wc_dev[g_wc_n]  = d;
    g_wc_elems[g_wc_n] = ti->n_element;
    g_wc_type[g_wc_n]  = ti->type;
    g_wc_off[g_wc_n]   = ti->offset;
    g_wc_n++;
    return d;
}

void gpu_op_shutdown(void) {
    for (int i = 0; i < g_wc_n; i++)
        if (g_wc_dev[i]) { cudaFree(g_wc_dev[i]); g_wc_dev[i] = NULL; }
    g_wc_n = 0;
}

/* ================================================================
 * Kernels
 * ================================================================ */

/* ---- OP_EMBED -------------------------------------------------
 * out[p*od + j] = dequant(te, tok[p]*od + j)  (row-major table), or
 * out[p*od + j] = dequant(te, j*n_vocab + tok[p])  (column-major).
 * The token ids are read on-device; the gather itself runs in
 * gpu.cu's dequant_gather_kernel via gpu_dequant_gather_dev(). */

/* ---- OP_RMS_NORM ----------------------------------------------
 * One block per row: out = x / rms(x) * w.  Mirrors rms_norm() with
 * a pre-dequantized weight (gpu_op_weight_f32). */
__global__ void k_rms_norm(const float *__restrict__ src, float *__restrict__ dst,
                           const float *__restrict__ w, u32 wn, u32 od,
                           u32 base, float eps) {
    __shared__ float red[32];
    const u32 p = blockIdx.x;
    const float *x = src + ((u64)base + p) * od;
    float *d       = dst + (u64)p * od;
    float ss = 0.0f;
    for (u32 i = threadIdx.x; i < od; i += blockDim.x) ss += x[i] * x[i];
    ss = block_sum(ss, red, threadIdx.x, blockDim.x);
    float scale = 1.0f / sqrtf(ss / (float)od + eps);
    for (u32 i = threadIdx.x; i < od; i += blockDim.x) {
        float wi = (i < wn) ? w[i] : 0.0f;
        d[i] = x[i] * scale * wi;
    }
}

/* ---- OP_RMS_NORM_HEADS ----------------------------------------
 * One block per (row, head): per-head rms(x)*w read at in_stride and
 * written at out_stride.  Mirrors op_rms_norm_heads(). */
__global__ void k_rms_norm_heads(const float *__restrict__ src, float *__restrict__ dst,
                                 const float *__restrict__ nw,
                                 u32 nh, u32 hd, u32 is, u32 os,
                                 u32 od, u32 base, float eps) {
    __shared__ float red[32];
    const u32 h = blockIdx.x, p = blockIdx.y;
    const float *sp = src + (((u64)base + p) * nh + h) * is;
    float *dp       = dst + (u64)p * od + (u64)h * os;
    float ss = 0.0f;
    for (u32 d = threadIdx.x; d < hd; d += blockDim.x) ss += sp[d] * sp[d];
    ss = block_sum(ss, red, threadIdx.x, blockDim.x);
    float scale = 1.0f / sqrtf(ss / (float)hd + eps);
    for (u32 d = threadIdx.x; d < hd; d += blockDim.x)
        dp[d] = sp[d] * scale * nw[d];
}

/* ---- OP_BIAS ---------------------------------------------------
 * dst[p*od + j] = src[(base+p)*od + j] + bw[j] for j < nb, plain copy
 * beyond (mirrors the memcpy + bias_add pair). */
__global__ void k_bias(const float *__restrict__ src, float *__restrict__ dst,
                       const float *__restrict__ bw, u32 nb, u32 od,
                       u32 base, u64 total) {
    u64 idx     = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    u64 gstride = (u64)gridDim.x * blockDim.x;
    for (; idx < total; idx += gstride) {
        u32 p = (u32)(idx / od), j = (u32)(idx % od);
        float b = (j < nb) ? bw[j] : 0.0f;
        dst[idx] = src[((u64)base + p) * od + j] + b;
    }
}

/* ---- OP_ADD / OP_MUL ------------------------------------------ */
__global__ void k_binary(const float *__restrict__ a, const float *__restrict__ b,
                         float *__restrict__ dst, int add, u32 od,
                         u32 base, u64 total) {
    u64 idx     = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    u64 gstride = (u64)gridDim.x * blockDim.x;
    for (; idx < total; idx += gstride) {
        u32 p = (u32)(idx / od), j = (u32)(idx % od);
        u64 off = ((u64)base + p) * od + j;
        dst[idx] = add ? (a[off] + b[off]) : (a[off] * b[off]);
    }
}

/* ---- OP_SILU --------------------------------------------------- */
__global__ void k_silu(const float *__restrict__ src, float *__restrict__ dst,
                       u32 od, u32 base, u64 total) {
    u64 idx     = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    u64 gstride = (u64)gridDim.x * blockDim.x;
    for (; idx < total; idx += gstride) {
        u32 p = (u32)(idx / od), j = (u32)(idx % od);
        float v = src[((u64)base + p) * od + j];
        dst[idx] = v / (1.0f + expf(-v));
    }
}

/* ---- OP_SOFTMAX ------------------------------------------------
 * One block per row, block-strided over the row width. */
__global__ void k_softmax(const float *__restrict__ src, float *__restrict__ dst,
                          u32 od, u32 base) {
    __shared__ float red[32];
    const u32 p = blockIdx.x;
    const float *x = src + ((u64)base + p) * od;
    float *d       = dst + (u64)p * od;
    float mx = op_neg_inf();
    for (u32 i = threadIdx.x; i < od; i += blockDim.x) mx = fmaxf(mx, x[i]);
    mx = block_max(mx, red, threadIdx.x, blockDim.x);
    float sum = 0.0f;
    for (u32 i = threadIdx.x; i < od; i += blockDim.x) {
        float e = expf(x[i] - mx);
        d[i] = e;
        sum += e;
    }
    sum = block_sum(sum, red, threadIdx.x, blockDim.x);
    for (u32 i = threadIdx.x; i < od; i += blockDim.x) d[i] /= sum;
}

/* ---- OP_ROPE_NEOX ----------------------------------------------
 * Thread per (row, head, i < head_dim/2): rotate the (i, i+half) pair
 * when i < rope_dim/2, plain copy otherwise.  Mirrors rope_partial()
 * degenerating to rope_neox for rope_dim == head_dim. */
__global__ void k_rope_neox(const float *__restrict__ src, float *__restrict__ dst,
                            float theta_base, u32 nh, u32 hdim, u32 rdim,
                            u32 od, u32 base, u32 pos, u64 total) {
    u64 idx     = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    u64 gstride = (u64)gridDim.x * blockDim.x;
    const u32 half = hdim / 2, npair = rdim / 2;
    for (; idx < total; idx += gstride) {
        u32 p = (u32)(idx / (nh * half));
        u32 rem = (u32)(idx % (nh * half));
        u32 h = rem / half, i = rem % half;
        u32 abspos = pos + base + p;
        const float *sp = src + ((u64)base + p) * od + (u64)h * hdim;
        float *dp       = dst + (u64)p * od + (u64)h * hdim;
        float a = sp[i], b = sp[i + half];
        if (i < npair) {
            float theta = 1.0f / powf(theta_base, (float)(2 * i) / (float)rdim);
            float c = cosf((float)abspos * theta);
            float s = sinf((float)abspos * theta);
            dp[i]       = a * c - b * s;
            dp[i + half] = a * s + b * c;
        } else {
            dp[i]       = a;
            dp[i + half] = b;
        }
    }
}

/* ---- OP_SIGMOID_GATE -------------------------------------------
 * out = a * sigmoid(gate), gate from the second half of each
 * 2*head_dim block of the fused source. */
__global__ void k_sigmoid_gate(const float *__restrict__ a, const float *__restrict__ gt,
                               float *__restrict__ dst, u32 hd, u32 od,
                               u32 gate_stride, u32 base, u64 total) {
    u64 idx     = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    u64 gstride = (u64)gridDim.x * blockDim.x;
    for (; idx < total; idx += gstride) {
        u32 p = (u32)(idx / od), j = (u32)(idx % od);
        u32 h = j / hd, d = j % hd;
        float av = a[((u64)base + p) * od + j];
        float gv = gt[((u64)base + p) * gate_stride + (u64)h * 2 * hd + hd + d];
        dst[idx] = av / (1.0f + expf(-gv));
    }
}

/* ---- OP_SSM_CONV -----------------------------------------------
 * Depthwise causal conv1d advancing the per-layer ring buffer one row
 * at a time.  Sequential over rows by construction, so the whole
 * batch runs in ONE block: rows are processed in order, channels in
 * parallel.  The state stores RAW pre-conv inputs (source rows), so
 * the conv reads the source directly instead of a saved copy. */
__global__ void k_ssm_conv(const float *__restrict__ src, float *__restrict__ dst,
                           const float *__restrict__ cw, float *__restrict__ state,
                           u32 od, u32 ck, u32 r, u32 base) {
    for (u32 p = 0; p < r; p++) {
        const float *sp = src + ((u64)base + p) * od;
        float *dp       = dst + (u64)p * od;
        for (u32 c = threadIdx.x; c < od; c += blockDim.x) {
            float sum = 0.0f;
            for (u32 k = 0; k < ck; k++) {
                float x = (k == ck - 1) ? sp[c] : state[(u64)k * od + c];
                sum += x * cw[(u64)c * ck + k];
            }
            dp[c] = sum;
        }
        __syncthreads();
        /* Shift the ring one slice at a time (in-place forward copy
         * would race across threads without the per-step barrier). */
        for (u32 s = 0; s + 1 < ck - 1; s++) {
            for (u32 c = threadIdx.x; c < od; c += blockDim.x)
                state[(u64)s * od + c] = state[(u64)(s + 1) * od + c];
            __syncthreads();
        }
        if (ck > 1) {
            u32 tail = (ck - 2) * od;
            for (u32 c = threadIdx.x; c < od; c += blockDim.x)
                state[(u64)tail + c] = sp[c];
        }
        __syncthreads();
    }
}

/* ---- OP_SSM_DELTA ----------------------------------------------
 * One block per value head, launched once per batch row (the state
 * advances sequentially).  Each block:
 *   1. computes its own decay/update gates from the row's alpha/beta,
 *   2. decays its [hd x hd] state slice,
 *   3. l2-normalizes its Q/K head slice in shared memory (GVA: value
 *      head grp reads key head grp % n_k; q carries the 1/sqrt(hd)
 *      scale, matching op_ssm_delta()),
 *   4. runs the delta recurrence column-parallel: thread j owns
 *      column j (mem[j] = S^T K, S += K delta^T, out = S^T Q),
 *   5. applies the per-head RMS norm and writes its output slice.
 * The fused source row is only read, never mutated. */
__global__ void k_ssm_delta(const float *__restrict__ fused_row,
                            const float *__restrict__ alpha_row,
                            const float *__restrict__ beta_row,
                            const float *__restrict__ a_log,
                            const float *__restrict__ dt_bias,
                            const float *__restrict__ nw,
                            float *__restrict__ state, float *__restrict__ dst_row,
                            u32 n_v, u32 n_k, u32 hd, u32 key_dim, float eps) {
    extern __shared__ float sm[];   /* q[hd] | k[hd] | v[hd] | red[32] */
    float *q = sm, *k = sm + hd, *v = sm + 2 * hd, *red = sm + 3 * hd;
    const u32 grp = blockIdx.x;
    const int tid = threadIdx.x;
    const u32 kh  = grp % n_k;
    const u64 d2  = (u64)hd * hd;
    float *Sg     = state + (u64)grp * d2;

    /* ssm.a already stores -exp(A_log): g = exp(a * softplus(alpha+dt)). */
    float g = expf(a_log[grp] * ((alpha_row[grp] + dt_bias[grp] > 20.0f)
                                     ? (alpha_row[grp] + dt_bias[grp])
                                     : log1pf(expf(alpha_row[grp] + dt_bias[grp]))));
    float b = 1.0f / (1.0f + expf(-beta_row[grp]));

    for (u64 i = tid; i < d2; i += blockDim.x) Sg[i] *= g;
    __syncthreads();

    const float *Qg = fused_row + (u64)kh * hd;
    const float *Kg = fused_row + key_dim + (u64)kh * hd;
    const float *Vg = fused_row + 2 * (u64)key_dim + (u64)grp * hd;

    float vq = 0.0f, vk = 0.0f, vv = 0.0f;
    if (tid < (int)hd) {
        vq = Qg[tid]; vk = Kg[tid]; vv = Vg[tid];
        q[tid] = vq; k[tid] = vk; v[tid] = vv;
    }
    __syncthreads();
    float ss = block_sum(vq * vq, red, tid, blockDim.x);
    vq *= 1.0f / sqrtf(ss + eps);
    ss = block_sum(vk * vk, red, tid, blockDim.x);
    vk *= 1.0f / sqrtf(ss + eps);
    vq *= 1.0f / sqrtf((float)hd);
    if (tid < (int)hd) { q[tid] = vq; k[tid] = vk; }
    __syncthreads();

    float out = 0.0f;
    if (tid < (int)hd) {
        float mem = 0.0f;
        for (u32 i = 0; i < hd; i++)
            mem += Sg[(u64)i * hd + tid] * k[i];
        float delta = (vv - mem) * b;
        for (u32 i = 0; i < hd; i++) {
            float *col = Sg + (u64)i * hd + tid;
            *col += k[i] * delta;
            out += *col * q[i];
        }
    }
    if (nw) {
        ss = block_sum((tid < (int)hd) ? out * out : 0.0f, red, tid, blockDim.x);
        if (tid < (int)hd)
            dst_row[(u64)grp * hd + tid] =
                out * (1.0f / sqrtf(ss / (float)hd + eps)) * nw[tid];
    } else if (tid < (int)hd) {
        dst_row[(u64)grp * hd + tid] = out;
    }
}

/* ---- OP_ATTN ---------------------------------------------------
 * Phase 1 — k_kv_write: copy the batch's K/V rows ([row][kv_dim])
 * into the head-major session cache ([head][pos][dim]).
 * Phase 2 — k_attn: one block per (head, query row); streaming
 * (tiled) softmax so the score tile lives in shared memory regardless
 * of context size: per tile of keys, dot products in parallel, block
 * max/sum reduction, accumulator rescale, weighted V accumulation. */
__global__ void k_kv_write(float *__restrict__ ck, float *__restrict__ cv,
                           const float *__restrict__ kd, const float *__restrict__ vd,
                           u32 n_kv, u32 khd, u32 kv_dim, u64 hs, u32 pos, u64 total) {
    u64 idx     = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    u64 gstride = (u64)gridDim.x * blockDim.x;
    for (; idx < total; idx += gstride) {
        u32 i   = (u32)(idx / ((u64)n_kv * khd));
        u32 rem = (u32)(idx % ((u64)n_kv * khd));
        u32 h = rem / khd, d = rem % khd;
        ck[(u64)h * hs + (u64)(pos + i) * khd + d] = kd[(u64)i * kv_dim + (u64)h * khd + d];
        cv[(u64)h * hs + (u64)(pos + i) * khd + d] = vd[(u64)i * kv_dim + (u64)h * khd + d];
    }
}

__global__ void k_attn(const float *__restrict__ qd, const float *__restrict__ ck,
                       const float *__restrict__ cv, float *__restrict__ dst,
                       u32 gqa, u32 hd, u32 khd, u32 q_dim, u64 hs,
                       u32 pos, float scale) {
    extern __shared__ float sm[];   /* acc[khd] | red[32] | p[T] | m | l */
    float *acc = sm, *red = sm + khd, *p = red + 32;
    float *msm = p + blockDim.x, *lsm = msm + 1;
    const u32 h = blockIdx.x, qi = blockIdx.y;
    const int tid = threadIdx.x;
    const float *qh = qd + (u64)qi * q_dim + (u64)h * hd;
    const float *Kb = ck + (u64)(h / gqa) * hs;
    const float *Vb = cv + (u64)(h / gqa) * hs;
    const u32 n_keys = pos + qi + 1;

    for (u32 d = tid; d < khd; d += blockDim.x) acc[d] = 0.0f;
    if (tid == 0) { *msm = op_neg_inf(); *lsm = 0.0f; }
    __syncthreads();

    for (u32 t0 = 0; t0 < n_keys; t0 += blockDim.x) {
        u32 tile = min(blockDim.x, n_keys - t0);
        float s = op_neg_inf();
        if ((u32)tid < tile) {
            const float *kt = Kb + (u64)(t0 + tid) * khd;
            float dot = 0.0f;
            for (u32 d = 0; d < khd; d++) dot += qh[d] * kt[d];
            s = dot * scale;
        }
        float tmax = block_max(s, red, tid, blockDim.x);
        float new_m = fmaxf(*msm, tmax);
        p[tid] = ((u32)tid < tile) ? expf(s - new_m) : 0.0f;
        float factor = expf(*msm - new_m);
        float tsum = block_sum(p[tid], red, tid, blockDim.x);
        if (tid == 0) { *lsm = *lsm * factor + tsum; *msm = new_m; }
        for (u32 d = tid; d < khd; d += blockDim.x) {
            float a = acc[d] * factor;
            for (u32 t2 = 0; t2 < tile; t2++)
                a += p[t2] * Vb[(u64)(t0 + t2) * khd + d];
            acc[d] = a;
        }
        __syncthreads();
    }
    float l = *lsm;
    for (u32 d = tid; d < khd; d += blockDim.x)
        dst[(u64)qi * q_dim + (u64)h * hd + d] = acc[d] / l;
}

/* ================================================================
 * Op entry points (mirror the op_* functions in graph.c)
 * ================================================================ */

static bool gpu_op_embed(GpuOpCtx *c) {
    TensorInfo *te = c->node->weights[0];
    bool te_trans  = (te->dim[0] == (i64)c->cfg->n_vocab);
    const u32 *tok = (const u32 *)c->g->node[(u32)c->node->src[0]].data;
    const u8 *raw  = gpu_weight_dev(te);
    if (!raw || !tok) return false;
    /* Row-major table: token t at [t*od, t*od+od); column-major:
     * token t at [t, t + (n_vocab-1)*n_vocab]. */
    u64 base_mul = te_trans ? (u64)c->od : 1;
    u64 stride   = te_trans ? 1 : (u64)c->cfg->n_vocab;
    return gpu_dequant_gather_dev(te->type, raw, tok, c->n, c->od,
                                  base_mul, stride, c->dst) == 0;
}

static bool gpu_op_rms_norm(GpuOpCtx *c) {
    TensorInfo *tw = c->node->weights[0];
    const float *w = gpu_op_weight_f32(tw);
    if (!w) return false;
    u32 wn = (u32)tw->n_element;
    k_rms_norm<<<c->r, GPU_OP_THREADS>>>(op_src(c, 0), c->dst, w, wn, c->od,
                                         c->base, DEFAULT_EPS);
    CHECK(cudaGetLastError());
    return true;
}

static bool gpu_op_rms_norm_heads(GpuOpCtx *c) {
    u32 nh = op_param(c, 0), hd = op_param(c, 1);
    u32 is = op_param(c, 2), os = op_param(c, 3);
    const float *nw = gpu_op_weight_f32(c->node->weights[0]);
    if (!nw) return false;
    dim3 grid(nh, c->r);
    k_rms_norm_heads<<<grid, GPU_OP_THREADS>>>(op_src(c, 0), c->dst, nw,
                                               nh, hd, is, os, c->od,
                                               c->base, DEFAULT_EPS);
    CHECK(cudaGetLastError());
    return true;
}

static bool gpu_op_matmul(GpuOpCtx *c) {
    TensorInfo *w = c->node->weights[0];
    bool tr       = (c->node->op == OP_MATMUL_T);
    u64 rows      = tr ? w->dim[1] : w->dim[0];
    u64 cols      = tr ? w->dim[0] : w->dim[1];
    const float *x = op_src(c, 0) + (u64)c->base * cols;
    return gpu_matmul_dev(w, x, c->dst, c->r, rows, cols, tr) == 0;
}

static bool gpu_op_bias(GpuOpCtx *c) {
    TensorInfo *tb = c->node->weights[0];
    const float *bw = gpu_op_weight_f32(tb);
    if (!bw) return false;
    u32 nb = tb->n_element < (u64)c->od ? (u32)tb->n_element : c->od;
    k_bias<<<op_block_count((u64)c->r * c->od), GPU_OP_THREADS>>>(
        op_src(c, 0), c->dst, bw, nb, c->od, c->base, (u64)c->r * c->od);
    CHECK(cudaGetLastError());
    return true;
}

static bool gpu_op_binary(GpuOpCtx *c) {
    bool add = (c->node->op == OP_ADD);
    k_binary<<<op_block_count((u64)c->r * c->od), GPU_OP_THREADS>>>(
        op_src(c, 0), op_src(c, 1), c->dst, add ? 1 : 0,
        c->od, c->base, (u64)c->r * c->od);
    CHECK(cudaGetLastError());
    return true;
}

static bool gpu_op_silu(GpuOpCtx *c) {
    k_silu<<<op_block_count((u64)c->r * c->od), GPU_OP_THREADS>>>(
        op_src(c, 0), c->dst, c->od, c->base, (u64)c->r * c->od);
    CHECK(cudaGetLastError());
    return true;
}

static bool gpu_op_softmax(GpuOpCtx *c) {
    k_softmax<<<c->r, GPU_OP_THREADS>>>(op_src(c, 0), c->dst, c->od, c->base);
    CHECK(cudaGetLastError());
    return true;
}

static bool gpu_op_rope_neox(GpuOpCtx *c) {
    u32 bits = op_param(c, 0);
    float theta;
    memcpy(&theta, &bits, sizeof(theta));
    u32 heads = op_param(c, 1);
    u32 hdim  = op_param(c, 2);
    u32 rdim  = op_param(c, 3);
    if (c->od != heads * hdim) return false;
    k_rope_neox<<<op_block_count((u64)c->r * heads * (hdim / 2)), GPU_OP_THREADS>>>(
        op_src(c, 0), c->dst, theta, heads, hdim, rdim, c->od, c->base, c->pos,
        (u64)c->r * heads * (hdim / 2));
    CHECK(cudaGetLastError());
    return true;
}

static bool gpu_op_sigmoid_gate(GpuOpCtx *c) {
    u32 nh = op_param(c, 0), hd = op_param(c, 1);
    if (c->od != nh * hd) return false;
    k_sigmoid_gate<<<op_block_count((u64)c->r * c->od), GPU_OP_THREADS>>>(
        op_src(c, 0), op_src(c, 1), c->dst, hd, c->od,
        (u64)nh * 2 * hd, c->base, (u64)c->r * c->od);
    CHECK(cudaGetLastError());
    return true;
}

static bool gpu_op_ssm_conv(GpuOpCtx *c) {
    u32 st = op_param(c, 0), ck = op_param(c, 1);
    if (st >= c->g->n_state) return false;
    float *state = (float *)c->g->state[st];
    const float *cw = gpu_op_weight_f32(c->node->weights[0]);
    if (!state || !cw) return false;
    /* Sequential over rows: one block owns the whole batch. */
    k_ssm_conv<<<1, GPU_OP_THREADS>>>(op_src(c, 0), c->dst, cw, state,
                                      c->od, ck, c->r, c->base);
    CHECK(cudaGetLastError());
    return true;
}

static bool gpu_op_ssm_delta(GpuOpCtx *c) {
    u32 st = op_param(c, 0), n_v = op_param(c, 1);
    u32 n_k = op_param(c, 2), hd = op_param(c, 3);
    if (st >= c->g->n_state) return false;
    u32 key_dim = n_k * hd;
    u64 val_dim = (u64)n_v * hd;
    if (c->od != (u32)val_dim) return false;
    float *state = (float *)c->g->state[st];
    const float *a_log = gpu_op_weight_f32(c->node->weights[0]);
    const float *dt_bi = gpu_op_weight_f32(c->node->weights[1]);
    const float *nw = c->node->weights[2] ? gpu_op_weight_f32(c->node->weights[2])
                                          : NULL;
    if (!state || !a_log || !dt_bi || (c->node->weights[2] && !nw)) return false;

    u32 bd = (hd + 31) & ~31u;
    if (bd > 1024 || hd == 0) return false;
    size_t shmem = (3 * (size_t)hd + 32) * sizeof(float);
    u64 fstride = 2 * (u64)key_dim + val_dim;
    const float *fused = op_src(c, 0);
    const float *alpha = op_src(c, 1);
    const float *beta  = op_src(c, 2);
    for (u32 p = 0; p < c->r; p++) {
        k_ssm_delta<<<n_v, bd, shmem>>>(
            fused + ((u64)c->base + p) * fstride,
            alpha + ((u64)c->base + p) * n_v,
            beta  + ((u64)c->base + p) * n_v,
            a_log, dt_bi, nw, state,
            c->dst + (u64)p * c->od,
            n_v, n_k, hd, key_dim, DEFAULT_EPS);
        CHECK(cudaGetLastError());
    }
    return true;
}

static bool gpu_op_attn(GpuOpCtx *c) {
    if (!c->s->cache.std) {
        slog(WARN, (char *)"gpu_op_attn: OP_ATTN requires the std KV cache");
        return false;
    }
    u32 layer = op_param(c, 0);
    AttnKvCache *akc = &c->s->cache.std[layer];
    if (!akc->k || !akc->v) return false;

    u32 n_head = c->cfg->n_head;
    u32 n_kv   = c->cfg->n_kv_head;
    u32 khd    = c->cfg->kv_head_dim;
    u32 gqa    = n_head / n_kv;
    u64 hs     = (u64)akc->cap * khd;

    /* K/V write-through, then causal attention over the cache rows
     * 0..pos+qi (history included).  Mirrors op_attn(). */
    u64 total = (u64)c->n * n_kv * khd;
    k_kv_write<<<op_block_count(total), GPU_OP_THREADS>>>(
        akc->k, akc->v, op_src(c, 1), op_src(c, 2),
        n_kv, khd, c->kv_dim, hs, c->pos, total);
    CHECK(cudaGetLastError());
    akc->n = c->pos + c->n;

    dim3 grid(n_head, c->n);
    size_t shmem = ((size_t)khd + 32 + GPU_OP_THREADS + 2) * sizeof(float);
    k_attn<<<grid, GPU_OP_THREADS, shmem>>>(
        op_src(c, 0), akc->k, akc->v, c->dst,
        gqa, c->cfg->head_dim, khd, c->q_dim, hs, c->pos, c->scale);
    CHECK(cudaGetLastError());
    (void)c->scr;
    return true;
}

/* ================================================================
 * Dispatch — the GPU twin of graph.c's op_table
 * ================================================================ */

bool gpu_graph_op(GpuOpCtx *c) {
    if (!c || !c->g || !c->s || !c->cfg || !c->node || !c->dst || c->od == 0)
        return false;
    switch (c->node->op) {
        case OP_EMBED:          return gpu_op_embed(c);
        case OP_RMS_NORM:       return gpu_op_rms_norm(c);
        case OP_RMS_NORM_HEADS: return gpu_op_rms_norm_heads(c);
        case OP_MATMUL:
        case OP_MATMULARRY:
        case OP_MATMUL_T:       return gpu_op_matmul(c);
        case OP_BIAS:           return gpu_op_bias(c);
        case OP_ADD:
        case OP_MUL:            return gpu_op_binary(c);
        case OP_SILU:           return gpu_op_silu(c);
        case OP_SOFTMAX:        return gpu_op_softmax(c);
        case OP_ROPE_NEOX:      return gpu_op_rope_neox(c);
        case OP_SIGMOID_GATE:   return gpu_op_sigmoid_gate(c);
        case OP_SSM_CONV:       return gpu_op_ssm_conv(c);
        case OP_SSM_DELTA:      return gpu_op_ssm_delta(c);
        case OP_ATTN:           return gpu_op_attn(c);
        /* OP_INPUT has nothing to compute; OP_H2D / OP_D2H stay with
         * the executor. */
        default:                return false;
    }
}
