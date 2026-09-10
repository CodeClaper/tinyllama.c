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
#include "cpu_op.h"

/* Row-major source tensor of edge `k`. */
static float *op_src(const OpCtx *c, int k) {
    return (float *)c->g->node[(u32)c->node->src[k]].data;
}

/* Op-specific parameter `k` of the node being executed. */
static u32 op_param(const OpCtx *c, int k) {
    return c->node->params[k];
}

static bool op_embed(OpCtx *c) {
    TensorInfo *te = c->node->weights[0];
    bool te_trans  = (te->dim[0] == (i64)c->cfg->n_vocab);
    u32 *tok = (u32 *)c->g->node[(u32)c->node->src[0]].data;

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

static bool op_silu(OpCtx *c)    { return op_unary(c, silu); }
static bool op_softmax(OpCtx *c) { return op_unary(c, softmax_n); }

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

            float *kt = kh_base;
            for (u32 t = 0; t < n_keys; t++, kt += khd) {
                float acc = 0.0f;
                for (u32 d = 0; d < khd; d++)
                    acc += qh[d] * kt[d];
                c->scr[t] = acc * c->scale;
            }
            softmax(c->scr, n_keys);

            float *oh = c->dst + (u64)qi * c->q_dim + (u64)h * hd;
            float *vt = vh_base;
            for (u32 t = 0; t < n_keys; t++, vt += khd) {
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
