#include <stdlib.h>
#include <math.h>
#include "def.h"
#include "sampler.h"

typedef struct {
    float prob;
    u32   idx;
} ProbIdx;

static int probidx_cmp(const void *a, const void *b) {
    float pa = ((const ProbIdx *)a)->prob;
    float pb = ((const ProbIdx *)b)->prob;
    if (pa > pb) return -1;
    if (pa < pb) return 1;
    return 0;
}

u32 sample_greedy(float *logits, u32 n_vocab) {
    u32 best = 0;
    float best_val = logits[0];
    for (u32 i = 1; i < n_vocab; i++) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best = i;
        }
    }
    return best;
}

/* Temperature + top-p (nucleus) sampling.
 * Modifies logits in-place (caller re-fills before next use).
 * Falls back to greedy when temperature <= 0 or top_p >= 1. */
u32 sample_token(float *logits, u32 n_vocab,
                  float temperature, float top_p) {
    if (temperature <= 1e-6f || top_p >= 1.0f)
        return sample_greedy(logits, n_vocab);

    /* Allocate sort buffer before mutating logits. */
    ProbIdx *pi = malloc(n_vocab * sizeof(ProbIdx));
    if (!pi) return sample_greedy(logits, n_vocab);

    /* Temperature scaling. */
    float inv_temp = 1.0f / temperature;
    float max_l = logits[0] * inv_temp;
    for (u32 i = 1; i < n_vocab; i++) {
        float v = logits[i] * inv_temp;
        if (v > max_l) max_l = v;
    }

    /* Softmax. */
    float sum = 0.0f;
    for (u32 i = 0; i < n_vocab; i++) {
        float v = expf(logits[i] * inv_temp - max_l);
        logits[i] = v;      /* reuse logits[] for probabilities */
        sum += v;
    }
    float inv_sum = 1.0f / sum;

    /* Fill (prob, idx) pairs. */
    for (u32 i = 0; i < n_vocab; i++) {
        pi[i].prob = logits[i] * inv_sum;
        pi[i].idx  = i;
    }

    /* Sort descending by probability. */
    qsort(pi, n_vocab, sizeof(ProbIdx), probidx_cmp);

    /* Find top-p nucleus. */
    float cum = 0.0f;
    u32 cutoff = n_vocab;
    for (u32 i = 0; i < n_vocab; i++) {
        cum += pi[i].prob;
        if (cum >= top_p) { cutoff = i + 1; break; }
    }
    if (cutoff < 1) cutoff = 1;

    /* Renormalise truncated distribution. */
    float renorm = 0.0f;
    for (u32 i = 0; i < cutoff; i++) renorm += pi[i].prob;
    float rn_inv = renorm > 0.0f ? 1.0f / renorm : 1.0f;

    /* Sample. */
    float r = (float)rand() / (float)RAND_MAX;
    float cdf = 0.0f;
    for (u32 i = 0; i < cutoff; i++) {
        cdf += pi[i].prob * rn_inv;
        if (r < cdf) {
            u32 result = pi[i].idx;
            free(pi);
            return result;
        }
    }

    /* Fallback (floating-point rounding). */
    u32 result = pi[cutoff - 1].idx;
    free(pi);
    return result;
}
