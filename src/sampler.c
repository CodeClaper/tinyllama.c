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

/* Bounded top-`cap` selection out of `vals[0..n)`.
 *
 * A min-heap of `cap` entries: one comparison per element plus a
 * sift-down only when an element beats the heap's minimum (~cap*ln(n/cap)
 * times), so the scan is O(n) for small cap.  Fills `heap[0..min(cap,n))`
 * in no particular order; the caller sorts. */
static u32 select_top(ProbIdx *heap, u32 cap, const float *vals, u32 n) {
    u32 filled = 0;
    for (u32 i = 0; i < n; i++) {
        float v = vals[i];
        if (filled < cap) {
            u32 c = filled++;
            heap[c].prob = v;
            heap[c].idx  = i;
            while (c > 0) {                        /* sift up */
                u32 p = (c - 1) >> 1;
                if (heap[p].prob <= heap[c].prob) break;
                ProbIdx t = heap[p]; heap[p] = heap[c]; heap[c] = t;
                c = p;
            }
        } else if (v > heap[0].prob) {
            heap[0].prob = v;
            heap[0].idx  = i;
            u32 p = 0;                             /* sift down */
            for (;;) {
                u32 l = 2 * p + 1, r = l + 1, m = p;
                if (l < cap && heap[l].prob < heap[m].prob) m = l;
                if (r < cap && heap[r].prob < heap[m].prob) m = r;
                if (m == p) break;
                ProbIdx t = heap[p]; heap[p] = heap[m]; heap[m] = t;
                p = m;
            }
        }
    }
    return filled;
}

/* Sort `n` entries descending.  Insertion sort for the small case (the
 * top-k path never exceeds a few dozen), qsort beyond that so the
 * grown-cap fallback stays O(n log n). */
static void sort_desc(ProbIdx *a, u32 n) {
    if (n > 64) {
        qsort(a, n, sizeof(ProbIdx), probidx_cmp);
        return;
    }
    for (u32 i = 1; i < n; i++) {
        ProbIdx v = a[i];
        u32 j = i;
        while (j > 0 && a[j - 1].prob < v.prob) { a[j] = a[j - 1]; j--; }
        a[j] = v;
    }
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

/* Temperature + top-k + top-p + min-p sampling.
 * Modifies logits in-place (caller re-fills before next use).
 * Falls back to greedy when no stochastic filter is active. */
u32 sample_token(float *logits, u32 n_vocab,
                  float temperature, u32 top_k, float top_p, float min_p,
                  float repeat_penalty, u32 repeat_last_n,
                  float frequency_penalty, float presence_penalty,
                  u32 *history_tokens, u32 n_history) {

    /* ---- Repeat penalty: matches llama.cpp's token_count approach.
     * For each unique token in the recent history window, apply penalty ONCE:
     *   if logit <= 0: logit *= repeat_penalty
     *   else:          logit /= repeat_penalty
     *   then:          logit -= count * frequency_penalty + (count > 0) * presence_penalty
     * Applied on logits before temperature scaling. ---- */
    if (repeat_penalty != 1.0f || frequency_penalty != 0.0f || presence_penalty != 0.0f) {
        if (repeat_last_n > 0 && history_tokens && n_history > 0) {
            u32 start = (n_history > repeat_last_n) ? (n_history - repeat_last_n) : 0;
            for (u32 j = start; j < n_history; j++) {
                u32 tid = history_tokens[j];
                if (tid >= n_vocab) continue;
                /* Skip if already processed this unique token */
                bool already_seen = false;
                for (u32 k = start; k < j; k++) {
                    if (history_tokens[k] == tid) {
                        already_seen = true;
                        break;
                    }
                }
                if (already_seen) continue;
                /* Count occurrences in window */
                int count = 1;
                for (u32 k = j + 1; k < n_history; k++) {
                    if (history_tokens[k] == tid) count++;
                }
                /* Apply penalty (once per unique token) */
                if (logits[tid] <= 0.0f)
                    logits[tid] *= repeat_penalty;
                else
                    logits[tid] /= repeat_penalty;
                logits[tid] -= (float)count * frequency_penalty + (float)(count > 0) * presence_penalty;
            }
        }
    }

    if (temperature <= DEFAULT_EPS)
        return sample_greedy(logits, n_vocab);

    bool no_filter = (top_k == 0) && (top_p >= 1.0f) && (min_p <= DEFAULT_EPS);
    if (no_filter)
        return sample_greedy(logits, n_vocab);

    /* Temperature scaling + softmax.  expf() runs over the whole
     * vocabulary because top-p needs the true normaliser; the survivors
     * are divided by inv_sum afterwards, only for the cap entries. */
    float inv_temp = 1.0f / temperature;
    float max_l = logits[0] * inv_temp;
    for (u32 i = 1; i < n_vocab; i++) {
        float v = logits[i] * inv_temp;
        if (v > max_l) max_l = v;
    }

    float sum = 0.0f;
    for (u32 i = 0; i < n_vocab; i++) {
        float v = expf(logits[i] * inv_temp - max_l);
        logits[i] = v;      /* reuse logits[] for the selection scan */
        sum += v;
    }
    float inv_sum = 1.0f / sum;

    /* Select only the candidates the filters can reach instead of
     * sorting the whole vocabulary.
     *
     * Every filter below inspects a descending prefix, and top-k caps
     * that prefix at `top_k`, so a full qsort over n_vocab entries buys
     * nothing: at 248k tokens it costs ~25 ms of the ~48 ms a token
     * takes, for ~4.5M comparisons and a 2 MB scratch allocation.  The
     * bounded heap needs one comparison per element and cap*8 bytes.
     *
     * With top_k == 0 no such bound exists, so the cap starts small and
     * is quadrupled until the filters are resolved inside it (or the
     * whole vocabulary has been selected) — exact, and still cheap
     * because a peaked distribution resolves at the first cap. */
    u32 cap = top_k > 0 ? top_k : 64;
    if (cap > n_vocab) cap = n_vocab;
    ProbIdx *pi = malloc((u64)cap * sizeof(ProbIdx));
    if (!pi) return sample_greedy(logits, n_vocab);

    u32 filled = 0, cutoff = 0;
    for (;;) {
        filled = select_top(pi, cap, logits, n_vocab);
        sort_desc(pi, filled);
        for (u32 i = 0; i < filled; i++)
            pi[i].prob *= inv_sum;     /* inv_sum only touches survivors */

        /* --- top-k: keep only the k most probable tokens --- */
        cutoff = filled;

        /* --- min-p: keep tokens with prob >= min_p * max_prob --- */
        if (min_p > DEFAULT_EPS && cutoff > 0) {
            float threshold = min_p * pi[0].prob;
            u32 new_cutoff = 0;
            for (u32 i = 0; i < cutoff; i++) {
                if (pi[i].prob >= threshold)
                    new_cutoff = i + 1;
                else
                    break;
            }
            if (new_cutoff > 0) cutoff = new_cutoff;
        }

        /* --- top-p (nucleus): smallest set with cumulative prob >= top_p --- */
        if (top_p < 1.0f) {
            float cum = 0.0f;
            u32 p_cutoff = cutoff;
            for (u32 i = 0; i < cutoff; i++) {
                cum += pi[i].prob;
                if (cum >= top_p) { p_cutoff = i + 1; break; }
            }
            if (p_cutoff < cutoff) cutoff = p_cutoff;
        }

        /* top-k already bounds the prefix, so one pass is always enough.
         * Otherwise the filters may want more than `cap` provided. */
        if (top_k > 0) break;
        if (cutoff < filled || cap >= n_vocab) break;
        u32 ncap = cap > n_vocab / 4 ? n_vocab : cap * 4;
        ProbIdx *np = realloc(pi, (u64)ncap * sizeof(ProbIdx));
        if (!np) break;
        pi = np;
        cap = ncap;
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
