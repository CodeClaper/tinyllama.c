/*
 * test_tpool.c — Unit tests for the thread pool
 *
 * minunit: http://www.jera.com/techinfo/jtns/jtn002.html
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../src/tpool.h"

/* ------------------------------------------------------------------
 * minunit
 * ------------------------------------------------------------------ */
#define mu_assert(test)           do { if (!(test)) return #test; } while (0)
#define mu_assert_msg(test, msg)  do { if (!(test)) { static char buf_[256]; snprintf(buf_, sizeof(buf_), "%s", msg); return buf_; } } while (0)
#define mu_run_test(test)         do {                                    \
        char *msg__ = test();                                            \
        tests_run__++;                                                   \
        if (msg__) {                                                     \
            printf("  FAIL: %s\n  -> %s\n", #test, msg__);               \
            tests_fail__++;                                              \
        } else {                                                         \
            printf("  PASS: %s\n", #test);                               \
        }                                                                \
    } while (0)

static int tests_run__   = 0;
static int tests_fail__  = 0;

/* ------------------------------------------------------------------
 * Helpers
 * ------------------------------------------------------------------ */

/* Count how many times each iteration index is visited, and which
 * thread IDs visit which indices. */
typedef struct {
    int    n;                   /* total iterations */
    int   *counts;              /* count[i] = how many times i was executed */
    int   *thread_ids;          /* thread_ids[i] = which thread did it (last) */
    int    max_thread_id;       /* max thread_id observed */
} parfor_state;

static void parfor_record(void *arg, int tid, int i) {
    parfor_state *s = (parfor_state *)arg;
    if (i >= 0 && i < s->n) {
        s->counts[i]++;
        s->thread_ids[i] = tid;
    }
    if (tid > s->max_thread_id) s->max_thread_id = tid;
}

static parfor_state *parfor_new(int n) {
    parfor_state *s = calloc(1, sizeof(parfor_state));
    s->n      = n;
    s->counts = calloc(n, sizeof(int));
    s->thread_ids = calloc(n, sizeof(int));
    for (int i = 0; i < n; i++) s->thread_ids[i] = -1;
    s->max_thread_id = -1;
    return s;
}

static void parfor_free(parfor_state *s) {
    free(s->counts);
    free(s->thread_ids);
    free(s);
}

/* ------------------------------------------------------------------
 * Tests
 * ------------------------------------------------------------------ */

static char *test_create_destroy(void) {
    tpool_t *pool = tpool_create(4);
    mu_assert(pool != NULL);
    mu_assert(pool->nthreads == 4);
    tpool_destroy(pool);
    return NULL;
}

static char *test_create_one(void) {
    tpool_t *pool = tpool_create(1);
    mu_assert(pool != NULL);
    mu_assert(pool->nthreads == 1);
    tpool_destroy(pool);
    return NULL;
}

static char *test_create_zero(void) {
    tpool_t *pool = tpool_create(0);
    mu_assert(pool != NULL);
    mu_assert(pool->nthreads == 1);
    tpool_destroy(pool);
    return NULL;
}

static char *test_empty_range(void) {
    tpool_t *pool = tpool_create(2);
    parfor_state *s = parfor_new(100);
    /* start == end → no work; should not crash */
    tpool_parallel_for(pool, 5, 5, parfor_record, s);
    for (int i = 0; i < 100; i++)
        mu_assert(s->counts[i] == 0);
    parfor_free(s);
    tpool_destroy(pool);
    return NULL;
}

static char *test_each_iteration_exactly_once(void) {
    const int N = 9973;  /* prime, not aligned to typical chunk sizes */
    tpool_t *pool = tpool_create(4);
    parfor_state *s = parfor_new(N);

    tpool_parallel_for(pool, 0, N, parfor_record, s);

    for (int i = 0; i < N; i++)
        mu_assert(s->counts[i] == 1);

    parfor_free(s);
    tpool_destroy(pool);
    return NULL;
}

static char *test_single_thread(void) {
    const int N = 1000;
    tpool_t *pool = tpool_create(1);
    parfor_state *s = parfor_new(N);

    tpool_parallel_for(pool, 0, N, parfor_record, s);

    for (int i = 0; i < N; i++) {
        mu_assert(s->counts[i] == 1);
        mu_assert(s->thread_ids[i] >= 0);
    }
    mu_assert(s->max_thread_id == 1);  /* main = nthreads */

    parfor_free(s);
    tpool_destroy(pool);
    return NULL;
}

static char *test_all_threads_participate(void) {
    const int N = 5000;
    const int T = 4;
    tpool_t *pool = tpool_create(T);
    parfor_state *s = parfor_new(N);

    tpool_parallel_for(pool, 0, N, parfor_record, s);

    /* Every thread including main should have processed at least one iteration. */
    for (int tid = 0; tid <= T; tid++) {
        int found = 0;
        for (int i = 0; i < N; i++)
            if (s->thread_ids[i] == tid) { found = 1; break; }
        mu_assert(found);
    }

    parfor_free(s);
    tpool_destroy(pool);
    return NULL;
}

static char *test_destroy_null(void) {
    tpool_destroy(NULL);
    return NULL;
}

/* ------------------------------------------------------------------
 * Run all
 * ------------------------------------------------------------------ */

int main(void) {
    printf("tpool tests\n");
    printf("-----------\n");

    mu_run_test(test_create_destroy);
    mu_run_test(test_create_one);
    mu_run_test(test_create_zero);
    mu_run_test(test_empty_range);
    mu_run_test(test_each_iteration_exactly_once);
    mu_run_test(test_single_thread);
    mu_run_test(test_all_threads_participate);
    mu_run_test(test_destroy_null);

    printf("\n%d / %d tests passed", tests_run__ - tests_fail__, tests_run__);
    if (tests_fail__) printf("  (%d FAILED)", tests_fail__);
    printf("\n");

    return tests_fail__ ? 1 : 0;
}
