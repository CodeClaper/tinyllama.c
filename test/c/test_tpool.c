#include <stdio.h>
#include <stdlib.h>
#include "minunit.h"
#include "../../src/tpool.h"
#include "../../src/mm.h"

typedef struct {
    int    n;
    int   *counts;
    int   *thread_ids;
    int    max_thread_id;
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
    parfor_state *s = scalloc(1, sizeof(parfor_state));
    s->n            = n;
    s->counts       = scalloc(n, sizeof(int));
    s->thread_ids   = scalloc(n, sizeof(int));
    for (int i = 0; i < n; i++) s->thread_ids[i] = -1;
    s->max_thread_id = -1;
    return s;
}

static void parfor_free(parfor_state *s) {
    sfree(s->counts);
    sfree(s->thread_ids);
    sfree(s);
}

MU_TEST(test_create_destroy) {
    tpool_t *pool = tpool_create(4);
    mu_check(pool != NULL);
    mu_assert_int_eq(4, pool->nthreads);
    tpool_destroy(pool);
}

MU_TEST(test_create_one) {
    tpool_t *pool = tpool_create(1);
    mu_check(pool != NULL);
    mu_assert_int_eq(1, pool->nthreads);
    tpool_destroy(pool);
}

MU_TEST(test_create_zero) {
    tpool_t *pool = tpool_create(0);
    mu_check(pool != NULL);
    mu_assert_int_eq(1, pool->nthreads);
    tpool_destroy(pool);
}

MU_TEST(test_empty_range) {
    tpool_t *pool = tpool_create(2);
    parfor_state *s = parfor_new(100);
    tpool_parallel_for(pool, 5, 5, parfor_record, s);
    for (int i = 0; i < 100; i++)
        mu_assert_int_eq(0, s->counts[i]);
    parfor_free(s);
    tpool_destroy(pool);
}

MU_TEST(test_each_iteration_exactly_once) {
    const int N = 9973;
    tpool_t *pool = tpool_create(4);
    parfor_state *s = parfor_new(N);

    tpool_parallel_for(pool, 0, N, parfor_record, s);

    for (int i = 0; i < N; i++)
        mu_assert_int_eq(1, s->counts[i]);

    parfor_free(s);
    tpool_destroy(pool);
}

MU_TEST(test_single_thread) {
    const int N = 1000;
    tpool_t *pool = tpool_create(1);
    parfor_state *s = parfor_new(N);

    tpool_parallel_for(pool, 0, N, parfor_record, s);

    for (int i = 0; i < N; i++) {
        mu_assert_int_eq(1, s->counts[i]);
        mu_check(s->thread_ids[i] >= 0);
    }
    mu_assert_int_eq(1, s->max_thread_id);

    parfor_free(s);
    tpool_destroy(pool);
}

MU_TEST(test_all_threads_participate) {
    const int N = 5000;
    const int T = 4;
    tpool_t *pool = tpool_create(T);
    parfor_state *s = parfor_new(N);

    tpool_parallel_for(pool, 0, N, parfor_record, s);

    for (int tid = 0; tid <= T; tid++) {
        int found = 0;
        for (int i = 0; i < N; i++)
            if (s->thread_ids[i] == tid) { found = 1; break; }
        mu_check(found);
    }

    parfor_free(s);
    tpool_destroy(pool);
}

MU_TEST(test_destroy_null) {
    tpool_destroy(NULL);
}

int main(void) {
    printf("tpool tests\n");
    printf("-----------\n");

    MU_RUN_TEST(test_create_destroy);
    MU_RUN_TEST(test_create_one);
    MU_RUN_TEST(test_create_zero);
    MU_RUN_TEST(test_empty_range);
    MU_RUN_TEST(test_each_iteration_exactly_once);
    MU_RUN_TEST(test_single_thread);
    MU_RUN_TEST(test_all_threads_participate);
    MU_RUN_TEST(test_destroy_null);

    MU_REPORT();
    return MU_EXIT_CODE;
}
