#include <stdio.h>
#include "minunit.h"
#include "../../src/pthreads.h"
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
    pthreads_t *pool = pthreads_create(4);
    mu_check(pool != NULL);
    mu_assert_int_eq(4, pool->nthreads);
    pthreads_destroy(pool);
}

MU_TEST(test_create_one) {
    pthreads_t *pool = pthreads_create(1);
    mu_check(pool != NULL);
    mu_assert_int_eq(1, pool->nthreads);
    pthreads_destroy(pool);
}

MU_TEST(test_create_zero) {
    pthreads_t *pool = pthreads_create(0);
    mu_check(pool != NULL);
    mu_assert_int_eq(1, pool->nthreads);
    pthreads_destroy(pool);
}

MU_TEST(test_empty_range) {
    pthreads_t *pool = pthreads_create(2);
    parfor_state *s = parfor_new(100);
    pthreads_parallel_for(pool, 5, 5, parfor_record, s);
    for (int i = 0; i < 100; i++)
        mu_assert_int_eq(0, s->counts[i]);
    parfor_free(s);
    pthreads_destroy(pool);
}

MU_TEST(test_each_iteration_exactly_once) {
    const int N = 9973;
    pthreads_t *pool = pthreads_create(4);
    parfor_state *s = parfor_new(N);

    pthreads_parallel_for(pool, 0, N, parfor_record, s);

    for (int i = 0; i < N; i++)
        mu_assert_int_eq(1, s->counts[i]);

    parfor_free(s);
    pthreads_destroy(pool);
}

MU_TEST(test_single_thread) {
    const int N = 1000;
    pthreads_t *pool = pthreads_create(1);
    parfor_state *s = parfor_new(N);

    pthreads_parallel_for(pool, 0, N, parfor_record, s);

    for (int i = 0; i < N; i++) {
        mu_assert_int_eq(1, s->counts[i]);
        mu_check(s->thread_ids[i] >= 0);
    }
    mu_assert_int_eq(0, s->max_thread_id);

    parfor_free(s);
    pthreads_destroy(pool);
}

MU_TEST(test_destroy_null) {
    pthreads_destroy(NULL);
}

typedef struct {
    volatile int acc;
    int total;
} IncCtx;

/* Regression test for the SIGSEGV seen when mat_vec_mul dispatched
 * through the pool with a stack-local ctx.  Before the generation-
 * counter fix, a slow worker could still be reading the (already
 * torn-down) caller's stack when pthreads_parallel_for returned,
 * because other workers bumped done_count past nthreads racing the
 * dispatcher.  This test reproduces the pattern at high repetition
 * with a deliberately stack-local argument and a long-running work
 * callback; without the fix it crashes with a use-after-free within
 * a handful of iterations. */
static void slow_inc(void *arg, int tid, int i) {
    (void)tid; (void)i;
    IncCtx *ctx = (IncCtx *)arg;
    __atomic_fetch_add(&ctx->acc, 1, __ATOMIC_SEQ_CST);
    /* Burn cycles to widen the race window for a straggling worker. */
    for (volatile int v = 0; v < 200; v++) { /* nop */ }
    mu_check(ctx->acc <= 100);
}

MU_TEST(test_rapid_reentrant_dispatch_stack_arg) {
    const int N = 100;
    const int T = 4;
    const int ITERS = 50;
    pthreads_t *pool = pthreads_create(T);
    long total = 0;
    for (int iter = 0; iter < ITERS; iter++) {
        IncCtx ctx = {.acc = 0, .total=N };
        /* stack-local, torn down on loop end */
        pthreads_parallel_for(pool, 0, N, slow_inc, &ctx);
        mu_assert_int_eq(N, ctx.acc);
        total += ctx.acc;
    }
    mu_assert_int_eq((long)ITERS * N, total);
    pthreads_destroy(pool);
}


MU_TEST(test_rapid_reentrant_dispatch_stack_sig) {
    const int N = 100;
    const int T = 4;
    const int ITERS = 50;
    pthreads_t *pool = pthreads_create(T);
    for (int iter = 0; iter < ITERS; iter++) {
        IncCtx ctx = {.acc = 0, .total=N };
        /* stack-local, torn down on loop end */
        pthreads_parallel_for(pool, 0, N, slow_inc, &ctx);
    }
    pthreads_destroy(pool);
}

int main(void) {
    printf("pthreads tests\n");
    printf("--------------\n");

    MU_RUN_TEST(test_create_destroy);
    MU_RUN_TEST(test_create_one);
    MU_RUN_TEST(test_create_zero);
    MU_RUN_TEST(test_empty_range);
    MU_RUN_TEST(test_each_iteration_exactly_once);
    MU_RUN_TEST(test_single_thread);
    MU_RUN_TEST(test_destroy_null);
    MU_RUN_TEST(test_rapid_reentrant_dispatch_stack_arg);
    MU_RUN_TEST(test_rapid_reentrant_dispatch_stack_sig);

    MU_REPORT();
    return MU_EXIT_CODE;
}
