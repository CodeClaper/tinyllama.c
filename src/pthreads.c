#include <stdlib.h>
#include "pthreads.h"
#include "mm.h"
#include "slog.h"
#include "utils.h"

/* ---- Worker loop --------------------------------------------------- */

/* Each thread runs this loop.  It sleeps until work is dispatched via
 * pthreads_parallel_for, then atomically claims loop indices from
 * pool->current until pool->end is reached.  When no work remains the
 * thread signals completion via done_count and returns to sleep.
 *
 * The initial lock acquisition (before the FOREVER) is intentional:
 * pthread_cond_wait requires the mutex to be locked on entry. */
static void *worker_loop(void *arg) {
    pthreads_t *pool = (pthreads_t *)arg;
    int tid;

    /* Resolve this thread's index within the pool. */
    pthread_mutex_lock(&pool->lock);
    for (tid = 0; tid < pool->nthreads; tid++)
        if (pthread_equal(pthread_self(), pool->threads[tid]))
            break;

    FOREVER {
        /* Wait for work (or shutdown).  cond_wait atomically releases
         * the mutex while waiting and re-acquires it on wakeup. */
        while (!pool->shutdown && pool->work == NULL)
            pthread_cond_wait(&pool->notify, &pool->lock);

        if (pool->shutdown) {
            pthread_mutex_unlock(&pool->lock);
            return NULL;
        }

        /* Drop the lock while doing actual work so other threads can
         * also claim work or signal completion concurrently. */
        pthread_mutex_unlock(&pool->lock);

        /* Atomically claim loop iterations via compare-and-swap on
         * pool->current until the entire range [start, end) is done. */
        FOREVER {
            int i;

            do { i = pool->current; }
            while (__sync_val_compare_and_swap(&pool->current, i, i + 1) != i);

            if (i >= pool->end) {
                pthread_mutex_lock(&pool->lock);
                pool->done_count++;
                pthread_cond_signal(&pool->notify);
                pthread_mutex_unlock(&pool->lock);
                break;
            }
            pool->work(pool->work_arg, tid, i);
            __sync_fetch_and_add(&pool->done_work, 1);
        }

        /* Re-acquire the mutex before looping back to wait on the
         * condition variable again. */
        pthread_mutex_lock(&pool->lock);
    }
}

/* ---- Pool lifecycle ------------------------------------------------- */

/* Allocate a thread pool with nthreads workers.  All threads are
 * created immediately and block until work arrives via
 * pthreads_parallel_for.  Returns NULL on failure. */
pthreads_t *pthreads_create(int nthreads) {
    if (nthreads < 1) nthreads = 1;

    pthreads_t *pool = scalloc(1, sizeof(pthreads_t));
    if (!pool) { slog(ERROR, "pthreads_create: calloc failed"); return NULL; }

    pool->nthreads = nthreads;
    pthread_mutex_init(&pool->lock, NULL);
    pthread_cond_init(&pool->notify, NULL);

    pool->threads = scalloc(nthreads, sizeof(pthread_t));
    if (!pool->threads) {
        slog(ERROR, "pthreads_create: thread array calloc failed");
        free(pool);
        return NULL;
    }

    for (int i = 0; i < nthreads; i++) {
        if (pthread_create(&pool->threads[i], NULL, worker_loop, pool) != 0) {
            slog(ERROR, "pthreads_create: pthread_create failed for thread %d", i);
            pool->nthreads = i;       /* only join threads that were created */
            pthreads_destroy(pool);
            return NULL;
        }
    }

    return pool;
}

/* Signal shutdown to all workers, join each thread, then free
 * pool resources.  Passing NULL is a no-op. */
void pthreads_destroy(pthreads_t *pool) {
    if (!pool) return;

    pthread_mutex_lock(&pool->lock);
    pool->shutdown = 1;
    pthread_cond_broadcast(&pool->notify);
    pthread_mutex_unlock(&pool->lock);

    for (int i = 0; i < pool->nthreads; i++)
        pthread_join(pool->threads[i], NULL);

    sfree(pool->threads);
    pthread_mutex_destroy(&pool->lock);
    pthread_cond_destroy(&pool->notify);
    sfree(pool);
}

/* ---- Parallel-for dispatch ----------------------------------------- */

/* Distribute a range [start, end) across the thread pool.
 * work(tid, i) is called for each i in the range from the thread
 * identified by tid.  The caller blocks until all iterations
 * complete. */
void pthreads_parallel_for(pthreads_t *pool, int start, int end, pthreads_work work, void *arg) {
    if (!pool || pool->nthreads < 1 || start >= end) return;

    /* Set up the work descriptor and wake all threads. */
    pthread_mutex_lock(&pool->lock);
    pool->start      = start;
    pool->end        = end;
    pool->current    = start;
    pool->done_count = 0;
    pool->done_work  = start;
    pool->work       = work;
    pool->work_arg   = arg;
    pthread_cond_broadcast(&pool->notify);
    pthread_mutex_unlock(&pool->lock);

    /* Block until every thread has reported done (done_count) and
     * every iteration has been processed (done_work >= end). */
    pthread_mutex_lock(&pool->lock);
    while (pool->done_count < pool->nthreads || pool->done_work < end)
        pthread_cond_wait(&pool->notify, &pool->lock);
    pool->work = NULL;
    pthread_mutex_unlock(&pool->lock);
}
