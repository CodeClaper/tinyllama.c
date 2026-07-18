#include <stdlib.h>
#include "tpool.h"
#include "slog.h"
#include "utils.h"

static void *worker_loop(void *arg) {
    tpool_t *pool = (tpool_t *)arg;
    int tid;

    pthread_mutex_lock(&pool->lock);
    /* Assign thread IDs in creation order. */
    for (tid = 0; tid < pool->nthreads; tid++)
        if (pthread_equal(pthread_self(), pool->threads[tid]))
            break;

    FOREVER {
        while (!pool->shutdown && pool->work == NULL)
            pthread_cond_wait(&pool->notify, &pool->lock);

        if (pool->shutdown) {
            pthread_mutex_unlock(&pool->lock);
            return NULL;
        }

        /* Snapshot the work. */
        tpool_work work = pool->work;
        void *arg_work  = pool->work_arg;
        int end         = pool->end;

        pthread_mutex_unlock(&pool->lock);

        /* Process chunks atomically. */
        FOREVER {
            int i;
            pthread_mutex_lock(&pool->lock);
            i = pool->current;
            if (i >= end) {
                pool->done_count++;
                pthread_cond_signal(&pool->notify);
                pthread_mutex_unlock(&pool->lock);
                break;
            }
            pool->current = i + 1;
            pthread_mutex_unlock(&pool->lock);

            work(arg_work, tid, i);
        }

        /* Wait for next batch or shutdown. */
        pthread_mutex_lock(&pool->lock);
    }
}

tpool_t *tpool_create(int nthreads) {
    if (nthreads < 1) nthreads = 1;

    tpool_t *pool = calloc(1, sizeof(tpool_t));
    if (!pool) { slog(ERROR, "tpool_create: calloc failed"); return NULL; }

    pool->nthreads = nthreads;
    pthread_mutex_init(&pool->lock, NULL);
    pthread_cond_init(&pool->notify, NULL);

    pool->threads = calloc(nthreads, sizeof(pthread_t));
    if (!pool->threads) { 
        slog(ERROR, "tpool_create: thread array calloc failed"); 
        free(pool); 
        return NULL; 
    }

    for (int i = 0; i < nthreads; i++) {
        if (pthread_create(&pool->threads[i], NULL, worker_loop, pool) != 0) {
            slog(ERROR, "tpool_create: pthread_create failed for thread %d", i);
            pool->nthreads = i;
            tpool_destroy(pool);
            return NULL;
        }
    }

    return pool;
}

void tpool_destroy(tpool_t *pool) {
    if (!pool) return;

    pthread_mutex_lock(&pool->lock);
    pool->shutdown = 1;
    pthread_cond_broadcast(&pool->notify);
    pthread_mutex_unlock(&pool->lock);

    for (int i = 0; i < pool->nthreads; i++)
        pthread_join(pool->threads[i], NULL);

    free(pool->threads);
    pthread_mutex_destroy(&pool->lock);
    pthread_cond_destroy(&pool->notify);
    free(pool);
}

void tpool_parallel_for(tpool_t *pool, int start, int end, tpool_work work, void *arg) {
    if (!pool || pool->nthreads < 1 || start >= end) return;

    /* Set work for workers. */
    pthread_mutex_lock(&pool->lock);
    pool->start      = start;
    pool->end        = end;
    pool->current    = start;
    pool->done_count = 0;
    pool->work       = work;
    pool->work_arg   = arg;
    pthread_cond_broadcast(&pool->notify);
    pthread_mutex_unlock(&pool->lock);

    /* Main thread also works. */
    FOREVER {
        int i;
        pthread_mutex_lock(&pool->lock);
        i = pool->current;
        if (i >= end) {
            pool->done_count++;
            pthread_mutex_unlock(&pool->lock);
            break;
        }
        pool->current = i + 1;
        pthread_mutex_unlock(&pool->lock);

        work(arg, pool->nthreads, i);  /* main thread id = nthreads */
    }

    /* Wait for all workers to finish. */
    pthread_mutex_lock(&pool->lock);
    while (pool->done_count < pool->nthreads)
        pthread_cond_wait(&pool->notify, &pool->lock);
    pool->work = NULL;
    pthread_mutex_unlock(&pool->lock);
}
