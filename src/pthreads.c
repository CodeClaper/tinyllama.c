#include <stdlib.h>
#include "pthreads.h"
#include "mm.h"
#include "slog.h"
#include "utils.h"

static void *worker_loop(void *arg) {
    pthreads_t *pool = (pthreads_t *)arg;
    int tid;

    pthread_mutex_lock(&pool->lock);
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

        pthread_mutex_unlock(&pool->lock);

        FOREVER {
            int i;
            pthread_mutex_lock(&pool->lock);
            i = pool->current;
            if (i >= pool->end) {
                pool->done_count++;
                pthread_cond_signal(&pool->notify);
                pthread_mutex_unlock(&pool->lock);
                break;
            }
            pool->current = i + 1;
            pthread_mutex_unlock(&pool->lock);

            pool->work(pool->work_arg, tid, i);
            __sync_fetch_and_add(&pool->done_work, 1);
        }

        pthread_mutex_lock(&pool->lock);
    }
}

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
            pool->nthreads = i;
            pthreads_destroy(pool);
            return NULL;
        }
    }

    return pool;
}

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

void pthreads_parallel_for(pthreads_t *pool, int start, int end, pthreads_work work, void *arg) {
    if (!pool || pool->nthreads < 1 || start >= end) return;

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

    pthread_mutex_lock(&pool->lock);
    while (pool->done_count < pool->nthreads || pool->done_work < end)
        pthread_cond_wait(&pool->notify, &pool->lock);
    pool->work = NULL;
    pthread_mutex_unlock(&pool->lock);
}
