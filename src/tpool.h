#ifndef TPOOL_H
#define TPOOL_H

#include <pthread.h>
#include <stddef.h>

typedef void (*tpool_work)(void *arg, int thread_id, int i);

typedef struct tpool {
    pthread_t       *threads;
    int             nthreads;
    volatile int    shutdown;
    pthread_mutex_t lock;
    pthread_cond_t  notify;
    volatile int    start;
    volatile int    end;
    volatile int    current;
    tpool_work      work;
    void            *work_arg;
    volatile int    done_count;
} tpool_t;

tpool_t *tpool_create(int nthreads);
void tpool_destroy(tpool_t *pool);
void tpool_parallel_for(tpool_t *pool, int start, int end, tpool_work work, void *arg);

#endif
