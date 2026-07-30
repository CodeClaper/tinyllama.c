#ifndef PTHREADS_H
#define PTHREADS_H

#include <pthread.h>
#include <stddef.h>

typedef void (*pthreads_work)(void *arg, int thread_id, int i);

typedef struct pthreads {
    pthread_t       *threads;
    int             nthreads;
    volatile int    shutdown;
    pthread_mutex_t lock;
    pthread_cond_t  notify;
    volatile int    start;
    volatile int    end;
    volatile int    current;
    pthreads_work   work;
    void            *work_arg;
    volatile int    done_count;
    volatile int    done_work;
} pthreads_t;

pthreads_t *pthreads_create(int nthreads);
void pthreads_destroy(pthreads_t *pool);
void pthreads_parallel_for(pthreads_t *pool, int start, int end, pthreads_work work, void *arg);

#endif
