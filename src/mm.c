#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include "mm.h"
#include "asctx.h"

/* Payload capacity of the initial block of the global context. */
#define MM_INIT_BLOCK_SIZE (8 * 1024)

/* The single AllocSet context backing every allocation in the process,
 * created lazily on first use.  The AllocSet free-list machinery is not
 * thread-safe, so every wrapper serialises access to it with mm_lock.
 * The lock is required now that request workers tokenise and log from
 * multiple threads. */
static AllocSetContext *mm_ctx;
static pthread_mutex_t mm_lock = PTHREAD_MUTEX_INITIALIZER;

static AllocSetContext *mm_get_ctx(void) {
    if (mm_ctx == NULL)
        mm_ctx = alloc_set_memory_context_create(MM_INIT_BLOCK_SIZE);
    return mm_ctx;
}

static void fatal(const char *msg) {
    fprintf(stderr, "Fatal for: %s\n", msg);
    exit(1);
}

void *smalloc(size_t n) {
    pthread_mutex_lock(&mm_lock);
    void *p = alloc_set_alloc(mm_get_ctx(), n ? n : 1);
    pthread_mutex_unlock(&mm_lock);
    if (!p) fatal("Out of memory");
    memset(p, '\0', n);
    return p;
}

void *scalloc(size_t n, size_t s) {
    size_t total;

    /* Overflow check, same as calloc. */
    if (n && s && n > SIZE_MAX / s) fatal("Out of memory");
    total = (n ? n : 1) * (s ? s : 1);

    pthread_mutex_lock(&mm_lock);
    void *p = alloc_set_alloc(mm_get_ctx(), total);
    pthread_mutex_unlock(&mm_lock);
    if (!p) fatal("Out of memory");
    memset(p, '\0', total);
    return p;
}

void *srealloc(void *p, size_t n) {
    /* alloc_set_realloc needs a live chunk; mirror realloc(NULL, n). */
    if (p == NULL) return smalloc(n);
    pthread_mutex_lock(&mm_lock);
    void *np = alloc_set_realloc(p, n);
    pthread_mutex_unlock(&mm_lock);
    if (!np) fatal("out of memory");
    return np;
}

char *sstrdup(char *s) {
    if (s == NULL) return NULL;
    size_t len = strlen(s);
    char *dup = smalloc(len + 1);
    memcpy(dup, s, len + 1);
    return dup;
}

void sfree(void *p) {
    /* alloc_set_free does chunk-header math; NULL must never reach it. */
    if (p) {
        pthread_mutex_lock(&mm_lock);
        alloc_set_free(p);
        pthread_mutex_unlock(&mm_lock);
    }
}
