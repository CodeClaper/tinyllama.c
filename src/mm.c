#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mm.h"
#include "asctx.h"

/* Payload capacity of the initial block of the global context. */
#define MM_INIT_BLOCK_SIZE (8 * 1024)

/* The single AllocSet context backing every allocation in the process,
 * created lazily on first use.  NOTE: the AllocSet free-list machinery
 * is not thread-safe.  This engine only calls the mm() wrappers from
 * the main thread (worker threads touch pre-allocated buffers only),
 * so one unlocked context is sufficient here. */
static AllocSetContext *mm_ctx;

static AllocSetContext *mm_get_ctx(void) {
    if (mm_ctx == NULL)
        mm_ctx = AllocSetMemoryContextCreate(MM_INIT_BLOCK_SIZE);
    return mm_ctx;
}

static void fatal(const char *msg) {
    fprintf(stderr, "Fatal for: %s\n", msg);
    exit(1);
}

void *smalloc(size_t n) {
    void *p = AllocSetAlloc(mm_get_ctx(), n ? n : 1);
    if (!p) fatal("Out of memory");
    memset(p, '\0', n);
    return p;
}

void *scalloc(size_t n, size_t s) {
    size_t total;

    /* Overflow check, same as calloc. */
    if (n && s && n > SIZE_MAX / s) fatal("Out of memory");
    total = (n ? n : 1) * (s ? s : 1);

    void *p = AllocSetAlloc(mm_get_ctx(), total);
    if (!p) fatal("Out of memory");
    memset(p, '\0', total);
    return p;
}

void *srealloc(void *p, size_t n) {
    /* AllocSetRealloc needs a live chunk; mirror realloc(NULL, n). */
    if (p == NULL) return smalloc(n);

    p = AllocSetRealloc(p, n);
    if (!p) fatal("out of memory");
    return p;
}

char *sstrdup(char *s) {
    if (s == NULL) return NULL;
    size_t len = strlen(s);
    char *dup = smalloc(len + 1);
    memcpy(dup, s, len + 1);
    return dup;
}

void sfree(void *p) {
    /* AllocSetFree does chunk-header math; NULL must never reach it. */
    if (p) AllocSetFree(p);
}
