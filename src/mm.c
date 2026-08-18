#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mm.h"
#include "mctx.h"

static void fatal(const char *msg) {
    fprintf(stderr, "Fatal for: %s\n", msg);
    exit(1);
}

void *smalloc(size_t n) {
    void *p = MemoryContextAlloc(n ? n : 1);
    if (!p) fatal("Out of memory");
    memset(p, '\0', n);
    return p;
}

void *scalloc(size_t n, size_t s) {
    void *p = MemoryContextCalloc(n, s);
    if (!p) fatal("Out of memory");
    return p;
}

void *srealloc(void *p, size_t n) {
    p = MemoryContextRealloc(p, n ? n : 1);
    if (!p) fatal("out of memory");
    return p;
}

char *sstrdup(char *s) {
    if (s == NULL) return NULL;
    else return MemoryContextStrdup(s);
}

void sfree(void *p) {
    if (p) free(p);
}
