#include <stddef.h>

void *smalloc(size_t n);
void *scalloc(size_t n, size_t s);
void *srealloc(void *p, size_t n);
char *sstrdup(char *s);
void sfree(void *p);
