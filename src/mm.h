#include <stddef.h>

void *smalloc(size_t n);
void *srealloc(void *p, size_t n);
char *sstrdup(const char *s);
void sfree(void *p);
