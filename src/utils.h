#include <stdbool.h>
#include <assert.h>
#include "def.h"

#ifdef DEBUG
#define Assert(condition) assert(condition)
#define AssertFalse(condition) assert(!(condition))
#define DBG_VEC(label, vec, n) do { \
    printf("  [C] %-30s: ", label); \
    for (u32 _i = 0; _i < (n) && _i < 10; _i++) \
        printf("%.6f ", (vec)[_i]); \
    printf("\n"); \
} while(0)
#else
#define Assert(condition) ((void)true)
#define AssertFalse(condition) ((void)false)
#define DBG_VEC(label, vec, n) ((void)0)
#endif

/* Anti-warning marco. */
#define UNUSED(v) ((void ) v)
#define FOREVER for(;;)

typedef enum { KB, MB, GB } SizeLevel;
typedef enum { SECOND, MILLISECOND, MICROSECOND, NANOSECOND } TimeLevel;

u64 next_pow2(u64 n);
u64 hash_bytes(void *ptr, u64 len);
int parse_int(char *s);
float parse_float(char *s);
long parse_long(char *s);
char *get_key_name(Key key);
char *get_datetime(TimeLevel level);
bool key_streq(Key key, char *s);
bool key_strcontains(Key key, char *s);
bool key_eq(Key k1, Key k2);
double size_convert(u64 bytes, SizeLevel level);
