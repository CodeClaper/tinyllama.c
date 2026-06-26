#include <stdbool.h>
#include "def.h"

/* Anti-warning marco. */
#define UNUSED(v) ((void ) v)
#define FOREVER for(;;)

typedef enum { 
    SECOND, 
    MILLISECOND, 
    MICROSECOND, 
    NANOSECOND 
} TimeLevel;

typedef enum {
    KB,
    MB,
    GB
} SizeLevel;

int parse_int(char *s);
float parse_float(char *s);
char *get_key_name(Key key);
char *get_datetime(TimeLevel level);
bool key_streq(Key key, char *s);
bool key_strcontains(Key key, char *s);
bool key_eq(Key k1, Key k2);
double size_convert(u64 bytes, SizeLevel level);
