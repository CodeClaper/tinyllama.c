#include <stdbool.h>
#include "def.h"

typedef enum { 
    SECOND, 
    MILLISECOND, 
    MICROSECOND, 
    NANOSECOND 
} TIME_LEVEL;


int parse_int(char *s);
float parse_float(char *s);
char *get_key_name(Key *key);
char *get_datetime(TIME_LEVEL level);
bool key_streq(Key *key, char *s);
