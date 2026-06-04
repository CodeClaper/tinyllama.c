
typedef enum { 
    SECOND, 
    MILLISECOND, 
    MICROSECOND, 
    NANOSECOND 
} TIME_LEVEL;


int parse_int(char *s);
float parse_float(char *s);
char *get_datetime(TIME_LEVEL level);
