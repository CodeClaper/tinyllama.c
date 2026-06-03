
typedef enum { 
    SECOND, 
    MILLISECOND, 
    MICROSECOND, 
    NANOSECOND 
} TIME_LEVEL;


char *get_datetime(TIME_LEVEL level);
