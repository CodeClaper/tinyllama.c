#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <bits/types/struct_timeval.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <limits.h>
#include <math.h>
#include "utils.h"
#include "mm.h"
#include "slog.h"

/* Parse string value to int. */
int parse_int(char *s) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (!s[0] || *end || v > INT_MAX || v < INT_MIN) {
        slog(ERROR, "Bad int string value: %s.", s);
    }
    return (int) v;
}

/* Parse string value to float. */
float parse_float(char *s) {
    char *end = NULL;
    float v = strtof(s, &end);
    if (!s[0] || *end || isinff(v) || v > FLT_MAX) {
        slog(ERROR, "Bad float string value: %s.", s);
    }
    return (float) v;
}

/* Get key name. blk.0.attn_q.weight*/
char *get_key_name(Key key) {
    char *s = smalloc(key.len + 1);
    memcpy(s, key.content, key.len);
    s[key.len] = '\0';
    return s;
}

/* Key and string equals. */
bool key_streq(Key key, char *s) {
    size_t len = strlen(s);
    return key.len == len && memcmp(key.content, s, len);
}

/* Keys equals. */
bool key_eq(Key k1, Key k2) {
    return k1.len == k2.len && memcmp(k1.content, k2.content, k1.len);
}

/* Get system datetime for ms level. 
 * Supports four time level, SECOND, MILLISECOND, MICROSECOND, NANOSECOND. */
char *get_datetime(TIME_LEVEL level) {
    struct timeval tv;
    time_t t;
    struct tm *ptm;

    gettimeofday(&tv, NULL);
    t = tv.tv_sec;
    ptm = localtime(&t);
    char *res = smalloc(30);

    switch (level) {
        case SECOND:
            sprintf(res, "%04d-%02d-%02d %02d:%02d:%02d",
                   ptm->tm_year + 1900, ptm->tm_mon + 1, ptm->tm_mday,
                   ptm->tm_hour, ptm->tm_min, ptm->tm_sec);
            break;
        case MILLISECOND:
            sprintf(res, "%04d-%02d-%02d %02d:%02d:%02d.%03d",
                   ptm->tm_year + 1900, ptm->tm_mon + 1, ptm->tm_mday,
                   ptm->tm_hour, ptm->tm_min, ptm->tm_sec, ((int)(tv.tv_usec)) / 1000);
            break;
        case MICROSECOND:
            sprintf(res, "%04d-%02d-%02d %02d:%02d:%02d.%06d",
                   ptm->tm_year + 1900, ptm->tm_mon + 1, ptm->tm_mday,
                   ptm->tm_hour, ptm->tm_min, ptm->tm_sec, (int)(tv.tv_usec));
            break;
        default:
            break;
    }

    return res;
}
