#include <errno.h>
#include <float.h>
#include <stdbool.h>
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

/* Get next pow2 value. */
u64 next_pow2(u64 n) {
    u64 p = 1;
    while (p < n) p <<= 1;
    return p;
}

/* Hash the bytes. */
u64 hash_bytes(void *ptr, u64 len) {
    const u8 *p = ptr;
    u64 h = 1469598103934665603ull;
    for (u64 i = 0; i < len; i++) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

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

/* Parse string value to long. */
long parse_long(char *s) {
    char *end = NULL;
    errno = 0;
    long v = strtol(s, &end, 10);
    if (!s[0] || *end || errno == ERANGE) {
        slog(ERROR, "Bad long string value: %s.", s);
    }
    return v;
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
    return key.len == len && memcmp(key.content, s, len) == 0;
}

/* Key contains. */
bool key_strcontains(Key key, char *s) {
    char *name = get_key_name(key);
    char *end = strstr(name, s);
    return end != NULL;
}

/* Keys equals. */
bool key_eq(Key k1, Key k2) {
    return k1.len == k2.len && memcmp(k1.content, k2.content, k1.len) == 0;
}

/* Get system datetime for ms level. 
 * Supports four time level, SECOND, MILLISECOND, MICROSECOND, NANOSECOND. */
char *get_datetime(TimeLevel level) {
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

/* Convert size. */
double size_convert(u64 bytes, SizeLevel level) {
    double c = 0;
    switch (level) {
        case KB:
            c = (double)bytes / 1024;
            break;
        case MB:
            c = (double)bytes / (1024 * 1024);
            break;
        case GB:
            c = (double)bytes / (1024 * 1024 * 1024);
            break;
    }
    return c;
}
