#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <stdarg.h>
#include "slog.h"
#include "utils.h"
#include "mm.h"

static char* LOG_LEVEL_NAME_LIST[] = { "INFO", "WARN", "EROR" };

void slog(LogLevel level, char *format, ...) {
    size_t len;
    va_list ap;

    /* Calculate the len. */
    va_start(ap, format);
    len = vsnprintf(NULL, 0, format, ap);
    if (len <= 0) {
        va_end(ap);
        return;
    }

    len = len + 1;
    char message[len];
    memset(message, 0, len);

    va_start(ap, format);
    vsnprintf(message, len, format, ap);
    va_end(ap);

    char *sys_time = get_datetime(MICROSECOND);
    char buff[len + 100];
    sprintf(buff, "[%s][%d][%s]:\t%s\n", 
            sys_time, getpid(), 
            LOG_LEVEL_NAME_LIST[level], message);
    if (level >= WARN) fprintf(stderr, "%s", buff);
    else fprintf(stdout, "%s", buff);
    sfree(sys_time);

    if (level >= ERROR) exit(100);
}

void slog_errno(char *format, ...) {
    size_t len;
    va_list ap;

    /* Calculate the len. */
    va_start(ap, format);
    len = vsnprintf(NULL, 0, format, ap);
    if (len <= 0) {
        va_end(ap);
        return;
    }

    len = len + 1;
    char message[len];
    memset(message, 0, len);

    va_start(ap, format);
    vsnprintf(message, len, format, ap);
    va_end(ap);

    char *sys_time = get_datetime(MICROSECOND);
    char buff[len + 100];
    sprintf(buff, "[%s][%d][%s]:\t%s, error: %s\n", 
            sys_time, getpid(), 
            LOG_LEVEL_NAME_LIST[ERROR], message, strerror(errno));
    fprintf(stderr, "%s", buff);
    sfree(sys_time);

    exit(100);
}

