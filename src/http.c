#include "http.h"
#include <stdio.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>

/* Find the body of an HTTP request (after \r\n\r\n).  Returns NULL
 * if the request is incomplete. */
char *http_body(char *req, int len) {
    for (int i = 0; i + 3 < len; i++) {
        if (req[i] == '\r' && req[i+1] == '\n' &&
            req[i+2] == '\r' && req[i+3] == '\n')
            return req + i + 4;
    }
    return NULL;
}

/* Extract an HTTP header value (returns pointer into buf, or ""). */
const char *http_header(char *buf, const char *name) {
    /* Headers end at \r\n\r\n */
    char *end = strstr(buf, "\r\n\r\n");
    if (!end) return "";
    int name_len = strlen(name);

    /* Walk lines after the request line */
    char *line = strstr(buf, "\r\n");
    if (!line) return "";
    line += 2;

    while (line < end) {
        if (strncasecmp(line, name, name_len) == 0 &&
            line[name_len] == ':') {
            const char *val = line + name_len + 1;
            while (*val == ' ') val++;
            return val;  /* points into buf, lives long enough */
        }
        char *next = strstr(line, "\r\n");
        if (!next) break;
        line = next + 2;
    }
    return "";
}

/* ---- JSON helpers (minimal, no-library) ------------------------ */

/* Extract a JSON string value associated with key.  Returns a
 * pointer into json (the value between the quotes), and sets *len
 * to the byte length.  Handles basic \" and \\ escapes. */
char *json_get_string(char *json, const char *key, int *len) {
    char search[128];
    int slen = snprintf(search, sizeof(search), "\"%s\"", key);
    if (slen < 0 || slen >= (int)sizeof(search)) return NULL;

    char *pos = json;
    while ((pos = strstr(pos, search)) != NULL) {
        pos += slen;
        /* Skip whitespace before colon */
        while (*pos == ' ' || *pos == '\t' || *pos == '\n' || *pos == '\r')
            pos++;
        if (*pos != ':') continue;
        pos++;
        while (*pos == ' ' || *pos == '\t' || *pos == '\n' || *pos == '\r')
            pos++;
        if (*pos != '"') continue;
        pos++; /* skip opening quote */
        char *start = pos;
        /* Scan to closing unescaped quote */
        while (*pos && *pos != '"') {
            if (*pos == '\\' && *(pos + 1)) pos++;
            pos++;
        }
        if (*pos != '"') continue;
        *len = (int)(pos - start);
        return start;
    }
    return NULL;
}

/* Return 1 if key exists with a literal true value, else 0. */
int json_get_bool(char *json, const char *key) {
    char search[128];
    int slen = snprintf(search, sizeof(search), "\"%s\"", key);
    if (slen < 0 || slen >= (int)sizeof(search)) return 0;

    char *pos = json;
    while ((pos = strstr(pos, search)) != NULL) {
        pos += slen;
        while (*pos == ' ' || *pos == '\t' || *pos == '\n' || *pos == '\r')
            pos++;
        if (*pos != ':') continue;
        pos++;
        while (*pos == ' ' || *pos == '\t' || *pos == '\n' || *pos == '\r')
            pos++;
        return strncmp(pos, "true", 4) == 0;
    }
    return 0;
}

/* Copy src[0..len] to dst, unescaping JSON escape sequences.
 * Returns the number of bytes written (excluding NUL). */
int json_unescape(char *dst, const char *src, int len) {
    int w = 0;
    for (int i = 0; i < len; i++) {
        if (src[i] == '\\' && i + 1 < len) {
            i++;
            switch (src[i]) {
                case '"':  dst[w++] = '"';  break;
                case '\\': dst[w++] = '\\'; break;
                case '/':  dst[w++] = '/';  break;
                case 'n':  dst[w++] = '\n'; break;
                case 'r':  dst[w++] = '\r'; break;
                case 't':  dst[w++] = '\t'; break;
                case 'u':  dst[w++] = '?';  i += 4; break; /* simplify */
                default:   dst[w++] = src[i]; break;
            }
        } else {
            dst[w++] = src[i];
        }
    }
    return w;
}

/* Escape a string for JSON output.  Returns number of bytes written
 * (excluding NUL).  Truncates if out_len is too small. */
int json_escape_str(char *out, int out_len, const char *in, int in_len) {
    int w = 0;
    for (int i = 0; i < in_len && w + 6 < out_len; i++) {
        unsigned char c = (unsigned char)in[i];
        switch (c) {
            case '"':  out[w++] = '\\'; out[w++] = '"';  break;
            case '\\': out[w++] = '\\'; out[w++] = '\\'; break;
            case '\n': out[w++] = '\\'; out[w++] = 'n';  break;
            case '\r': out[w++] = '\\'; out[w++] = 'r';  break;
            case '\t': out[w++] = '\\'; out[w++] = 't';  break;
            case '\b': out[w++] = '\\'; out[w++] = 'b';  break;
            case '\f': out[w++] = '\\'; out[w++] = 'f';  break;
            default:
                if (c < 0x20) {
                    w += snprintf(out + w, out_len - w, "\\u%04x", c);
                } else {
                    out[w++] = (char)c;
                }
        }
    }
    return w;
}

static void write_all(int fd, const char *buf, int len) {
    int off = 0;
    while (off < len) {
        ssize_t w = write(fd, buf + off, (size_t)(len - off));
        if (w <= 0) break;
        off += (int)w;
    }
}

/* Send a minimal HTTP response with a JSON body. */
void http_respond(int fd, int status, const char *status_msg,
                  const char *body) {
    int body_len = body ? (int)strlen(body) : 0;
    char hdr[256];
    int hl = snprintf(hdr, sizeof(hdr),
                      "HTTP/1.1 %d %s\r\n"
                      "Content-Type: application/json\r\n"
                      "Content-Length: %d\r\n"
                      "Connection: close\r\n"
                      "\r\n",
                      status, status_msg, body_len);
    write_all(fd, hdr, hl);
    if (body && body_len > 0)
        write_all(fd, body, body_len);
}

/* Start an SSE (server-sent events) response.  The connection is closed
 * when the stream ends, which delimits the body for the client. */
void http_stream_start(int fd, int status, const char *status_msg) {
    char hdr[256];
    int hl = snprintf(hdr, sizeof(hdr),
                      "HTTP/1.1 %d %s\r\n"
                      "Content-Type: text/event-stream\r\n"
                      "Cache-Control: no-cache\r\n"
                      "X-Accel-Buffering: no\r\n"
                      "Connection: close\r\n"
                      "\r\n",
                      status, status_msg);
    write_all(fd, hdr, hl);
}

void http_stream_write(int fd, const char *data, int len) {
    write_all(fd, data, len);
}
