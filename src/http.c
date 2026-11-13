#include "http.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>
#include <errno.h>
#include "mm.h"
#include "slog.h"

/* Largest request we are willing to buffer (headers + body). */
#define MAX_REQUEST_BYTES (8 * 1024 * 1024)

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

/* Read a complete HTTP request body from fd.  On success returns
 * HTTP_BODY_OK and stores a NUL-terminated copy in *body_out (the caller
 * owns it); on failure returns a status describing why the request could
 * not be read and sets *body_out to NULL.  A single read() is not enough:
 * large JSON requests (chat history, tool definitions) arrive split
 * across TCP segments, and the body may not share a segment with the
 * headers. */
HttpBodyResult http_read_body(int fd, char **body_out) {
    size_t cap = 16384;
    size_t len = 0;
    char *buf = smalloc(cap);
    char *body = NULL;
    size_t header_len = 0;
    long content_length = -1;
    bool complete = false;
    bool too_large = false;

    *body_out = NULL;

    /* Read until the headers and the full Content-Length body arrive. */
    for (;;) {
        if (len + 1 >= cap) {
            if (cap >= MAX_REQUEST_BYTES) {
                too_large = true;
                break;
            }
            size_t new_cap = cap * 2;
            if (new_cap > MAX_REQUEST_BYTES) new_cap = MAX_REQUEST_BYTES;
            buf = srealloc(buf, new_cap);
            cap = new_cap;
            if (body) body = buf + header_len; /* realloc may have moved it */
        }

        ssize_t n = read(fd, buf + len, cap - len - 1);
        if (n < 0) {
            if (errno == EINTR) continue;
            slog_errno("Client read error");
            break;
        }
        if (n == 0) break; /* peer closed the connection */

        len += (size_t)n;
        buf[len] = '\0';

        if (!body) {
            body = http_body(buf, (int)len);
            if (body) {
                header_len = (size_t)(body - buf);
                const char *cl_str = http_header(buf, "Content-Length");
                content_length = cl_str[0] ? strtol(cl_str, NULL, 10) : -1;
            }
        }

        if (body && (content_length < 0 ||
                     len - header_len >= (size_t)content_length)) {
            complete = true;
            break;
        }
    }

    if (too_large) {
        sfree(buf);
        return HTTP_BODY_TOO_LARGE;
    }
    if (!body) {
        sfree(buf);
        return HTTP_BODY_INCOMPLETE_HEADER;
    }

    size_t body_len = content_length >= 0
                    ? (size_t)content_length : len - header_len;
    if (body_len <= 0) {
        sfree(buf);
        return HTTP_BODY_EMPTY;
    }
    if (!complete) {
        sfree(buf);
        return HTTP_BODY_INCOMPLETE;
    }

    char *copy = smalloc(body_len + 1);
    memcpy(copy, body, body_len);
    copy[body_len] = '\0';
    sfree(buf);
    *body_out = copy;
    return HTTP_BODY_OK;
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
