#ifndef __HTTP_H__
#define __HTTP_H__

char *http_body(char *req, int len);
const char *http_header(char *buf, const char *name);
char *json_get_string(char *json, const char *key, int *len);
int json_get_bool(char *json, const char *key);
int json_unescape(char *dst, const char *src, int len);
int json_escape_str(char *out, int out_len, const char *in, int in_len);
void http_respond(int fd, int status, const char *status_msg, const char *body);
void http_stream_start(int fd, int status, const char *status_msg);
void http_stream_write(int fd, const char *data, int len);

#endif
