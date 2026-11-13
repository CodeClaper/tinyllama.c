#include <signal.h>
#include <pthread.h>
#include <stdatomic.h>
#include "anet.h"
#include "el.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <ctype.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#ifdef __linux__
#include <sys/eventfd.h>
#endif
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include "def.h"
#include "slog.h"
#include "utils.h"
#include "mm.h"
#include "core.h"
#include "tokenizer.h"
#include "sampler.h"
#include "anet.h"
#include "http.h"
#include "graph.h"
#include "tool.h"
#include "banner.h"

#define JSON_WRAPPER_RESERVE 256 
#define DEFAULT_N_PARALLEL   1
#define JOB_QUEUE_CAP        256
#define MAX_CHAT_MESSAGES    256
#define MAX_REQUEST_TOOLS    64
#define MAX_TOOL_ARGUMENTS   TOOL_ARGS_MAX

typedef enum {
    TOOL_CHOICE_AUTO,
    TOOL_CHOICE_NONE,
    TOOL_CHOICE_REQUIRED,
    TOOL_CHOICE_NAMED
} ToolChoice;

/* A parsed-but-not-yet-executed request.  The body is copied out of the
 * connection's read buffer so the worker owns it for the whole run. */
typedef struct {
    int   fd;
    char *body;
} ClientJob;

typedef struct WorkerPool WorkerPool;

typedef struct {
    WorkerPool *pool;
    int         id;
    pthread_t   thread;
} Worker;

struct WorkerPool {
    pthread_mutex_t lock;
    pthread_cond_t  cond;
    ClientJob       queue[JOB_QUEUE_CAP];
    int             head;
    int             tail;
    int             count;
    bool            stop;

    Engine         *engine;
    Session       **sessions;   /* one per worker, indexed by worker id */
    Worker         *workers;
    int             n_workers;
};

typedef struct  {
    Engine      *engine;
    Session    **sessions;
    int          n_parallel;
    EventLoop   *el;
    int          serverfd;
    WorkerPool  *pool;
} Server;

static volatile sig_atomic_t g_signal_count = 0;
static _Atomic unsigned long long g_tool_call_seq = 0;
static int g_event_fd = -1;
#ifdef __linux__
/* Linux uses an eventfd to wake the select loop from the signal handler. */
#else
/* Other platforms (macOS) use a self-pipe instead of eventfd. */
static int g_signal_wfd = -1;
#endif

static void signal_handler(int sig) {
    UNUSED(sig);
    if (g_signal_count > 0) _exit(130);
    g_signal_count = 1;
#ifdef __linux__
    u64 val = 1;
    if (g_event_fd >= 0) (void)write(g_event_fd, &val, sizeof(val));
#else
    u8 val = 1;
    if (g_signal_wfd >= 0) (void)write(g_signal_wfd, &val, sizeof(val));
#endif
}

/* Callback for the signal eventfd/self-pipe — wake up and stop the event loop. */
static void signal_callback(EventLoop *el, int fd, int mask, void *privdata) {
    UNUSED(mask);
    UNUSED(privdata);
#ifdef __linux__
    u64 val;
#else
    u8 val;
#endif
    (void)read(fd, &val, sizeof(val));
    el->stop = true;
}

/* Length of the longest prefix of buf[0..len] that ends on a complete
 * UTF-8 code point.  Token pieces can split a multibyte character, so we
 * hold back the trailing partial sequence until the next token arrives. */
static int utf8_complete_prefix(const char *buf, int len) {
    int i = 0;
    while (i < len) {
        unsigned char c = (unsigned char)buf[i];
        int need;
        if (c < 0x80)                need = 1;
        else if ((c & 0xE0) == 0xC0) need = 2;
        else if ((c & 0xF0) == 0xE0) need = 3;
        else if ((c & 0xF8) == 0xF0) need = 4;
        else { i++; continue; }
        if (i + need > len) break;
        i += need;
    }
    return i;
}

static void sse_emit_role(int fd, long created) {
    char buf[512];
    int n = snprintf(buf, sizeof(buf),
        "data: {\"id\":\"chatcmpl-%ld\",\"object\":\"chat.completion.chunk\","
        "\"created\":%ld,\"model\":\"default\",\"choices\":[{\"index\":0,"
        "\"delta\":{\"role\":\"assistant\",\"content\":\"\"},"
        "\"finish_reason\":null}]}\n\n",
        created, created);
    if (n > 0)
        http_stream_write(fd, buf, n < (int)sizeof(buf) ? n : (int)sizeof(buf) - 1);
}

static void sse_emit_content(int fd, long created, const char *text, int len) {
    char esc[16384];
    int el = json_escape_str(esc, (int)sizeof(esc) - 1, text, len);
    esc[el] = '\0';

    char buf[16384 + 512];
    int n = snprintf(buf, sizeof(buf),
        "data: {\"id\":\"chatcmpl-%ld\",\"object\":\"chat.completion.chunk\","
        "\"created\":%ld,\"model\":\"default\",\"choices\":[{\"index\":0,"
        "\"delta\":{\"content\":\"%s\"},\"finish_reason\":null}]}\n\n",
        created, created, esc);
    if (n > 0)
        http_stream_write(fd, buf, n < (int)sizeof(buf) ? n : (int)sizeof(buf) - 1);
}

static void sse_emit_tool_call(int fd, long created, const ToolCall *call) {
    char name[1024];
    int name_len = json_escape_str(name, sizeof(name) - 1,
                                   call->name, (int)strlen(call->name));
    name[name_len] = '\0';
    char args[MAX_TOOL_ARGUMENTS * 6 + 1];
    int args_len = json_escape_str(args, sizeof(args) - 1,
                                   call->arguments, call->arguments_len);
    args[args_len] = '\0';
    char buf[MAX_TOOL_ARGUMENTS * 6 + 2048];
    int n = snprintf(buf, sizeof(buf),
        "data: {\"id\":\"chatcmpl-%ld\",\"object\":\"chat.completion.chunk\","
        "\"created\":%ld,\"model\":\"default\",\"choices\":[{\"index\":0,"
        "\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"%s\","
        "\"type\":\"function\",\"function\":{\"name\":\"%s\","
        "\"arguments\":\"\"}}]},\"finish_reason\":null}]}\n\n"
        "data: {\"id\":\"chatcmpl-%ld\",\"object\":\"chat.completion.chunk\","
        "\"created\":%ld,\"model\":\"default\",\"choices\":[{\"index\":0,"
        "\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{"
        "\"arguments\":\"%s\"}}]},\"finish_reason\":null}]}\n\n",
        created, created, call->id, name, created, created, args);
    if (n > 0)
        http_stream_write(fd, buf, n < (int)sizeof(buf) ? n : (int)sizeof(buf) - 1);
}

static void sse_emit_done(int fd, long created, const char *finish_reason) {
    char buf[512];
    int n = snprintf(buf, sizeof(buf),
        "data: {\"id\":\"chatcmpl-%ld\",\"object\":\"chat.completion.chunk\","
        "\"created\":%ld,\"model\":\"default\",\"choices\":[{\"index\":0,"
        "\"delta\":{},\"finish_reason\":\"%s\"}]}\n\n"
        "data: [DONE]\n\n",
        created, created, finish_reason);
    if (n > 0)
        http_stream_write(fd, buf, n < (int)sizeof(buf) ? n : (int)sizeof(buf) - 1);
}

/* ---- Minimal JSON conversation parser -----------------------------
 * Extracts the ordered "messages": [{ "role": ..., "content": ... }]
 * array.  Everything else in the request body (tools, sampling
 * parameters) is ignored here. */

static char *json_skip_ws(char *p) {
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    return p;
}

/* `p` points at the opening quote of a JSON string.  Returns the
 * position just past the closing quote (or the NUL terminator). */
static char *json_string_end(char *p) {
    p++;
    while (*p) {
        if (*p == '\\' && p[1]) { p += 2; continue; }
        if (*p == '"') return p + 1;
        p++;
    }
    return p;
}

/* `p` points at the opening brace of a JSON object.  Returns the
 * position of the matching closing brace, or NULL. */
static char *json_object_end(char *p) {
    int depth = 0;
    while (*p) {
        if (*p == '"') { p = json_string_end(p); continue; }
        if (*p == '{') depth++;
        else if (*p == '}' && --depth == 0) return p;
        p++;
    }
    return NULL;
}

static char *json_array_end(char *p) {
    if (!p || *p != '[') return NULL;
    int depth = 0;
    while (*p) {
        if (*p == '"') { p = json_string_end(p); continue; }
        if (*p == '[') depth++;
        else if (*p == ']' && --depth == 0) return p;
        p++;
    }
    return NULL;
}

static char *json_value_end(char *p) {
    if (!p) return NULL;
    if (*p == '"') return json_string_end(p);
    if (*p == '{') {
        char *end = json_object_end(p);
        return end ? end + 1 : NULL;
    }
    if (*p == '[') {
        char *end = json_array_end(p);
        return end ? end + 1 : NULL;
    }
    while (*p && *p != ',' && *p != '}' && *p != ']' &&
           *p != ' ' && *p != '\t' && *p != '\n' && *p != '\r') p++;
    return p;
}

/* Find a field directly on an object, without matching a nested field of
 * the same name. */
static char *json_direct_value(char *object, const char *key) {
    if (!object || *object != '{') return NULL;
    char *end = json_object_end(object);
    if (!end) return NULL;
    char *p = json_skip_ws(object + 1);
    while (p < end && *p == '"') {
        char *key_end = json_string_end(p);
        if (!key_end || key_end > end) return NULL;
        size_t key_len = (size_t)(key_end - p - 2);
        char *value = json_skip_ws(key_end);
        if (*value != ':') return NULL;
        value = json_skip_ws(value + 1);
        char *value_end = json_value_end(value);
        if (!value_end || value_end > end) return NULL;
        if (key_len == strlen(key) && memcmp(p + 1, key, key_len) == 0)
            return value;
        p = json_skip_ws(value_end);
        if (*p == ',') p = json_skip_ws(p + 1);
        else break;
    }
    return NULL;
}

static int request_tools(char *body, ToolDef *tools, int max,
                         char **raw_tools, int *raw_tools_len) {
    *raw_tools = NULL;
    *raw_tools_len = 0;
    char *array = json_direct_value(body, "tools");
    if (!array || *array != '[') return 0;
    char *array_end = json_array_end(array);
    if (!array_end) return 0;
    *raw_tools = array;
    *raw_tools_len = (int)(array_end - array + 1);

    int count = 0;
    char *p = json_skip_ws(array + 1);
    while (p < array_end && count < max) {
        if (*p != '{') break;
        char *item_end = json_object_end(p);
        if (!item_end || item_end > array_end) break;
        char *function = json_direct_value(p, "function");
        char *name = json_direct_value(function, "name");
        if (name && *name == '"') {
            char *name_end = json_string_end(name);
            int name_len = (int)(name_end - name - 2);
            if (name_end && name_len > 0 && name_len < TOOL_NAME_MAX) {
                ToolDef *tool = &tools[count];
                memset(tool, 0, sizeof(*tool));
                int decoded = json_unescape(tool->name, name + 1, name_len);
                tool->name[decoded] = '\0';

                char *params = json_direct_value(function, "parameters");
                char *props = json_direct_value(params, "properties");
                if (props && *props == '{') {
                    char *props_end = json_object_end(props);
                    char *q = json_skip_ws(props + 1);
                    while (props_end && q < props_end && *q == '"' &&
                           tool->n_params < TOOL_MAX_PARAMS) {
                        char *key_end = json_string_end(q);
                        if (!key_end || key_end > props_end) break;
                        int key_len = (int)(key_end - q - 2);
                        char *colon = json_skip_ws(key_end);
                        if (colon >= props_end || *colon != ':') break;
                        char *value = json_skip_ws(colon + 1);
                        char *value_end = json_value_end(value);
                        if (!value_end || value_end > props_end) break;
                        if (key_len > 0 && key_len < TOOL_NAME_MAX) {
                            int n = tool->n_params++;
                            int d = json_unescape(tool->params[n],
                                                  q + 1, key_len);
                            tool->params[n][d] = '\0';
                            tool->required[n] = false;
                        }
                        q = json_skip_ws(value_end);
                        if (*q == ',') q = json_skip_ws(q + 1);
                        else break;
                    }
                }

                char *req = json_direct_value(params, "required");
                if (req && *req == '[') {
                    char *req_end = json_array_end(req);
                    char *q = json_skip_ws(req + 1);
                    while (req_end && q < req_end && *q == '"') {
                        char *end = json_string_end(q);
                        if (!end || end > req_end) break;
                        int len = (int)(end - q - 2);
                        char required[TOOL_NAME_MAX];
                        if (len > 0 && len < TOOL_NAME_MAX) {
                            int d = json_unescape(required, q + 1, len);
                            required[d] = '\0';
                            for (int i = 0; i < tool->n_params; i++)
                                if (strcmp(tool->params[i], required) == 0)
                                    tool->required[i] = true;
                        }
                        q = json_skip_ws(end);
                        if (*q == ',') q = json_skip_ws(q + 1);
                        else break;
                    }
                }
                count++;
            }
        }
        p = json_skip_ws(item_end + 1);
        if (*p == ',') p = json_skip_ws(p + 1);
        else break;
    }
    return count;
}

static ToolChoice request_tool_choice(char *body, char *name, int name_cap) {
    char *choice = json_direct_value(body, "tool_choice");
    if (!choice || (*choice == '"' && strncmp(choice + 1, "auto\"", 5) == 0))
        return TOOL_CHOICE_AUTO;
    if (*choice == '"') {
        if (strncmp(choice + 1, "none\"", 5) == 0) return TOOL_CHOICE_NONE;
        if (strncmp(choice + 1, "required\"", 9) == 0) return TOOL_CHOICE_REQUIRED;
        return TOOL_CHOICE_AUTO;
    }
    if (*choice == '{') {
        char *function = json_direct_value(choice, "function");
        char *field = json_direct_value(function, "name");
        if (field && *field == '"') {
            char *end = json_string_end(field);
            int len = (int)(end - field - 2);
            if (end && len > 0 && len < name_cap) {
                int decoded = json_unescape(name, field + 1, len);
                name[decoded] = '\0';
                return TOOL_CHOICE_NAMED;
            }
        }
    }
    return TOOL_CHOICE_AUTO;
}

static char *tool_prompt_create(const char *raw_tools, int raw_tools_len,
                                ToolChoice choice, const char *forced_name,
                                const ToolOps *tool_ops) {
    static const char before[] =
        "# Tools\n\nYou have access to the following functions:\n\n<tools>\n";
    static const char after[] = "\n</tools>\n";
    const char *hint = (tool_ops && tool_ops->prompt_hint)
                     ? tool_ops->prompt_hint : "";
    const char *required = choice == TOOL_CHOICE_REQUIRED
        ? "You must call one function.\n" : "";
    char named[256] = "";
    if (choice == TOOL_CHOICE_NAMED)
        (void)snprintf(named, sizeof(named), "Call only the function '%s'.\n", forced_name);
    size_t size = sizeof(before) + sizeof(after) + strlen(hint) +
                  strlen(required) + strlen(named) +
                  (size_t)raw_tools_len + 64;
    char *prompt = smalloc(size);
    int n = snprintf(prompt, size, "%s%s%s%.*s%s%s",
                     required, named, before, raw_tools_len, raw_tools,
                     after, hint);
    if (n < 0 || (size_t)n >= size) {
        sfree(prompt);
        return NULL;
    }
    return prompt;
}

/* Find `key`'s value within the object spanning [start, end). */
static char *json_object_value(char *start, char *end, const char *key) {
    size_t klen = strlen(key);
    for (char *p = start; p < end; ) {
        if (*p == '"') {
            char *q = json_string_end(p);
            int len = (int)(q - p - 2);
            if (len == (int)klen && memcmp(p + 1, key, klen) == 0) {
                char *v = json_skip_ws(q);
                if (v < end && *v == ':')
                    return json_skip_ws(v + 1);
            }
            p = q;
        } else {
            p++;
        }
    }
    return NULL;
}

/* Parse the "messages" array into out[].  role/content are heap
 * allocated and must be released with chat_messages_free().  Returns
 * the number of messages stored (0 when absent/empty). */
static int json_parse_messages(char *body, ChatMessage *out, int max,
                               const ToolOps *tool_ops) {
    char latest_call_id[64] = "";
    char latest_call_name[TOOL_NAME_MAX] = "";
    char *arr = NULL;
    for (char *p = body; (p = strstr(p, "\"messages\"")) != NULL; p += 10) {
        char *q = json_skip_ws(p + 10);
        if (*q != ':') continue;
        q = json_skip_ws(q + 1);
        if (*q == '[') { arr = q; break; }
    }
    if (!arr) return 0;

    int n = 0;
    char *p = json_skip_ws(arr + 1);
    while (*p && *p != ']') {
        if (*p != '{') break;
        char *end = json_object_end(p);
        if (!end) break;

        char *rv = json_object_value(p, end, "role");
        char *cv = json_object_value(p, end, "content");
        if (rv && *rv == '"' && n < max) {
            char *rs = rv + 1;
            int   rl = (int)(json_string_end(rv) - rs - 1);

            const char *cs = "";
            int         cl = 0;
            if (cv && *cv == '"') {
                cs = cv + 1;
                cl = (int)(json_string_end(cv) - cs - 1);
            }

            char *role = smalloc((size_t)rl + 1);
            int ul = json_unescape(role, rs, rl);
            role[ul] = '\0';

            char *content = smalloc((size_t)cl + 1);
            ul = json_unescape(content, cs, cl);
            content[ul] = '\0';

            /* Reconstruct prior assistant tool calls so a client can send
             * the OpenAI tool result in the next request. */
            char *calls = json_direct_value(p, "tool_calls");
            if (tool_ops && tool_ops->format_call &&
                strcmp(role, "assistant") == 0 && calls && *calls == '[') {
                char *calls_end = json_array_end(calls);
                char *cp = json_skip_ws(calls + 1);
                bool added_call = false;
                int  first_call = n;
                while (calls_end && cp < calls_end && n < max && *cp == '{') {
                    char *call_end = json_object_end(cp);
                    if (!call_end || call_end > calls_end) break;
                    char *fn = json_direct_value(cp, "function");
                    char *name_value = json_direct_value(fn, "name");
                    char *args_value = json_direct_value(fn, "arguments");
                    char *id_value = json_direct_value(cp, "id");
                    if (name_value && *name_value == '"' &&
                        args_value && *args_value == '"') {
                        char *name_end = json_string_end(name_value);
                        char *args_end = json_string_end(args_value);
                        int name_len = (int)(name_end - name_value - 2);
                        int args_len = (int)(args_end - args_value - 2);
                        if (name_end && args_end && name_len > 0 &&
                            name_len < TOOL_NAME_MAX && args_len >= 0 &&
                            args_len <= MAX_TOOL_ARGUMENTS) {
                            char tool_name[TOOL_NAME_MAX];
                            int decoded_name = json_unescape(tool_name,
                                name_value + 1, name_len);
                            tool_name[decoded_name] = '\0';
                            if (id_value && *id_value == '"') {
                                char *id_end = json_string_end(id_value);
                                int id_len = (int)(id_end - id_value - 2);
                                if (id_end && id_len > 0 &&
                                    id_len < (int)sizeof(latest_call_id)) {
                                    int decoded_id = json_unescape(latest_call_id,
                                        id_value + 1, id_len);
                                    latest_call_id[decoded_id] = '\0';
                                    (void)snprintf(latest_call_name,
                                        sizeof(latest_call_name), "%s", tool_name);
                                }
                            }
                            char *args = smalloc((size_t)args_len + 1);
                            int decoded_args = json_unescape(args,
                                args_value + 1, args_len);
                            args[decoded_args] = '\0';
                            char *call_text = tool_ops->format_call(tool_name, args);
                            sfree(args);
                            if (call_text) {
                                if (n == first_call && content && content[0]) {
                                    size_t combined_len = strlen(content) +
                                                          strlen(call_text) + 3;
                                    char *combined = smalloc(combined_len);
                                    (void)snprintf(combined, combined_len,
                                                   "%s\n\n%s", content, call_text);
                                    sfree(call_text);
                                    call_text = combined;
                                }
                                out[n].role = sstrdup("assistant");
                                out[n].content = call_text;
                                out[n].end_marker = tool_ops->call_end;
                                n++;
                                added_call = true;
                            }
                        }
                    }
                    cp = json_skip_ws(call_end + 1);
                    if (*cp == ',') cp = json_skip_ws(cp + 1);
                    else break;
                }
                if (added_call) {
                    sfree(role);
                    sfree(content);
                    p = json_skip_ws(end + 1);
                    if (*p == ',') p = json_skip_ws(p + 1);
                    continue;
                }
            }

            /* Tool results use the architecture's own turn role. */
            if (strcmp(role, "tool") == 0) {
                char *name_value = json_direct_value(p, "name");
                char *call_id_value = json_direct_value(p, "tool_call_id");
                if (tool_ops && tool_ops->result_role &&
                    strcmp(tool_ops->result_role, "tool") != 0) {
                    sfree(role);
                    role = sstrdup((char *)tool_ops->result_role);
                    if (tool_ops->result_prefix || tool_ops->result_suffix) {
                        const char *pre = tool_ops->result_prefix
                                        ? tool_ops->result_prefix : "";
                        const char *suf = tool_ops->result_suffix
                                        ? tool_ops->result_suffix : "";
                        size_t wrapped_len = strlen(pre) + strlen(content) +
                                             strlen(suf) + 1;
                        char *wrapped = smalloc(wrapped_len);
                        (void)snprintf(wrapped, wrapped_len, "%s%s%s",
                                       pre, content, suf);
                        sfree(content);
                        content = wrapped;
                    }
                    name_value = NULL;
                } else if ((!name_value || *name_value != '"') && call_id_value &&
                    *call_id_value == '"' && latest_call_id[0]) {
                    char *id_end = json_string_end(call_id_value);
                    int id_len = (int)(id_end - call_id_value - 2);
                    char id[sizeof(latest_call_id)];
                    if (id_len > 0 && id_len < (int)sizeof(id)) {
                        int decoded_id = json_unescape(id, call_id_value + 1, id_len);
                        id[decoded_id] = '\0';
                        if (strcmp(id, latest_call_id) == 0) {
                            name_value = NULL;
                            sfree(role);
                            role = sstrdup(latest_call_name);
                        }
                    }
                }
                if (name_value && *name_value == '"') {
                    char *name_end = json_string_end(name_value);
                    int name_len = (int)(name_end - name_value - 2);
                    if (name_end && name_len > 0 && name_len < TOOL_NAME_MAX) {
                        char *tool_name = smalloc((size_t)name_len + 1);
                        int decoded = json_unescape(tool_name,
                            name_value + 1, name_len);
                        tool_name[decoded] = '\0';
                        sfree(role);
                        role = tool_name;
                    }
                }
            }

            out[n].role    = role;
            out[n].content = content;
            out[n].end_marker = NULL;
            n++;
        }

        p = json_skip_ws(end + 1);
        if (*p == ',') p = json_skip_ws(p + 1);
    }
    return n;
}

static void chat_messages_free(ChatMessage *msgs, int n) {
    for (int i = 0; i < n; i++) {
        sfree((void *)msgs[i].role);
        sfree((void *)msgs[i].content);
    }
}

/* Resolve the output cap from the request.  max_completion_tokens (the
 * current OpenAI field) takes precedence over the legacy max_tokens;
 * absent or 0 keeps the session default.  Returns false for negatives. */
static bool request_max_tokens(char *body, u32 *out) {
    long v;
    if (!json_get_int(body, "max_completion_tokens", &v) &&
        !json_get_int(body, "max_tokens", &v))
        return true;
    if (v < 0) return false;
    if (v > 0) *out = (u32)v;
    return true;
}

/* Execute one request on a worker thread using its dedicated session.
 * `body` is the NUL-terminated HTTP body copied by the reader thread. */
static void handle_request(Engine *engine, Session *s, int fd, char *body) {
    Vocab   *v = engine->vocab;

    const ToolOps *tool_ops = s->ops.tools;

    /* 2. Parse the ordered conversation from the JSON body. */
    ChatMessage msgs[MAX_CHAT_MESSAGES + 1];
    int n_msgs = json_parse_messages(body, msgs, MAX_CHAT_MESSAGES, tool_ops);
    ToolDef tools[MAX_REQUEST_TOOLS];
    char *raw_tools = NULL;
    int raw_tools_len = 0;
    int n_tools = request_tools(body, tools, MAX_REQUEST_TOOLS,
                                &raw_tools, &raw_tools_len);
    char forced_tool[TOOL_NAME_MAX] = "";
    ToolChoice tool_choice = request_tool_choice(body, forced_tool,
                                                  sizeof(forced_tool));
    bool tools_enabled = n_tools > 0 && tool_choice != TOOL_CHOICE_NONE;
    if (tool_choice == TOOL_CHOICE_NAMED) {
        bool found = false;
        for (int i = 0; i < n_tools; i++)
            if (strcmp(forced_tool, tools[i].name) == 0) found = true;
        if (!found) {
            chat_messages_free(msgs, n_msgs);
            http_respond(fd, 400, "Bad Request",
                         "{\"error\":\"tool_choice names an unknown function\"}");
            close(fd);
            return;
        }
    }
    if (tools_enabled && !tool_ops) {
        char err[256];
        (void)snprintf(err, sizeof(err),
                       "{\"error\":\"Tool calling is not supported by the "
                       "%s architecture\"}", engine->model->arch_name);
        chat_messages_free(msgs, n_msgs);
        http_respond(fd, 400, "Bad Request", err);
        close(fd);
        return;
    }
    char *tool_prompt = tools_enabled
        ? tool_prompt_create(raw_tools, raw_tools_len, tool_choice,
                             forced_tool, tool_ops)
        : NULL;
    if (tools_enabled && !tool_prompt) {
        chat_messages_free(msgs, n_msgs);
        http_respond(fd, 500, "Internal Server Error",
                     "{\"error\":\"Could not construct tool prompt\"}");
        close(fd);
        return;
    }

    /* 3. Build chat prompt tokens from every turn, so earlier rounds
     * carried in "messages" are part of the prompt. */
    u32 prompt_tokens[4096];
    int max_pt = (int)(sizeof(prompt_tokens) / sizeof(prompt_tokens[0]));
    int n_prompt = 0;

    if (n_msgs > 0 && s->ops.chat_conversation) {
        if (tool_prompt) {
            /* The Qwen/Llama templates put the tools block inside the
             * leading system turn, followed by the caller's system text. */
            ChatMessage prompt_msgs[MAX_CHAT_MESSAGES + 1];
            int first_nonsys = 0;
            size_t sys_len = strlen(tool_prompt) + 1;
            while (first_nonsys < n_msgs && msgs[first_nonsys].role &&
                   strcmp(msgs[first_nonsys].role, "system") == 0) {
                if (msgs[first_nonsys].content)
                    sys_len += strlen(msgs[first_nonsys].content) + 2;
                first_nonsys++;
            }
            char *merged_sys = smalloc(sys_len);
            size_t used = (size_t)snprintf(merged_sys, sys_len, "%s",
                                           tool_prompt);
            for (int i = 0; i < first_nonsys; i++) {
                if (!msgs[i].content || !msgs[i].content[0]) continue;
                used += (size_t)snprintf(merged_sys + used, sys_len - used,
                                         "\n\n%s", msgs[i].content);
            }
            prompt_msgs[0].role = "system";
            prompt_msgs[0].content = merged_sys;
            prompt_msgs[0].end_marker = NULL;
            int rest = n_msgs - first_nonsys;
            if (rest > 0)
                memcpy(prompt_msgs + 1, msgs + first_nonsys,
                       (size_t)rest * sizeof(*msgs));
            n_prompt = s->ops.chat_conversation(s, prompt_msgs, rest + 1,
                                                prompt_tokens, max_pt);
            sfree(merged_sys);
        } else {
            n_prompt = s->ops.chat_conversation(s, msgs, n_msgs,
                                                prompt_tokens, max_pt);
        }
    }

    /* Fallback: bodies with no parseable "messages" array, or an arch
     * without a conversation template. */
    if (n_prompt == 0) {
        char user_buf[4096]  = {0};
        char sys_buf[1024]   = {0};
        const char *user_msg = "";
        const char *sys_msg  = NULL;

        /* Look for system message */
        int slen = 0;
        char *sys_raw = json_get_string(body, "content", &slen);
        if (sys_raw) {
            /* Check if this content belongs to a system-role message */
            char *role_search = sys_raw - 40; /* rough lookback */
            if (role_search < body) role_search = body;
            if (strstr(role_search, "\"system\"")) {
                int ul = json_unescape(sys_buf, sys_raw,
                                       slen < (int)sizeof(sys_buf) - 1
                                       ? slen : (int)sizeof(sys_buf) - 1);
                sys_buf[ul] = '\0';
                sys_msg = sys_buf;
            }
        }

        /* Look for all "content" fields — pick the last one preceded by "user" */
        {
            char *scan = body;
            char *last_user_content = NULL;
            int   last_user_len = 0;
            int   clen = 0;
            while ((scan = json_get_string(scan, "content", &clen)) != NULL) {
                /* Look backwards for "role": "user" within ~60 chars */
                char *check = scan - 60;
                if (check < body) check = body;
                if (strstr(check, "\"user\"")) {
                    last_user_content = scan;
                    last_user_len = clen;
                }
                scan += clen + 2;
            }
            if (last_user_content) {
                int ul = json_unescape(user_buf, last_user_content,
                                       last_user_len < (int)sizeof(user_buf) - 1
                                       ? last_user_len : (int)sizeof(user_buf) - 1);
                user_buf[ul] = '\0';
                user_msg = user_buf;
            }
        }
        if (!user_msg[0]) {
            chat_messages_free(msgs, n_msgs);
            http_respond(fd, 400, "Bad Request",
                         "{\"error\":\"No user message found\"}");
            close(fd);
            return;
        }

        char *combined_sys = NULL;
        if (tool_prompt) {
            size_t sys_len = sys_msg ? strlen(sys_msg) : 0;
            size_t size = strlen(tool_prompt) + sys_len + 3;
            combined_sys = smalloc(size);
            (void)snprintf(combined_sys, size, "%s%s%s", tool_prompt,
                           sys_len ? "\n\n" : "", sys_msg ? sys_msg : "");
            sys_msg = combined_sys;
        }
        n_prompt = s->ops.chat_prompt(s, user_msg, sys_msg,
                                      prompt_tokens, max_pt);
        sfree(combined_sys);
    }

    chat_messages_free(msgs, n_msgs);
    sfree(tool_prompt);

    if (n_prompt == 0) {
        http_respond(fd, 400, "Bad Request",
                    "{\"error\":\"No valid tokens in input\"}");
        close(fd);
        return;
    }

    /* 4. Reset session & prefill. */
    s->ops.reset(s);

    /* Output cap: an explicit request value overrides the session
     * default, but never exceeds the free context. */
    u32 max_tokens = s->max_tokens > 0 ? s->max_tokens : 256;
    if (!request_max_tokens(body, &max_tokens)) {
        http_respond(fd, 400, "Bad Request",
                     "{\"error\":\"max_tokens must be >= 0\"}");
        close(fd);
        return;
    }
    /* The batch is appended at the current position; the executor
     * extends the session KV cache. */
    u32 room = s->ctx_size - s->n_tokens;
    if (n_prompt > (int)room) {
        char err[256];
        (void)snprintf(err, sizeof(err),
                       "{\"error\":\"Context length exceeded: prompt is %d tokens, "
                       "but only %u tokens are available (context size %u)\"}",
                       n_prompt, room, s->ctx_size);
        http_respond(fd, 400, "Bad Request", err);
        close(fd);
        return;
    }
    bool ok = n_prompt > 0 && graph_execute(s, prompt_tokens, (u32)n_prompt, s->logits);
    if (!ok) {
        http_respond(fd, 500, "Internal Server Error",
                     "{\"error\":\"Forward pass failed\"}");
        close(fd);
        return;
    }
    /* Prefill consumed part of the context; clamp to what remains. */
    u32 gen_room = s->ctx_size - s->n_tokens;
    if (max_tokens > gen_room) max_tokens = gen_room;

    /* 5. Generate. */
    bool stream = json_get_bool(body, "stream");
    bool buffered_stream = stream && tools_enabled;
    char resp_body[65536];
    int  resp_used = 0;
    char utf8_pending[8];
    int  utf8_pending_len = 0;
    time_t now = time(NULL);

    u32 next_token = sample_token(s->logits, s->cfg.n_vocab, &s->sample,
                                    s->tokens, s->n_tokens);
    u32 n_gen = 0;
    i32 call_token = (tools_enabled && tool_ops && tool_ops->call_end)
                   ? vocab_lookup(v, tool_ops->call_end)
                   : (i32)VOCAB_ID_NONE;
    bool hit_tool_marker = false;

    /* The opening <think> is part of the prompt, not the sampled
     * output.  Echo it so clients see the complete reasoning block. */
    bool show_think = !s->no_think &&
                      v->think_start_id != (i32)VOCAB_ID_NONE &&
                      next_token != (u32)v->eos_id;

    if (stream && !buffered_stream) {
        http_stream_start(fd, 200, "OK");
        sse_emit_role(fd, (long)now);
        if (show_think)
            sse_emit_content(fd, (long)now, "<think>\n", 8);
    } else if (show_think) {
        memcpy(resp_body, "<think>\n", 8);
        resp_used = 8;
    }

    for (u32 i = 0; i < max_tokens; i++) {
        if (call_token != (i32)VOCAB_ID_NONE && next_token == (u32)call_token) {
            hit_tool_marker = true;
            break;
        }
        if (next_token == (u32)v->eos_id) break;
        if (next_token < v->n_vocab && v->token[next_token].content) {
            Key *tk = &v->token[next_token];
            if (stream && !buffered_stream) {
                char piece[2048];
                int plen = s->ops.decode
                    ? s->ops.decode((const u8 *)tk->content, (int)tk->len,
                                    piece, (int)sizeof(piece) - 1)
                    : 0;
                if (plen < 0) plen = 0;
                if (plen > (int)sizeof(piece) - 1) plen = (int)sizeof(piece) - 1;

                char merged[2048 + 8];
                int mlen = utf8_pending_len;
                memcpy(merged, utf8_pending, (size_t)utf8_pending_len);
                if (plen > (int)sizeof(merged) - mlen)
                    plen = (int)sizeof(merged) - mlen;
                memcpy(merged + mlen, piece, (size_t)plen);
                mlen += plen;

                int emit = utf8_complete_prefix(merged, mlen);
                if (emit > 0) sse_emit_content(fd, (long)now, merged, emit);
                utf8_pending_len = mlen - emit;
                memcpy(utf8_pending, merged + emit, (size_t)utf8_pending_len);
            } else {
                int remaining = (int)sizeof(resp_body) - resp_used - 1;
                int added = s->ops.decode
                    ? s->ops.decode((const u8 *)tk->content, (int)tk->len,
                                    resp_body + resp_used, remaining)
                    : 0;
                if (added > 0) {
                    resp_used += added;
                    resp_body[resp_used] = '\0';
                    /* Weak models keep generating after the closing tag
                     * (repeating </tool_call> or rambling), which makes
                     * the strict parser reject an otherwise valid call.
                     * Stop as soon as the terminator is complete. */
                    if (tools_enabled &&
                        (strstr(resp_body, "</tool_call>") ||
                         strstr(resp_body, "</function>"))) {
                        hit_tool_marker = true;
                        break;
                    }
                }
            }
        }
        n_gen++;
        /* Incremental decode: one new row at the current position;
         * attention reads the cached history. */
        if (s->n_tokens >= s->ctx_size) break;
        if (!graph_execute(s, &next_token, 1, s->logits)) break;
        next_token = sample_token(s->logits, s->cfg.n_vocab, &s->sample,
                                    s->tokens, s->n_tokens);
    }

    bool hit_eos = next_token == (u32)v->eos_id;
    resp_body[resp_used] = '\0';

    /* A model may print the call terminator as text when the vocabulary
     * lacks it as a special token; strip it before parsing. */
    int parse_len = resp_used;
    if (tool_ops && tool_ops->call_end) {
        int marker_len = (int)strlen(tool_ops->call_end);
        if (parse_len >= marker_len &&
            memcmp(resp_body + parse_len - marker_len,
                   tool_ops->call_end, (size_t)marker_len) == 0)
            parse_len -= marker_len;
    }
    ToolCall tool_call;
    bool has_tool_call = tools_enabled && tool_ops && tool_ops->parse &&
        tool_ops->parse(resp_body, parse_len, tools, n_tools, &tool_call);
    if (has_tool_call) {
        hit_tool_marker = true;
        unsigned long long seq = atomic_fetch_add_explicit(
            &g_tool_call_seq, 1, memory_order_relaxed);
        (void)snprintf(tool_call.id, sizeof(tool_call.id), "call_%ld_%ld_%llu",
                       (long)now, (long)getpid(), seq);
    }

    /* 6a. Stream the response as SSE events. */
    if (stream) {
        if (buffered_stream) {
            http_stream_start(fd, 200, "OK");
            sse_emit_role(fd, (long)now);
            if (has_tool_call) {
                sse_emit_tool_call(fd, (long)now, &tool_call);
                sse_emit_done(fd, (long)now, "tool_calls");
            } else {
                int off = 0;
                while (off < resp_used) {
                    int chunk = resp_used - off;
                    if (chunk > 12000) chunk = 12000;
                    int complete = utf8_complete_prefix(resp_body + off, chunk);
                    if (complete == 0) complete = chunk;
                    sse_emit_content(fd, (long)now, resp_body + off, complete);
                    off += complete;
                }
                sse_emit_done(fd, (long)now,
                              (hit_eos || hit_tool_marker) ? "stop" : "length");
            }
        } else {
            if (utf8_pending_len > 0)
                sse_emit_content(fd, (long)now, utf8_pending, utf8_pending_len);
            sse_emit_done(fd, (long)now,
                          (hit_eos || hit_tool_marker) ? "stop" : "length");
        }
        close(fd);
        return;
    }

    /* 6b. Build OpenAI-compatible JSON response. */
    char json_buf[65536];
    if (has_tool_call) {
        char escaped_name[1024];
        int name_len = json_escape_str(escaped_name, sizeof(escaped_name) - 1,
                                       tool_call.name, (int)strlen(tool_call.name));
        escaped_name[name_len] = '\0';
        char escaped_args[MAX_TOOL_ARGUMENTS * 6 + 1];
        int args_len = json_escape_str(escaped_args, sizeof(escaped_args) - 1,
                                       tool_call.arguments,
                                       tool_call.arguments_len);
        escaped_args[args_len] = '\0';
        (void)snprintf(json_buf, sizeof(json_buf),
            "{\"id\":\"chatcmpl-%ld\",\"object\":\"chat.completion\","
            "\"created\":%ld,\"model\":\"default\",\"choices\":[{"
            "\"index\":0,\"message\":{\"role\":\"assistant\","
            "\"content\":null,\"tool_calls\":[{\"id\":\"%s\","
            "\"type\":\"function\",\"function\":{\"name\":\"%s\","
            "\"arguments\":\"%s\"}}]},\"finish_reason\":\"tool_calls\"}],"
            "\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%u,"
            "\"total_tokens\":%d}}",
            (long)now, (long)now, tool_call.id, escaped_name, escaped_args,
            n_prompt, n_gen, n_prompt + (int)n_gen);
        http_respond(fd, 200, "OK", json_buf);
        close(fd);
        return;
    }

    /* JSON-escape the generated text.  Reserve 256 bytes for the
     * JSON wrapper so the final snprintf can't overflow json_buf. */
    char escaped[65536 - JSON_WRAPPER_RESERVE];
    int esc_len = json_escape_str(escaped, (int)sizeof(escaped) - 1,
                                  resp_body, resp_used);
    escaped[esc_len] = '\0';

    (void)snprintf(json_buf, sizeof(json_buf),
                  "{"
                  "\"id\":\"chatcmpl-%ld\","
                  "\"object\":\"chat.completion\","
                  "\"created\":%ld,"
                  "\"model\":\"default\","
                  "\"choices\":[{"
                      "\"index\":0,"
                      "\"message\":{"
                          "\"role\":\"assistant\","
                          "\"content\":\"%s\""
                      "},"
                      "\"finish_reason\":\"%s\""
                  "}],"
                  "\"usage\":{"
                      "\"prompt_tokens\":%d,"
                      "\"completion_tokens\":%u,"
                      "\"total_tokens\":%d"
                  "}"
                  "}",
                  (long)now, (long)now,
                  escaped,
                  (hit_eos || hit_tool_marker) ? "stop" : "length",
                  n_prompt, n_gen, n_prompt + (int)n_gen);

    http_respond(fd, 200, "OK", json_buf);

    close(fd);
}

/* ---- Request worker pool -----------------------------------------
 * The event loop only accepts connections and reads complete requests;
 * inference runs on n_parallel worker threads, each owning a private
 * Session (tokens, logits, KV cache, graph arena) so requests no longer
 * reset each other's state. */

static bool worker_pool_submit(WorkerPool *pool, int fd, char *body) {
    bool ok = true;
    pthread_mutex_lock(&pool->lock);
    if (pool->stop || pool->count == JOB_QUEUE_CAP) {
        ok = false;
    } else {
        pool->queue[pool->tail].fd   = fd;
        pool->queue[pool->tail].body = body;
        pool->tail = (pool->tail + 1) % JOB_QUEUE_CAP;
        pool->count++;
        pthread_cond_signal(&pool->cond);
    }
    pthread_mutex_unlock(&pool->lock);
    return ok;
}

static void *worker_main(void *arg) {
    Worker *w = (Worker *)arg;
    WorkerPool *pool = w->pool;
    Session *s = pool->sessions[w->id];

    for (;;) {
        pthread_mutex_lock(&pool->lock);
        while (pool->count == 0 && !pool->stop)
            pthread_cond_wait(&pool->cond, &pool->lock);
        if (pool->count == 0 && pool->stop) {
            pthread_mutex_unlock(&pool->lock);
            break;
        }
        ClientJob job = pool->queue[pool->head];
        pool->head = (pool->head + 1) % JOB_QUEUE_CAP;
        pool->count--;
        pthread_mutex_unlock(&pool->lock);

        /* Drain remaining work even after shutdown was requested. */
        handle_request(pool->engine, s, job.fd, job.body);
        sfree(job.body);
    }
    return NULL;
}

static WorkerPool *worker_pool_create(Engine *engine, Session **sessions, int n) {
    WorkerPool *pool = scalloc(1, sizeof(*pool));
    pthread_mutex_init(&pool->lock, NULL);
    pthread_cond_init(&pool->cond, NULL);
    pool->engine    = engine;
    pool->sessions  = sessions;
    pool->n_workers = n;
    pool->workers   = scalloc((size_t)n, sizeof(Worker));

    for (int i = 0; i < n; i++) {
        pool->workers[i].pool = pool;
        pool->workers[i].id   = i;
        if (pthread_create(&pool->workers[i].thread, NULL, worker_main,
                           &pool->workers[i]) != 0) {
            slog(ERROR, "Failed to create request worker %d", i);
        }
    }
    return pool;
}

static void worker_pool_destroy(WorkerPool *pool) {
    if (!pool) return;
    pthread_mutex_lock(&pool->lock);
    pool->stop = true;
    pthread_cond_broadcast(&pool->cond);
    pthread_mutex_unlock(&pool->lock);
    for (int i = 0; i < pool->n_workers; i++)
        pthread_join(pool->workers[i].thread, NULL);
    pthread_mutex_destroy(&pool->lock);
    pthread_cond_destroy(&pool->cond);
    sfree(pool->workers);
    sfree(pool);
}

/* Callback for a readable client socket: read one complete request,
 * hand the body to the worker pool, and stop watching the fd. */
static void client_read_proc(EventLoop *el, int fd, int mask, void *privdata) {
    UNUSED(mask);
    Server *server = (Server *)privdata;

    char *body = NULL;
    HttpBodyResult res = http_read_body(fd, &body);
    if (res != HTTP_BODY_OK) {
        int status = 400;
        const char *status_msg = "Bad Request";
        const char *error = "Bad Request";

        switch (res) {
            case HTTP_BODY_TOO_LARGE:
                status = 413;
                status_msg = "Payload Too Large";
                error = "Request body too large";
                break;
            case HTTP_BODY_INCOMPLETE_HEADER:
                error = "Incomplete HTTP request";
                break;
            case HTTP_BODY_EMPTY:
                error = "Empty HTTP body";
                break;
            case HTTP_BODY_INCOMPLETE:
                error = "Incomplete HTTP body";
                break;
            default:
                break;
        }

        char err[256];
        (void)snprintf(err, sizeof(err), "{\"error\":\"%s\"}", error);
        http_respond(fd, status, status_msg, err);
        delete_file_event(el, fd, ELOOP_READABLE);
        close(fd);
        return;
    }

    /* The worker now owns the fd; the event loop must not touch it. */
    delete_file_event(el, fd, ELOOP_READABLE);
    if (!worker_pool_submit(server->pool, fd, body)) {
        sfree(body);
        http_respond(fd, 503, "Service Unavailable",
                     "{\"error\":\"Server busy\"}");
        close(fd);
    }
}


/* Callback for accepting new connections. */
static void accept_proc(EventLoop *el, int fd, int mask, void *privdata) {
    UNUSED(mask);
    Server *server = (Server *)privdata;

    int cfd, port;
    char ip[128];

    cfd = server_accept(fd, ip, &port);
    if (cfd == ANET_ERR) slog_errno("Accept failed");
    slog(INFO, "Accepted %s:%d attached.", ip, port, pthread_self());
    if (create_file_event(el, cfd, ELOOP_READABLE, client_read_proc, server) == ELOOP_ERR)
        slog(ERROR, "Create file event fail.");
}

/* Useage. */
static void usage(FILE *file, int exit_code) {
    fprintf(file, "Usage:   server [options]\n");
    fprintf(file, "Example: server -m model.gguf -p 9987 -t 0.9 -tp 0.9\n");
    fprintf(file, "Options:\n");
    fprintf(file, "  -m  | --model             <string>  The model file path to run\n");
    fprintf(file, "  -h  | --host              <string>  The server host, defalt 127.0.0.1\n");
    fprintf(file, "  -p  | --port              <int>     The Port to listening, default 9987\n");
    fprintf(file, "  -c  | --ctx               <int>     The context size, default auto (model context, capped by VRAM)\n");
    fprintf(file, "  -ub | --ubatch            <int>     Max tokens per forward pass, default 512\n");
    fprintf(file, "  -n  | --tokens            <int>     The default token size, defalt 393216\n");
    fprintf(file, "  -t  | --temp              <float>   Temperature for sampling, default 0.8\n");
    fprintf(file, "  -T  | --threads           <int>     Number of threads, default CPU cores\n");
    fprintf(file, "  -np | --parallel          <int>     Concurrent request slots, default 1\n");
    fprintf(file, "  -nt | --no-thinking       <none>    Disable thinking mode\n");
    fprintf(file, "  -ngl| --n-gpu-layers      <int>     Layers on GPU (-1=auto, 0=CPU, default -1)\n");
    fprintf(file, "  -tp | --topp              <float>   Top-p (nucleus) threshold, default 0.9\n");
    fprintf(file, "  -tk | --topk              <int>     Top-k sampling, default 40\n");
    fprintf(file, "  -P  | --minp              <float>   Min-p threshold, default 0.05\n");
    fprintf(file, "  -rp | --repeat-penalty    <float>   Repeat penalty, default 1.0 (1.0=disabled)\n");
    fprintf(file, "  -rl | --repeat-last-n     <int>     Repeat penalty lookback, default 64\n");
    fprintf(file, "  -fp | --frequency-penalty <float>   Frequency penalty, default 0.0 (0.0=disabled)\n");
    fprintf(file, "  -pp | --presence-penalty  <float>   Presence penalty, default 0.0 (0.0=disabled)\n");
    fprintf(file, "  -i  | --inspect           <none>    Inspect the engine/model\n");
    fprintf(file, "  -g  | --graph             <none>    Draw the graph workflow and exit\n");
    exit(exit_code);
}

/* Parse current arg value. */
static char *parse_arg(int argc, char *argv[], int *i, const char *opt) {
    if (*i + 1 >= argc) 
        slog(ERROR, "Missing value for option [%s]", opt);
    return argv[++(*i)];
}

/* Parse options. */
static ServerOptions parse_options(int argc, char *argv[]) {
    ServerOptions so = {
        .host = "127.0.0.1",
        .port = 8080,
        .ctx_size = 0,
        .n_ubatch = DEFAULT_N_UBATCH,
        .default_tokens = 393216,
        .sample = {
            .temperature = 0.8f,
            .top_k = DEFAULT_TOP_K,
            .top_p = DEFAULT_TOP_P,
            .min_p = DEFAULT_MIN_P,
            .repeat_penalty = DEFAULT_REPEAT_PENALTY,
            .repeat_last_n = DEFAULT_REPEAT_LAST_N,
            .frequency_penalty = DEFAULT_FREQUENCY_PENALTY,
            .presence_penalty = DEFAULT_PRESENCE_PENALTY,
        },
        .nthread = DEFAULT_N_THREAD,
        .n_gpu_layers = -1,
        .n_parallel = DEFAULT_N_PARALLEL
    };
    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) usage(stdout, EXIT_SUCCESS);
        else if (!strcmp(arg, "-m")   || !strcmp(arg, "--model")) so.engine.model_path = parse_arg(argc, argv, &i, arg);
        else if (!strcmp(arg, "-h")   || !strcmp(arg, "--host")) so.host = parse_arg(argc, argv, &i, arg);
        else if (!strcmp(arg, "-p")   || !strcmp(arg, "--port")) so.port = parse_int(parse_arg(argc, argv, &i, arg));
        else if (!strcmp(arg, "-c")   || !strcmp(arg, "--ctx")) so.ctx_size = parse_int(parse_arg(argc, argv, &i, arg));
        else if (!strcmp(arg, "-ub")  || !strcmp(arg, "--ubatch")) so.n_ubatch = parse_int(parse_arg(argc, argv, &i, arg));
        else if (!strcmp(arg, "-n")   || !strcmp(arg, "--tokens")) so.default_tokens = parse_int(parse_arg(argc, argv, &i, arg));
        else if (!strcmp(arg, "-t")   || !strcmp(arg, "--temp")) so.sample.temperature = parse_float(parse_arg(argc, argv, &i, arg));
        else if (!strcmp(arg, "-T")   || !strcmp(arg, "--threads")) so.nthread = parse_int(parse_arg(argc, argv, &i, arg));
        else if (!strcmp(arg, "-np")  || !strcmp(arg, "--parallel")) so.n_parallel = parse_int(parse_arg(argc, argv, &i, arg));
        else if (!strcmp(arg, "-nt")  || !strcmp(arg, "--no-thinking")) so.no_think = true;
        else if (!strcmp(arg, "-ngl") || !strcmp(arg, "--n-gpu-layers")) so.n_gpu_layers = parse_int(parse_arg(argc, argv, &i, arg));
        else if (!strcmp(arg, "-tp")  || !strcmp(arg, "--topp")) so.sample.top_p = parse_float(parse_arg(argc, argv, &i, arg));
        else if (!strcmp(arg, "-tk")  || !strcmp(arg, "--topk")) so.sample.top_k = (u32)parse_int(parse_arg(argc, argv, &i, arg));
        else if (!strcmp(arg, "-P")   || !strcmp(arg, "--minp")) so.sample.min_p = parse_float(parse_arg(argc, argv, &i, arg));
        else if (!strcmp(arg, "-rp")  || !strcmp(arg, "--repeat-penalty")) so.sample.repeat_penalty = parse_float(parse_arg(argc, argv, &i, arg));
        else if (!strcmp(arg, "-rl")  || !strcmp(arg, "--repeat-last-n")) so.sample.repeat_last_n = (u32)parse_int(parse_arg(argc, argv, &i, arg));
        else if (!strcmp(arg, "-fp")  || !strcmp(arg, "--frequency-penalty")) so.sample.frequency_penalty = parse_float(parse_arg(argc, argv, &i, arg));
        else if (!strcmp(arg, "-pp")  || !strcmp(arg, "--presence-penalty")) so.sample.presence_penalty = parse_float(parse_arg(argc, argv, &i, arg));
        else if (!strcmp(arg, "-i")   || !strcmp(arg, "--inspect")) { so.inspect = true; so.engine.inspect = true; }
        else if (!strcmp(arg, "-g")   || !strcmp(arg, "--graph")) so.show_graph = true;
        else {
            fprintf(stderr, "Unkonow option: %s.\n", arg);
            usage(stderr, 2);
        }
    }
    if (so.nthread <= 0) {
        long n = sysconf(_SC_NPROCESSORS_ONLN);
        so.nthread = (int)(n > 0 ? n : 1);
    }
    if (so.n_parallel <= 0) {
        fprintf(stderr, "Parallel slots must be >= 1, got %d\n", so.n_parallel);
        usage(stderr, 2);
    }
    if (so.sample.temperature < 0.0f) {
        fprintf(stderr, "Temperature must be >= 0.0, got %.2f\n", so.sample.temperature);
        usage(stderr, 2);
    }
    if (so.sample.top_p <= 0.0f || so.sample.top_p > 1.0f) {
        fprintf(stderr, "Top-p must be in (0.0, 1.0], got %.2f\n", so.sample.top_p);
        usage(stderr, 2);
    }
    if (so.sample.min_p < 0.0f || so.sample.min_p > 1.0f) {
        fprintf(stderr, "Min-p must be in [0.0, 1.0], got %.2f\n", so.sample.min_p);
        usage(stderr, 2);
    }
    return so;
}


static int setup_server_eventloop(Server *server, ServerOptions so) {
    int fd, retval;

    server->el = smalloc(sizeof(EventLoop));
    server->el->numkeys = 0;
    server->el->maxfd = 0;
    server->el->fileEventHead = NULL;
    server->el->stop = false;

    fd = create_tcp_server(so.host, so.port);
    if (fd == ANET_ERR) {
        slog(WARN, "Create tcp socket server fail");
        return ELOOP_ERR;
    }

    retval = create_file_event(server->el, fd, ELOOP_READABLE, accept_proc, server);
    if (retval == ELOOP_ERR) return ELOOP_ERR;
    server->serverfd = fd;

    return ELOOP_OK;
}


static void sessions_destroy(Session **sessions, int n) {
    if (!sessions) return;
    for (int i = 0; i < n; i++) {
        if (!sessions[i]) continue;
        graph_free(sessions[i]->graph);
        sessions[i]->graph = NULL;
        session_free(sessions[i]);
    }
    sfree(sessions);
}

static void server_resource_close(Server *server) {
    /* Workers must stop before the sessions they use are freed. */
    worker_pool_destroy(server->pool);
    sessions_destroy(server->sessions, server->n_parallel);
    engine_close(server->engine);
    el_free(server->el);
    if (server->serverfd >= 0) close(server->serverfd);
    if (g_event_fd >= 0) close(g_event_fd);
#ifndef __linux__
    if (g_signal_wfd >= 0) close(g_signal_wfd);
#endif
}

int main(int argc, char *argv[]) {
    banner_print();

    /* 1. Parse server options. */
    if (argc < 2) usage(stderr, 2);
    signal(SIGPIPE, SIG_IGN);
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = signal_handler;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);

    srand((unsigned)time(NULL) ^ (unsigned)getpid());
    ServerOptions so = parse_options(argc, argv);

    /* 2. Load model. */
    Engine *engine = engine_open(&so.engine);
    if (so.inspect) { 
        engine_summary(engine);
        engine_close(engine);
        return 0;
    }

    /* No explicit -c: use the model's context length when it fits in
     * device memory, otherwise the largest value that fits across all
     * parallel slots. */
    if (so.ctx_size <= 0)
        so.ctx_size = (int)engine_auto_ctx_size(engine, (u32)so.n_ubatch,
                                                so.n_gpu_layers,
                                                so.n_parallel);

    /* 3. Create one session per concurrent request slot.  Each owns its
     * own tokens, logits, KV cache and graph arena, so requests never
     * clobber each other.  The -T thread budget is split across slots. */
    int threads_per_session = so.nthread;
    if (so.n_parallel > 1) {
        threads_per_session = so.nthread / so.n_parallel;
        if (threads_per_session < 1) threads_per_session = 1;
    }

    Session **sessions = scalloc((size_t)so.n_parallel, sizeof(*sessions));
    for (int i = 0; i < so.n_parallel; i++) {
        Session *s = session_create(engine, (u32)so.ctx_size,
                                    (u32)so.n_ubatch, threads_per_session,
                                    so.n_gpu_layers, so.n_parallel);
        if (!s) {
            sessions_destroy(sessions, i);
            engine_close(engine);
            slog(ERROR, "Failed to create session %d/%d.", i + 1, so.n_parallel);
        }
        s->max_tokens = so.default_tokens;
        s->sample     = so.sample;
        s->no_think   = so.no_think;
        sessions[i]   = s;

        /* Warmup: build the graph, plan the arena and preload weights
         * onto the device backend so the first request only pays
         * compute.  Done serially: GPU weight caches are global. */
        if (!graph_prepare(s))
            slog(WARN, "Graph warmup failed for session %d; first request "
                 "will be slower", i + 1);
    }
    if (so.n_parallel > 1)
        slog(INFO, "Parallel slots: %d (threads per slot: %d)",
             so.n_parallel, threads_per_session);

    /* Show what we loaded, like the chat CLI does. */
    engine_print_info(engine, sessions[0], so.engine.model_path, true);

#if defined(GPU_BUILD) || defined(METAL_BUILD)
    if (so.n_parallel > 1)
        slog(WARN, "GPU backend shares global scratch buffers; forward "
             "passes across the %d slots will be serialised", so.n_parallel);
#endif

    /* Draw the workflow and exit (diagnostic, like --inspect). */
    if (so.show_graph) {
        graph_drawer(sessions[0]->graph);
        sessions_destroy(sessions, so.n_parallel);
        engine_close(engine);
        return 0;
    }

    /* 4. Set up server, worker pool & signal handling. */
    Server server;
    memset(&server, 0, sizeof(server));
    server.engine     = engine;
    server.sessions   = sessions;
    server.n_parallel = so.n_parallel;
    server.serverfd   = -1;

    if (setup_server_eventloop(&server, so) == ELOOP_ERR) {
        server_resource_close(&server);
        slog(ERROR, "Failed to setup server event loop.");
    }
    server.pool = worker_pool_create(engine, sessions, so.n_parallel);

#ifdef __linux__
    g_event_fd = eventfd(0, EFD_NONBLOCK);
#else
    int sigpipe[2];
    if (pipe(sigpipe) == 0) {
        g_event_fd = sigpipe[0];
        g_signal_wfd = sigpipe[1];
        fcntl(g_event_fd, F_SETFL, fcntl(g_event_fd, F_GETFL, 0) | O_NONBLOCK);
        fcntl(g_signal_wfd, F_SETFL, fcntl(g_signal_wfd, F_GETFL, 0) | O_NONBLOCK);
    } else {
        g_event_fd = -1;
    }
#endif
    if (g_event_fd >= 0) {
        create_file_event(server.el, g_event_fd, ELOOP_READABLE, signal_callback, NULL);
    }

    slog(INFO, "Server: listening on http://%s:%d", so.host, so.port);

    /* 5. Run event loop (blocks until shutdown). */
    el_main(server.el);

    /* 6. Cleanup. */
    slog(INFO, "Server: shutdown requested, draining requests");
    server_resource_close(&server);
    return 0;
}
