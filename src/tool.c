#include <ctype.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "def.h"
#include "mm.h"
#include "tool.h"

/* ---- Bounded JSON scanning helpers ----------------------------- */

static const char *tool_skip_ws(const char *p, const char *limit) {
    while (p < limit &&
           (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
    return p;
}

/* `p` points at a JSON string.  Returns the position just past its closing
 * quote, or NULL when the string is unterminated or mis-escaped. */
static const char *tool_string_end(const char *p, const char *limit) {
    if (p >= limit || *p != '"') return NULL;
    for (p++; p < limit; p++) {
        unsigned char c = (unsigned char)*p;
        if (c == '"') return p + 1;
        if (c < 0x20) return NULL;
        if (c == '\\') {
            p++;
            if (p >= limit) return NULL;
            if (*p == 'u') {
                for (int i = 0; i < 4; i++)
                    if (p + 1 + i >= limit ||
                        !isxdigit((unsigned char)p[1 + i]))
                        return NULL;
                p += 4;
            } else if (!strchr("\\\"/bfnrt", *p)) {
                return NULL;
            }
        }
    }
    return NULL;
}

static const char *tool_value_end(const char *p, const char *limit, int depth) {
    if (depth > 64) return NULL;
    p = tool_skip_ws(p, limit);
    if (p >= limit) return NULL;
    if (*p == '"') return tool_string_end(p, limit);
    if (*p == '{') {
        p = tool_skip_ws(p + 1, limit);
        if (p < limit && *p == '}') return p + 1;
        for (;;) {
            p = tool_string_end(p, limit);
            if (!p) return NULL;
            p = tool_skip_ws(p, limit);
            if (p >= limit || *p++ != ':') return NULL;
            p = tool_value_end(p, limit, depth + 1);
            if (!p) return NULL;
            p = tool_skip_ws(p, limit);
            if (p >= limit) return NULL;
            if (*p == '}') return p + 1;
            if (*p++ != ',') return NULL;
            p = tool_skip_ws(p, limit);
        }
    }
    if (*p == '[') {
        p = tool_skip_ws(p + 1, limit);
        if (p < limit && *p == ']') return p + 1;
        for (;;) {
            p = tool_value_end(p, limit, depth + 1);
            if (!p) return NULL;
            p = tool_skip_ws(p, limit);
            if (p >= limit) return NULL;
            if (*p == ']') return p + 1;
            if (*p++ != ',') return NULL;
        }
    }
    if (limit - p >= 4 && memcmp(p, "true", 4) == 0) return p + 4;
    if (limit - p >= 5 && memcmp(p, "false", 5) == 0) return p + 5;
    if (limit - p >= 4 && memcmp(p, "null", 4) == 0) return p + 4;
    const char *start = p;
    if (*p == '-') p++;
    if (p < limit && *p == '0') p++;
    else {
        if (p >= limit || *p < '1' || *p > '9') return NULL;
        while (p < limit && *p >= '0' && *p <= '9') p++;
    }
    if (p < limit && *p == '.') {
        p++;
        if (p >= limit || *p < '0' || *p > '9') return NULL;
        while (p < limit && *p >= '0' && *p <= '9') p++;
    }
    if (p < limit && (*p == 'e' || *p == 'E')) {
        p++;
        if (p < limit && (*p == '+' || *p == '-')) p++;
        if (p >= limit || *p < '0' || *p > '9') return NULL;
        while (p < limit && *p >= '0' && *p <= '9') p++;
    }
    return p == start ? NULL : p;
}

/* Find a direct field of the object at `obj`.  Nested objects are not
 * searched.  On success stores the value span in [*value, *value_end). */
static bool tool_direct_value(const char *obj, const char *limit,
                              const char *key,
                              const char **value, const char **value_end) {
    if (obj >= limit || *obj != '{') return false;
    const char *end = tool_value_end(obj, limit, 0);
    if (!end) return false;
    const char *p = tool_skip_ws(obj + 1, end - 1);
    size_t key_len = strlen(key);
    while (p < end - 1) {
        if (*p != '"') return false;
        const char *key_end = tool_string_end(p, end - 1);
        if (!key_end) return false;
        const char *colon = tool_skip_ws(key_end, end - 1);
        if (colon >= end - 1 || *colon != ':') return false;
        const char *v = tool_skip_ws(colon + 1, end - 1);
        const char *ve = tool_value_end(v, end - 1, 0);
        if (!ve) return false;
        if ((size_t)(key_end - p - 2) == key_len &&
            memcmp(p + 1, key, key_len) == 0) {
            *value = v;
            *value_end = ve;
            return true;
        }
        p = tool_skip_ws(ve, end - 1);
        if (p >= end - 1) break;
        if (*p++ != ',') return false;
        p = tool_skip_ws(p, end - 1);
    }
    return false;
}

static int tool_unescape(char *dst, const char *src, int len, int cap) {
    int w = 0;
    for (int i = 0; i < len; i++) {
        if (src[i] == '\\' && i + 1 < len) {
            i++;
            if (w >= cap - 1) break;
            switch (src[i]) {
                case '"':  dst[w++] = '"';  break;
                case '\\': dst[w++] = '\\'; break;
                case '/':  dst[w++] = '/';  break;
                case 'b':  dst[w++] = '\b'; break;
                case 'f':  dst[w++] = '\f'; break;
                case 'n':  dst[w++] = '\n'; break;
                case 'r':  dst[w++] = '\r'; break;
                case 't':  dst[w++] = '\t'; break;
                case 'u':
                    dst[w++] = '?';
                    i += 4;
                    break;
                default:   dst[w++] = src[i]; break;
            }
        } else if (w < cap - 1) {
            dst[w++] = src[i];
        }
    }
    dst[w] = '\0';
    return w;
}

static bool tool_name_known(const char *name, const ToolDef *tools,
                            int n_tools) {
    for (int i = 0; i < n_tools; i++)
        if (strcmp(name, tools[i].name) == 0) return true;
    return false;
}

static bool tool_args_copy(ToolCall *out, const char *src, int len) {
    if (len < 0 || len >= TOOL_ARGS_MAX) return false;
    memcpy(out->arguments, src, (size_t)len);
    out->arguments[len] = '\0';
    out->arguments_len = len;
    return true;
}

/* ---- Small bounded string builder ------------------------------ */

typedef struct {
    char  *buf;
    size_t len;
    size_t cap;
} ToolBuf;

static bool tb_reserve(ToolBuf *b, size_t extra) {
    if (b->len + extra + 1 <= b->cap) return true;
    size_t cap = b->cap ? b->cap : 256;
    while (cap < b->len + extra + 1) cap *= 2;
    b->buf = srealloc(b->buf, cap);
    b->cap = cap;
    return true;
}

static bool tb_put(ToolBuf *b, const char *s, size_t n) {
    if (!tb_reserve(b, n)) return false;
    memcpy(b->buf + b->len, s, n);
    b->len += n;
    b->buf[b->len] = '\0';
    return true;
}

static bool tb_puts(ToolBuf *b, const char *s) {
    return tb_put(b, s, strlen(s));
}

static bool tb_printf(ToolBuf *b, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    va_list cp;
    va_copy(cp, ap);
    int need = vsnprintf(NULL, 0, fmt, cp);
    va_end(cp);
    if (need < 0 || !tb_reserve(b, (size_t)need)) {
        va_end(ap);
        return false;
    }
    vsnprintf(b->buf + b->len, b->cap - b->len, fmt, ap);
    va_end(ap);
    b->len += (size_t)need;
    return true;
}

static bool tb_json_string(ToolBuf *b, const char *s, int len) {
    if (!tb_put(b, "\"", 1)) return false;
    for (int i = 0; i < len; i++) {
        unsigned char c = (unsigned char)s[i];
        switch (c) {
            case '"':  if (!tb_puts(b, "\\\"")) return false; break;
            case '\\': if (!tb_puts(b, "\\\\")) return false; break;
            case '\n': if (!tb_puts(b, "\\n"))  return false; break;
            case '\r': if (!tb_puts(b, "\\r"))  return false; break;
            case '\t': if (!tb_puts(b, "\\t"))  return false; break;
            case '\b': if (!tb_puts(b, "\\b"))  return false; break;
            case '\f': if (!tb_puts(b, "\\f"))  return false; break;
            default:
                if (c < 0x20) {
                    if (!tb_printf(b, "\\u%04x", c)) return false;
                } else if (!tb_put(b, (const char *)&s[i], 1)) {
                    return false;
                }
                break;
        }
    }
    return tb_puts(b, "\"");
}

static char *tb_take(ToolBuf *b) {
    if (!b->buf) return sstrdup("");
    return b->buf;
}

static void tb_free(ToolBuf *b) {
    if (b->buf) sfree(b->buf);
    memset(b, 0, sizeof(*b));
}

/* Decide whether an XML parameter body is already a JSON value. */
static bool tool_json_literal(const char *v, const char *limit) {
    if (v >= limit) return false;
    if (!strchr("{[tfn-0123456789\"", *v)) return false;
    const char *end = tool_value_end(v, limit, 0);
    return end && tool_skip_ws(end, limit) == limit;
}

/* Append a parameter value to a JSON object under construction. */
static bool tb_json_value(ToolBuf *b, const char *v, const char *limit) {
    if (tool_json_literal(v, limit))
        return tb_put(b, v, (size_t)(limit - v));
    return tb_json_string(b, v, (int)(limit - v));
}

/* ---- JSON call object ------------------------------------------ */

/* Parse:
 *   {"name": "...", "arguments"|"parameters": {...}}
 *   {"function": {"name": "...", "arguments"|"parameters": {...}}}
 * Also accepts `arguments` as a JSON string containing an object. */
static bool tool_parse_call_object(const char *json, int json_len,
                                   const ToolDef *tools, int n_tools,
                                   ToolCall *out) {
    const char *limit = json + json_len;
    const char *p = tool_skip_ws(json, limit);
    if (p >= limit || *p != '{') return false;
    const char *obj_end = tool_value_end(p, limit, 0);
    if (!obj_end || tool_skip_ws(obj_end, limit) != limit) return false;

    const char *name = NULL, *name_end = NULL;
    const char *args = NULL, *args_end = NULL;
    if (!tool_direct_value(p, obj_end, "name", &name, &name_end)) {
        const char *fn = NULL, *fn_end = NULL;
        if (!tool_direct_value(p, obj_end, "function", &fn, &fn_end))
            return false;
        if (!tool_direct_value(fn, fn_end, "name", &name, &name_end))
            return false;
        if (!tool_direct_value(fn, fn_end, "arguments", &args, &args_end) &&
            !tool_direct_value(fn, fn_end, "parameters", &args, &args_end))
            return false;
    } else if (!tool_direct_value(p, obj_end, "arguments", &args, &args_end) &&
               !tool_direct_value(p, obj_end, "parameters", &args, &args_end)) {
        return false;
    }

    if (!name || name >= name_end || *name != '"') return false;
    int name_len = (int)(name_end - name - 2);
    if (name_len <= 0 || name_len >= TOOL_NAME_MAX) return false;
    tool_unescape(out->name, name + 1, name_len, TOOL_NAME_MAX);
    if (!tool_name_known(out->name, tools, n_tools)) return false;

    args = tool_skip_ws(args, args_end);
    if (args >= args_end) return false;
    if (*args == '"') {
        char decoded[TOOL_ARGS_MAX];
        int decoded_len = tool_unescape(decoded, args + 1,
                                        (int)(args_end - args - 2),
                                        TOOL_ARGS_MAX);
        const char *d_end = decoded + decoded_len;
        const char *ve = tool_value_end(decoded, d_end, 0);
        if (!ve || tool_skip_ws(ve, d_end) != d_end)
            return false;
        if (!tool_args_copy(out, decoded, (int)(ve - decoded))) return false;
    } else {
        if (*args != '{' && *args != '[') return false;
        const char *value_end = tool_value_end(args, args_end, 0);
        if (!value_end || tool_skip_ws(value_end, args_end) != args_end)
            return false;
        if (!tool_args_copy(out, args, (int)(value_end - args))) return false;
    }
    out->valid = true;
    return true;
}

/* ---- Qwen2 JSON / Hermes calls --------------------------------- */

static bool qwen25_tool_parse(const char *text, int len,
                              const ToolDef *tools, int n_tools,
                              ToolCall *out) {
    memset(out, 0, sizeof(*out));
    const char *limit = text + len;

    const char *p = text;
    const char *think = strstr(p, "</think>");
    if (think && think < limit) p = think + strlen("</think>");
    p = tool_skip_ws(p, limit);

    if (limit - p >= 3 && memcmp(p, "to=", 3) == 0) {
        p += 3;
        const char *name_start = p;
        while (p < limit && *p != ' ' && *p != '\t' &&
               *p != '\r' && *p != '\n') p++;
        int name_len = (int)(p - name_start);
        if (name_len <= 0 || name_len >= TOOL_NAME_MAX) return false;
        tool_unescape(out->name, name_start, name_len, TOOL_NAME_MAX);
        if (!tool_name_known(out->name, tools, n_tools)) return false;

        p = tool_skip_ws(p, limit);
        if (p >= limit || *p != '{') return false;
        const char *args_end = tool_value_end(p, limit, 0);
        if (!args_end || tool_skip_ws(args_end, limit) != limit) return false;
        if (!tool_args_copy(out, p, (int)(args_end - p))) return false;
        out->valid = true;
        return true;
    }

    if (limit - p >= 11 && memcmp(p, "<tool_call>", 11) == 0) {
        p += 11;
        const char *close = strstr(p, "</tool_call>");
        if (!close || close > limit) return false;
        if (!tool_parse_call_object(p, (int)(close - p), tools, n_tools, out))
            return false;
        return tool_skip_ws(close + strlen("</tool_call>"), limit) == limit;
    }
    return false;
}

static char *qwen25_format_call(const char *name, const char *arguments) {
    size_t size = strlen(name) + strlen(arguments) + 64;
    char *s = smalloc(size);
    (void)snprintf(s, size,
                   "<tool_call>\n{\"name\":\"%s\",\"arguments\":%s}\n"
                   "</tool_call>", name, arguments);
    return s;
}

const ToolOps qwen25_tool_ops = {
    .parse         = qwen25_tool_parse,
    .format_call   = qwen25_format_call,
    .prompt_hint   = "\nIf you choose to call a function ONLY reply in the "
                     "following format with NO suffix:\n"
                     "<tool_call>\n{\"name\":\"example_function_name\","
                     "\"arguments\":{...}}\n</tool_call>\n",
    .call_end      = "<|im_end|>",
    .result_role   = "user",
    .result_prefix = "<tool_response>\n",
    .result_suffix = "\n</tool_response>",
};

/* ---- Qwen3.5 XML calls ----------------------------------------- */

/* Parse the XML form used by the Qwen3.5 chat template:
 *   <tool_call>
 *   <function=name>
 *   <parameter=key>
 *   value
 *   </parameter>
 *   </function>
 *   </tool_call>
 * Falls back to the Qwen2 JSON form when the body is not XML. */
typedef struct {
    char        key[TOOL_NAME_MAX];
    const char *value;
    const char *value_end;
} QwenParam;

static const char *tool_find(const char *p, const char *limit,
                             const char *tag) {
    const char *q = strstr(p, tag);
    return (q && q < limit) ? q : NULL;
}

static int tool_param_index(const ToolDef *tool, const char *key) {
    for (int i = 0; i < tool->n_params; i++)
        if (strcmp(tool->params[i], key) == 0) return i;
    return -1;
}

static bool qwen35_tool_parse(const char *text, int len,
                              const ToolDef *tools, int n_tools,
                              ToolCall *out) {
    memset(out, 0, sizeof(*out));
    const char *limit = text + len;

    const char *p = text;
    const char *think = strstr(p, "</think>");
    if (think && think < limit) p = think + strlen("</think>");
    p = tool_skip_ws(p, limit);

    /* Accept a missing <tool_call> wrapper or leading prose. */
    const char *call_tag = tool_find(p, limit, "<tool_call>");
    const char *fn_tag   = tool_find(p, limit, "<function=");
    const char *par_tag  = tool_find(p, limit, "<parameter=");
    if (!call_tag && !fn_tag && !par_tag)
        return qwen25_tool_parse(text, len, tools, n_tools, out);
    p = call_tag ? call_tag + strlen("<tool_call>")
                 : (fn_tag ? fn_tag : par_tag);
    p = tool_skip_ws(p, limit);

    bool have_name = false;
    const char *next_fn = tool_find(p, limit, "<function=");
    const char *next_par = tool_find(p, limit, "<parameter=");
    if (next_fn && (!next_par || next_fn < next_par)) {
        p = next_fn + strlen("<function=");
        const char *name_start = p;
        while (p < limit && *p != '>') p++;
        if (p >= limit) return false;
        int name_len = (int)(p - name_start);
        if (name_len <= 0 || name_len >= TOOL_NAME_MAX) return false;
        tool_unescape(out->name, name_start, name_len, TOOL_NAME_MAX);
        if (!tool_name_known(out->name, tools, n_tools)) return false;
        have_name = true;
        p++;
    }

    QwenParam params[TOOL_MAX_PARAMS];
    int n_params = 0;
    for (;;) {
        p = tool_skip_ws(p, limit);
        if (limit - p >= 11 && memcmp(p, "</function>", 11) == 0) {
            p += 11;
            break;
        }
        if (limit - p >= 12 && memcmp(p, "</tool_call>", 12) == 0)
            break;
        const char *param = tool_find(p, limit, "<parameter=");
        if (!param) break;
        p = param + strlen("<parameter=");
        const char *key_start = p;
        while (p < limit && *p != '>') p++;
        if (p >= limit) return false;
        int key_len = (int)(p - key_start);
        p++;

        /* The value ends at the first closing/next tag.  Missing
         * </parameter> and closing tags are tolerated. */
        const char *value_end = limit;
        const char *tags[] = {
            "</parameter>", "<parameter=", "</function>", "</tool_call>"
        };
        for (size_t i = 0; i < sizeof(tags) / sizeof(tags[0]); i++) {
            const char *t = tool_find(p, limit, tags[i]);
            if (t && t < value_end) value_end = t;
        }
        const char *v = p;
        const char *ve = value_end;
        while (v < ve && (*v == '\n' || *v == '\r' ||
                          *v == ' ' || *v == '\t')) v++;
        while (ve > v && (ve[-1] == '\n' || ve[-1] == '\r' ||
                          ve[-1] == ' ' || ve[-1] == '\t')) ve--;

        if (n_params < TOOL_MAX_PARAMS) {
            params[n_params].key[0] = '\0';
            if (key_len > 0 && key_len < TOOL_NAME_MAX)
                (void)tool_unescape(params[n_params].key, key_start,
                                    key_len, TOOL_NAME_MAX);
            params[n_params].value = v;
            params[n_params].value_end = ve;
            n_params++;
        }
        p = value_end;
        if (limit - p >= 12 && memcmp(p, "</parameter>", 12) == 0)
            p += 12;
        if (p >= limit) break;
    }

    if (!have_name && n_params == 0)
        return qwen25_tool_parse(text, len, tools, n_tools, out);

    const ToolDef *tool = NULL;
    if (have_name) {
        for (int i = 0; i < n_tools; i++)
            if (strcmp(tools[i].name, out->name) == 0) { tool = &tools[i]; break; }
        if (!tool) return false;
    } else {
        /* Infer the function from parameter names.  It must match exactly
         * one advertised function and all of that function's required
         * parameters must be present. */
        int candidates = 0;
        for (int i = 0; i < n_tools; i++) {
            const ToolDef *t = &tools[i];
            if (t->n_params == 0) continue;
            bool ok = true;
            for (int j = 0; j < n_params; j++) {
                if (!params[j].key[0]) { ok = false; break; }
                if (tool_param_index(t, params[j].key) < 0) { ok = false; break; }
            }
            if (!ok) continue;
            for (int j = 0; j < t->n_params; j++) {
                if (!t->required[j]) continue;
                bool present = false;
                for (int k = 0; k < n_params; k++)
                    if (strcmp(params[k].key, t->params[j]) == 0)
                        present = true;
                if (!present) { ok = false; break; }
            }
            if (ok) { candidates++; tool = t; }
        }
        if (candidates != 1 || !tool) return false;
        (void)snprintf(out->name, sizeof(out->name), "%s", tool->name);
    }

    /* Repair an empty <parameter=> using the only missing required
     * parameter.  Anything more ambiguous is rejected. */
    int empty[TOOL_MAX_PARAMS], n_empty = 0;
    for (int i = 0; i < n_params; i++)
        if (!params[i].key[0] && n_empty < TOOL_MAX_PARAMS)
            empty[n_empty++] = i;
    if (n_empty > 0) {
        int missing[TOOL_MAX_PARAMS], n_missing = 0;
        for (int i = 0; i < tool->n_params; i++) {
            if (!tool->required[i]) continue;
            bool present = false;
            for (int j = 0; j < n_params; j++)
                if (strcmp(params[j].key, tool->params[i]) == 0)
                    present = true;
            if (!present && n_missing < TOOL_MAX_PARAMS)
                missing[n_missing++] = i;
        }
        if (n_missing != n_empty) return false;
        for (int i = 0; i < n_empty; i++)
            (void)snprintf(params[empty[i]].key, TOOL_NAME_MAX, "%s",
                           tool->params[missing[i]]);
    }

    /* Every required parameter must now be present. */
    for (int i = 0; i < tool->n_params; i++) {
        if (!tool->required[i]) continue;
        bool present = false;
        for (int j = 0; j < n_params; j++)
            if (strcmp(params[j].key, tool->params[i]) == 0)
                present = true;
        if (!present) return false;
    }

    ToolBuf args = {0};
    if (!tb_puts(&args, "{")) goto fail;
    for (int i = 0; i < n_params; i++) {
        if (i > 0 && !tb_puts(&args, ",")) goto fail;
        if (!tb_json_string(&args, params[i].key,
                            (int)strlen(params[i].key))) goto fail;
        if (!tb_puts(&args, ":")) goto fail;
        if (!tb_json_value(&args, params[i].value,
                           params[i].value_end)) goto fail;
    }
    if (!tb_puts(&args, "}")) goto fail;
    if (!tool_args_copy(out, args.buf, (int)args.len)) goto fail;
    tb_free(&args);
    out->valid = true;
    return true;

fail:
    tb_free(&args);
    return false;
}

/* Build the XML `<tool_call>` form from a serialized arguments object. */
static char *qwen35_format_call(const char *name, const char *arguments) {
    ToolBuf b = {0};
    if (!tb_printf(&b, "<tool_call>\n<function=%s>\n", name)) goto fail;
    const char *p = arguments;
    const char *limit = arguments + strlen(arguments);
    p = tool_skip_ws(p, limit);
    if (p < limit && *p == '{') {
        const char *end = tool_value_end(p, limit, 0);
        if (!end) goto fail;
        p = tool_skip_ws(p + 1, end - 1);
        while (p < end - 1) {
            if (*p != '"') goto fail;
            const char *key_end = tool_string_end(p, end - 1);
            if (!key_end) goto fail;
            const char *colon = tool_skip_ws(key_end, end - 1);
            if (colon >= end - 1 || *colon != ':') goto fail;
            const char *v = tool_skip_ws(colon + 1, end - 1);
            const char *ve = tool_value_end(v, end - 1, 0);
            if (!ve) goto fail;
            char key[TOOL_NAME_MAX];
            int key_len = (int)(key_end - p - 2);
            (void)tool_unescape(key, p + 1, key_len, TOOL_NAME_MAX);
            if (!tb_printf(&b, "<parameter=%s>\n", key)) goto fail;
            if (v < ve && *v == '"') {
                char *value = smalloc((size_t)(ve - v));
                int value_len = tool_unescape(value, v + 1,
                                              (int)(ve - v - 2),
                                              (int)(ve - v));
                if (!tb_put(&b, value, (size_t)value_len)) {
                    sfree(value);
                    goto fail;
                }
                sfree(value);
            } else if (!tb_put(&b, v, (size_t)(ve - v))) {
                goto fail;
            }
            if (!tb_puts(&b, "\n</parameter>\n")) goto fail;
            p = tool_skip_ws(ve, end - 1);
            if (p >= end - 1) break;
            if (*p++ != ',') goto fail;
            p = tool_skip_ws(p, end - 1);
        }
    }
    if (!tb_puts(&b, "</function>\n</tool_call>")) goto fail;
    return tb_take(&b);

fail:
    tb_free(&b);
    return NULL;
}

const ToolOps qwen35_tool_ops = {
    .parse         = qwen35_tool_parse,
    .format_call   = qwen35_format_call,
    .prompt_hint   =
        "\nIf you choose to call a function ONLY reply in the following "
        "format with NO suffix:\n\n"
        "<tool_call>\n"
        "<function=example_function_name>\n"
        "<parameter=example_parameter_1>\n"
        "value_1\n"
        "</parameter>\n"
        "<parameter=example_parameter_2>\n"
        "This is the value for the second parameter\n"
        "that can span\n"
        "multiple lines\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>\n\n"
        "<IMPORTANT>\n"
        "Reminder:\n"
        "- Function calls MUST follow the specified format: an inner "
        "<function=...></function> block must be nested within "
        "<tool_call></tool_call> XML tags\n"
        "- Required parameters MUST be specified\n"
        "- You may provide optional reasoning for your function call in "
        "natural language BEFORE the function call, but NOT after\n"
        "- If there is no function call available, answer the question like "
        "normal with your current knowledge and do not tell the user about "
        "function calls\n"
        "</IMPORTANT>\n",
    .call_end      = "<|im_end|>",
    .result_role   = "user",
    .result_prefix = "<tool_response>\n",
    .result_suffix = "\n</tool_response>",
};

/* ---- Llama 3.x ------------------------------------------------- */

static bool llama3_tool_parse(const char *text, int len,
                              const ToolDef *tools, int n_tools,
                              ToolCall *out) {
    memset(out, 0, sizeof(*out));
    const char *limit = text + len;
    const char *p = tool_skip_ws(text, limit);
    static const char tag[] = "<|python_tag|>";
    if (limit - p >= (int)sizeof(tag) - 1 &&
        memcmp(p, tag, sizeof(tag) - 1) == 0)
        p += sizeof(tag) - 1;
    p = tool_skip_ws(p, limit);
    return tool_parse_call_object(p, (int)(limit - p), tools, n_tools, out);
}

static char *llama3_format_call(const char *name, const char *arguments) {
    size_t size = strlen(name) + strlen(arguments) + 64;
    char *s = smalloc(size);
    (void)snprintf(s, size,
                   "<|python_tag|>{\"name\":\"%s\",\"parameters\":%s}",
                   name, arguments);
    return s;
}

const ToolOps llama3_tool_ops = {
    .parse       = llama3_tool_parse,
    .format_call = llama3_format_call,
    .prompt_hint = "Emit only <|python_tag|>{\"name\":\"<function-name>\","
                   "\"parameters\":{...}} with no surrounding prose. ",
    .call_end    = "<|eot_id|>",
    .result_role = "ipython",
};
