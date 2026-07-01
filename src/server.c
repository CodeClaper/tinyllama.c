#include <signal.h>
#include <pthread.h>
#include "anet.h"
#include "el.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <ctype.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/eventfd.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>
#include "def.h"
#include "slog.h"
#include "utils.h"
#include "mm.h"
#include "core.h"
#include "anet.h"
#include "http.h"

#define JSON_WRAPPER_RESERVE 256 
typedef struct  {
    Engine *engine;
    Session *session;
    EventLoop *el;
    int serverfd;
} Server;

static volatile sig_atomic_t g_signal_count = 0;
static int g_event_fd = -1;

static void signal_handler(int sig) {
    UNUSED(sig);
    if (g_signal_count > 0) _exit(130);
    g_signal_count = 1;
    u64 val = 1;
    (void)write(g_event_fd, &val, sizeof(val));
}

/* Callback for signal eventfd — wake up and stop the event loop. */
static void signal_callback(EventLoop *el, int fd, int mask, void *privdata) {
    UNUSED(mask);
    UNUSED(privdata);
    u64 val;
    (void)read(fd, &val, sizeof(val));
    el->stop = true;
}

/* GPT-2 style pre-tokenizer character classification.
 * 1=letter (including non-ASCII UTF-8 bytes), 2=digit,
 * 3=whitespace, 4=other (punctuation/symbol) */
static int char_category(int c) {
    unsigned char uc = (unsigned char)c;
    if (uc >= 0x80)              return 1;
    if (isalpha(uc))             return 1;
    if (isdigit(uc))             return 2;
    if (isspace(uc))             return 3;
    return 4;
}

/* Try to match a special token (format: <|...|>) at text[pos].
 * Returns token ID on success, VOCAB_ID_NONE on failure.
 * On success, *matched_len is set to bytes consumed. */
static i32 match_special_token(Vocab *v, const char *text, int text_len, int *matched_len) {
    if (text_len < 3) return (i32)VOCAB_ID_NONE;
    if (text[0] != '<' || text[1] != '|') return (i32)VOCAB_ID_NONE;

    int max_try = text_len < 64 ? text_len : 64;
    int best_len = 0;
    i32 best_id  = (i32)VOCAB_ID_NONE;
    for (int len = 2; len <= max_try; len++) {
        i32 id;
        if (vocab_lookup_len(v, text, len, &id)) {
            best_len = len;
            best_id  = id;
        }
    }
    if (best_len > 0) {
        *matched_len = best_len;
        return best_id;
    }
    return (i32)VOCAB_ID_NONE;
}

/* Byte-level encode a text chunk and apply BPE merges within it.
 * Returns the number of tokens produced. */
static int bpe_encode_chunk(Vocab *v, const char *chunk, int chunk_len,
                            u32 *out, int max_out) {
    u32 syms[512];
    int n = 0;

    for (int i = 0; i < chunk_len && n < (int)(sizeof(syms)/sizeof(syms[0])); i++) {
        i32 id = v->byte_token_ids[(unsigned char)chunk[i]];
        if (id >= 0)
            syms[n++] = (u32)id;
    }

    while (n > 1) {
        i32 best_rank = INT32_MAX;
        int best_idx  = -1;
        for (int i = 0; i < n - 1; i++) {
            i32 rank = vocab_merge_rank(v, (i32)syms[i], (i32)syms[i + 1]);
            if (rank >= 0 && rank < best_rank) {
                best_rank = rank;
                best_idx  = i;
            }
        }
        if (best_idx < 0) break;
        i32 merged = vocab_merge_result(v, (i32)syms[best_idx], (i32)syms[best_idx + 1]);
        if (merged == (i32)VOCAB_ID_NONE) break;
        syms[best_idx] = (u32)merged;
        memmove(&syms[best_idx + 1], &syms[best_idx + 2],
                (size_t)(n - best_idx - 2) * sizeof(u32));
        n--;
    }

    if (n > max_out) n = max_out;
    memcpy(out, syms, (size_t)n * sizeof(u32));
    return n;
}

/* BPE tokenizer for GPT-2 style vocabularies (Qwen2, LLaMA, etc.).
 * 1. Detects special tokens (<|...|>) by direct vocab lookup.
 * 2. Applies GPT-2 regex pre-tokenization to split text into chunks.
 * 3. Byte-encodes and BPE-merges within each chunk independently. */
static int tokenize_bpe(Vocab *v, const char *text, int text_len, u32 *tokens, int max_tokens) {
    #define BPE_MAX_SYMBOLS 4096
    u32 symbols[BPE_MAX_SYMBOLS];
    int n = 0, pos = 0;

    while (pos < text_len && n < BPE_MAX_SYMBOLS) {
        /* Step 0: try to match a special token (<|...|>) first. */
        int special_len = 0;
        i32 special_id = match_special_token(v, text + pos, text_len - pos, &special_len);
        if (special_id >= 0) {
            symbols[n++] = (u32)special_id;
            pos += special_len;
            continue;
        }

        /* Step 1: locate the next pre-tokenizer chunk. */

        /* Contractions: 's, 't, 're, 've, 'm, 'll, 'd */
        if (text[pos] == '\'' && pos + 1 < text_len) {
            char nxt = text[pos + 1];
            int clen = 0;
            if (nxt == 's' || nxt == 't' || nxt == 'm' || nxt == 'd')
                clen = 2;
            else if ((nxt == 'r' || nxt == 'v') && pos + 2 < text_len &&
                     text[pos + 2] == 'e')
                clen = 3;
            else if (nxt == 'l' && pos + 2 < text_len && text[pos + 2] == 'l')
                clen = 3;
            if (clen > 0) {
                int added = bpe_encode_chunk(v, text + pos, clen,
                                             symbols + n, BPE_MAX_SYMBOLS - n);
                n += added; pos += clen;
                continue;
            }
        }

        /* Whitespace: single leading space is absorbed by the next chunk
         * (GPT-2 regex semantics); multi-space runs form their own chunk. */
        if (char_category(text[pos]) == 3) {
            int ws_start = pos;
            int ws_len = 0;
            while (pos < text_len && char_category(text[pos]) == 3) {
                pos++; ws_len++;
            }
            if (ws_len != 1 || pos >= text_len) {
                int added = bpe_encode_chunk(v, text + ws_start, ws_len,
                                             symbols + n, BPE_MAX_SYMBOLS - n);
                n += added;
                continue;
            }
            /* Single space before non-whitespace: let next chunk absorb it. */
            pos = ws_start;
        }

        /* Non-whitespace chunk (letter, digit, or other). */
        int chunk_start = pos;
        if (pos < text_len && char_category(text[pos]) == 3)
            pos++;                    /* absorb optional leading space */
        if (pos >= text_len) continue;

        int cat = char_category(text[pos]);
        if (cat == 1) {
            while (pos < text_len && char_category(text[pos]) == 1) pos++;
        } else if (cat == 2) {
            while (pos < text_len && char_category(text[pos]) == 2) pos++;
        } else {
            while (pos < text_len && char_category(text[pos]) == 4) pos++;
        }

        if (pos == chunk_start) continue;

        int added = bpe_encode_chunk(v, text + chunk_start, pos - chunk_start,
                                     symbols + n, BPE_MAX_SYMBOLS - n);
        n += added;
    }

    if (n > max_tokens) n = max_tokens;
    memcpy(tokens, symbols, (size_t)n * sizeof(u32));
    return n;
    #undef BPE_MAX_SYMBOLS
}

/* Tokenize dispatch — selects tokenization strategy based on the
 * tokenizer type declared in the GGUF metadata. */
/* Chat template (Qwen-style) — build token ID array directly.
 * Pre-looks up format tokens from the vocabulary; tokenizes user/system
 * message content via BPE. Returns total number of prompt tokens. */
static int build_chat_tokens(Vocab *v, const char *user_msg, const char *sys_msg,
                             u32 *tokens, int max_tokens) {
    int n = 0;

    /* Pre-lookup format tokens. */
    i32 im_start_id = v->im_start_id;
    i32 im_end_id   = v->im_end_id;
    if (im_start_id == (i32)VOCAB_ID_NONE) {
        im_start_id = vocab_lookup(v, "<|im_start|>");
        if (im_start_id == (i32)VOCAB_ID_NONE) return 0;
    }
    if (im_end_id == (i32)VOCAB_ID_NONE) {
        im_end_id = vocab_lookup(v, "<|im_end|>");
        if (im_end_id == (i32)VOCAB_ID_NONE) return 0;
    }
    i32 nl_id = v->byte_token_ids[(unsigned char)'\n'];

    #define ADD(id) do { \
        if (n < max_tokens) tokens[n++] = (u32)(id); \
    } while (0)
    #define ADD_TEXT(txt) do { \
        int tlen = (int)strlen(txt); \
        int added = tokenize_bpe(v, txt, tlen, tokens + n, max_tokens - n); \
        n += added; \
    } while (0)

    if (sys_msg && sys_msg[0]) {
        ADD(im_start_id);
        ADD_TEXT("system");
        ADD(nl_id);
        ADD_TEXT(sys_msg);
        ADD(im_end_id);
        ADD(nl_id);
    }
    ADD(im_start_id);
    ADD_TEXT("user");
    ADD(nl_id);
    ADD_TEXT(user_msg);
    ADD(im_end_id);
    ADD(nl_id);
    ADD(im_start_id);
    ADD_TEXT("assistant");
    ADD(nl_id);

    #undef ADD
    #undef ADD_TEXT
    return n;
}

static u32 sample_greedy(float *logits, u32 n_vocab) {
    u32 best = 0;
    float best_val = logits[0];
    for (u32 i = 1; i < n_vocab; i++) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best = i;
        }
    }
    return best;
}


static void client_read_proc(EventLoop *el, int fd, int mask, void *privdata) {
    UNUSED(mask);
    Server *server = (Server *)privdata;
    Session *s = server->session;
    Vocab   *v = server->engine->vocab;

    char buf[8192];
    ssize_t n = read(fd, buf, sizeof(buf) - 1);
    if (n <= 0) {
        if (n < 0) slog_errno("Client read error");
        delete_file_event(el, fd, ELOOP_READABLE);
        close(fd);
        return;
    }
    buf[n] = '\0';

    /* 1. Parse HTTP request. */
    char *body = http_body(buf, (int)n);
    if (!body) {
        http_respond(fd, 400, "Bad Request",
                     "{\"error\":\"Incomplete HTTP request\"}");
        delete_file_event(el, fd, ELOOP_READABLE);
        close(fd);
        return;
    }

    /* Check Content-Length and read remaining body if needed. */
    {
        const char *cl_str = http_header(buf, "Content-Length");
        int cl = cl_str[0] ? atoi(cl_str) : 0;
        int body_len = (int)((buf + n) - body);
        if (cl > 0 && body_len < cl) {
            int to_read = cl - body_len;
            if (to_read > 0 && (int)sizeof(buf) - n > to_read) {
                ssize_t more = read(fd, buf + n, to_read);
                if (more > 0) {
                    n += more;
                    buf[n] = '\0';
                }
            }
        }
    }

    /* 2. Extract messages from JSON body. */
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
        http_respond(fd, 400, "Bad Request",
                     "{\"error\":\"No user message found\"}");
        delete_file_event(el, fd, ELOOP_READABLE);
        close(fd);
        return;
    }

    /* 3. Build chat prompt tokens. */
    u32 prompt_tokens[4096];
    int max_pt = (int)(sizeof(prompt_tokens) / sizeof(prompt_tokens[0]));
    int n_prompt = build_chat_tokens(v, user_msg, sys_msg, prompt_tokens, max_pt);
    if (n_prompt == 0) {
        http_respond(fd, 400, "Bad Request",
                    "{\"error\":\"No valid tokens in input\"}");
        delete_file_event(el, fd, ELOOP_READABLE);
        close(fd);
        return;
    }

    /* 4. Reset session & prefill. */
    s->ops.reset(s);
    for (int i = 0; i < n_prompt; i++) {
        if (!s->ops.forward(s, prompt_tokens[i], s->logits)) {
            http_respond(fd, 500, "Internal Server Error",
                         "{\"error\":\"Forward pass failed\"}");
            delete_file_event(el, fd, ELOOP_READABLE);
            close(fd);
            return;
        }
    }

    /* 5. Generate. */
    u32 max_tokens = s->max_tokens > 0 ? s->max_tokens : 256;
    char resp_body[65536];
    int  resp_used = 0;

    u32 next_token = sample_greedy(s->logits, s->cfg.n_vocab);
    u32 n_gen = 0;

    for (u32 i = 0; i < max_tokens; i++) {
        if (next_token == (u32)v->eos_id) break;
        if (next_token < v->n_vocab && v->token[next_token].content) {
            Key *tk = &v->token[next_token];
            int remaining = (int)sizeof(resp_body) - resp_used - 1;
            if ((int)tk->len < remaining) {
                memcpy(resp_body + resp_used, tk->content, tk->len);
                resp_used += tk->len;
            }
        }
        n_gen++;
        if (!s->ops.forward(s, next_token, s->logits)) break;
        next_token = sample_greedy(s->logits, s->cfg.n_vocab);
    }
    resp_body[resp_used] = '\0';

    /* 6. Build OpenAI-compatible JSON response. */
    char json_buf[65536];
    /* JSON-escape the generated text.  Reserve 256 bytes for the
     * JSON wrapper so the final snprintf can't overflow json_buf. */
    char escaped[65536 - JSON_WRAPPER_RESERVE];
    int esc_len = json_escape_str(escaped, (int)sizeof(escaped) - 1,
                                  resp_body, resp_used);
    escaped[esc_len] = '\0';

    time_t now = time(NULL);
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
                  (next_token == (u32)v->eos_id) ? "stop" : "length",
                  n_prompt, n_gen, n_prompt + (int)n_gen);

    http_respond(fd, 200, "OK", json_buf);

    delete_file_event(el, fd, ELOOP_READABLE);
    close(fd);
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
    fprintf(file, "Example: server model.gguf -p 9987 \n");
    fprintf(file, "Options:\n");
    fprintf(file, "  -m | --model   <string>  The model file path to run\n");
    fprintf(file, "  -h | --host    <string>  The server host, defalt 127.0.0.1\n");
    fprintf(file, "  -p | --port    <int>     The Port to listening, default 9987\n");
    fprintf(file, "  -c | --ctx     <int>     The context size, defalt 4096\n");
    fprintf(file, "  -n | --tokens  <int>     The default token size, defalt 393216\n");
    fprintf(file, "  -i | --inspect <none>    Inspect the engine/model\n");
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
        .port = 7788,
        .ctx_size = 4096,
        .default_tokens = 393216
    };
    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) usage(stdout, EXIT_SUCCESS);
        else if (!strcmp(arg, "-m") || !strcmp(arg, "--model")) so.engine.model_path = parse_arg(argc, argv, &i, arg);
        else if (!strcmp(arg, "-h") || !strcmp(arg, "--host")) so.host = parse_arg(argc, argv, &i, arg);
        else if (!strcmp(arg, "-p") || !strcmp(arg, "--port")) so.port = parse_int(parse_arg(argc, argv, &i, arg));
        else if (!strcmp(arg, "-c") || !strcmp(arg, "--ctx")) so.ctx_size = parse_int(parse_arg(argc, argv, &i, arg));
        else if (!strcmp(arg, "-n") || !strcmp(arg, "--tokens")) so.default_tokens = parse_int(parse_arg(argc, argv, &i, arg));
        else if (!strcmp(arg, "-i") || !strcmp(arg, "--inspect")) { so.inspect = true; so.engine.inspect = true; }
        else {
            fprintf(stderr, "Unkonow option: %s.\n", arg);
            usage(stderr, 2);
        }
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


static void server_resource_close(Server *server) {
    session_free(server->session);
    engine_close(server->engine);
    el_free(server->el);
    if (server->serverfd >= 0) close(server->serverfd);
    if (g_event_fd >= 0) close(g_event_fd);
}

int main(int argc, char *argv[]) {
    if (argc < 2) usage(stderr, 2);
    signal(SIGPIPE, SIG_IGN);
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = signal_handler;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);

    ServerOptions so = parse_options(argc, argv);
    Engine *engine = engine_open(&so.engine);
    if (so.inspect) { 
        engine_summary(engine);
        engine_close(engine);
        return 0;
    }

    Session *session = session_create(engine, (u32)so.ctx_size);
    if (!session) {
        engine_close(engine);
        slog(ERROR, "Failed to create session.");
    }

    Server server;
    server.engine = engine;
    server.session = session;

    if (setup_server_eventloop(&server, so) == ELOOP_ERR) {
        server_resource_close(&server);
        slog(ERROR, "Failed to setup server event loop.");
    }

    g_event_fd = eventfd(0, EFD_NONBLOCK);
    if (g_event_fd >= 0) {
        create_file_event(server.el, g_event_fd, ELOOP_READABLE, signal_callback, NULL);
    }

    slog(INFO, "Server: listening on http://%s:%d", so.host, so.port);

    el_main(server.el);

    slog(INFO, "Server: shutdown requested, draining requests");
    server_resource_close(&server);
    return 0;
}
