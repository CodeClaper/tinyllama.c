#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <time.h>
#include <unistd.h>
#include "def.h"
#include "utils.h"
#include "mm.h"
#include "core.h"
#include "tokenizer.h"
#include "sampler.h"
#include "slog.h"
#include "linenoise.h"

typedef struct {
    EngineOptons engine;
    int ctx_size;
    u32 n_tokens;
    float temperature;
    u32 top_k;
    float top_p;
    float min_p;
    float repeat_penalty;
    u32 repeat_last_n;
    float frequency_penalty;
    float presence_penalty;
    int nthread;
    const char *system;
} ChatOptions;

/* Usage. */
static void usage(FILE *file, int exit_code) {
    fprintf(file, "Usage:   chat [options]\n");
    fprintf(file, "Example: chat -m model.gguf -s \"You are a helpful assistant.\"\n");
    fprintf(file, "Options:\n");
    fprintf(file, "  -m  | --model             <string>  The model file path to run\n");
    fprintf(file, "  -c  | --ctx               <int>     The context size, default 4096\n");
    fprintf(file, "  -n  | --tokens            <int>     Number of tokens to generate, default 393216\n");
    fprintf(file, "  -T  | --threads           <int>     Number of threads, default CPU cores\n");
    fprintf(file, "  -s  | --system            <string>  System input\n");
    fprintf(file, "  -t  | --temp              <float>   Temperature for sampling, default 0.8\n");
    fprintf(file, "  -tp | --topp              <float>   Top-p (nucleus) threshold, default 0.9\n");
    fprintf(file, "  -tk | --topk              <int>     Top-k sampling, default 40\n");
    fprintf(file, "  -P  | --minp              <float>   Min-p threshold, default 0.05\n");
    fprintf(file, "  -rp | --repeat-penalty    <float>   Repeat penalty, default 1.0 (1.0=disabled)\n");
    fprintf(file, "  -rl | --repeat-last-n     <int>     Repeat penalty lookback, default 64\n");
    fprintf(file, "  -fp | --frequency-penalty <float>   Frequency penalty, default 0.0 (0.0=disabled)\n");
    fprintf(file, "  -pp | --presence-penalty  <float>   Presence penalty, default 0.0 (0.0=disabled)\n");
    exit(exit_code);
}

/* Fatala. */
static void fatal(char *format, ...) {
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

    fprintf(stderr, "%s\n", message);
    exit(3);
}

/* Parse current arg value. */
static char *parse_arg(int argc, char *argv[], int *i, const char *opt) {
    if (*i + 1 >= argc)
        fatal("Missing value for option [%s]", opt);
    return argv[++(*i)];
}


/* Parse options. */
static ChatOptions parse_options(int argc, char *argv[]) {
    ChatOptions co = {
        .ctx_size = 4096,
        .n_tokens = 393216,
        .temperature = 0.8f,
        .top_k = DEFAULT_TOP_K,
        .top_p = 0.9f,
        .min_p = DEFAULT_MIN_P,
        .repeat_penalty = DEFAULT_REPEAT_PENALTY,
        .repeat_last_n = DEFAULT_REPEAT_LAST_N,
        .frequency_penalty = DEFAULT_FREQUENCY_PENALTY,
        .presence_penalty = DEFAULT_PRESENCE_PENALTY,
        .nthread = -1,
        .system = NULL
    };
    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) usage(stdout, EXIT_SUCCESS);
        else if (!strcmp(arg, "-m")  || !strcmp(arg, "--model")) co.engine.model_path = parse_arg(argc, argv, &i, arg);
        else if (!strcmp(arg, "-c")  || !strcmp(arg, "--ctx")) co.ctx_size = parse_int(parse_arg(argc, argv, &i, arg));
        else if (!strcmp(arg, "-n")  || !strcmp(arg, "--tokens")) co.n_tokens = (u32)parse_int(parse_arg(argc, argv, &i, arg));
        else if (!strcmp(arg, "-T")  || !strcmp(arg, "--threads")) co.nthread = parse_int(parse_arg(argc, argv, &i, arg));
        else if (!strcmp(arg, "-t")  || !strcmp(arg, "--temp")) co.temperature = parse_float(parse_arg(argc, argv, &i, arg));
        else if (!strcmp(arg, "-tp") || !strcmp(arg, "--topp")) co.top_p = parse_float(parse_arg(argc, argv, &i, arg));
        else if (!strcmp(arg, "-tk") || !strcmp(arg, "--topk")) co.top_k = (u32)parse_int(parse_arg(argc, argv, &i, arg));
        else if (!strcmp(arg, "-P")  || !strcmp(arg, "--minp")) co.min_p = parse_float(parse_arg(argc, argv, &i, arg));
        else if (!strcmp(arg, "-rp") || !strcmp(arg, "--repeat-penalty")) co.repeat_penalty = parse_float(parse_arg(argc, argv, &i, arg));
        else if (!strcmp(arg, "-rl") || !strcmp(arg, "--repeat-last-n")) co.repeat_last_n = (u32)parse_int(parse_arg(argc, argv, &i, arg));
        else if (!strcmp(arg, "-fp") || !strcmp(arg, "--frequency-penalty")) co.frequency_penalty = parse_float(parse_arg(argc, argv, &i, arg));
        else if (!strcmp(arg, "-pp") || !strcmp(arg, "--presence-penalty")) co.presence_penalty = parse_float(parse_arg(argc, argv, &i, arg));
        else if (!strcmp(arg, "-s")  || !strcmp(arg, "--system")) co.system = parse_arg(argc, argv, &i, arg);
        else {
            fprintf(stderr, "Unknown option: %s.\n", arg);
            usage(stderr, 2);
        }
    }
    if (co.nthread <= 0) {
        long n = sysconf(_SC_NPROCESSORS_ONLN);
        co.nthread = (int)(n > 0 ? n : 1);
    }
    if (co.n_tokens < 1) co.n_tokens = 1;
    if (!co.engine.model_path) {
        fprintf(stderr, "Model path is required (-m / --model)");
        usage(stderr, 2);
    }
    if (co.temperature < 0.0f) {
        fprintf(stderr, "Temperature must be >= 0.0, got %.2f\n", co.temperature);
        usage(stderr, 2);
    }
    if (co.top_p <= 0.0f || co.top_p > 1.0f) {
        fprintf(stderr, "Top-p must be in (0.0, 1.0], got %.2f\n", co.top_p);
        usage(stderr, 2);
    }
    if (co.min_p < 0.0f || co.min_p > 1.0f) {
        fprintf(stderr, "Min-p must be in [0.0, 1.0], got %.2f\n", co.min_p);
        usage(stderr, 2);
    }
    return co;
}

/* Decode a token to UTF-8 text into buf. Returns bytes written. */
static int decode_token(Session *s, Vocab *v, u32 id, char *buf, int max_len) {
    if (id >= v->n_vocab) return 0;
    Key *tk = &v->token[id];
    if (!tk->content) return 0;
    if (s->ops.decode)
        return s->ops.decode((const u8 *)tk->content, (int)tk->len, buf, max_len);
    return 0;
}

/* Strict UTF-8 validation: rejects truncated sequences, invalid lead /
 * continuation bytes, overlong encodings, surrogates and > U+10FFFF. */
static bool utf8_valid(const char *s, size_t len) {
    size_t i = 0;
    while (i < len) {
        unsigned char c = (unsigned char)s[i];
        u32   cp;
        size_t n;
        if (c < 0x80) { i++; continue; }
        else if ((c & 0xE0) == 0xC0) { n = 2; cp = c & 0x1Fu; }
        else if ((c & 0xF0) == 0xE0) { n = 3; cp = c & 0x0Fu; }
        else if ((c & 0xF8) == 0xF0) { n = 4; cp = c & 0x07u; }
        else return false;
        if (i + n > len) return false;
        for (size_t j = 1; j < n; j++) {
            unsigned char cc = (unsigned char)s[i + j];
            if ((cc & 0xC0) != 0x80) return false;
            cp = (cp << 6) | (u32)(cc & 0x3Fu);
        }
        if ((n == 2 && cp < 0x80u) || (n == 3 && cp < 0x800u) ||
            (n == 4 && cp < 0x10000u) || cp > 0x10FFFFu ||
            (cp >= 0xD800u && cp <= 0xDFFFu))
            return false;
        i += n;
    }
    return true;
}

/* Strip trailing whitespace (including \n, \r) from a string in-place. */
static void strip_trailing(char *s) {
    int len = (int)strlen(s);
    while (len > 0 && (s[len - 1] == '\n' || s[len - 1] == '\r' ||
                       s[len - 1] == ' '  || s[len - 1] == '\t'))
        s[--len] = '\0';
}

/* High-resolution time in seconds. */
static double time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(int argc, char *argv[]) {
    /* Suppress INFO logs during chat to keep output clean. */
    slog_set_level(WARN);

    srand((unsigned)time(NULL) ^ (unsigned)getpid());
    if (argc < 2) usage(stderr, 4);
    ChatOptions co = parse_options(argc, argv);

    /* Load model. */
    Engine *engine = engine_open(&co.engine);
    if (!engine) fatal("Failed to load model");

    /* Create session. */
    Session *session = session_create(engine, (u32)co.ctx_size, co.nthread);
    if (!session) fatal("Failed to create session");
    session->temperature = co.temperature;
    session->top_k = co.top_k;
    session->top_p = co.top_p;
    session->min_p = co.min_p;
    session->max_tokens       = co.n_tokens;
    session->repeat_penalty   = co.repeat_penalty;
    session->repeat_last_n    = co.repeat_last_n;
    session->frequency_penalty = co.frequency_penalty;
    session->presence_penalty  = co.presence_penalty;

    Vocab *v = engine->vocab;
    u32 max_tokens = session->max_tokens > 0 ? session->max_tokens : 256;
    bool first_turn = true;

    /* Banner. */
    printf("Chat with %s. Type /quit to exit, /clear to reset.\n", engine->model->arch_name);

    /* Green prompt and input. */
    linenoiseSetInputColor("\033[32m");

    char *input = NULL;
    u32 prompt_tokens[4096];
    int max_pt = (int)(sizeof(prompt_tokens) / sizeof(prompt_tokens[0]));

    FOREVER {
        free(input);
        input = linenoise("\033[32mYou > \033[0m");
        if (input == NULL) break; /* EOF, Ctrl-D or Ctrl-C */
        strip_trailing(input);
        if (input[0] != '\0') linenoiseHistoryAdd(input);

        if (!utf8_valid(input, strlen(input))) {
            fprintf(stderr, "Invalid UTF-8 input, line skipped.\n");
            continue;
        }

        /* Commands. */
        if (!strcmp(input, "/quit") || !strcmp(input, "/exit")) break;
        if (!strcmp(input, "/clear")) {
            session->ops.reset(session);
            first_turn = true;
            printf("[Conversation cleared]\n");
            continue;
        }
        if (input[0] == '\0') continue; /* skip empty lines */

        /* Build prompt tokens. */
        int n_prompt;
        if (first_turn) {
            n_prompt = build_chat_tokens(v, input, co.system, prompt_tokens, max_pt);
            if (n_prompt == 0) fatal("No valid tokens in input");
            first_turn = false;
        } else {
            n_prompt = build_continuation_tokens(v, input, prompt_tokens, max_pt);
            if (n_prompt == 0) fatal("No valid tokens in input");
        }
        if (n_prompt >= max_pt)
            fprintf(stderr, "Warning: prompt truncated to %d tokens.\n", max_pt);

        /* Forward pass. */
        if (!session->ops.prefill(session, prompt_tokens, n_prompt, session->logits))
            fatal("Forward pass failed");

        /* Sample first token. */
        u32 next_token;
        if (co.temperature > 0.0f)
            next_token = sample_token(session->logits, session->cfg.n_vocab,
                                       co.temperature, co.top_k, co.top_p, co.min_p,
                                       co.repeat_penalty, co.repeat_last_n,
                                       co.frequency_penalty, co.presence_penalty,
                                       session->tokens, session->n_tokens);
        else
            next_token = sample_greedy(session->logits, session->cfg.n_vocab);

        /* Generate. */
        u32 n_gen = 0;
        double t0 = time_sec();
        for (u32 i = 0; i < max_tokens; i++) {
            if (next_token == (u32)v->eos_id) break;
            char dec[64];
            int dlen = decode_token(session, v, next_token, dec,
                                    (int)sizeof(dec) - 1);
            if (dlen > 0) {
                dec[dlen] = '\0';
                fputs(dec, stdout);
                fflush(stdout);
            }
            n_gen++;
            if (!session->ops.generate(session, next_token, session->logits))
                break;
            if (co.temperature > 0.0f)
                next_token = sample_token(session->logits, session->cfg.n_vocab,
                                           co.temperature, co.top_k, co.top_p, co.min_p,
                                           co.repeat_penalty, co.repeat_last_n,
                                           co.frequency_penalty, co.presence_penalty,
                                           session->tokens, session->n_tokens);
            else
                next_token = sample_greedy(session->logits, session->cfg.n_vocab);
        }
        double dt = time_sec() - t0;
        double tok_s = dt > 0.0 ? (double)n_gen / dt : 0.0;
        printf("\n[%u tokens, %.1f tok/s]\n", n_gen, tok_s);
    }

    printf("Bye.\n");
    linenoiseFree(input);
    session_free(session);
    engine_close(engine);
    return 0;
}
