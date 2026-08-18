#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <unistd.h>
#include <sys/resource.h>
#include "def.h"
#include "slog.h"
#include "utils.h"
#include "mm.h"
#include "mctx.h"
#include "core.h"
#include "tokenizer.h"
#include "sampler.h"

typedef struct {
    EngineOptons engine;
    int ctx_size;
    u32 n_tokens;
    float temperature;
    u32 top_k;
    float top_p;
    float min_p;
    float frequency_penalty;
    float presence_penalty;
    const char *input;
    const char *output;
    int repeat;
    int nthread;
} BenchOptions;

/* Usage. */
static void usage(FILE *file, int exit_code) {
    fprintf(file, "Usage:   bench [options]\n");
    fprintf(file, "Example: bench -m model.gguf -i \"Who is Isaac Newton?\"\n");
    fprintf(file, "Options:\n");
    fprintf(file, "  -m  | --model   <string>  The model file path to run\n");
    fprintf(file, "  -c  | --ctx     <int>     The context size, default 4096\n");
    fprintf(file, "  -n  | --tokens  <int>     Number of tokens to generate, default 128\n");
    fprintf(file, "  -T  | --threads <int>     Number of threads, default CPU cores\n");
    fprintf(file, "  -t  | --temp    <float>   Temperature for sampling, default 0.8\n");
    fprintf(file, "  -tp | --topp    <float>   Top-p (nucleus) threshold, default 0.9\n");
    fprintf(file, "  -tk | --topk    <int>     Top-k sampling, default 40\n");
    fprintf(file, "  -P  | --minp    <float>   Min-p threshold, default 0.05\n");
    fprintf(file, "  -i  | --input   <string>  User input\n");
    fprintf(file, "  -o  | --output  <string>  Output file for generated text (optional)\n");
    fprintf(file, "  -r  | --repeat  <int>     Repeat count, default 1\n");
    exit(exit_code);
}

/* Parse current arg value. */
static char *parse_arg(int argc, char *argv[], int *i, const char *opt) {
    if (*i + 1 >= argc)
        slog(ERROR, "Missing value for option [%s]", opt);
    return argv[++(*i)];
}

/* Parse options. */
static BenchOptions parse_options(int argc, char *argv[]) {
    BenchOptions bo = {
        .ctx_size = 4096,
        .n_tokens = 128,
        .temperature = 0.8f,
        .top_k = DEFAULT_TOP_K,
        .top_p = 0.9f,
        .min_p = DEFAULT_MIN_P,
        .input = NULL,
        .output = NULL,
        .repeat = 1,
        .nthread = -1,
    };
    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) usage(stdout, EXIT_SUCCESS);
        else if (!strcmp(arg, "-m") || !strcmp(arg, "--model")) bo.engine.model_path = parse_arg(argc, argv, &i, arg);
        else if (!strcmp(arg, "-c") || !strcmp(arg, "--ctx")) bo.ctx_size = parse_int(parse_arg(argc, argv, &i, arg));
        else if (!strcmp(arg, "-n") || !strcmp(arg, "--tokens")) bo.n_tokens = (u32)parse_int(parse_arg(argc, argv, &i, arg));
        else if (!strcmp(arg, "-T") || !strcmp(arg, "--threads")) bo.nthread = parse_int(parse_arg(argc, argv, &i, arg));
        else if (!strcmp(arg, "-t") || !strcmp(arg, "--temp")) bo.temperature = parse_float(parse_arg(argc, argv, &i, arg));
        else if (!strcmp(arg, "-tp") || !strcmp(arg, "--topp")) bo.top_p = parse_float(parse_arg(argc, argv, &i, arg));
        else if (!strcmp(arg, "-tk") || !strcmp(arg, "--topk")) bo.top_k = (u32)parse_int(parse_arg(argc, argv, &i, arg));
        else if (!strcmp(arg, "-P") || !strcmp(arg, "--minp")) bo.min_p = parse_float(parse_arg(argc, argv, &i, arg));
        else if (!strcmp(arg, "-i") || !strcmp(arg, "--input")) bo.input = parse_arg(argc, argv, &i, arg);
        else if (!strcmp(arg, "-o") || !strcmp(arg, "--output")) bo.output = parse_arg(argc, argv, &i, arg);
        else if (!strcmp(arg, "-r") || !strcmp(arg, "--repeat")) bo.repeat = parse_int(parse_arg(argc, argv, &i, arg));
        else {
            fprintf(stderr, "Unknown option: %s.\n", arg);
            usage(stderr, 2);
        }
    }
    if (!bo.engine.model_path) slog(ERROR, "Model path is required (-m / --model)");
    if (!bo.input) slog(ERROR, "Input text is required (-i / --input)");
    if (bo.repeat < 1) bo.repeat = 1;
    if (bo.nthread <= 0) {
        long n = sysconf(_SC_NPROCESSORS_ONLN);
        bo.nthread = (int)(n > 0 ? n : 1);
    }
    if (bo.n_tokens < 1) bo.n_tokens = 1;
    if (bo.temperature < 0.0f) {
        fprintf(stderr, "Temperature must be >= 0.0, got %.2f\n", bo.temperature);
        usage(stderr, 2);
    }
    if (bo.top_p <= 0.0f || bo.top_p > 1.0f) {
        fprintf(stderr, "Top-p must be in (0.0, 1.0], got %.2f\n", bo.top_p);
        usage(stderr, 2);
    }
    if (bo.min_p < 0.0f || bo.min_p > 1.0f) {
        fprintf(stderr, "Min-p must be in [0.0, 1.0], got %.2f\n", bo.min_p);
        usage(stderr, 2);
    }
    return bo;
}

/* Get high-resolution time in seconds. */
static double time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
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

int main(int argc, char *argv[]) {
    /* Initialize the memory-context subsystem. */
    MemoryContextInit();

    if (argc < 2) usage(stderr, 3);

    BenchOptions opts = parse_options(argc, argv);

    /* Seed random for sampling. */
    srand((unsigned)time(NULL) ^ (unsigned)getpid());

    /* Load model. */
    slog(INFO, "Loading model: %s", opts.engine.model_path);
    Engine *engine = engine_open(&opts.engine);
    if (!engine) slog(ERROR, "Failed to load model");

    slog(INFO, "Loaded model (%s)", engine->model->arch_name);

    /* Create session. */
    Session *session = session_create(engine, (u32)opts.ctx_size, opts.nthread);
    if (!session) slog(ERROR, "Failed to create session");
    session->temperature = opts.temperature;
    session->top_k = opts.top_k;
    session->top_p = opts.top_p;
    session->min_p = opts.min_p;

    Vocab *v = engine->vocab;

    /* Tokenize input. */
    slog(INFO, "Tokenizing input...");
    u32 prompt_tokens[8192];
    int max_pt = (int)(sizeof(prompt_tokens) / sizeof(prompt_tokens[0]));
    int n_prompt = build_chat_tokens(v, opts.input, NULL, prompt_tokens, max_pt);
    if (n_prompt == 0) {
        /* Fallback: direct tokenization without chat template. */
        slog(WARN, "Chat template not available, tokenizing raw input");
        n_prompt = tokenize_bpe(v, opts.input, (int)strlen(opts.input),
                                prompt_tokens, max_pt);
    }
    if (n_prompt == 0) slog(ERROR, "No valid tokens in input");
    slog(INFO, "Prompt tokens: %d", n_prompt);

    /* Benchmark iterations. */
    double *prefill_times = smalloc((size_t)opts.repeat * sizeof(double));
    double *gen_times     = smalloc((size_t)opts.repeat * sizeof(double));
    u32    *gen_counts    = smalloc((size_t)opts.repeat * sizeof(u32));

    /* Buffer for generated text. */
    char *output_text = NULL;
    int output_len = 0;
    if (opts.output) {
        output_text = smalloc(65536);
    }

    slog(INFO, "Running benchmark: %d iterations, %u tokens each", opts.repeat, opts.n_tokens);

    for (int r = 0; r < opts.repeat; r++) {
        /* Reset session. */
        session->ops.reset(session);

        /* --- Prefill --- */
        double t0 = time_sec();
        if (!session->ops.prefill(session, prompt_tokens, (u32)n_prompt, session->logits))
            slog(ERROR, "Forward pass failed at prefill");
        double t1 = time_sec();
        prefill_times[r] = (t1 - t0) * 1000.0; /* ms */

        /* --- Generate --- */
        u32 next_token;
        if (opts.temperature > 0.0f)
            next_token = sample_token(session->logits, session->cfg.n_vocab,
                                      opts.temperature, opts.top_k, opts.top_p, opts.min_p,
                                      session->repeat_penalty, session->repeat_last_n,
                                      session->frequency_penalty, session->presence_penalty,
                                      session->tokens, session->n_tokens);
        else
            next_token = sample_greedy(session->logits, session->cfg.n_vocab);

        if (opts.output) output_len = 0;

        double t_gen0 = time_sec();
        u32 n_gen = 0;
        for (u32 i = 0; i < opts.n_tokens; i++) {
            if (next_token == (u32)v->eos_id) break;
            fputc('.', stdout); fflush(stdout);

            /* Decode token for output. */
            if (opts.output && output_len < 65000) {
                char dec[64];
                int dlen = decode_token(session, v, next_token, dec, (int)sizeof(dec) - 1);
                if (dlen > 0) {
                    dec[dlen] = '\0';
                    int remaining = 65000 - output_len;
                    int to_copy = dlen < remaining ? dlen : remaining;
                    memcpy(output_text + output_len, dec, (size_t)to_copy);
                    output_len += to_copy;
                }
            }

            n_gen++;
            if (!session->ops.generate(session, next_token, session->logits))
                break;

            if (opts.temperature > 0.0f)
                next_token = sample_token(session->logits, session->cfg.n_vocab,
                                          opts.temperature, opts.top_k, opts.top_p, opts.min_p,
                                          session->repeat_penalty, session->repeat_last_n,
                                          session->frequency_penalty, session->presence_penalty,
                                          session->tokens, session->n_tokens);
            else
                next_token = sample_greedy(session->logits, session->cfg.n_vocab);
        }
        fputc('\n', stdout);
        double t_gen1 = time_sec();
        gen_times[r]  = (t_gen1 - t_gen0) * 1000.0; /* ms */
        gen_counts[r] = n_gen;

        fprintf(stderr, "\r[%d/%d] TTFT: %.1f ms | Prefill: %.1f ms | Generate: %.1f ms (%d tokens, %.1f tok/s)    ",
                r + 1, opts.repeat, prefill_times[r], prefill_times[r], gen_times[r], n_gen,
                n_gen / ((t_gen1 - t_gen0) > 0 ? (t_gen1 - t_gen0) : 1e-9));
    }
    fprintf(stderr, "\n");

    /* Compute statistics. */
    double sum_prefill = 0, sum_gen = 0;
    u64 sum_n_gen = 0;
    double min_prefill = prefill_times[0], max_prefill = prefill_times[0];
    double min_gen = gen_times[0], max_gen = gen_times[0];
    for (int r = 0; r < opts.repeat; r++) {
        sum_prefill += prefill_times[r];
        sum_gen     += gen_times[r];
        sum_n_gen   += gen_counts[r];
        if (prefill_times[r] < min_prefill) min_prefill = prefill_times[r];
        if (prefill_times[r] > max_prefill) max_prefill = prefill_times[r];
        if (gen_times[r] < min_gen) min_gen = gen_times[r];
        if (gen_times[r] > max_gen) max_gen = gen_times[r];
    }

    /* Peak memory. */
    struct rusage usage;
    long peak_mem_kb = 0;
    if (getrusage(RUSAGE_SELF, &usage) == 0)
        peak_mem_kb = usage.ru_maxrss;

    /* Print results. */
    printf("\n===== TinyLLaMA Benchmark =====\n");
    printf("Model:              %s\n", opts.engine.model_path);
    printf("Architecture:       %s\n", engine->model->arch_name);
    printf("Context Size:       %d\n", opts.ctx_size);
    printf("Threads:            %d\n", opts.nthread);
    printf("Prompt Tokens:      %d\n", n_prompt);
    printf("Target Tokens:      %u\n", opts.n_tokens);
    printf("Repeats:            %d\n", opts.repeat);
    printf("Temperature:        %.2f\n", opts.temperature);
    printf("Top-k:              %u\n", opts.top_k);
    printf("Top-p:              %.2f\n", opts.top_p);
    printf("Min-p:              %.2f\n", opts.min_p);
    printf("Repeat Penalty:     %.2f\n", session->repeat_penalty);
    printf("Repeat Last N:      %u\n", session->repeat_last_n);
    printf("Frequency Penalty:  %.2f\n", opts.frequency_penalty);
    printf("Presence Penalty:   %.2f\n", opts.presence_penalty);
    printf("────────────────────────────────────────────────────────────────────────────────────────────────\n");
    for (int r = 0; r < opts.repeat; r++) {
        double prefill_tok_s = (double)n_prompt / (prefill_times[r] / 1000.0);
        double gen_tok_s     = (double)gen_counts[r] / (gen_times[r] / 1000.0);
        printf("[%d/%d] TTFT: %8.1f ms | Prefill: %8.1f ms (%7.1f tok/s) | Generate: %8.1f ms (%d tok, %7.1f tok/s)\n",
               r + 1, opts.repeat,
               prefill_times[r],
               prefill_times[r], prefill_tok_s,
               gen_times[r], gen_counts[r], gen_tok_s);
    }
    printf("────────────────────────────────────────────────────────────────────────────────────────────────\n");

    double avg_prefill = sum_prefill / opts.repeat;
    double avg_gen     = sum_gen / opts.repeat;
    double avg_prefill_tok_s = (double)n_prompt * opts.repeat / (sum_prefill / 1000.0);
    double avg_gen_tok_s     = (double)sum_n_gen / (sum_gen / 1000.0);
    printf("Avg TTFT:       %8.1f ms\n", avg_prefill);
    printf("Avg Prefill:    %8.1f ms (%7.1f tok/s)\n", avg_prefill, avg_prefill_tok_s);
    printf("Avg Generate:   %8.1f ms (%7.1f tok/s)\n", avg_gen, avg_gen_tok_s);
    printf("Min/Max TTFT:     %.1f / %.1f ms\n", min_prefill, max_prefill);
    printf("Min/Max Generate: %.1f / %.1f ms\n", min_gen, max_gen);
    if (peak_mem_kb > 0)
        printf("Peak Memory:    %.0f MB\n", (double)peak_mem_kb / 1024.0);
    printf("════════════════════════════════════════════════════════════════════════════════════════════════\n");

    /* Write output file. */
    if (opts.output && output_len > 0) {
        output_text[output_len] = '\0';
        FILE *fout = fopen(opts.output, "w");
        if (fout) {
            fprintf(fout, "%s", output_text);
            fclose(fout);
            slog(INFO, "Output written to %s", opts.output);
        } else {
            slog(WARN, "Failed to write output to %s", opts.output);
        }
    }

    /* Cleanup. */
    sfree(prefill_times);
    sfree(gen_times);
    sfree(gen_counts);
    sfree(output_text);
    session_free(session);
    engine_close(engine);

    return 0;
}
