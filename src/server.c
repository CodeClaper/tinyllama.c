#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "def.h"
#include "slog.h"
#include "utils.h"

/* Useage. */
static void usage(FILE *file, int exit_code) {
    fprintf(file, "Usage:   server [options]\n");
    fprintf(file, "Example: server model.bin -n 256 -i \"Once upon a time\"\n");
    fprintf(file, "Options:\n");
    fprintf(file, "  -t <float>  temperature in [0,inf], default 1.0\n");
    fprintf(file, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
    fprintf(file, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(file, "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n");
    fprintf(file, "  -i <string> input prompt\n");
    fprintf(file, "  -z <string> optional path to custom tokenizer\n");
    fprintf(file, "  -m <string> mode: generate|chat, default: generate\n");
    fprintf(file, "  -y <string> (optional) system prompt in chat mode\n");
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
    for (int i = 0; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) usage(stdout, EXIT_SUCCESS);
        else if (!strcmp(arg, "-m") || !strcmp(arg, "--model")) so.engine.model_path = parse_arg(argc, argv, &i, arg);
        else if (!strcmp(arg, "-h") || !strcmp(arg, "--host")) so.host = parse_arg(argc, argv, &i, arg);
        else if (!strcmp(arg, "-p") || !strcmp(arg, "--port")) so.port = parse_int(parse_arg(argc, argv, &i, arg));
        else if (!strcmp(arg, "-c") || !strcmp(arg, "--ctx")) so.ctx_size = parse_int(parse_arg(argc, argv, &i, arg));
        else if (!strcmp(arg, "-n") || !strcmp(arg, "--tokens")) so.default_tokens = parse_int(parse_arg(argc, argv, &i, arg));
        else {
            slog(WARN, "Unkonow option: %s", arg);
            usage(stderr, 2);
        }
    }
    return so;
} 

int main(int argc, char *argv[]) {
    ServerOptions so = parse_options(argc, argv);
    printf("Hello from tinyllama.c. \n");
}
