#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include "def.h"
#include "slog.h"
#include "utils.h"
#include "mm.h"
#include "core.h"

static void signal_handler(int sig) {
    UNUSED(sig);
    _exit(300);
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
        else if (!strcmp(arg, "-i") || !strcmp(arg, "--inspect")) so.inspect = true;
        else {
            fprintf(stderr, "Unkonow option: %s.\n", arg);
            usage(stderr, 2);
        }
    }
    return so;
}


int main(int argc, char *argv[]) {
    signal(SIGPIPE, SIG_IGN);
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = signal_handler;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);

    ServerOptions so = parse_options(argc, argv);
    Engine *en = engine_load(&so.engine);
    if (so.inspect) engine_summary(en);
}
