#include <stdio.h>
#include <stdlib.h>

/* Usage. */
static void usage(FILE *file, int exit_code) {
    fprintf(file, "Usage:   chat [options]\n");
    fprintf(file, "Example: chat -m model.gguf -s \"You are a helpful assistant.\"\n");
    fprintf(file, "Options:\n");
    fprintf(file, "  -m | --model   <string>  The model file path to run\n");
    fprintf(file, "  -c | --ctx     <int>     The context size, default 4096\n");
    fprintf(file, "  -n | --tokens  <int>     Number of tokens to generate, default 128\n");
    fprintf(file, "  -T | --threads <int>     Number of threads, default 1\n");
    fprintf(file, "  -t | --temp    <float>   Temperature for sampling, default 0.0 (greedy)\n");
    fprintf(file, "  -p | --topp    <float>   Top-p (nucleus) threshold, default 1.0\n");
    fprintf(file, "  -s | --system  <string>  System input\n");
    exit(exit_code);
}

int main(int argc, char *argv[]) {
    if (argc < 2) usage(stderr, 4);
}
