#include <stdio.h>
#include <stdlib.h>

/* Useage. */
static void usage(FILE *file, int exit_code) {
    fprintf(file, "Usage:   bench [options]\n");
    fprintf(file, "Example: bench -m model.gguf -i Who is Issac Newton?\n");
    fprintf(file, "Options:\n");
    fprintf(file, "  -m | --model   <string>  The model file path to run\n");
    fprintf(file, "  -c | --ctx     <int>     The context size, defalt 4096\n");
    fprintf(file, "  -n | --tokens  <int>     The default token size, defalt 393216\n");
    fprintf(file, "  -t | --temp    <float>   Temperature for sampling, default 1.0\n");
    fprintf(file, "  -p | --topp    <float>   Top-p (nucleus) threshold, default 0.9\n");
    fprintf(file, "  -i | --input   <string>  User input\n");
    exit(exit_code);
}

int main(int argc, char *argv[]) {
    if (argc < 2) usage(stderr, 3);
    return 0;
}
