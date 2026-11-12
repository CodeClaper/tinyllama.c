#ifndef TINYLLAMA_BANNER_H
#define TINYLLAMA_BANNER_H

#include <stdio.h>

/* Startup banner shared by the chat and server entry points. */
static const char TINYLLAMA_BANNER[] =
    "\n"
    "█████  █████  █   █  █   █  █      █       ███   █   █   ███\n"
    " ▒█▒▒▒  ▒█▒▒▒ ██  █▒ █▒  █▒ █▒     █▒     █ ▒▒█  ██ ██▒ █ ▒▒█\n"
    "  █▒░░░  █▒░░░█▒█ █▒░ █░█ ▒░█▒░    █▒░    █████▒  █▒█ █▒░█████▒\n"
    "  █▒░    █▒░  █▒░██▒░  █ ▒ ░█▒░    █▒░    █▒▒▒█▒░█▒░▒█▒░█▒▒▒█▒░\n"
    "  █▒░  █████  █▒░ █▒░  █▒ ░ █████  █████  █▒░░█▒░█▒░ █▒░█▒░░█▒░\n"
    "   ▒░   ▒▒▒▒▒  ▒░  ▒░   ▒░   ▒▒▒▒▒  ▒▒▒▒▒  ▒░  ▒░ ▒░  ▒░ ▒░  ▒░\n"
    "    ░    ░░░░░  ░   ░    ░    ░░░░░  ░░░░░  ░   ░  ░   ░  ░   ░\n"
    "        tinyllama.c  |  pure C LLM inference  |  GGUF v3\n";

static inline void banner_print(void) {
    fputs(TINYLLAMA_BANNER, stdout);
    fflush(stdout);
}

#endif /* TINYLLAMA_BANNER_H */
