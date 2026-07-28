# tinyllama.c

Tinyllama.c is a tiny and simple inference engine for LLMs, written in pure C. It loads GGUF v3 model files (the format used by llama.cpp) via mmap for zero-copy access and runs them as an OpenAI-compatible HTTP chat server.

## Features

- Pure C, no external ML frameworks
- mmap-based zero-copy GGUF v3 model loading
- OpenAI-compatible chat completions API (`/v1/chat/completions`)
- Multi-threaded inference via thread pool
- Supports multiple model architectures:
  - LLaMA
  - Qwen2
  - DeepSeek (including MLA)
  - Falcon
- Multiple quantization formats (Q4_0, Q4_K, Q5_K, Q6_K, Q8_0, IQ2_XXS, IQ3_XXS, etc.)
- epoll-based event loop for HTTP serving
- CPU backends with x86 and ARM optimizations

## Dependencies

- **Compiler:** GCC
- **Libraries:** libm (math), pthreads, Linux kernel (eventfd, epoll)
- **OS:** Linux only (uses `eventfd`, `epoll`, `mmap`)

## Build

```bash
# Release build (O3, march=native, LTO)
make

# Debug build (O0, debug symbols, -DDEBUG)
DEBUG=1 make

# Run tests
make check

# Clean
make clean
```

The binary is produced at `src/server`.

## Usage

```bash
# Run with a GGUF model file
./src/server -m path/to/model.gguf

# Full options
./src/server -m model.gguf                    # Model file (required)
             -p 8080                          # Port (default: 8080)
             -h 127.0.0.1                     # Host (default: 127.0.0.1)
             -c 4096                          # Context size (default: 4096)
             -n 393216                        # Max tokens (default: 393216)
             -t 1.0                           # Temperature (default: 1.0)
             -T 1                             # Thread count (default: 1)
             --topp 0.9                       # Top-p sampling (default: 0.9)
             -i                               # Inspect model metadata only

# Inspect a model without running the server
./src/server -m model.gguf -i
```

Send requests to the server:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
  "model": "qwen2.5",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ]
}'
```

## Project Structure

```
├── src/
│   ├── server.c        — CLI entry point, HTTP server, option parsing
│   ├── core.c          — Engine/model loading, GGUF parsing
│   ├── mm.c            — Safe memory allocators
│   ├── slog.c          — Logging
│   ├── utils.c         — String parsing, key comparison, etc.
│   ├── el.c            — epoll event loop
│   ├── anet.c          — TCP networking helpers
│   ├── http.c          — HTTP request/response helpers
│   ├── tokenizer.c     — Tokenizer (BPE/SPM/WPM)
│   ├── sampler.c       — Token sampling
│   ├── tpool.c         — Thread pool
│   ├── quants.c        — Quantized tensor dequantization
│   ├── model/          — Architecture-specific forward passes
│   │   ├── llama.c
│   │   ├── qwen2.c
│   │   ├── deepseek.c
│   │   └── falcon.c
│   └── cpu/            — CPU-optimized kernels
│       ├── quants_cpu.c
│       ├── quants_x86.c
│       └── quants_arm.c
└── test/               — Unit tests
```
