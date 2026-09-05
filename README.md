# tinyllama.c

Tinyllama.c is a tiny and simple inference engine for LLMs, written in pure C. It loads GGUF v3 model files (the format used by llama.cpp) via mmap for zero-copy access. It provides three modes of operation:

- **server** — OpenAI-compatible HTTP chat server
- **chat** — Interactive CLI chat
- **bench** — CLI inference benchmark

## Features

- Pure C, no external ML frameworks
- mmap-based zero-copy GGUF v3 model loading
- OpenAI-compatible chat completions API (`/v1/chat/completions`)
- Interactive CLI chat with conversation history
- Inference benchmark with time and throughput statistics
- Multi-threaded inference via thread pool
- Supports multiple model architectures:
  - LLaMA
  - Qwen2
  - DeepSeek (including MLA)
  - Falcon
- Multiple quantization formats (Q4_0, Q4_K, Q5_K, Q6_K, Q8_0, IQ2_XXS, IQ3_XXS, etc.)
- epoll-based event loop for HTTP serving
- Optional CUDA GPU acceleration with automatic CPU fallback (`GPU=1 make`)
- Optional Metal GPU acceleration on macOS with automatic CPU fallback (`METAL=1 make`)
- CPU backends with x86 and ARM optimizations

## Supported Models

| Model Family | GGUF Architecture ID | Status |
|---|---|---|
| Qwen2 / Qwen2.5 | `qwen2` | Supported |
| Qwen3 / Qwen3.5 | `qwen35` | WIP |
| LLaMA / Llama 2 / Llama 3 / Llama 3.1 / Llama 3.2 | `llama` | WIP |
| DeepSeek-V2 | `deepseek2` | WIP |
| Falcon | `falcon` | WIP |

## Dependencies

- **Compiler:** GCC
- **Libraries:** libm (math), pthreads, Linux kernel (eventfd, epoll)
- **Optional:** CUDA toolkit (nvcc) for the GPU backend, or macOS with Xcode command line tools for the Metal backend
- **OS:** Linux only (uses `eventfd`, `epoll`, `mmap`)

## Build

```bash
# Release build (O3, march=native, LTO) — produces server, chat, and bench
make

# Debug build (release flags + -g -DDEBUG; enables asserts, runs at near-release speed)
DEBUG=1 make

# Trace build (enables DBG_VEC tensor-print macro in model code)
DEBUG_VEC=1 make

# AddressSanitizer build (on top of release or DEBUG flags; ~2-4x slower, use for memory debugging)
SAN=1 make

# CUDA GPU build (requires nvcc; falls back to CPU per-op when the GPU path is unavailable)
GPU=1 make

# Metal GPU build (macOS only; requires clang + Metal framework)
METAL=1 make

# Run tests
make check

# Clean
make clean
```

Binaries are produced at `src/server`, `src/chat`, and `src/bench`.

## Usage

### Server — OpenAI-compatible HTTP API

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

### Chat — Interactive CLI

```bash
# Start an interactive chat session
./src/chat -m path/to/model.gguf

# Full options
./src/chat -m model.gguf                      # Model file (required)
           -c 4096                            # Context size (default: 4096)
           -n 128                             # Max tokens per response (default: 128)
           -T 1                               # Thread count (default: 1)
           -t 0.8                             # Temperature (default: 0.8)
           -tp 0.9                            # Top-p sampling (default: 0.9)
           -tk 40                             # Top-k sampling (default: 40)
           -P 0.05                            # Min-p threshold (default: 0.05)
           -s "You are a helpful assistant."  # System prompt

# With a custom system prompt
./src/chat -m model.gguf -s "You are a Python programming expert."
```

During the chat session, use `/clear` to reset the conversation and `/quit` to exit.

### Bench — Inference Benchmark

```bash
# Run a benchmark with a prompt
./src/bench -m path/to/model.gguf -i "Who is Isaac Newton?"

# Full options
./src/bench -m model.gguf                     # Model file (required)
            -c 4096                           # Context size (default: 4096)
            -n 128                            # Tokens to generate per iteration (default: 128)
            -T 1                              # Thread count (default: 1)
            -t 0.8                            # Temperature (default: 0.8)
            -tp 0.9                           # Top-p sampling (default: 0.9)
            -tk 40                            # Top-k sampling (default: 40)
            -P 0.05                           # Min-p threshold (default: 0.05)
            -i "Some prompt"                  # Input prompt (required)
            -r 5                              # Repeat count (default: 1)
            -o output.txt                     # Save generated text to file

# Run 10 iterations and save output
./src/bench -m model.gguf -i "Explain quantum computing." -r 10 -o output.txt
```

The benchmark runs the prompt through prefill and generation phases, repeating `-r` times, then prints timing statistics including average tokens/second for both phases and peak memory usage. A GPU build prints the active device (`CUDA`, `Metal`, or `CPU`).

## Benchmark

Benchmark results on a consumer desktop (12 threads, context 4096, 13 prompt tokens, 128 generated tokens). Offloading matmuls to the GPU speeds up generation by roughly 6-14x over CPU-only:

| Model | Device | Quant | Peak Mem | TTFT | Prefill | Generate | tok/s |
|---|---|---|---|---|---|---|---|
| Qwen2.5 1.5B Instruct | CPU | Q4_K_M | 1198 MB | 341.0 ms | 341.0 ms (38.1 tok/s) | 11386.6 ms | 11.2 |
| Qwen2.5 1.5B Instruct | CUDA | Q4_K_M | 1403 MB | 480.8 ms | 480.8 ms (27.0 tok/s) | 1739.0 ms | 73.6 |
| Qwen2.5 1.5B Instruct | CPU | Q3_K_M | 1043 MB | 427.9 ms | 427.9 ms (30.4 tok/s) | 10054.4 ms | 9.8 |
| Qwen2.5 1.5B Instruct | CUDA | Q3_K_M | 1249 MB | 374.7 ms | 374.7 ms (34.7 tok/s) | 1772.0 ms | 72.2 |
| Qwen3.5 0.8B | CPU | BF16 | 1883 MB | 799.2 ms | 799.2 ms (16.3 tok/s) | 24080.8 ms | 5.3 |
| Qwen3.5 0.8B | CUDA | BF16 | 2087 MB | 414.2 ms | 414.2 ms (31.4 tok/s) | 1665.0 ms | 76.9 |

> CUDA row built with `GPU=1 make`; matmuls above a size threshold run on the GPU, smaller ones fall back to CPU. The GPU path trades slower prefill on the small prompt (fixed kernel-launch overhead per op) for a large generation throughput win.

## Project Structure

```
- src/
  - server.c          — CLI entry point, HTTP server, option parsing
  - chat.c            — Interactive CLI chat
  - bench.c           — Inference benchmark
  - core.c / core.h   — Engine/model loading, GGUF parsing
  - mm.c              — Safe memory allocators
  - mm.h
  - slog.c            — Logging
  - slog.h
  - utils.c           — String parsing, key comparison, etc.
  - utils.h
  - el.c              — epoll event loop
  - el.h
  - anet.c            — TCP networking helpers
  - anet.h
  - http.c            — HTTP request/response helpers
  - http.h
  - tokenizer.c       — Tokenizer (BPE/SPM/WPM)
  - tokenizer.h
  - sampler.c         — Token sampling
  - sampler.h
  - pthreads.c        — Thread pool
  - pthreads.h
  - quants.c          — Quantized tensor dequantization
  - quants.h
  - kvcache.c         — KV cache
  - kvcache.h
  - rax.c             — Radix tree
  - rax.h
  - model/            — Architecture-specific forward passes
    - model.h
    - llama.c         — LLaMA architecture
    - qwen25.c        — Qwen2.5 architecture
    - qwen35.c        — Qwen3.5 architecture
    - deepseek.c      — DeepSeek architecture
    - falcon.c        — Falcon architecture
  - backend/          — Hardware backends
    - cpu/            — CPU-optimized kernels
      - cpu.c         — Kernel dispatcher
      - cpu.h
      - cpu_x86.c     — x86 SIMD optimizations
      - cpu_arm.c     — ARM NEON optimizations
    - gpu/            — GPU-optimized kernels
      - gpu.cu        — CUDA persistent-weight dequant + matmul/matvec backend
      - gpu.h
    - metal/          — Metal GPU backend (macOS only)
      - metal.m       — Objective-C host side
      - metal.h
      - metal.metal   — MSL kernels (embedded via generated metal_msl.h)

- test/               — Unit tests
  - Makefile
  - c/                — Test sources
    - minunit.h
    - test_pthreads.c — Thread pool tests
    - test_qwen25.c   — Qwen2 integration tests
    - test_utils.c    — Utility function tests
  - py/               — Test utilities
    - generate_test_vectors.py
```
