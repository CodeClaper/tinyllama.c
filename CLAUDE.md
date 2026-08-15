# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
# Release build
make

# Debug build (release flags + -g -Wall -gdwarf-2 -g3 -DDEBUG; enables asserts)
DEBUG=1 make

# Trace build (adds -DDEBUG_VEC, enabling the DBG_VEC tensor-print macro in model code)
DEBUG_VEC=1 make

# AddressSanitizer build (on top of release or DEBUG flags; ~2-4x slower)
SAN=1 make

# Run tests
make check

# Clean
make clean
```

The root Makefile delegates to `src/Makefile` and `test/Makefile`. The compiler is `gcc`, linking with `-lm`.

## Architecture

This is a pure-C LLM inference engine that loads GGUF v3 model files (the format used by llama.cpp) via mmap for zero-copy access.

### Module layering

```
server.c          — CLI entry point, option parsing
  └─ core.c       — Engine/model loading, GGUF parsing, summary
       ├─ mm.c    — Safe memory allocators (smalloc/scalloc/srealloc/sstrdup/sfree)
       ├─ slog.c  — Logging with timestamps (INFO/SUCCESS/WARN/ERROR); ERROR exits
       └─ utils.c — String parsing, Key comparison, datetime formatting, size conversion
```

### Key types (defined in `src/def.h`)

- **`Key`** — A length-prefixed string slice (u8 len, char *content), not null-terminated. Used throughout for GGUF metadata key names.
- **`KV`** — A key-value metadata entry from the GGUF header.
- **`TensorInfo`** — Tensor metadata: name (Key), dimensions, GGUF type, file offset, element count.
- **`Model`** — The mmap'd GGUF file. Holds the file descriptor, raw mapped bytes, arrays of KV and TensorInfo, and the GGUF alignment.
- **`Engine`** — Wraps a Model pointer. Currently minimal, will grow as inference is added.
- **`EngineOptons`** / **`ServerOptions`** — Configuration structs (note: `EngineOptons` is misspelled — intentional as-is in the codebase).

### GGUF parsing flow (`src/core.c`)

1. `model_load()` mmaps the file, checks the GGUF magic (`0x46554747` little-endian), validates version == 3, reads tensor/kv counts.
2. `kv_load()` iterates through metadata key-value pairs, extracting alignment info.
3. `tensor_load()` reads each tensor's name, dimensions, type, and offset.
4. A `Cursor` struct walks sequentially through the mmap'd bytes — all reads advance `post`. On error, `cursor_error()` records the byte offset.

The `gguf_types[]` table maps GGUF quantization type IDs (q4_0, q4_k, iq2_xxs, etc.) to block element counts and byte sizes, used by `tensor_bytes()` to compute tensor storage size.

### Logging conventions

- `slog(ERROR, ...)` and `slog_errno(...)` both call `exit(100)` after printing.
- `WARN` and above go to stderr; `INFO` and `SUCCESS` go to stdout.
- Every log line includes `[timestamp][pid][LEVEL]` prefix.
- `slog_errno()` appends `strerror(errno)` to the message.
