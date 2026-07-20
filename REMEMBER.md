# Inference improvements vs llama.cpp

## Critical (massive speedup)

1. **Zero multi-threaded compute** — `pthreads_t` pool allocated but never used in forward passes (`qwen2_forward`/`qwen2_forward_one`). Every GEMM/GEMV and attention loop runs single-threaded. ll.cpp parallelizes across rows in every mat_mul and across heads in attention.

2. **Transposed `mat_vec_mul` is O(n²) dequant calls** (`core.c:190-196`): Calls `tensor_get_f32()` per element when weights are non-contiguous. Should batch-dequantize columns into a buffer then dot-product.

3. **`mat_mat_mul` uses scalar inner loop** (`core.c:273-275`, `294-296`): After dequantizing a row, the batch dot-product is a plain scalar loop. No explicit SIMD reduction and no thread parallelism across rows.

## High

4. **Attention is entirely scalar** (`qwen2.c:230-245`, `556-568`):
   - Q·K^T scores use nested scalar loops instead of SIMD fused dot products
   - Softmax + V-weighted sum is scalar — no SIMD reduction
   - Prefill attention re-reads KV cache from scratch per query position (O(batch × seq_len × head_dim) with no tiling)

5. **No KV cache quantization** — f32 KV cache is 4× memory vs Q8_0. ll.cpp supports quantized KV cache (Q8_0, Q6_K) to improve attention memory bandwidth.

6. **No RoPE precomputation** — `rope()` (`core.c:307-320`) recomputes `powf`/`cosf`/`sinf` per token per layer per head. Precompute sin/cos tables once per position.

## Medium

7. **Attention output buffering** (`qwen2.c:207-212`, `474-481`): KV cache write uses one `memcpy` per head. Fuse RoPE + KV write into one loop to avoid redundant passes.

8. **No AVX2/AVX-512** — x86 path uses SSE only (128-bit). ll.cpp uses AVX2 (256-bit) baseline with AVX-512 kernels. 2×+ throughput left on table for modern x86.

9. **Thread pool uses per-item mutex lock** (`pthreads.c:33-41`): Every work item acquires/releases pool lock. High contention for fine-grained parallelism. ll.cpp uses barrier-based dispatch.

10. **Softmax two-pass overhead** — fine per-token but adds O(n) per call in prefill path.

## Architectural (future)

11. **Only Qwen2 has a working forward pass** — Llama, DeepSeek (MLA), Falcon are stubs. Llama is the most popular family and should be prioritized.

12. **No GPU backend** — `src/gpu/quants_gpu.c` / `.h` are both empty. Even a simple CUDA mat_mul kernel would transform prefill speed.
