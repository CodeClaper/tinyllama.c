"""
Tokenize text in Qwen2.5 chat format and print the token embeddings.

The embedding dequantization matches the memory layout used in src/arch/qwen2.c:
  ws->x[i] = tensor_get_f32(te, base, (u64)i * n_vocab + token);
i.e. n_vocab is the contiguous (innermost) dimension in the GGUF file, so
the numpy reshape must use Fortran order.

Usage:
    python3 test/py/token_embding.py
"""

import numpy as np
import gguf
import os
import sys
from llama_cpp import Llama

MODEL_PATH = "/home/zc/Work/model/qwen2.5-1.5b-instruct-q3_k_m.gguf"
NPY_PATH = os.path.join(os.path.dirname(__file__), "token_embd.npy")
INPUT_TEXT = "yes"

QK_BLOCK_SIZE = 256
QK_BLOCK_BYTES = 110


def extract_token_embedding(model_path: str) -> np.ndarray:
    """Extract and dequantize token_embd.weight from a GGUF model.
    Returns (n_vocab, n_embd) float32 array matching the C memory layout."""
    reader = gguf.GGUFReader(model_path)

    tensor = None
    for t in reader.tensors:
        if t.name == "token_embd.weight":
            tensor = t
            break
    if tensor is None:
        raise ValueError("token_embd.weight not found")

    n_embd, n_vocab = int(tensor.shape[0]), int(tensor.shape[1])
    qt = gguf.GGMLQuantizationType(tensor.tensor_type)
    n_blocks = tensor.n_elements // QK_BLOCK_SIZE

    print(f"Tensor: {tensor.name}  shape=[{n_embd}, {n_vocab}]  "
          f"quant={qt.name}  blocks={n_blocks:,}")

    with open(model_path, "rb") as f:
        f.seek(tensor.data_offset)
        raw_bytes = f.read(tensor.n_bytes)

    block_data = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(n_blocks, QK_BLOCK_BYTES)

    CHUNK = 50000
    chunks = []
    for start in range(0, n_blocks, CHUNK):
        end = min(start + CHUNK, n_blocks)
        sys.stdout.write(f"\rDequantizing [{start:,}..{end:,}] / {n_blocks:,}")
        sys.stdout.flush()
        chunks.append(gguf.dequantize(block_data[start:end], qt))
    print()

    flat = np.concatenate(chunks).ravel()  # (n_elements,) in GGUF memory order

    # GGUF memory layout: shape [n_embd, n_vocab] in C/row-major order,
    # meaning n_vocab is contiguous.  The C code at qwen2.c:136 does:
    #   ws->x[i] = tensor_get_f32(te, base, i * n_vocab + token);
    # So flat[dim * n_vocab + tok] is the value for embedding dim `dim` of token `tok`.
    # To get a (n_vocab, n_embd) matrix where row tok is that token's embedding,
    # use Fortran-order reshape:
    embedding = flat.reshape(n_vocab, n_embd, order="F")

    print(f"Output: ({n_vocab}, {n_embd})  dtype={embedding.dtype}  "
          f"size={embedding.nbytes / 1024**2:.1f} MB")
    return embedding


def main():
    # ---- 1. Load or extract token embeddings ----
    if os.path.exists(NPY_PATH):
        embedding = np.load(NPY_PATH)
        print(f"Loaded from: {NPY_PATH}")
        print(f"Shape: {embedding.shape}  dtype: {embedding.dtype}")
    else:
        embedding = extract_token_embedding(MODEL_PATH)
        np.save(NPY_PATH, embedding)
        print(f"Saved to: {NPY_PATH}")

    n_vocab, n_embd = embedding.shape

    # ---- 2. Tokenize prompt ----
    llm = Llama(model_path=MODEL_PATH, verbose=False)

    prompt = (
        f"<|im_start|>user\n{INPUT_TEXT}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    tokens = llm.tokenize(prompt.encode("utf-8"), add_bos=False, special=True)
    print(f"\nPrompt: {prompt!r}")
    print(f"Tokens ({len(tokens)}): {tokens}\n")

    # ---- 3. Print each token's text + embedding (first 10 dims) ----
    for tid in tokens:
        piece = llm.detokenize([tid]).decode("utf-8", errors="replace")
        emb = embedding[tid]  # shape (n_embd,)
        print(f"token {tid:>6}: {piece!r:10s}  "
              f"emb[:10] = [{', '.join(f'{v: .8f}' for v in emb[:10])}]  "
              f"norm={np.linalg.norm(emb):.4f}")


if __name__ == "__main__":
    main()
