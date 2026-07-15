from llama_cpp import Llama
from llama_cpp import llama_cpp as llama_c
import ctypes

llm = Llama("/home/zc/Work/model/qwen2.5-1.5b-instruct-q3_k_m.gguf",
            n_ctx=256, verbose=False, logits_all=True)

# ---- Model metadata ----
print("=== Model Metadata ===")
md = llm.metadata
print(f"  n_embd     = {llm.n_embd()}")
print(f"  n_vocab    = {llm.n_vocab()}")
print(f"  n_ctx      = {llm.n_ctx()}")

# Search metadata for key params
import re
interesting = ['head', 'layer', 'rope', 'freq', 'norm', 'eps', 'dim', 'block']
for k, v in sorted(md.items()):
    kl = str(k).lower()
    if any(x in kl for x in interesting):
        if isinstance(v, (bytes, bytearray)):
            if len(v) < 100:
                print(f"  {k} = {v}")
        else:
            print(f"  {k} = {v}")

# ---- Tokenize ----
print("\n=== Tokens ===")
tokens = llm.tokenize(b"<|im_start|>user\nyes<|im_end|>\n<|im_start|>assistant\n", add_bos=True, special=True)
print(f"  n_tokens = {len(tokens)}")
print(f"  tokens   = {tokens}")

# ---- Test 1: Single BOS token ----
print("\n=== Single BOS token (151644) ===")
bos_token = 151644
llm.reset()
llm.eval([bos_token])
n_vocab = llm.n_vocab()
FloatArrayType = ctypes.c_float * n_vocab
logits_ptr = llama_c.llama_get_logits_ith(llm.ctx, 0)
logits_arr = FloatArrayType.from_address(ctypes.addressof(logits_ptr.contents))
bos_logits = [logits_arr[i] for i in range(10)]
print(f"  BOS logits (first 10): {[f'{v:.6f}' for v in bos_logits]}")

# ---- Test 2: Eval all tokens ----
print("\n=== Per-position Logits (first 10 each) ===")
llm.reset()
llm.eval(tokens)

for i, tok in enumerate(tokens):
    logits_ptr = llama_c.llama_get_logits_ith(llm.ctx, i)
    logits_arr = FloatArrayType.from_address(ctypes.addressof(logits_ptr.contents))
    vals = [logits_arr[j] for j in range(10)]
    print(f"  pos[{i}] token={tok}: {[f'{v:.6f}' for v in vals]}")

# ---- Final logits (last position) ----
print("\n=== Final Logits (last position) ===")
logits_ptr = llama_c.llama_get_logits_ith(llm.ctx, len(tokens) - 1)
logits_arr = FloatArrayType.from_address(ctypes.addressof(logits_ptr.contents))
logits_list = [logits_arr[i] for i in range(n_vocab)]

print(f"  First 10: {[f'{logits_list[i]:.6f}' for i in range(10)]}")

top5 = sorted(range(n_vocab), key=lambda i: logits_list[i], reverse=True)[:5]
print(f"  Top-5:")
for idx in top5:
    token_str = llm.detokenize([idx]).decode("utf-8", errors="replace")
    print(f"    [{idx}] {repr(token_str)}: {logits_list[idx]:.6f}")
