import ctypes
import numpy as np
from llama_cpp import Llama, llama_cpp

MODEL_PATH = "/home/zc/Work/model/qwen2.5-1.5b-instruct-q3_k_m.gguf"  # fill in
llm = Llama(model_path=MODEL_PATH, n_ctx=256, verbose=False, logits_all=True)
n_embd = llm.n_embd()
n_vocab = llm.n_vocab()
n_layer = llm.n_batch  # not correct; need to find n_layer
# Actually, get metadata:
ctx = llm._ctx
print(f"n_embd={n_embd}, n_vocab={n_vocab}")

# Print hyperparams for cross-check with C
import llama_cpp._internals as _internals
# These might not be directly exposed; try model metadata
model = llm._model

# For token embedding, access raw tensor
# llama_get_model_tensor might be available
# Try direct approach:
prompt_txt = b"<|im_start|>user\nyes<|im_end|>\n<|im_start|>assistant\n"
tokens = llm.tokenize(prompt_txt, add_bos=True, special=True)
print(f"\nTokens ({len(tokens)}): {tokens}")

# Eval and compare logits
llm.reset()
llm.eval(tokens)
logits = llm.scores[len(tokens) - 1]
print(f"\nOutput logits (first 20):")
for i, v in enumerate(logits[:20]):
    print(f"  [{i}] {v}")
