from llama_cpp import Llama, llama_cpp
import ctypes

llm = Llama("/home/zc/Work/model/qwen2.5-1.5b-instruct-q3_k_m.gguf",
            n_ctx=256, verbose=False)

# Apply chat template manually to see the exact tokens
messages = [{"role": "user", "content": "yes"}]

# Try to get tokenized prompt
try:
    # llama_cpp has tokenize method
    prompt = llm.create_chat_completion(messages=messages, max_tokens=1, temperature=0)
    # Can't see the tokens, but can see the output
    print(f"Response: {prompt['choices'][0]['message']['content'][:50]}")
except Exception as e:
    print(f"Error: {e}")

# Try tokenizing
tokens = llm.tokenize(b"<|im_start|>user\nyes<|im_end|>\n<|im_start|>assistant\n",
add_bos=True, special=True)
print(f"Prompt tokens: {tokens}")

# Also try with the chat handler
tokens2 = llm.tokenize(b"<|im_start|>user\nyes<|im_end|>\n<|im_start|>assistant\n",
add_bos=True)
print(f"Prompt tokens (no special): {tokens2}")
