from gguf import GGUFReader

# 替换为你的 GGUF 文件路径
reader = GGUFReader("/home/zc/Work/model/Qwen3.5-0.8B-BF16.gguf")

# GGUF 中的键名通常带有架构前缀，例如 "llama.embedding_length"
# 我们可以通过循环遍历直接匹配包含 "embedding_length" 的键
n_embd = None
for key in reader.fields.keys():
    if "embedding_length" in key:
        n_embd = reader.fields[key].parts[-1][0]
        print(f"键名: {key}, n_embd (embedding_length) = {n_embd}")
        break

if n_embd is None:
    print("未在元数据中找到 embedding_length。")
