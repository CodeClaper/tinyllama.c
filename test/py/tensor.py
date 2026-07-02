from gguf import GGUFReader

# 加载 GGUF 文件
reader = GGUFReader("/home/zc/Work/model/Qwen3.5-0.8B-BF16.gguf")

# 1. 获取核心配置值 (元数据)
# 键名会根据模型架构带有前缀，例如 llama.embedding_length
metadata = {field.name: field.parts[-1] for field in reader.fields.values()}

n_embd = metadata.get("llama.embedding_length") or metadata.get("general.embedding_length")
n_vocab = metadata.get("llama.vocab_size") or metadata.get("general.vocab_size")

print(f"--- 元数据配置 ---")
print(f"n_embd: {n_embd}")
print(f"n_vocab: {n_vocab}\n")

# 2. 获取 Tensor 维度信息
print(f"--- Tensor 维度信息 (前5个示例) ---")
for i, tensor in enumerate(reader.tensors):
    if i < 5:  # 仅打印前几个展示结构
        # tensor.shape 是一个列表，例如 [4096, 32000]
        print(f"Tensor 名: {tensor.name:25} | 维度 (Shape): {tensor.shape}")
