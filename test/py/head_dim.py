from gguf import GGUFReader

# 1. 加载 GGUF 文件
reader = GGUFReader("/home/zc/Work/model/Qwen3.5-0.8B-BF16.gguf")

# 2. 尝试直接获取标准键（注意：不同架构的架构前缀不同，如 llama.attention.key_length）
key_length_field = reader.get_field(f"qwen35.attention.key_length")

if key_length_field:
    print(key_length_field)
    # 键存在，直接读取（通常返回 uint32/uint64）
    head_dim = key_length_field.parts[-1][0]
    print(f"直接获取模型的 head_dim 为: {head_dim}")
else:
    # 键不存在，通过经典公式隐式计算：隐藏层维度 / 注意力头数
    hidden_size = reader.get_field(f"qwen35.embedding_length").parts[-1][0]
    num_heads = reader.get_field(f"qwen35.attention.head_count").parts[-1][0]
    head_dim = hidden_size // num_heads
    print(f"推倒获取模型的 head_dim 为: {head_dim}")

