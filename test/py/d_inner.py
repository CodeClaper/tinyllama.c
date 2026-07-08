from gguf import GGUFReader

# 加载你的 Qwen3.5 GGUF 文件
reader = GGUFReader("/home/zc/Work/model/Qwen3.5-0.8B-BF16.gguf")

# 1. 尝试直接获取封装好的特定元数据（适用于SSM架构部分）
# 不同的实现可能对应 ssm.d_inner 或以模型架构命名的 key
d_inner = None
for fields in reader.fields.values():
    if "d_inner" in fields.name:
        d_inner = fields.parts[-1][0]
        print(f"直接获取到 d_inner: {d_inner}")
        break

# 2. 如果未直接提供 d_inner，通过常规模型参数计算
if d_inner is None:
    # 查找模型标准隐藏层大小 (hidden_size)
    hidden_size = None
    for fields in reader.fields.values():
        if "embedding_length" in fields.name or "hidden_size" in fields.name:
            hidden_size = fields.parts[-1][0]
            break
            
    if hidden_size:
        # 在标准的 Mamba / 线性注意力实现中，d_inner 默认是 hidden_size 的 2 倍
        d_inner = 2 * hidden_size
        print(f"通过 hidden_size 隐式推导 d_inner: {d_inner}")
