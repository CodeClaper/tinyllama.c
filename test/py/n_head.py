from gguf import GGUFReader

# 1. 加载 GGUF 文件
reader = GGUFReader("/home/zc/Work/model/Qwen3.5-0.8B-BF16.gguf")

# 3. 拼接出注意力头的 Key 并读取
head_count_key = f"qwen35.attention.head_count"
head_count_field = reader.get_field(head_count_key)

if head_count_field:
    # 获取具体的头数数值
    n_head = head_count_field.parts[head_count_field.data[0]][0]
    print(f"n_head ({head_count_key}): {n_head}")
else:
    print(f"未找到元数据键: {head_count_key}")
