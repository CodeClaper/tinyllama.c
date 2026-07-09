import gguf
import torch

def get_all_head_q_vectors(gguf_path, layer_idx=0):
    reader = gguf.GGUFReader(gguf_path)
    
    # 1. 使用修正后的方法获取头数量
    n_heads = None
    for key, field in reader.fields.items():
        if "head_count" in key:
            n_heads = int(field.parts[-1])
            break
    if n_heads is None:
        raise KeyError("无法找到注意力头数量")
    
    # 2. 加载指定层的 Q 投影权重
    tensor_name = f"blk.{layer_idx}.attn_q.weight"
    tensor = None
    for t in reader.tensors:
        if t.name == tensor_name:
            tensor = t
            break
    if tensor is None:
        raise ValueError(f"Tensor '{tensor_name}' not found.")
    
    # 3. 反量化
    if tensor.tensor_type != gguf.GGMLQuantizationType.F32:
        np_array = gguf.dequantize(tensor.data, tensor.tensor_type)
    else:
        np_array = tensor.data
    q_weight = torch.from_numpy(np_array)
    
    # 4. 切分
    n_embd = q_weight.shape[0]
    head_dim = n_embd // n_heads
    head_q_vectors = torch.split(q_weight, head_dim, dim=-1)
    
    return list(head_q_vectors), n_heads, head_dim, n_embd

# 使用示例
gguf_path = "/home/zc/Work/model/Qwen3.5-0.8B-BF16.gguf"
head_vectors, n_heads, head_dim, n_embd = get_all_head_q_vectors(gguf_path, layer_idx=3)
print(f"n_heads: {n_heads}, head_dim: {head_dim}, n_embd: {n_embd}, head_vectors: {head_vectors}")
