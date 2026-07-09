import gguf
import torch

# 1. 加载GGUF文件
reader = gguf.GGUFReader("/home/zc/Work/model/Qwen3.5-0.8B-BF16.gguf")  # 替换为你的模型路径

# 2. 按名称查找 'attn_q' 张量
# 在GGUF中，张量名称格式为 "blk.0.attn_q.weight"，其中0是层索引[reference:3]
target_tensor_name = "blk.3.attn_q.weight"  # 示例：获取第0层的attn_q
tensor = None
for t in reader.tensors:
    if t.name == target_tensor_name:
        tensor = t
        break

if tensor is None:
    raise ValueError(f"Tensor '{target_tensor_name}' not found in the model.")

# 3. (重要) 检查并反量化张量
# GGUF中的张量可能是量化格式（如Q4_0, Q8_0），需要先反量化为numpy数组[reference:4]
if tensor.tensor_type != gguf.GGMLQuantizationType.F32:
    # tensor.data是原始字节，tensor.tensor_type指明其量化类型
    np_array = gguf.dequantize(tensor.data, tensor.tensor_type)
else:
    # 如果已经是FP32，可以直接读取
    np_array = tensor.data

# 4. 转换为PyTorch张量 (可选)
torch_tensor = torch.from_numpy(np_array)

# 现在可以使用 torch_tensor 了
print(f"成功获取张量 '{target_tensor_name}'，形状为: {torch_tensor.shape}")
