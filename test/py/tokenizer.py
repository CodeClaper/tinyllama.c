from llama_cpp import Llama

# 1. 加载模型（请替换为你的实际 GGUF 模型路径）
llm = Llama(model_path="/home/zc/Work/model/Qwen3.5-0.8B-BF16.gguf", vocab_only=True) # 如果只为了tokenize，可以加上 vocab_only=True 节省内存

# 2. 准备需要切分的文本
text = b"<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n" # 注意：部分版本接收 bytes，部分接收 str。建议先用 bytes

# 3. 获取 Token ID 列表
# add_bos 参数决定是否在开头自动加上 <s> (BOS) 标记
token_ids = llm.tokenize(text, add_bos=True, special=True)

# 4. 打印 Token IDs 结果
print("--- Token IDs ---")
print(token_ids)

# 5. 如果你想把每个 ID 还原并打印出对应的文本片段（Piece）：
print("\n--- Token Pieces ---")
for tid in token_ids:
    piece = llm.detokenize([tid])
    # 解码为字符串并打印
    print(f"ID: {tid:<6} -> Piece: {piece.decode('utf-8', errors='ignore')}")
