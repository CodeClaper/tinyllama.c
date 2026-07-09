import sys
import struct
import numpy as np
from gguf import GGUFReader
from gguf.constants import GGMLQuantizationType

ELEM_SIZE = {
    GGMLQuantizationType.F32:  4,
    GGMLQuantizationType.F16:  2,
    GGMLQuantizationType.BF16: 2,
}

MODEL_PATH = "/home/zc/Work/model/Qwen3.5-0.8B-BF16.gguf"
TOKEN_ID   = int(sys.argv[1]) if len(sys.argv) > 1 else 0
N_DIMS     = int(sys.argv[2]) if len(sys.argv) > 2 else 16


def decode_f16(u16_val):
    """Convert F16 (IEEE 754 half) to float32."""
    sign = (u16_val >> 15) & 1
    exp  = (u16_val >> 10) & 0x1f
    mant = u16_val & 0x3ff
    if exp == 0:
        if mant == 0:
            f32_bits = sign << 31
        else:
            e = 1 - 15
            while (mant & 0x400) == 0:
                mant <<= 1
                e -= 1
            mant &= 0x3ff
            f32_bits = (sign << 31) | ((e + 127) << 23) | (mant << 13)
    elif exp == 31:
        f32_bits = (sign << 31) | (0xff << 23) | (mant << 13)
    else:
        f32_bits = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13)
    return struct.unpack('<f', struct.pack('<I', f32_bits))[0]


def decode_bf16(u16_val):
    """Convert BF16 (brain float) to float32."""
    return struct.unpack('<f', struct.pack('<I', u16_val << 16))[0]


def main():
    reader = GGUFReader(MODEL_PATH)

    te = None
    for t in reader.tensors:
        if t.name == "token_embd.weight":
            te = t
            break

    if te is None:
        print("ERROR: token_embd.weight tensor not found")
        sys.exit(1)

    tt = te.tensor_type
    n_bytes = te.n_bytes
    shape   = te.shape
    elem_sz = ELEM_SIZE.get(tt)

    n_embd  = int(shape[0])
    n_vocab = int(shape[1]) if len(shape) >= 2 else 1

    print(f"Tensor: {te.name}")
    print(f"  type    = {tt.name}, shape = {list(shape)}")
    print(f"  n_embd  = {n_embd}, n_vocab = {n_vocab}")
    print(f"  token   = {TOKEN_ID}")

    if TOKEN_ID >= n_vocab:
        print(f"ERROR: token {TOKEN_ID} >= n_vocab {n_vocab}")
        sys.exit(1)

    raw = bytearray(te.data)

    # [n_embd, n_vocab] layout: token j is at row_start + j
    stride = n_vocab

    if tt == GGMLQuantizationType.F32:
        data = np.frombuffer(raw, dtype=np.uint32)
        vals = np.frombuffer(raw, dtype=np.float32)[TOKEN_ID::stride][:n_embd]
    elif tt == GGMLQuantizationType.F16:
        data = np.frombuffer(raw, dtype=np.uint16)
        indices = np.arange(TOKEN_ID, len(data), stride, dtype=np.uint64)[:n_embd]
        vals = np.array([decode_f16(int(data[i])) for i in indices], dtype=np.float32)
    elif tt == GGMLQuantizationType.BF16:
        data = np.frombuffer(raw, dtype=np.uint16)
        indices = np.arange(TOKEN_ID, len(data), stride, dtype=np.uint64)[:n_embd]
        vals = np.array([decode_bf16(int(data[i])) for i in indices], dtype=np.float32)
    else:
        print(f"ERROR: unsupported type {tt.name}")
        sys.exit(1)

    print(f"\nFirst {N_DIMS} embedding dims for token {TOKEN_ID}:")
    for i in range(min(N_DIMS, len(vals))):
        print(f"  dim[{i:4d}] = {vals[i]: .10e}")

    print(f"\nStats over all {n_embd} dims:")
    print(f"  min  = {vals.min():.10e}")
    print(f"  max  = {vals.max():.10e}")
    print(f"  mean = {vals.mean():.10e}")
    print(f"  std  = {vals.std():.10e}")


if __name__ == "__main__":
    main()
