"""Read token embeddings directly from GGUF v3 file for comparison with C output."""
import struct
import mmap
import sys

MODEL_PATH = "/home/zc/Work/model/qwen2.5-1.5b-instruct-q3_k_m.gguf"  # fill in

def read_gguf_string(data, offset):
    length = struct.unpack_from("<Q", data, offset)[0]
    offset += 8
    s = data[offset:offset + length].decode("utf-8", errors="replace")
    return s, offset + length

def f16_to_f32(bits):
    sign = (bits >> 15) & 1
    exp = (bits >> 10) & 0x1F
    mant = bits & 0x3FF
    if exp == 0:
        val = (mant / 1024.0) * (2.0 ** -14)
    elif exp == 31:
        val = float('nan') if mant != 0 else float('inf')
    else:
        val = (1.0 + mant / 1024.0) * (2.0 ** (exp - 15))
    return -val if sign else val

def main():
    with open(MODEL_PATH, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    magic = struct.unpack_from("<I", mm, 0)[0]
    if magic != 0x46554747:
        print(f"Bad magic: 0x{magic:08X}")
        return
    version = struct.unpack_from("<I", mm, 4)[0]
    n_tensor = struct.unpack_from("<Q", mm, 8)[0]
    n_kv = struct.unpack_from("<Q", mm, 16)[0]
    print(f"GGUF v{version}, n_tensor={n_tensor}, n_kv={n_kv}")

    off = 24
    alignment = 32

    # Parse KV pairs
    for _ in range(n_kv):
        key, off = read_gguf_string(mm, off)
        vtype = struct.unpack_from("<I", mm, off)[0]
        off += 4

        # GGUF v3 value types
        SZ = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 7:1, 10:8, 11:8, 12:8}
        if vtype in SZ:
            if key == "general.alignment" and vtype == 4:
                alignment = struct.unpack_from("<I", mm, off)[0]
            off += SZ[vtype]
        elif vtype == 8:  # STRING
            _, off = read_gguf_string(mm, off)
        elif vtype == 9:  # ARRAY
            atype = struct.unpack_from("<I", mm, off)[0]
            off += 4
            alen = struct.unpack_from("<Q", mm, off)[0]
            off += 8
            esz = SZ.get(atype, 4)
            if atype == 8:  # array of strings
                for _ in range(alen):
                    _, off = read_gguf_string(mm, off)
            else:
                off += alen * esz
        else:
            print(f"Unknown KV type {vtype} for key '{key}'")
            return

    print(f"Alignment: {alignment}")

    # Align to alignment boundary
    off = (off + alignment - 1) // alignment * alignment

    # Parse tensor infos
    tensor_infos = []
    for i in range(n_tensor):
        name, off = read_gguf_string(mm, off)
        ndim = struct.unpack_from("<I", mm, off)[0]
        off += 4
        dims = []
        n_el = 1
        for d in range(ndim):
            dim = struct.unpack_from("<Q", mm, off)[0]
            off += 8
            dims.append(dim)
            n_el *= dim
        ggml_type = struct.unpack_from("<I", mm, off)[0]
        off += 4
        tensor_offset = struct.unpack_from("<Q", mm, off)[0]
        off += 8
        tensor_infos.append((name, dims, ggml_type, tensor_offset, n_el))
        if i < 3 or name == "token_embd.weight":
            print(f"  tensor[{i}]: {name} dims={dims} type={ggml_type} offset={tensor_offset} n_el={n_el}")

    print()

    # Find token_embd tensor
    for name, dims, ggml_type, toff, n_el in tensor_infos:
        if name == "token_embd.weight":
            n_vocab, n_embd = dims[0], dims[1]
            print(f"token_embd: dims={dims}, type={ggml_type}, offset={toff}, n_vocab={n_vocab}, n_embd={n_embd}")

            tokens_to_check = {
                151644: "<|im_start|>",
                872: "user",
                198: "\\n",
                9693: "yes",
                151645: "<|im_end|>",
                77091: "assistant",
            }

            for tid, tname in tokens_to_check.items():
                print(f"\nembed[{tid}] ({tname}) first 10:")
                if ggml_type == 1:  # F16
                    eoff = toff + tid * n_embd * 2
                    for i in range(10):
                        bits = struct.unpack_from("<H", mm, eoff + i * 2)[0]
                        val = f16_to_f32(bits)
                        print(f"  [{i}] {val:.8e}")
                elif ggml_type == 0:  # F32
                    eoff = toff + tid * n_embd * 4
                    for i in range(10):
                        val = struct.unpack_from("<f", mm, eoff + i * 4)[0]
                        print(f"  [{i}] {val:.8e}")
            break

    mm.close()

if __name__ == "__main__":
    main()
