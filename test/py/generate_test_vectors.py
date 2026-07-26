#!/usr/bin/env python3
"""
Generate GPT-2 byte-level encoding test vectors for qwen2_decode validation.

Usage:
  python3 generate_test_vectors.py              # print comparison report
  python3 generate_test_vectors.py --json       # output test vectors as JSON
  python3 generate_test_vectors.py --c          # output C arrays for test code
  python3 generate_test_vectors.py --verify     # cross-validate with HF tokenizers

Ground truth: the standard GPT-2 bytes_to_unicode() mapping from the original
OpenAI GPT-2 code, also used by HuggingFace tokenizers and tiktoken.

This script also compares the C code's simplified range-based mapping and
reports any discrepancies.  The C code's qwen2_decode / gpt2_encode_bytes
differs from the standard for 35 byte values in ranges 0x7F-0xA0, 0xAD.
"""

import json
import sys
from functools import lru_cache


# ============================================================================
# Standard GPT-2 byte <-> unicode mapping (original OpenAI implementation)
# Verified identical to HuggingFace tokenizers.ByteLevel decoder (256/256).
# ============================================================================

@lru_cache(maxsize=1)
def standard_bytes_to_unicode() -> dict:
    """Return dict[int, str]: raw byte value -> unicode character."""
    # Printable ASCII 0x21-0x7E ("!".."~") plus Latin-1 supplement chars
    # that encode to a single UTF-8 byte: U+00A1..U+00AC, U+00AE..U+00FF
    bs = (
        list(range(ord("!"), ord("~") + 1))        # 33..126
        + list(range(ord("¡"), ord("¬") + 1))       # 161..172
        + list(range(ord("®"), ord("ÿ") + 1))       # 174..255
    )
    cs = bs[:]  # these bytes map to chr(byte) directly

    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1

    return dict(zip(bs, [chr(c) for c in cs]))


@lru_cache(maxsize=1)
def standard_unicode_to_bytes() -> dict:
    """Inverse of bytes_to_unicode: unicode char -> raw byte."""
    return {v: k for k, v in standard_bytes_to_unicode().items()}


# ============================================================================
# Standard encode / decode
# ============================================================================

def standard_encode(raw: bytes) -> bytes:
    """Raw bytes -> GPT-2 byte-level encoded UTF-8 bytes."""
    b2u = standard_bytes_to_unicode()
    return "".join(b2u[b] for b in raw).encode("utf-8")


def standard_decode(encoded: bytes) -> bytes:
    """GPT-2 byte-level encoded UTF-8 bytes -> raw bytes."""
    u2b = standard_unicode_to_bytes()
    return bytes(u2b[ch] for ch in encoded.decode("utf-8"))


# ============================================================================
# C code's range-based encode / decode (for discrepancy reporting)
# ============================================================================

def c_style_encode(raw: bytes) -> bytes:
    """Raw bytes -> encoded using C code's range-based scheme."""
    out = bytearray()
    for b in raw:
        if 0x21 <= b <= 0x7E:
            out.append(b)
        elif b <= 0x3F:
            out.extend([0xC4, b + 0x80])
        elif b <= 0x7F:
            out.extend([0xC5, b + 0x40])
        elif b <= 0xBF:
            out.extend([0xC2, b])
        else:
            out.extend([0xC3, b - 0x40])
    return bytes(out)


def c_style_decode(encoded: bytes) -> bytes:
    """Encoded -> raw bytes using C code's range-based scheme."""
    out = bytearray()
    i = 0
    while i < len(encoded):
        c = encoded[i]
        if 0x21 <= c <= 0x7E:
            out.append(c)
            i += 1
        elif 0xC2 <= c <= 0xC5 and i + 1 < len(encoded):
            nc = encoded[i + 1]
            if c == 0xC2:
                out.append(nc)
            elif c == 0xC3:
                out.append(nc + 0x40)
            elif c == 0xC4:
                out.append(nc - 0x80)
            elif c == 0xC5:
                out.append(nc - 0x40)
            i += 2
        else:
            out.append(c)
            i += 1
    return bytes(out)


# ============================================================================
# Cross-validation with HuggingFace tokenizers (if available)
# ============================================================================

def verify_against_hf_tokenizers() -> dict:
    """Cross-validate standard mapping against HuggingFace ByteLevel decoder.

    Returns dict with validation results.  Only works when the `tokenizers`
    package is installed (pip install tokenizers).
    """
    try:
        from tokenizers import decoders as _decoders
    except ImportError:
        return {"available": False, "error": "tokenizers package not installed"}

    dec = _decoders.ByteLevel()
    b2u = standard_bytes_to_unicode()

    ok, fail = 0, 0
    failures = []
    for b in range(256):
        ch = b2u[b]
        result = dec.decode([ch])
        expected = bytes([b]).decode("utf-8", errors="replace")
        if result == expected:
            ok += 1
        else:
            fail += 1
            failures.append({"byte": b, "char": ch, "got": result, "expected": expected})

    return {
        "available": True,
        "match": ok,
        "mismatch": fail,
        "failures": failures,
        "ok": fail == 0,
    }


# ============================================================================
# C array generator
# ============================================================================

def format_c_array(name: str, data: bytes, indent: int = 8) -> str:
    """Format bytes as a C u8 array initializer."""
    prefix = " " * indent
    hex_bytes = [f"0x{b:02X}" for b in data]
    lines = []
    for i in range(0, len(hex_bytes), 12):
        lines.append(prefix + ", ".join(hex_bytes[i:i + 12]) + ",")
    body = "\n".join(lines)
    return f"static const u8 {name}[] = {{\n{body}\n}};"


def generate_c_vectors(strings: list[str]) -> str:
    """Generate C test arrays for all given strings."""
    parts = [
        "/*",
        " * AUTO-GENERATED by test/py/generate_test_vectors.py",
        " * Ground truth: standard GPT-2 bytes_to_unicode() mapping",
        " * Verified against HuggingFace tokenizers.ByteLevel decoder.",
        " *",
        " * To regenerate:  python3 test/py/generate_test_vectors.py --c",
        " */",
        "",
    ]

    for s in strings:
        utf8 = s.encode("utf-8")
        encoded = standard_encode(utf8)
        safe = "".join(c if c.isalnum() else "_" for c in s)[:40]

        parts.append(f"/* ---- {s!r} ---- */")
        parts.append(f"/* UTF-8: {utf8.hex(' ')} */")
        parts.append(f"/* encoded: {encoded.hex(' ')} */")
        parts.append(format_c_array(f"raw_{safe}", encoded))
        parts.append(f"/* expected decoded bytes (original UTF-8) */")
        parts.append(format_c_array(f"expected_{safe}", utf8))
        parts.append("")

    return "\n".join(parts)


# ============================================================================
# Report
# ============================================================================

TEST_STRINGS = [
    "!Aa~",
    " ",
    "Hello, World!",
    "你好",
    "こんにちは",
    "Hello World! こんにちは",
    "안녕하세요",
    "안녕! Hello?",
    "😀🎉",
    "Hello 😀 你好 こんにちは 안녕",
    "Hi !",
]


def main():
    mismatches = []
    b2u = standard_bytes_to_unicode()
    for b in range(256):
        std = b2u[b].encode("utf-8")
        cst = c_style_encode(bytes([b]))
        if std != cst:
            mismatches.append((b, std, cst, b2u[b]))

    # ---- Report ----
    print("=" * 72)
    print("GPT-2 Byte-Level Encoding: Standard vs C-code Comparison")
    print("=" * 72)

    if mismatches:
        print(f"\n  {len(mismatches)} / 256 bytes differ between standard GPT-2")
        print(f"  mapping and the C code's range-based approximation.\n")
        print(f"  {'Byte':>6}  {'Standard':>14}  {'C-style':>14}  Char")
        print(f"  {'----':>6}  {'--------':>14}  {'-------':>14}  ----")
        for b, std, cst, ch in mismatches:
            print(f"  0x{b:02X}    {std.hex():>14}  {cst.hex():>14}  {ch!r}")
    else:
        print("\n  All 256 bytes match.\n")

    # ---- Roundtrip ----
    print()
    print("=" * 72)
    print("Roundtrip (all 256 bytes)")
    print("=" * 72)

    all_bytes = bytes(range(256))
    std_enc = standard_encode(all_bytes)
    std_dec = standard_decode(std_enc)
    print(f"  Standard:     encode->decode {'OK' if std_dec == all_bytes else 'FAIL'}  "
          f"({len(std_enc)} enc bytes)")

    c_enc = c_style_encode(all_bytes)
    c_dec = c_style_decode(c_enc)
    print(f"  C-style:      encode->decode {'OK' if c_dec == all_bytes else 'FAIL'}  "
          f"({len(c_enc)} enc bytes)")

    cross_std_to_c = c_style_decode(std_enc)
    diff = sum(1 for a, b in zip(all_bytes, cross_std_to_c) if a != b)
    print(f"  Std-enc -> C-decode:     {'OK' if diff == 0 else f'FAIL ({diff} wrong)'}")

    try:
        cross_c_to_std = standard_decode(c_enc)
        diff2 = sum(1 for a, b in zip(all_bytes, cross_c_to_std) if a != b)
        print(f"  C-enc -> Std-decode:    {'OK' if diff2 == 0 else f'FAIL ({diff2} wrong)'}")
    except KeyError:
        print(f"  C-enc -> Std-decode:     FAIL (crash: C-style produces chars "
              f"outside standard mapping)")

    # ---- HF tokenizers verification ----
    print()
    print("=" * 72)
    print("Cross-validation: HuggingFace tokenizers.ByteLevel")
    print("=" * 72)
    result = verify_against_hf_tokenizers()
    if result["available"]:
        if result["ok"]:
            print(f"  Standard mapping == HF ByteLevel decoder: {result['match']}/{result['match'] + result['mismatch']}")
        else:
            print(f"  FAIL: {result['mismatch']} mismatches with HF ByteLevel")
            for f in result["failures"][:5]:
                print(f"    byte 0x{f['byte']:02X}: got {f['got']!r}, expected {f['expected']!r}")
    else:
        print(f"  SKIP: {result['error']}")

    # ---- Test vectors ----
    print()
    print("=" * 72)
    print("Test vectors (standard GPT-2 encoding)")
    print("=" * 72)

    for s in TEST_STRINGS:
        utf8 = s.encode("utf-8")
        enc = standard_encode(utf8)
        dec = standard_decode(enc)
        ok = "OK" if dec == utf8 else "FAIL"
        print(f"\n  {s!r}")
        print(f"    UTF-8:     {utf8.hex(' ')}")
        print(f"    encoded:   {enc.hex(' ')}  ({len(enc)} bytes)")
        print(f"    roundtrip: {ok}")

    # ---- Output formats ----
    if "--c" in sys.argv:
        print()
        print(generate_c_vectors(TEST_STRINGS))

    if "--json" in sys.argv:
        out = {
            "num_mismatches": len(mismatches),
            "mismatches": [
                {
                    "byte": b,
                    "byte_hex": f"0x{b:02X}",
                    "standard_hex": std.hex(),
                    "c_style_hex": cst.hex(),
                    "standard_char": ch,
                    "standard_codepoint": f"U+{ord(ch):04X}",
                }
                for b, std, cst, ch in mismatches
            ],
            "test_vectors": {},
        }
        for s in TEST_STRINGS:
            utf8 = s.encode("utf-8")
            enc = standard_encode(utf8)
            out["test_vectors"][s] = {
                "utf8_hex": utf8.hex(),
                "encoded_hex": enc.hex(),
                "encoded_bytes": list(enc),
            }
        print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
