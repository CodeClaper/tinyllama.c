#include <stdio.h>
#include <string.h>
#include "minunit.h"
#include "../../src/model/model.h"

/* Helper: call qwen2_ops.decode and null-terminate, return the decoded length. */
static int decode(const u8 *raw, int raw_len, char *out, int max_len) {
    return qwen2_ops.decode(raw, raw_len, out, max_len);
}

/* ================================================================
 * Printable ASCII (0x21-0x7E): direct passthrough
 * ================================================================ */

MU_TEST(test_decode_ascii_basic) {
    u8 raw[] = {0x21, 0x41, 0x61, 0x7E}; /* "!Aa~" */
    char out[64] = {0};
    int n = decode(raw, 4, out, sizeof(out));
    mu_assert_int_eq(4, n);
    mu_assert_string_eq("!Aa~", out);
}

MU_TEST(test_decode_ascii_full_range) {
    /* Every byte in 0x21..0x7E should pass through unchanged. */
    u8 raw[94];
    char expected[95];
    for (int i = 0; i < 94; i++) {
        raw[i] = (u8)(0x21 + i);
        expected[i] = (char)raw[i];
    }
    expected[94] = '\0';
    char out[128] = {0};
    int n = decode(raw, 94, out, sizeof(out));
    mu_assert_int_eq(94, n);
    mu_assert_string_eq(expected, out);
}

/* ================================================================
 * C2 prefix → bytes 0x80..0xBF
 * ================================================================ */

MU_TEST(test_decode_c2_range) {
    /* 0xC2 0x80 → 0x80,  0xC2 0xBF → 0xBF */
    u8 raw[] = {0xC2, 0x80, 0xC2, 0xBF};
    char out[64] = {0};
    int n = decode(raw, 4, out, sizeof(out));
    mu_assert_int_eq(2, n);
    mu_assert_int_eq(0x80, (u8)out[0]);
    mu_assert_int_eq(0xBF, (u8)out[1]);
}

/* ================================================================
 * C3 prefix → bytes 0xC0..0xFF
 * ================================================================ */

MU_TEST(test_decode_c3_range) {
    /* 0xC3 0x80 → 0xC0,  0xC3 0xBF → 0xFF */
    u8 raw[] = {0xC3, 0x80, 0xC3, 0xBF};
    char out[64] = {0};
    int n = decode(raw, 4, out, sizeof(out));
    mu_assert_int_eq(2, n);
    mu_assert_int_eq(0xC0, (u8)out[0]);
    mu_assert_int_eq(0xFF, (u8)out[1]);
}

/* ================================================================
 * C4 prefix → bytes 0x00..0x3F
 * ================================================================ */

MU_TEST(test_decode_c4_range) {
    /* 0xC4 0x80 → 0x00,  0xC4 0xBF → 0x3F */
    u8 raw[] = {0xC4, 0x80, 0xC4, 0xBF};
    char out[64] = {0};
    int n = decode(raw, 4, out, sizeof(out));
    mu_assert_int_eq(2, n);
    mu_assert_int_eq(0x00, (u8)out[0]);
    mu_assert_int_eq(0x3F, (u8)out[1]);
}

/* ================================================================
 * C5 prefix → bytes 0x40..0x7F
 * ================================================================ */

MU_TEST(test_decode_c5_range) {
    /* 0xC5 0x80 → 0x40,  0xC5 0xBF → 0x7F */
    u8 raw[] = {0xC5, 0x80, 0xC5, 0xBF};
    char out[64] = {0};
    int n = decode(raw, 4, out, sizeof(out));
    mu_assert_int_eq(2, n);
    mu_assert_int_eq(0x40, (u8)out[0]);
    mu_assert_int_eq(0x7F, (u8)out[1]);
}

/* ================================================================
 * Special byte: space (0x20) → 0xC4 0xA0
 * ================================================================ */

MU_TEST(test_decode_space) {
    u8 raw[] = {0xC4, 0xA0}; /* encoded space */
    char out[64] = {0};
    int n = decode(raw, 2, out, sizeof(out));
    mu_assert_int_eq(1, n);
    mu_assert_int_eq(' ', out[0]);
}

/* ================================================================
 * Chinese text roundtrip (UTF-8 → GPT2-encode → decode → UTF-8)
 * "你好" = E4 BD A0 E5 A5 BD
 * E4→C3 A4  BD→C2 BD  A0→C2 A0  E5→C3 A5  A5→C2 A5  BD→C2 BD
 * ================================================================ */

MU_TEST(test_decode_chinese) {
    /* GPT-2 encoded bytes for "你好" */
    u8 raw[] = {0xC3, 0xA4, 0xC2, 0xBD, 0xC2, 0xA0,
                0xC3, 0xA5, 0xC2, 0xA5, 0xC2, 0xBD};
    char out[64] = {0};
    int n = decode(raw, 12, out, sizeof(out));
    mu_assert_int_eq(6, n);
    /* Verify raw UTF-8 bytes for "你好" */
    u8 expected[] = {0xE4, 0xBD, 0xA0, 0xE5, 0xA5, 0xBD};
    mu_assert_int_eq(6, n);
    for (int i = 0; i < 6; i++)
        mu_assert_int_eq(expected[i], (u8)out[i]);
}

/* ================================================================
 * Mixed content: printable ASCII + encoded non-printable + C2/C3/C4/C5
 * ================================================================ */

MU_TEST(test_decode_mixed) {
    /* "Hi" + space(encoded) + "!" + byte 0x80(encoded) + byte 0xFF(encoded) */
    u8 raw[] = {
        'H', 'i',                   /* 0x48, 0x69 → direct */
        0xC4, 0xA0,                 /* space → 0x20 */
        '!',                        /* 0x21 → direct */
        0xC2, 0x80,                 /* → 0x80 */
        0xC3, 0xBF,                 /* → 0xFF */
    };
    char out[64] = {0};
    int n = decode(raw, 9, out, sizeof(out));
    mu_assert_int_eq(6, n);
    mu_assert_int_eq('H',  out[0]);
    mu_assert_int_eq('i',  out[1]);
    mu_assert_int_eq(' ',  out[2]);
    mu_assert_int_eq('!',  out[3]);
    mu_assert_int_eq(0x80, (u8)out[4]);
    mu_assert_int_eq(0xFF, (u8)out[5]);
}

/* ================================================================
 * Edge case: empty input
 * ================================================================ */

MU_TEST(test_decode_empty) {
    char out[64] = {0};
    int n = decode(NULL, 0, out, sizeof(out));
    mu_assert_int_eq(0, n);
}

/* ================================================================
 * Edge case: single-byte C2-C5 at end of input (no next byte)
 * ================================================================ */

MU_TEST(test_decode_truncated_c2) {
    u8 raw[] = {0xC2}; /* missing next byte → passthrough */
    char out[64] = {0};
    int n = decode(raw, 1, out, sizeof(out));
    mu_assert_int_eq(1, n);
    mu_assert_int_eq(0xC2, (u8)out[0]);
}

MU_TEST(test_decode_truncated_c3) {
    u8 raw[] = {0xC3};
    char out[64] = {0};
    int n = decode(raw, 1, out, sizeof(out));
    mu_assert_int_eq(1, n);
    mu_assert_int_eq(0xC3, (u8)out[0]);
}

MU_TEST(test_decode_truncated_c4) {
    u8 raw[] = {0xC4};
    char out[64] = {0};
    int n = decode(raw, 1, out, sizeof(out));
    mu_assert_int_eq(1, n);
    mu_assert_int_eq(0xC4, (u8)out[0]);
}

MU_TEST(test_decode_truncated_c5) {
    u8 raw[] = {0xC5};
    char out[64] = {0};
    int n = decode(raw, 1, out, sizeof(out));
    mu_assert_int_eq(1, n);
    mu_assert_int_eq(0xC5, (u8)out[0]);
}

/* ================================================================
 * Edge case: max_len boundary — output truncated
 * ================================================================ */

MU_TEST(test_decode_max_len_boundary) {
    u8 raw[] = {0x21, 0x22, 0x23, 0x24}; /* 4 printable ASCII */
    char out[3] = {0};
    int n = decode(raw, 4, out, 3);
    mu_assert_int_eq(3, n); /* only 3 bytes fit before hitting max_len */
    mu_assert_int_eq(0x21, (u8)out[0]);
    mu_assert_int_eq(0x22, (u8)out[1]);
    mu_assert_int_eq(0x23, (u8)out[2]);
}

MU_TEST(test_decode_max_len_zero) {
    u8 raw[] = {0x21};
    char out[64] = {0};
    int n = decode(raw, 1, out, 0);
    mu_assert_int_eq(0, n);
}

/* ================================================================
 * Edge case: C2-C5 with out-of-range next byte → passthrough
 * ================================================================ */

MU_TEST(test_decode_c2_unusual_next) {
    /* 0xC2 0x20: the function does NOT validate that nc is in [0x80,0xBF].
     * byte = (int)0x20 = 32 (space), which passes the >=0 && <=255 check.
     * So 1 byte output: 0x20 (space), and the second input byte is consumed. */
    u8 raw[] = {0xC2, 0x20};
    char out[64] = {0};
    int n = decode(raw, 2, out, sizeof(out));
    mu_assert_int_eq(1, n);
    mu_assert_int_eq(' ', out[0]);
}

/* ================================================================
 * Edge case: bytes outside known ranges (0x00-0x20, 0x7F, 0xC0-C1, >0xC5)
 * should all pass through as-is
 * ================================================================ */

MU_TEST(test_decode_unmapped_bytes) {
    /* Bytes 0x00, 0x1F, 0x7F, 0xC0, 0xC1, 0xC6 are all pass-through */
    u8 raw[] = {0x00, 0x1F, 0x7F, 0xC0, 0xC1, 0xC6};
    char out[64] = {0};
    int n = decode(raw, 6, out, sizeof(out));
    mu_assert_int_eq(6, n);
    for (int i = 0; i < 6; i++)
        mu_assert_int_eq(raw[i], (u8)out[i]);
}

/* ================================================================
 * Full byte coverage: verify all 256 bytes roundtrip correctly
 * ================================================================ */

MU_TEST(test_decode_full_byte_coverage) {
    /* Build input that encodes every byte 0x00..0xFF via the GPT-2 scheme,
     * decode it, and verify we get back the original bytes. */
    u8 encoded[512];
    u8 expected[256];
    int in_len = 0;
    int out_idx = 0;

    for (int b = 0; b < 256; b++) {
        expected[out_idx++] = (u8)b;
        if (b >= 0x21 && b <= 0x7E) {
            encoded[in_len++] = (u8)b;
        } else if (b >= 0x00 && b <= 0x3F) {
            encoded[in_len++] = 0xC4;
            encoded[in_len++] = (u8)(b + 0x80);
        } else if (b >= 0x40 && b <= 0x7F) {
            encoded[in_len++] = 0xC5;
            encoded[in_len++] = (u8)(b + 0x40);
        } else if (b >= 0x80 && b <= 0xBF) {
            encoded[in_len++] = 0xC2;
            encoded[in_len++] = (u8)b;
        } else { /* 0xC0..0xFF */
            encoded[in_len++] = 0xC3;
            encoded[in_len++] = (u8)(b - 0x40);
        }
    }

    char out[512] = {0};
    int n = decode(encoded, in_len, out, sizeof(out));
    mu_assert_int_eq(256, n);
    for (int i = 0; i < 256; i++)
        mu_assert_int_eq(expected[i], (u8)out[i]);
}

/* ================================================================
 * Edge case: all printable "hello world" as single segment
 * ================================================================ */

MU_TEST(test_decode_hello_world) {
    const char *msg = "Hello, World!";
    int len = (int)strlen(msg);
    char out[64] = {0};
    int n = decode((const u8 *)msg, len, out, sizeof(out));
    mu_assert_int_eq(len, n);
    mu_assert_string_eq(msg, out);
}

/* ================================================================
 * Test suite
 * ================================================================ */

MU_TEST_SUITE(test_decode_ascii) {
    MU_RUN_TEST(test_decode_ascii_basic);
    MU_RUN_TEST(test_decode_ascii_full_range);
    MU_RUN_TEST(test_decode_hello_world);
}

MU_TEST_SUITE(test_decode_multibyte) {
    MU_RUN_TEST(test_decode_c2_range);
    MU_RUN_TEST(test_decode_c3_range);
    MU_RUN_TEST(test_decode_c4_range);
    MU_RUN_TEST(test_decode_c5_range);
    MU_RUN_TEST(test_decode_space);
}

MU_TEST_SUITE(test_decode_complex) {
    MU_RUN_TEST(test_decode_chinese);
    MU_RUN_TEST(test_decode_mixed);
    MU_RUN_TEST(test_decode_full_byte_coverage);
}

MU_TEST_SUITE(test_decode_edge_cases) {
    MU_RUN_TEST(test_decode_empty);
    MU_RUN_TEST(test_decode_truncated_c2);
    MU_RUN_TEST(test_decode_truncated_c3);
    MU_RUN_TEST(test_decode_truncated_c4);
    MU_RUN_TEST(test_decode_truncated_c5);
    MU_RUN_TEST(test_decode_max_len_boundary);
    MU_RUN_TEST(test_decode_max_len_zero);
    MU_RUN_TEST(test_decode_c2_unusual_next);
    MU_RUN_TEST(test_decode_unmapped_bytes);
}

/* ================================================================
 * Main
 * ================================================================ */

int main(void) {
    printf("qwen2 decode tests\n");
    printf("------------------\n");

    printf("\n--- ASCII ---\n");
    MU_RUN_SUITE(test_decode_ascii);

    printf("\n--- Multi-byte ranges ---\n");
    MU_RUN_SUITE(test_decode_multibyte);

    printf("\n--- Complex content ---\n");
    MU_RUN_SUITE(test_decode_complex);

    printf("\n--- Edge cases ---\n");
    MU_RUN_SUITE(test_decode_edge_cases);

    MU_REPORT();
    return MU_EXIT_CODE;
}
