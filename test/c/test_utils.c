#include <stdio.h>
#include <string.h>
#include <math.h>
#include "minunit.h"
#include "../../src/utils.h"
#include "../../src/mm.h"

/* next_pow2 */

MU_TEST(test_next_pow2_one) {
    mu_assert_int_eq(1, next_pow2(1));
}

MU_TEST(test_next_pow2_two) {
    mu_assert_int_eq(2, next_pow2(2));
}

MU_TEST(test_next_pow2_three) {
    mu_assert_int_eq(4, next_pow2(3));
}

MU_TEST(test_next_pow2_large) {
    mu_assert_int_eq(1024, next_pow2(1023));
    mu_assert_int_eq(1024, next_pow2(1024));
    mu_assert_int_eq(2048, next_pow2(1025));
}

MU_TEST(test_next_pow2_zero) {
    mu_assert_int_eq(1, next_pow2(0));
}

/* hash_bytes */

MU_TEST(test_hash_bytes_empty) {
    u64 h = hash_bytes(NULL, 0);
    mu_check(h == 1469598103934665603ull);
}

MU_TEST(test_hash_bytes_consistent) {
    char data[] = "hello";
    u64 h1 = hash_bytes(data, 5);
    u64 h2 = hash_bytes(data, 5);
    mu_assert_int_eq(h1, h2);
}

MU_TEST(test_hash_bytes_different) {
    char a[] = "hello";
    char b[] = "world";
    u64 ha = hash_bytes(a, 5);
    u64 hb = hash_bytes(b, 5);
    mu_check(ha != hb);
}

/* parse_int */

MU_TEST(test_parse_int_zero) {
    mu_assert_int_eq(0, parse_int("0"));
}

MU_TEST(test_parse_int_positive) {
    mu_assert_int_eq(42, parse_int("42"));
}

MU_TEST(test_parse_int_negative) {
    mu_assert_int_eq(-7, parse_int("-7"));
}

/* parse_float */

MU_TEST(test_parse_float_zero) {
    mu_assert_double_eq(0.0, parse_float("0"));
}

MU_TEST(test_parse_float_basic) {
    float v = parse_float("3.14");
    mu_check(fabs(v - 3.14f) < DEFAULT_EPS);
}

MU_TEST(test_parse_float_negative) {
    float v = parse_float("-2.5");
    mu_check(fabs(v - (-2.5f)) < DEFAULT_EPS);
}

/* parse_long */

MU_TEST(test_parse_long_zero) {
    mu_assert_int_eq(0, parse_long("0"));
}

MU_TEST(test_parse_long_positive) {
    mu_assert_int_eq(1234567890L, parse_long("1234567890"));
}

MU_TEST(test_parse_long_negative) {
    mu_assert_int_eq(-987654321L, parse_long("-987654321"));
}

/* get_key_name */

MU_TEST(test_get_key_name_simple) {
    Key k = {3, "foo"};
    char *name = get_key_name(k);
    mu_assert_string_eq("foo", name);
    sfree(name);
}

MU_TEST(test_get_key_name_empty) {
    Key k = {0, ""};
    char *name = get_key_name(k);
    mu_assert_string_eq("", name);
    sfree(name);
}

/* key_streq */

MU_TEST(test_key_streq_match) {
    Key k = {5, "hello"};
    mu_check(key_streq(k, "hello"));
}

MU_TEST(test_key_streq_mismatch) {
    Key k = {5, "hello"};
    mu_check(!key_streq(k, "world"));
}

MU_TEST(test_key_streq_different_len) {
    Key k = {5, "hello"};
    mu_check(!key_streq(k, "hello!"));
}

/* key_strcontains */

MU_TEST(test_key_strcontains_found) {
    Key k = {19, "blk.0.attn_q.weight"};
    mu_check(key_strcontains(k, "attn"));
}

MU_TEST(test_key_strcontains_not_found) {
    Key k = {10, "test.value"};
    mu_check(!key_strcontains(k, "xyz"));
}

MU_TEST(test_key_strcontains_exact) {
    Key k = {19, "blk.0.attn_q.weight"};
    mu_check(key_strcontains(k, "blk.0.attn_q.weight"));
}

/* key_eq */

MU_TEST(test_key_eq_equal) {
    Key a = {5, "hello"};
    Key b = {5, "hello"};
    mu_check(key_eq(a, b));
}

MU_TEST(test_key_eq_different_len) {
    Key a = {5, "hello"};
    Key b = {3, "hel"};
    mu_check(!key_eq(a, b));
}

MU_TEST(test_key_eq_different_content) {
    Key a = {5, "hello"};
    Key b = {5, "world"};
    mu_check(!key_eq(a, b));
}

/* get_datetime */

MU_TEST(test_get_datetime_second_format) {
    char *res = get_datetime(SECOND);
    mu_check(strlen(res) == 19);
    mu_check(res[4] == '-');
    mu_check(res[7] == '-');
    mu_check(res[10] == ' ');
    mu_check(res[13] == ':');
    mu_check(res[16] == ':');
    sfree(res);
}

MU_TEST(test_get_datetime_millisecond_format) {
    char *res = get_datetime(MILLISECOND);
    mu_check(strlen(res) == 23);
    mu_check(res[19] == '.');
    sfree(res);
}

MU_TEST(test_get_datetime_microsecond_format) {
    char *res = get_datetime(MICROSECOND);
    mu_check(strlen(res) == 26);
    mu_check(res[19] == '.');
    sfree(res);
}

/* size_convert */

MU_TEST(test_size_convert_kb) {
    mu_assert_double_eq(1.0, size_convert(1024, KB));
}

MU_TEST(test_size_convert_mb) {
    mu_assert_double_eq(1.0, size_convert(1024 * 1024, MB));
}

MU_TEST(test_size_convert_gb) {
    mu_assert_double_eq(1.0, size_convert(1024 * 1024 * 1024, GB));
}

MU_TEST(test_size_convert_zero) {
    mu_assert_double_eq(0.0, size_convert(0, KB));
    mu_assert_double_eq(0.0, size_convert(0, MB));
    mu_assert_double_eq(0.0, size_convert(0, GB));
}

MU_TEST(test_size_convert_fractional) {
    mu_assert_double_eq(0.5, size_convert(512, KB));
}

int main(void) {
    printf("utils tests\n");
    printf("-----------\n");

    MU_RUN_TEST(test_next_pow2_one);
    MU_RUN_TEST(test_next_pow2_two);
    MU_RUN_TEST(test_next_pow2_three);
    MU_RUN_TEST(test_next_pow2_large);
    MU_RUN_TEST(test_next_pow2_zero);

    MU_RUN_TEST(test_hash_bytes_empty);
    MU_RUN_TEST(test_hash_bytes_consistent);
    MU_RUN_TEST(test_hash_bytes_different);

    MU_RUN_TEST(test_parse_int_zero);
    MU_RUN_TEST(test_parse_int_positive);
    MU_RUN_TEST(test_parse_int_negative);

    MU_RUN_TEST(test_parse_float_zero);
    MU_RUN_TEST(test_parse_float_basic);
    MU_RUN_TEST(test_parse_float_negative);

    MU_RUN_TEST(test_parse_long_zero);
    MU_RUN_TEST(test_parse_long_positive);
    MU_RUN_TEST(test_parse_long_negative);

    MU_RUN_TEST(test_get_key_name_simple);
    MU_RUN_TEST(test_get_key_name_empty);

    MU_RUN_TEST(test_key_streq_match);
    MU_RUN_TEST(test_key_streq_mismatch);
    MU_RUN_TEST(test_key_streq_different_len);

    MU_RUN_TEST(test_key_strcontains_found);
    MU_RUN_TEST(test_key_strcontains_not_found);
    MU_RUN_TEST(test_key_strcontains_exact);

    MU_RUN_TEST(test_key_eq_equal);
    MU_RUN_TEST(test_key_eq_different_len);
    MU_RUN_TEST(test_key_eq_different_content);

    MU_RUN_TEST(test_get_datetime_second_format);
    MU_RUN_TEST(test_get_datetime_millisecond_format);
    MU_RUN_TEST(test_get_datetime_microsecond_format);

    MU_RUN_TEST(test_size_convert_kb);
    MU_RUN_TEST(test_size_convert_mb);
    MU_RUN_TEST(test_size_convert_gb);
    MU_RUN_TEST(test_size_convert_zero);
    MU_RUN_TEST(test_size_convert_fractional);

    MU_REPORT();
    return MU_EXIT_CODE;
}
