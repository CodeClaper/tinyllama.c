#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include "def.h"
#include "core.h"
#include "utils.h"
#include "tokenizer.h"

/* GPT-2 style pre-tokenizer character classification.
 * 1=letter (including non-ASCII UTF-8 bytes), 2=digit,
 * 3=whitespace, 4=other (punctuation/symbol) */
static int char_category(int c) {
    unsigned char uc = (unsigned char)c;
    if (uc >= 0x80)              return 1;
    if (isalpha(uc))             return 1;
    if (isdigit(uc))             return 2;
    if (isspace(uc))             return 3;
    return 4;
}

/* Try to match a special token (format: <|...|>) at text[pos].
 * Returns token ID on success, VOCAB_ID_NONE on failure.
 * On success, *matched_len is set to bytes consumed. */
static i32 match_special_token(Vocab *v, const char *text, int text_len, int *matched_len) {
    if (text_len < 3) return (i32)VOCAB_ID_NONE;
    if (text[0] != '<' || text[1] != '|') return (i32)VOCAB_ID_NONE;

    int max_try = text_len < 64 ? text_len : 64;
    int best_len = 0;
    i32 best_id  = (i32)VOCAB_ID_NONE;
    for (int len = 2; len <= max_try; len++) {
        i32 id;
        if (vocab_lookup_len(v, text, len, &id)) {
            best_len = len;
            best_id  = id;
        }
    }
    if (best_len > 0) {
        *matched_len = best_len;
        return best_id;
    }
    return (i32)VOCAB_ID_NONE;
}

/* Byte-level encode a text chunk and apply BPE merges within it.
 * Returns the number of tokens produced. */
static int bpe_encode_chunk(Vocab *v, const char *chunk, int chunk_len,
                            u32 *out, int max_out) {
    u32 syms[512];
    int n = 0;

    for (int i = 0; i < chunk_len && n < (int)(sizeof(syms)/sizeof(syms[0])); i++) {
        i32 id = v->byte_token_ids[(unsigned char)chunk[i]];
        if (id >= 0)
            syms[n++] = (u32)id;
    }

    while (n > 1) {
        i32 best_rank = INT32_MAX;
        int best_idx  = -1;
        for (int i = 0; i < n - 1; i++) {
            i32 rank = vocab_merge_rank(v, (i32)syms[i], (i32)syms[i + 1]);
            if (rank >= 0 && rank < best_rank) {
                best_rank = rank;
                best_idx  = i;
            }
        }
        if (best_idx < 0) break;
        i32 merged = vocab_merge_result(v, (i32)syms[best_idx], (i32)syms[best_idx + 1]);
        if (merged == (i32)VOCAB_ID_NONE) break;
        syms[best_idx] = (u32)merged;
        memmove(&syms[best_idx + 1], &syms[best_idx + 2],
                (size_t)(n - best_idx - 2) * sizeof(u32));
        n--;
    }

    if (n > max_out) n = max_out;
    memcpy(out, syms, (size_t)n * sizeof(u32));
    return n;
}

/* BPE tokenizer for GPT-2 style vocabularies (Qwen2, LLaMA, etc.).
 * 1. Detects special tokens (<|...|>) by direct vocab lookup.
 * 2. Applies GPT-2 regex pre-tokenization to split text into chunks.
 * 3. Byte-encodes and BPE-merges within each chunk independently. */
int tokenize_bpe(Vocab *v, const char *text, int text_len, u32 *tokens, int max_tokens) {
    #define BPE_MAX_SYMBOLS 4096
    u32 symbols[BPE_MAX_SYMBOLS];
    int n = 0, pos = 0;

    while (pos < text_len && n < BPE_MAX_SYMBOLS) {
        /* Step 0: try to match a special token (<|...|>) first. */
        int special_len = 0;
        i32 special_id = match_special_token(v, text + pos, text_len - pos, &special_len);
        if (special_id >= 0) {
            symbols[n++] = (u32)special_id;
            pos += special_len;
            continue;
        }

        /* Step 1: locate the next pre-tokenizer chunk. */

        /* Contractions: 's, 't, 're, 've, 'm, 'll, 'd */
        if (text[pos] == '\'' && pos + 1 < text_len) {
            char nxt = text[pos + 1];
            int clen = 0;
            if (nxt == 's' || nxt == 't' || nxt == 'm' || nxt == 'd')
                clen = 2;
            else if ((nxt == 'r' || nxt == 'v') && pos + 2 < text_len &&
                     text[pos + 2] == 'e')
                clen = 3;
            else if (nxt == 'l' && pos + 2 < text_len && text[pos + 2] == 'l')
                clen = 3;
            if (clen > 0) {
                int added = bpe_encode_chunk(v, text + pos, clen,
                                             symbols + n, BPE_MAX_SYMBOLS - n);
                n += added; pos += clen;
                continue;
            }
        }

        /* Whitespace: single leading space is absorbed by the next chunk
         * (GPT-2 regex semantics); multi-space runs form their own chunk. */
        if (char_category(text[pos]) == 3) {
            int ws_start = pos;
            int ws_len = 0;
            while (pos < text_len && char_category(text[pos]) == 3) {
                pos++; ws_len++;
            }
            if (ws_len != 1 || pos >= text_len) {
                int added = bpe_encode_chunk(v, text + ws_start, ws_len,
                                             symbols + n, BPE_MAX_SYMBOLS - n);
                n += added;
                continue;
            }
            /* Single space before non-whitespace: let next chunk absorb it. */
            pos = ws_start;
        }

        /* Non-whitespace chunk (letter, digit, or other). */
        int chunk_start = pos;
        if (pos < text_len && char_category(text[pos]) == 3)
            pos++;                    /* absorb optional leading space */
        if (pos >= text_len) continue;

        int cat = char_category(text[pos]);
        if (cat == 1) {
            while (pos < text_len && char_category(text[pos]) == 1) pos++;
        } else if (cat == 2) {
            while (pos < text_len && char_category(text[pos]) == 2) pos++;
        } else {
            while (pos < text_len && char_category(text[pos]) == 4) pos++;
        }

        if (pos == chunk_start) continue;

        int added = bpe_encode_chunk(v, text + chunk_start, pos - chunk_start,
                                     symbols + n, BPE_MAX_SYMBOLS - n);
        n += added;
    }

    if (n > max_tokens) n = max_tokens;
    memcpy(tokens, symbols, (size_t)n * sizeof(u32));
    return n;
    #undef BPE_MAX_SYMBOLS
}

/* Chat template (Qwen-style) — build token ID array directly.
 * Pre-looks up format tokens from the vocabulary; tokenizes user/system
 * message content via BPE. Returns total number of prompt tokens. */
int build_chat_tokens(Vocab *v, const char *user_msg, const char *sys_msg,
                      u32 *tokens, int max_tokens) {
    int n = 0;

    /* Pre-lookup format tokens. */
    i32 im_start_id = v->im_start_id;
    i32 im_end_id   = v->im_end_id;
    if (im_start_id == (i32)VOCAB_ID_NONE) {
        im_start_id = vocab_lookup(v, "<|im_start|>");
        if (im_start_id == (i32)VOCAB_ID_NONE) return 0;
    }
    if (im_end_id == (i32)VOCAB_ID_NONE) {
        im_end_id = vocab_lookup(v, "<|im_end|>");
        if (im_end_id == (i32)VOCAB_ID_NONE) return 0;
    }
    i32 nl_id = v->byte_token_ids[(unsigned char)'\n'];

    #define ADD(id) do { \
        if (n < max_tokens) tokens[n++] = (u32)(id); \
    } while (0)
    #define ADD_TEXT(txt) do { \
        int tlen = (int)strlen(txt); \
        int added = tokenize_bpe(v, txt, tlen, tokens + n, max_tokens - n); \
        n += added; \
    } while (0)

    if (sys_msg && sys_msg[0]) {
        ADD(im_start_id);
        ADD_TEXT("system");
        ADD(nl_id);
        ADD_TEXT(sys_msg);
        ADD(im_end_id);
        ADD(nl_id);
    }
    ADD(im_start_id);
    ADD_TEXT("user");
    ADD(nl_id);
    ADD_TEXT(user_msg);
    ADD(im_end_id);
    ADD(nl_id);
    ADD(im_start_id);
    ADD_TEXT("assistant");
    ADD(nl_id);

    #undef ADD
    #undef ADD_TEXT
    return n;
}
