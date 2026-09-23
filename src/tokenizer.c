#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include "def.h"
#include "core.h"
#include "mm.h"
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
static int tokenize_bpe(Vocab *v, const char *text, int text_len, u32 *tokens, int max_tokens) {
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

/* Number of bytes in the UTF-8 sequence starting at lead byte c. */
static int utf8_char_len(unsigned char c) {
    if (c < 0x80) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;
}

/* SentencePiece (unigram) encoder.
 *
 * Mirrors llama.cpp's llm_tokenizer_spm_session: the text is escaped
 * (spaces -> U+2581), split into UTF-8 characters, then adjacent
 * characters are greedily merged into the highest-scoring known piece
 * until no merge is possible.  Symbols that never form a piece fall
 * back to per-byte "<0xHH>" tokens.  add_prefix_space prepends the
 * SentencePiece word-boundary marker, as when text follows a special
 * token or starts a prompt. */
static int tokenize_spm(Vocab *v, const char *text, int text_len, u32 *tokens,
                 int max_tokens, bool add_prefix_space) {
    if (!v->scores)   /* no unigram scores: best effort via byte fallback */
        return tokenize_bpe(v, text, text_len, tokens, max_tokens);

    /* Normalize: optional word-boundary marker, then ' ' -> U+2581. */
    char *norm = smalloc((size_t)text_len * 3 + 3);
    if (!norm) return 0;
    int nn = 0;
    if (add_prefix_space) {
        norm[nn++] = (char)0xE2; norm[nn++] = (char)0x96; norm[nn++] = (char)0x81;
    }
    for (int i = 0; i < text_len; i++) {
        if (text[i] == ' ') {
            norm[nn++] = (char)0xE2; norm[nn++] = (char)0x96; norm[nn++] = (char)0x81;
        } else {
            norm[nn++] = text[i];
        }
    }

    /* Split into UTF-8 characters, linked into a doubly-linked list. */
    typedef struct { const char *text; int n, prev, next; } SpmSym;
    SpmSym *sym = smalloc((size_t)(nn + 1) * sizeof(SpmSym));
    if (!sym) { sfree(norm); return 0; }
    int nsym = 0;
    for (int off = 0; off < nn; ) {
        int cl = utf8_char_len((unsigned char)norm[off]);
        if (off + cl > nn) cl = 1;
        sym[nsym].text = norm + off;
        sym[nsym].n    = cl;
        sym[nsym].prev = nsym - 1;
        sym[nsym].next = nsym + 1;
        nsym++;
        off += cl;
    }
    if (nsym == 0) { sfree(sym); sfree(norm); return 0; }
    sym[nsym - 1].next = -1;

    /* Greedy merge: repeatedly join the adjacent pair whose concatenation
     * is a known piece with the highest score (leftmost on ties). */
    for (;;) {
        int best = -1;
        float best_score = 0.0f;
        for (int i = 0; i >= 0 && sym[i].next >= 0; i = sym[i].next) {
            int r   = sym[i].next;
            int len = sym[i].n + sym[r].n;
            if (len > 255) continue;            /* Key.len is u8 */
            i32 id;
            if (!vocab_lookup_len(v, sym[i].text, len, &id)) continue;
            if ((u32)id >= v->n_vocab) continue;
            float sc = v->scores[id];
            if (best < 0 || sc > best_score ||
                (sc == best_score && i < best)) {
                best = i;
                best_score = sc;
            }
        }
        if (best < 0) break;
        int r = sym[best].next;
        sym[best].n += sym[r].n;
        sym[r].n = 0;
        sym[best].next = sym[r].next;
        if (sym[r].next >= 0) sym[sym[r].next].prev = best;
    }

    /* Emit: known pieces as-is, everything else byte by byte. */
    int n = 0;
    for (int i = 0; i >= 0; i = sym[i].next) {
        if (sym[i].n == 0) continue;
        i32 id;
        if (vocab_lookup_len(v, sym[i].text, sym[i].n, &id)) {
            if (n < max_tokens) tokens[n++] = (u32)id;
        } else {
            for (int j = 0; j < sym[i].n; j++) {
                i32 bid = v->byte_token_ids[(unsigned char)sym[i].text[j]];
                if (bid >= 0 && n < max_tokens) tokens[n++] = (u32)bid;
            }
        }
    }
    sfree(sym);
    sfree(norm);
    return n;
}

/* Generic text encoder: SentencePiece for SPM vocabs, BPE otherwise.
 * Starts a fresh fragment, so SPM text gets the leading word-boundary
 * marker (matching llama.cpp's initial is_prev_special == true). */
static int tokenize(Vocab *v, const char *text, int text_len, u32 *tokens, int max_tokens) {
    if (v->tokenizer_type == TOKENIZER_TYPE_SPM)
        return tokenize_spm(v, text, text_len, tokens, max_tokens, true);
    return tokenize_bpe(v, text, text_len, tokens, max_tokens);
}

/* ChatML prompt template, shared by the Qwen family (and any other
 * arch whose ops table points here).  Lays out the first turn:
 *   [<|im_start|>system\n{sys}<|im_end|>\n]
 *   <|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n
 * Returns 0 when the vocabulary lacks the ChatML markers. */
int chatml_prompt(Session *s, const char *user_msg, const char *sys_msg,
                  u32 *tokens, int max_tokens) {
    Vocab *v = s->en->vocab;
    int n = 0;

    /* Pre-lookup format tokens. */
    i32 im_start_id = v->im_start_id;
    i32 im_end_id   = v->im_end_id;
    if (im_start_id == (i32)VOCAB_ID_NONE)
        im_start_id = vocab_lookup(v, "<|im_start|>");
    if (im_end_id == (i32)VOCAB_ID_NONE)
        im_end_id = vocab_lookup(v, "<|im_end|>");
    if (im_start_id == (i32)VOCAB_ID_NONE || im_end_id == (i32)VOCAB_ID_NONE)
        return 0;
    i32 nl_id = v->byte_token_ids[(unsigned char)'\n'];

    #define ADD(id) do { \
        if (n < max_tokens) tokens[n++] = (u32)(id); \
    } while (0)
    #define ADD_TEXT(txt) do { \
        if (n < max_tokens) { \
            int tlen = (int)strlen(txt); \
            n += tokenize(v, txt, tlen, tokens + n, max_tokens - n); \
        } \
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

/* ChatML continuation: closes the previous assistant block and opens a
 * new user→assistant round:
 *   <|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n */
int chatml_continuation(Session *s, const char *user_msg,
                        u32 *tokens, int max_tokens) {
    Vocab *v = s->en->vocab;
    int n = 0;

    i32 im_start_id = v->im_start_id;
    i32 im_end_id   = v->im_end_id;
    if (im_start_id == (i32)VOCAB_ID_NONE)
        im_start_id = vocab_lookup(v, "<|im_start|>");
    if (im_end_id == (i32)VOCAB_ID_NONE)
        im_end_id = vocab_lookup(v, "<|im_end|>");
    if (im_start_id == (i32)VOCAB_ID_NONE || im_end_id == (i32)VOCAB_ID_NONE)
        return 0;
    i32 nl_id = v->byte_token_ids[(unsigned char)'\n'];

    #define ADD(id) do { \
        if (n < max_tokens) tokens[n++] = (u32)(id); \
    } while (0)
    #define ADD_TEXT(txt) do { \
        if (n < max_tokens) { \
            int tlen = (int)strlen(txt); \
            n += tokenize(v, txt, tlen, tokens + n, max_tokens - n); \
        } \
    } while (0)

    ADD(im_end_id);
    ADD(nl_id);
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

/* Encode one Gemma template fragment.  SPM text that follows a special
 * token gets the word-boundary marker, matching llama.cpp. */
static int gemma_text(Vocab *v, const char *txt, u32 *out, int max_out, bool prefix_space) {
    int len = (int)strlen(txt);
    if (v->tokenizer_type == TOKENIZER_TYPE_SPM)
        return tokenize_spm(v, txt, len, out, max_out, prefix_space);
    return tokenize_bpe(v, txt, len, out, max_out);
}

/* Gemma chat template.  Gemma has no dedicated system role, so the
 * system prompt is merged into the first user turn:
 *   <bos><start_of_turn>user\n{sys}\n\n{user}<end_of_turn>\n
 *        <start_of_turn>model\n
 * Returns 0 when the vocabulary lacks the turn markers. */
int gemma_chat_prompt(Session *s, const char *user_msg, const char *sys_msg,
                       u32 *tokens, int max_tokens) {
    Vocab *v = s->en->vocab;
    i32 st = v->start_of_turn_id;
    i32 et = v->end_of_turn_id;
    if (st == (i32)VOCAB_ID_NONE || et == (i32)VOCAB_ID_NONE) return 0;
    int n = 0;

    #define ADD(id) do { \
        if (n < max_tokens) tokens[n++] = (u32)(id); \
    } while (0)
    #define ADD_TEXT(txt, sp) do { \
        if (n < max_tokens) \
            n += gemma_text(v, txt, tokens + n, max_tokens - n, sp); \
    } while (0)

    if (v->bos_id != (i32)VOCAB_ID_NONE) ADD(v->bos_id);
    ADD(st);
    ADD_TEXT("user", true);      /* word-boundary marker after <start_of_turn> */
    ADD_TEXT("\n", false);
    if (sys_msg && sys_msg[0]) {
        ADD_TEXT(sys_msg, false);
        ADD_TEXT("\n\n", false);
    }
    ADD_TEXT(user_msg, false);
    ADD(et);
    ADD_TEXT("\n", false);
    ADD(st);
    ADD_TEXT("model", true);
    ADD_TEXT("\n", false);

    #undef ADD
    #undef ADD_TEXT
    return n;
}

/* Gemma continuation: close the previous model turn and open the next
 * user/model round:
 *   <end_of_turn>\n<start_of_turn>user\n{user}<end_of_turn>\n
 *                  <start_of_turn>model\n */
int gemma_chat_continuation(Session *s, const char *user_msg,
                             u32 *tokens, int max_tokens) {
    Vocab *v = s->en->vocab;
    i32 st = v->start_of_turn_id;
    i32 et = v->end_of_turn_id;
    if (st == (i32)VOCAB_ID_NONE || et == (i32)VOCAB_ID_NONE) return 0;
    int n = 0;

    #define ADD(id) do { \
        if (n < max_tokens) tokens[n++] = (u32)(id); \
    } while (0)
    #define ADD_TEXT(txt, sp) do { \
        if (n < max_tokens) \
            n += gemma_text(v, txt, tokens + n, max_tokens - n, sp); \
    } while (0)

    ADD(et);
    ADD_TEXT("\n", false);
    ADD(st);
    ADD_TEXT("user", true);
    ADD_TEXT("\n", false);
    ADD_TEXT(user_msg, false);
    ADD(et);
    ADD_TEXT("\n", false);
    ADD(st);
    ADD_TEXT("model", true);
    ADD_TEXT("\n", false);

    #undef ADD
    #undef ADD_TEXT
    return n;
}
