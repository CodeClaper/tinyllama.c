#ifndef __TOKENIZER_H__
#define __TOKENIZER_H__

#include "def.h"

int tokenize_bpe(Vocab *v, const char *text, int text_len, u32 *tokens, int max_tokens);
int build_chat_tokens(Vocab *v, const char *user_msg, const char *sys_msg,
                      u32 *tokens, int max_tokens);
int build_continuation_tokens(Vocab *v, const char *user_msg,
                               u32 *tokens, int max_tokens);

#endif
