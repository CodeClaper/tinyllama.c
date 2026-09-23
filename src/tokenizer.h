#ifndef __TOKENIZER_H__
#define __TOKENIZER_H__

#include "def.h"

int chatml_prompt(Session *s, const char *user_msg, const char *sys_msg, u32 *tokens, int max_tokens);
int chatml_continuation(Session *s, const char *user_msg, u32 *tokens, int max_tokens);
int gemma3_chat_prompt(Session *s, const char *user_msg, const char *sys_msg, u32 *tokens, int max_tokens);
int gemma3_chat_continuation(Session *s, const char *user_msg, u32 *tokens, int max_tokens);

#endif
