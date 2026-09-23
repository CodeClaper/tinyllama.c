#ifndef __TOKENIZER_H__
#define __TOKENIZER_H__

#include "def.h"

/* Chat prompt builders, referenced from each arch's ArchOps.  A family
 * shares one pair: ChatML (Qwen) and Gemma (<start_of_turn>). */
int chatml_prompt(Session *s, const char *user_msg, const char *sys_msg, u32 *tokens, int max_tokens);
int chatml_continuation(Session *s, const char *user_msg, u32 *tokens, int max_tokens);
int gemma_chat_prompt(Session *s, const char *user_msg, const char *sys_msg, u32 *tokens, int max_tokens);
int gemma_chat_continuation(Session *s, const char *user_msg, u32 *tokens, int max_tokens);

#endif
