# Multi-turn Chat Design

## Overview

Enhance `src/chat.c` from single-shot interaction to an interactive multi-turn
chat loop.  The session's KV cache is preserved across turns so earlier
conversation context is re-used without re-prefilling.

## Architecture

### Main loop

```
load model → create session → set sampling params
print banner

first_turn = true
while (true):
    print "You > "
    read input (fgets, 4096-byte buffer)

    - EOF or empty read              → break
    - "/quit" or "/exit"             → break
    - "/clear"                       → session->ops.reset(session); first_turn = true; continue
    - empty line (after stripping)   → continue

    if first_turn:
        tokens = build_chat_tokens(v, input, sys_msg)   // full template
        forward(session, tokens, n_prompt, logits)       // prefill
        first_turn = false
    else:
        tokens = build_continuation_tokens(v, input)     // continuation markers
        forward(session, tokens, n_cont, logits)         // incremental forward

    print "Assistant > "
    generate + stream tokens until EOS or max_tokens
    print "\n"
```

### Token accumulation strategy

First turn builds the full Qwen chat template via `build_chat_tokens()`:

```
<|im_start|>system\n<sys_msg><|im_end|>\n          (optional)
<|im_start|>user\n<user_msg><|im_end|>\n
<|im_start|>assistant\n
```

Subsequent turns append only the delta tokens via the new
`build_continuation_tokens()`:

```
<|im_end|>\n<|im_start|>user\n<user_msg><|im_end|>\n<|im_start|>assistant\n
```

This closes the previous assistant block, opens a new user block with the
message, then opens a new assistant block ready for generation.  The KV cache
from all prior tokens is preserved — no recomputation of history.

### New function: `build_continuation_tokens()`

Declared in `src/tokenizer.h`, implemented in `src/tokenizer.c`.
Same signature shape as `build_chat_tokens()` but only takes a user message
(no system prompt) and produces the continuation token sequence.

```
int build_continuation_tokens(Vocab *v, const char *user_msg,
                               u32 *tokens, int max_tokens);
```

Internally pre-looks up `<|im_end|>`, `<|im_start|>`, newline byte token, and
"user"/"assistant" text, then tokenizes the user message via BPE.

### Special commands

- `/quit`, `/exit` — exit the chat loop cleanly
- `/clear` — reset the session (KV cache + token ring buffer) and restart from
  first-turn state.  The system prompt (if any) is re-applied on the next user
  message.
- Empty input lines are silently skipped.

### Input handling

- Single-line via `fgets` with a 4096-byte buffer.
- Trailing newline stripped before processing.
- EOF (Ctrl+D) triggers clean exit.

### Output

- Tokens streamed character-by-character as they are generated.
- `"Assistant > "` prefix before generation, trailing newline after EOS/max.

### Sampling

- Same as current: greedy when temperature is 0, temperature + top-p otherwise.
- `max_tokens` defaults to session's `max_tokens` if set, else 256.

## Files changed

| File | Change |
|------|--------|
| `src/chat.c` | Main loop, continuation branch, command handling, banner |
| `src/tokenizer.h` | Declare `build_continuation_tokens()` |
| `src/tokenizer.c` | Implement `build_continuation_tokens()` |

## Non-goals

- Multi-line user input
- `/system` to change system prompt mid-session
- Saving/loading chat history
- Readline-style line editing
