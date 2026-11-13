#ifndef __TOOL_H__
#define __TOOL_H__

#include "def.h"

/* Qwen3.5 XML tool calls (<tool_call><function=...>). */
extern const ToolOps qwen35_tool_ops;

/* Qwen2.x JSON / Hermes-style tool calls. */
extern const ToolOps qwen25_tool_ops;

/* Llama 3.x `<|python_tag|>` function calls. */
extern const ToolOps llama3_tool_ops;

#endif
