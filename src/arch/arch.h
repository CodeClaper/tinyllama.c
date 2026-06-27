#ifndef __ARCH_H__
#define __ARCH_H__

#include "../def.h"

extern const ArchOps llama_ops;
extern const ArchOps qwen2_ops;
extern const ArchOps deepseek_ops;
extern const ArchOps falcon_ops;

void arch_config_init(Engine *en, ArchConfig *cfg);
const char *arch_key_prefix(ModelArch arch);

#endif
