#include "def.h"

Model *model_load(const char *path);
Engine *engine_load(EngineOptons *opts);
void engine_close(Engine *en);
void engine_summary(Engine *en);
