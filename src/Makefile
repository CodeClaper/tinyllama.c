CC = gcc
ARCH ?= $(shell uname -m)
UNAME_S := $(shell uname -s)
CFLAGS = -O3 -march=native -ffast-math -flto=auto -pthread
LDADDCFLAGS = -lm -flto=auto -pthread
ifdef DEBUG
	CFLAGS += -g -Wall -gdwarf-2 -g3 -D DEBUG
endif
ifdef DEBUG_VEC
	CFLAGS += -D DEBUG_VEC
endif
ifdef SAN
	CFLAGS += -fsanitize=address
	LDADDCFLAGS += -fsanitize=address
endif
ifneq (,$(filter x86_64 i386 i686,$(ARCH)))
	CFLAGS += -msse4.1
endif
# macOS: Apple clang rejects -march=native and -flto=auto.
ifeq ($(UNAME_S),Darwin)
	CC = clang
	CFLAGS := $(filter-out -march=native -flto=auto,$(CFLAGS))
	LDADDCFLAGS := $(filter-out -flto=auto,$(LDADDCFLAGS))
endif

# CUDA GPU backend (optional).  Build with: GPU=1 make
ifdef GPU
	NVCC ?= $(shell command -v nvcc 2>/dev/null)
	ifeq ($(NVCC),)
		$(error GPU=1 requested but nvcc was not found in PATH)
	endif
	CUDA_HOME ?= $(abspath $(NVCC)/../..)
	CUDA_LIB ?= $(if $(wildcard $(CUDA_HOME)/lib64),$(CUDA_HOME)/lib64,$(CUDA_HOME)/lib)
	NVCCFLAGS = -O3 -arch=native --std=c++17
	CFLAGS += -D GPU_BUILD
	GPU_OBJS = backend/gpu/gpu.o
	LDADDCFLAGS += -L$(CUDA_LIB) -lcudart -lstdc++
else
	GPU_OBJS =
endif

# Metal GPU backend (optional, macOS only).  Build with: METAL=1 make
ifdef METAL
	ifeq ($(UNAME_S),Linux)
		$(error METAL=1 requires macOS)
	endif
	ifdef GPU
		$(error GPU=1 and METAL=1 are mutually exclusive)
	endif
	CFLAGS += -D METAL_BUILD
	METAL_OBJS = backend/metal/metal.o
	LDADDCFLAGS += -framework Metal -framework Foundation
else
	METAL_OBJS =
endif

OBJS = server.o core.o quants.o mm.o slog.o utils.o el.o anet.o http.o graph.o \
       tokenizer.o sampler.o pthreads.o kvcache.o rax.o asctx.o linenoise.o \
       model/llama.o model/qwen25.o model/qwen35.o model/deepseek.o model/falcon.o \
       backend/cpu/cpu.o $(GPU_OBJS) $(METAL_OBJS)

MAIN_OBJS = core.o quants.o mm.o slog.o utils.o graph.o \
              tokenizer.o sampler.o pthreads.o kvcache.o rax.o asctx.o linenoise.o \
              model/llama.o model/qwen25.o model/qwen35.o model/deepseek.o model/falcon.o \
              backend/cpu/cpu.o $(GPU_OBJS) $(METAL_OBJS)

all: server chat bench

server: $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $@ $(LDADDCFLAGS)

chat: chat.o $(MAIN_OBJS)
	$(CC) $(CFLAGS) chat.o $(MAIN_OBJS) -o $@ $(LDADDCFLAGS)

bench: bench.o $(MAIN_OBJS)
	$(CC) $(CFLAGS) bench.o $(MAIN_OBJS) -o $@ $(LDADDCFLAGS)

server.o: server.c def.h el.h http.h
	$(CC) $(CFLAGS) -c server.c -o $@

chat.o: chat.c
	$(CC) $(CFLAGS) -c chat.c -o $@

bench.o: bench.c
	$(CC) $(CFLAGS) -c bench.c -o $@

http.o: http.c http.h
	$(CC) $(CFLAGS) -c http.c -o $@

core.o: core.c core.h def.h model/model.h
	$(CC) $(CFLAGS) -c core.c -o $@

quants.o: quants.c quants.h def.h
	$(CC) $(CFLAGS) -c quants.c -o $@

graph.o: graph.c graph.h def.h
	$(CC) $(CFLAGS) -c graph.c -o $@

mm.o: mm.c def.h mm.h asctx.h
	$(CC) $(CFLAGS) -c mm.c -o $@

slog.o: slog.c def.h slog.h
	$(CC) $(CFLAGS) -c slog.c -o $@

utils.o: utils.c def.h utils.h
	$(CC) $(CFLAGS) -c utils.c -o $@

el.o: el.c el.h
	$(CC) $(CFLAGS) -c el.c -o $@

anet.o: anet.c anet.h
	$(CC) $(CFLAGS) -c anet.c -o $@

tokenizer.o: tokenizer.c tokenizer.h core.h def.h utils.h
	$(CC) $(CFLAGS) -c tokenizer.c -o $@

sampler.o: sampler.c sampler.h def.h
	$(CC) $(CFLAGS) -c sampler.c -o $@

pthreads.o: pthreads.c pthreads.h slog.h
	$(CC) $(CFLAGS) -c pthreads.c -o $@

model/llama.o: model/llama.c model/model.h def.h
	$(CC) $(CFLAGS) -c model/llama.c -o $@

model/qwen25.o: model/qwen25.c model/model.h def.h
	$(CC) $(CFLAGS) -c model/qwen25.c -o $@

model/qwen35.o: model/qwen35.c model/model.h def.h
	$(CC) $(CFLAGS) -c model/qwen35.c -o $@

model/deepseek.o: model/deepseek.c model/model.h def.h
	$(CC) $(CFLAGS) -c model/deepseek.c -o $@

model/falcon.o: model/falcon.c model/model.h def.h
	$(CC) $(CFLAGS) -c model/falcon.c -o $@

backend/cpu/cpu.o: backend/cpu/cpu.c backend/cpu/cpu_arm.c backend/cpu/cpu_x86.c backend/cpu/cpu.h def.h quants.h
	$(CC) $(CFLAGS) -c backend/cpu/cpu.c -o $@

backend/gpu/gpu.o: backend/gpu/gpu.cu backend/gpu/gpu.h def.h
	$(NVCC) $(NVCCFLAGS) -c backend/gpu/gpu.cu -o $@

# metal_msl.h embeds the MSL kernel source as a C string so the
# host binary can compile the library at runtime (newLibraryWithSource);
# metal.metal stays the single source of truth.
backend/metal/metal_msl.h: backend/metal/metal.metal
	@echo "  GEN $@"
	@printf '/* Generated from metal.metal - do not edit. */\n#ifndef __METAL_MSL_H__\n#define __METAL_MSL_H__\nstatic const char *metal_msl =\n' > $@
	@sed -e 's/\\/\\\\/g' -e 's/"/\\"/g' -e 's/^/"/' -e 's/$$/\\n"/' backend/metal/metal.metal >> $@
	@printf ';\n#endif /* __METAL_MSL_H__ */\n' >> $@

backend/metal/metal.o: backend/metal/metal.m backend/metal/metal.h backend/metal/metal_msl.h def.h
	$(CC) $(CFLAGS) -fobjc-arc -c backend/metal/metal.m -o $@

kvcache.o: kvcache.c kvcache.h def.h
	$(CC) $(CFLAGS) -c kvcache.c -o $@

asctx.o: asctx.c asctx.h def.h
	$(CC) $(CFLAGS) -c asctx.c -o $@

rax.o: rax.c rax.h rax_malloc.h
	$(CC) $(CFLAGS) -c rax.c -o $@

linenoise.o: linenoise.c linenoise.h
	$(CC) $(CFLAGS) -c linenoise.c -o $@

.PHONY: clean all

clean:
	rm -f *.o model/*.o backend/cpu/*.o backend/gpu/*.o backend/metal/*.o backend/metal/metal_msl.h server bench rax.o
