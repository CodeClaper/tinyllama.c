CC = gcc
ARCH ?= $(shell uname -m)
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
	GPU_OBJS = gpu/quants_gpu.o
	LDADDCFLAGS += -L$(CUDA_LIB) -lcudart -lstdc++
else
	GPU_OBJS =
endif

OBJS = server.o core.o quants.o mm.o slog.o utils.o el.o anet.o http.o \
       tokenizer.o sampler.o pthreads.o kvcache.o rax.o \
       model/llama.o model/qwen25.o model/qwen35.o model/deepseek.o model/falcon.o \
       cpu/quants_cpu.o $(GPU_OBJS)

MAIN_OBJS = core.o quants.o mm.o slog.o utils.o \
              tokenizer.o sampler.o pthreads.o kvcache.o rax.o \
              model/llama.o model/qwen25.o model/qwen35.o model/deepseek.o model/falcon.o \
              cpu/quants_cpu.o $(GPU_OBJS)

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

mm.o: mm.c def.h mm.h
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

cpu/quants_cpu.o: cpu/quants_cpu.c cpu/quants_arm.c cpu/quants_x86.c cpu/quants_cpu.h def.h quants.h
	$(CC) $(CFLAGS) -c cpu/quants_cpu.c -o $@

gpu/quants_gpu.o: gpu/quants_gpu.cu gpu/quants_gpu.h def.h
	$(NVCC) $(NVCCFLAGS) -c gpu/quants_gpu.cu -o $@

kvcache.o: kvcache.c kvcache.h def.h
	$(CC) $(CFLAGS) -c kvcache.c -o $@

rax.o: rax.c rax.h
	$(CC) $(CFLAGS) -c rax.c -o $@

.PHONY: clean all

clean:
	rm -f *.o model/*.o cpu/*.o gpu/*.o server bench rax.o
