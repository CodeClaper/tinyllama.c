CC      = gcc
ARCH    ?= $(shell uname -m)
CFLAGS  = -O0 -g -Wall -pthread -Ic
ifneq (,$(filter x86_64 i386 i686,$(ARCH)))
	CFLAGS += -msse4.1
endif
LDFLAGS = -lpthread
SRC_DIR  = ../src

all: test_pthreads test_utils test_qwen25

test_pthreads: c/test_pthreads.c pthreads.o slog.o utils.o mm.o mctx.o asctx.o bit.o
	$(CC) $(CFLAGS) -I$(SRC_DIR) $^ -o $@ $(LDFLAGS)

pthreads.o: $(SRC_DIR)/pthreads.c $(SRC_DIR)/pthreads.h $(SRC_DIR)/slog.h
	$(CC) $(CFLAGS) -I$(SRC_DIR) -c $(SRC_DIR)/pthreads.c -o $@

slog.o: $(SRC_DIR)/slog.c $(SRC_DIR)/slog.h
	$(CC) $(CFLAGS) -I$(SRC_DIR) -c $(SRC_DIR)/slog.c -o $@

utils.o: $(SRC_DIR)/utils.c $(SRC_DIR)/utils.h $(SRC_DIR)/def.h
	$(CC) $(CFLAGS) -I$(SRC_DIR) -c $(SRC_DIR)/utils.c -o $@

mm.o: $(SRC_DIR)/mm.c $(SRC_DIR)/mm.h $(SRC_DIR)/def.h
	$(CC) $(CFLAGS) -I$(SRC_DIR) -c $(SRC_DIR)/mm.c -o $@

mctx.o: $(SRC_DIR)/mctx.c $(SRC_DIR)/mctx.h $(SRC_DIR)/asctx.h $(SRC_DIR)/def.h
	$(CC) $(CFLAGS) -I$(SRC_DIR) -c $(SRC_DIR)/mctx.c -o $@

asctx.o: $(SRC_DIR)/asctx.c $(SRC_DIR)/asctx.h $(SRC_DIR)/mctx.h $(SRC_DIR)/bit.h $(SRC_DIR)/def.h
	$(CC) $(CFLAGS) -I$(SRC_DIR) -c $(SRC_DIR)/asctx.c -o $@

bit.o: $(SRC_DIR)/bit.c $(SRC_DIR)/bit.h $(SRC_DIR)/def.h
	$(CC) $(CFLAGS) -I$(SRC_DIR) -c $(SRC_DIR)/bit.c -o $@

test_utils: c/test_utils.c utils.o slog.o mm.o mctx.o asctx.o bit.o
	$(CC) $(CFLAGS) -I$(SRC_DIR) $^ -o $@ $(LDFLAGS) -lm

qwen25.o: $(SRC_DIR)/model/qwen25.c $(SRC_DIR)/model/model.h $(SRC_DIR)/def.h
	$(CC) $(CFLAGS) -I$(SRC_DIR) -c $(SRC_DIR)/model/qwen25.c -o $@

qwen35.o: $(SRC_DIR)/model/qwen35.c $(SRC_DIR)/model/model.h $(SRC_DIR)/def.h
	$(CC) $(CFLAGS) -I$(SRC_DIR) -c $(SRC_DIR)/model/qwen35.c -o $@

core.o: $(SRC_DIR)/core.c $(SRC_DIR)/core.h $(SRC_DIR)/def.h $(SRC_DIR)/model/model.h
	$(CC) $(CFLAGS) -I$(SRC_DIR) -c $(SRC_DIR)/core.c -o $@

quants.o: $(SRC_DIR)/quants.c $(SRC_DIR)/quants.h $(SRC_DIR)/def.h
	$(CC) $(CFLAGS) -I$(SRC_DIR) -c $(SRC_DIR)/quants.c -o $@

quants_cpu.o: $(SRC_DIR)/cpu/quants_cpu.c $(SRC_DIR)/cpu/quants_cpu.h $(SRC_DIR)/def.h $(SRC_DIR)/quants.h
	$(CC) $(CFLAGS) -I$(SRC_DIR) -c $(SRC_DIR)/cpu/quants_cpu.c -o $@

llama.o: $(SRC_DIR)/model/llama.c $(SRC_DIR)/model/model.h $(SRC_DIR)/def.h
	$(CC) $(CFLAGS) -I$(SRC_DIR) -c $(SRC_DIR)/model/llama.c -o $@

deepseek.o: $(SRC_DIR)/model/deepseek.c $(SRC_DIR)/model/model.h $(SRC_DIR)/def.h
	$(CC) $(CFLAGS) -I$(SRC_DIR) -c $(SRC_DIR)/model/deepseek.c -o $@

falcon.o: $(SRC_DIR)/model/falcon.c $(SRC_DIR)/model/model.h $(SRC_DIR)/def.h
	$(CC) $(CFLAGS) -I$(SRC_DIR) -c $(SRC_DIR)/model/falcon.c -o $@

test_qwen25: c/test_qwen25.c qwen25.o qwen35.o llama.o deepseek.o falcon.o core.o quants.o quants_cpu.o pthreads.o mm.o mctx.o asctx.o bit.o slog.o utils.o
	$(CC) $(CFLAGS) -I$(SRC_DIR) $^ -o $@ $(LDFLAGS) -lm

.PHONY: all check clean

check: test_pthreads test_utils test_qwen25
	./test_pthreads
	./test_utils
	./test_qwen25

clean:
	rm -f test_pthreads test_qwen25 *.o
