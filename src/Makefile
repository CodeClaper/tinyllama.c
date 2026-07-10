CC = gcc
ifdef DEBUG
	CFLAGS = -O0 -g -Wall -gdwarf-2 -g3 -D DEBUG
else
	CFLAGS = -O3 -march=native -ffast-math
endif
LDADDCFLAGS = -lm

OBJS = server.o core.o mm.o slog.o utils.o el.o anet.o http.o \
       arch/llama.o arch/qwen2.o arch/deepseek.o arch/falcon.o

all: server

server: $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $@ $(LDADDCFLAGS)

server.o: server.c def.h el.h http.h
	$(CC) $(CFLAGS) -c server.c -o $@

http.o: http.c http.h
	$(CC) $(CFLAGS) -c http.c -o $@

core.o: core.c core.h def.h arch/arch.h
	$(CC) $(CFLAGS) -c core.c -o $@

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

arch/llama.o: arch/llama.c arch/arch.h def.h
	$(CC) $(CFLAGS) -c arch/llama.c -o $@

arch/qwen2.o: arch/qwen2.c arch/arch.h def.h
	$(CC) $(CFLAGS) -c arch/qwen2.c -o $@

arch/deepseek.o: arch/deepseek.c arch/arch.h def.h
	$(CC) $(CFLAGS) -c arch/deepseek.c -o $@

arch/falcon.o: arch/falcon.c arch/arch.h def.h
	$(CC) $(CFLAGS) -c arch/falcon.c -o $@

.PHONY: clean all

clean:
	rm -f *.o arch/*.o server
