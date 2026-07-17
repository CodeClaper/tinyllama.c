CC = gcc
ifdef DEBUG
	CFLAGS = -O0 -g -Wall -gdwarf-2 -g3 -D DEBUG
else
	CFLAGS = -O3 -march=native -ffast-math -flto
endif
LDADDCFLAGS = -lm -flto

OBJS = server.o core.o quants.o mm.o slog.o utils.o el.o anet.o http.o \
       model/llama.o model/qwen2.o model/deepseek.o model/falcon.o \
       cpu/quants_cpu.o

all: server

server: $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $@ $(LDADDCFLAGS)

server.o: server.c def.h el.h http.h
	$(CC) $(CFLAGS) -c server.c -o $@

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

model/llama.o: model/llama.c model/model.h def.h
	$(CC) $(CFLAGS) -c model/llama.c -o $@

model/qwen2.o: model/qwen2.c model/model.h def.h
	$(CC) $(CFLAGS) -c model/qwen2.c -o $@

model/deepseek.o: model/deepseek.c model/model.h def.h
	$(CC) $(CFLAGS) -c model/deepseek.c -o $@

model/falcon.o: model/falcon.c model/model.h def.h
	$(CC) $(CFLAGS) -c model/falcon.c -o $@

cpu/quants_cpu.o: cpu/quants_cpu.c cpu/quants_cpu.h def.h quants.h
	$(CC) $(CFLAGS) -c cpu/quants_cpu.c -o $@

.PHONY: clean all

clean:
	rm -f *.o model/*.o cpu/*.o server
