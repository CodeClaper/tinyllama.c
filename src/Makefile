CC = gcc
CFLAGS = -O1
LDADDCFLAGS = -lm

all: server

server: server.o server.o mm.o slog.o utils.o
	$(CC) $(CFLAGS) server.o mm.o slog.o utils.o -o $@ $(LDADDCFLAGS)

server.o: server.c def.h
	$(CC) $(CFLAGS) -c server.c -o $@

mm.o: mm.c def.h mm.h
	$(CC) $(CFLAGS) -c mm.c -o $@

slog.o: slog.c def.h slog.h
	$(CC) $(CFLAGS) -c slog.c -o $@

utils.o: utils.c def.h utils.h
	$(CC) $(CFLAGS) -c utils.c -o $@

.PHONY: clean all

clean:
	rm -f *.o tllm 
