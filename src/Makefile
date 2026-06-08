CC = gcc
ifdef DEBUG
	CFLAGS = -O0 -g -Wall -gdwarf-2 -g3 -D DEBUG
else
	CFLAGS = -O3
endif
LDADDCFLAGS = -lm

all: server

server: server.o server.o core.o mm.o slog.o utils.o
	$(CC) $(CFLAGS) server.o core.o mm.o slog.o utils.o -o $@ $(LDADDCFLAGS)

server.o: server.c def.h
	$(CC) $(CFLAGS) -c server.c -o $@

core.o: core.c core.h def.h
	$(CC) $(CFLAGS) -c core.c -o $@

mm.o: mm.c def.h mm.h
	$(CC) $(CFLAGS) -c mm.c -o $@

slog.o: slog.c def.h slog.h
	$(CC) $(CFLAGS) -c slog.c -o $@

utils.o: utils.c def.h utils.h
	$(CC) $(CFLAGS) -c utils.c -o $@

.PHONY: clean all

clean:
	rm -f *.o tllm 
