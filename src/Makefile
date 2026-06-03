CC = gcc
CFLAGS = -O1
LDADDCFLAGS = -lm

all: run

run: run.o run.o mm.o slog.o utils.o
	$(CC) $(CFLAGS) run.o mm.o slog.o utils.o -o $@ $(LDADDCFLAGS)

run.o: run.c def.h
	$(CC) $(CFLAGS) -c run.c -o $@

mm.o: mm.c def.h mm.h
	$(CC) $(CFLAGS) -c mm.c -o $@

slog.o: slog.c def.h slog.h
	$(CC) $(CFLAGS) -c slog.c -o $@

utils.o: utils.c def.h utils.h
	$(CC) $(CFLAGS) -c utils.c -o $@

.PHONY: clean all

clean:
	rm -f *.o run 
