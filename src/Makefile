CC = gcc
CFLAGS = -O1
LDADDCFLAGS = -lm

all: run

run: run.o
	$(CC) $(CFLAGS) run.o -o $@ $(LDADDCFLAGS)

run.o: run.c lib.h
	$(CC) $(CFLAGS) -c run.c -o $@


.PHONY: clean all

clean:
	rm -f *.o run 
