CC      = gcc
CFLAGS  = -O0 -g -Wall -pthread -Ic
LDFLAGS = -lpthread
SRC_DIR  = ../src

all: test_tpool test_utils

test_tpool: c/test_tpool.c tpool.o slog.o utils.o mm.o
	$(CC) $(CFLAGS) -I$(SRC_DIR) $^ -o $@ $(LDFLAGS)

tpool.o: $(SRC_DIR)/tpool.c $(SRC_DIR)/tpool.h $(SRC_DIR)/slog.h
	$(CC) $(CFLAGS) -I$(SRC_DIR) -c $(SRC_DIR)/tpool.c -o $@

slog.o: $(SRC_DIR)/slog.c $(SRC_DIR)/slog.h
	$(CC) $(CFLAGS) -I$(SRC_DIR) -c $(SRC_DIR)/slog.c -o $@

utils.o: $(SRC_DIR)/utils.c $(SRC_DIR)/utils.h $(SRC_DIR)/def.h
	$(CC) $(CFLAGS) -I$(SRC_DIR) -c $(SRC_DIR)/utils.c -o $@

mm.o: $(SRC_DIR)/mm.c $(SRC_DIR)/mm.h $(SRC_DIR)/def.h
	$(CC) $(CFLAGS) -I$(SRC_DIR) -c $(SRC_DIR)/mm.c -o $@

test_utils: c/test_utils.c utils.o slog.o mm.o
	$(CC) $(CFLAGS) -I$(SRC_DIR) $^ -o $@ $(LDFLAGS) -lm

.PHONY: all check clean

check: test_tpool test_utils
	./test_tpool
	./test_utils

clean:
	rm -f test_tpool *.o
