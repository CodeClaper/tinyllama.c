
run: run.o
	$(CC) $(CFLAGS) run.o -o $@ $(LDADDCFLAGS)

run.o:
	$(CC) $(CFLAGS) -c run.c -o $@


.PHONY: run clean

clean:
	rm -f *.o run 
