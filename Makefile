SUBDIRS := src test

all:
	$(foreach dir, $(SUBDIRS), $(MAKE) -C $(dir);)

.PHONY: all check clean

check:
	@$(MAKE) -C test check

clean:
	$(foreach dir, $(SUBDIRS), $(MAKE) -C $(dir) clean;)

