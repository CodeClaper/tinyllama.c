SUBDIRS := src test

NAME    := $(notdir $(CURDIR))
VERSION ?= $(shell git describe --tags --always 2>/dev/null || echo 0.0.0)
ARCHIVE := $(NAME)-$(VERSION).tar.gz

.PHONY: all archive check clean

all:
	$(foreach dir, $(SUBDIRS), $(MAKE) -C $(dir);)

archive:
	@git status --porcelain --untracked-files=no | grep -q . && \
		echo "warning: uncommitted changes are not included; commit them first" >&2 || true
	@git archive --format=tar.gz --prefix=$(NAME)/ -o $(ARCHIVE) HEAD
	@echo "  ARCH $(ARCHIVE)"
	@echo "  Deploy: scp $(ARCHIVE) user@host:/opt/ && tar xzf $(ARCHIVE) && make -C $(NAME)"

check:
	@$(MAKE) -C test check

clean:
	$(foreach dir, $(SUBDIRS), $(MAKE) -C $(dir) clean;)

