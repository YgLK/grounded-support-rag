SHELL := /bin/zsh

CONFIG_FILE ?= support_graph.toml
SECRETS_FILE ?= .env
DOMAIN ?= dmv
SUBSET ?= smoke
MAX_CONCURRENCY ?= 1
NOTES ?=

.PHONY: eval-smoke

eval-smoke:
	uv run support-graph \
		--config-file $(CONFIG_FILE) \
		--secrets-file $(SECRETS_FILE) \
		eval \
		--domain $(DOMAIN) \
		--subset $(SUBSET) \
		--max-concurrency $(MAX_CONCURRENCY) \
		$(if $(NOTES),--notes "$(NOTES)",)
