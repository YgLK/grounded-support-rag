SHELL := /bin/zsh

CONFIG_FILE ?= support_graph.toml
SECRETS_FILE ?= .env
DOMAIN ?= dmv
SUBSET ?= smoke
MAX_CONCURRENCY ?= 1
NOTES ?=
UI_HOST ?= 127.0.0.1
UI_PORT ?= 8008

.PHONY: postgres-up ui eval-smoke

postgres-up:
	docker compose up -d postgres

ui:
	uv run grounded-support-rag \
		--config-file $(CONFIG_FILE) \
		--secrets-file $(SECRETS_FILE) \
		ui \
		--host $(UI_HOST) \
		--port $(UI_PORT)

eval-smoke:
	uv run grounded-support-rag \
		--config-file $(CONFIG_FILE) \
		--secrets-file $(SECRETS_FILE) \
		eval \
		--domain $(DOMAIN) \
		--subset $(SUBSET) \
		--max-concurrency $(MAX_CONCURRENCY) \
		$(if $(NOTES),--notes "$(NOTES)",)
