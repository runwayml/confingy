# Shell configuration
SHELL := /bin/bash

# Targets
.PHONY: pytest
pytest:
	uv run --group dev --extra viz pytest -vv

.PHONY: mypy
mypy:
	uv run --group dev --extra viz mypy -p confingy

.PHONY: format-check
format-check:
	uv run --group dev --extra viz ruff format --check

.PHONY: lint
lint:
	uv run --group dev --extra viz ruff check

.PHONY: docs
docs:
	uv run --group dev mkdocs build --site-dir _build/docs

.PHONY: serve-docs
serve-docs:
	uv run --group dev mkdocs serve
