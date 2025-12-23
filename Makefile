.PHONY: help install test lint

help:
	@echo "Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make test       - Run unit tests with pytest"
	@echo "  make lint       - Run and fix linting checks with ruff"

install:
	uv sync

test:
	uv run pytest -v

lint:
	uv run ruff check .
	uv run ruff format --check .