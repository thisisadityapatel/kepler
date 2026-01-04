.PHONY: help install test lint

help:
	@echo "Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make test       - Run unit tests with pytest"
	@echo "  make lint       - Run and fix linting checks with ruff"
	@echo "  make hf_cache   - Hugging face locally downloaded models data"

install:
	uv sync
	uv pip install -r requirements.txt

test:
	uv run pytest -v

lint:
	uv run ruff check --fix .
	uv run ruff format .

hf_cache:
	@echo "\nCache Location: ~/.cache/huggingface/"
	@echo "\nTotal Size:"
	@du -sh ~/.cache/huggingface/
	@echo "\nSize by Model:"
	@du -sh ~/.cache/huggingface/hub/models--* 2>/dev/null || echo "No models cached"
	@echo "\nAll Cached Items:"
	@ls -lh ~/.cache/huggingface/hub/ 2>/dev/null || echo "Cache directory not found"