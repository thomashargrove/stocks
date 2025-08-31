.PHONY: run test test-verbose clean

STOCK_DATA_DIR ?= ./data

run:
	uv run refresh.py $(STOCK_DATA_DIR)

test:
	uv run pytest

test-verbose:
	uv run pytest -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete