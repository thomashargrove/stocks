# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python stock analysis project that uses the yfinance_cache library to fetch and analyze stock market data. The project is in early development with a simple structure.

## Python Environment

- **Python Version**: 3.13 (specified in .python-version)
- **Project Configuration**: Uses pyproject.toml for project metadata
- **Package Manager**: UV (uv.lock present)

## Key Dependencies

- **yfinance-cache** (>=0.7.13): Stock market data fetching library with caching capabilities

## Common Commands

Since this project uses UV package manager:

```bash
# Install dependencies
uv sync

# Run the main application
uv run refresh.py

# Run tests
make test

# Run tests with verbose output
make test-verbose

# Clean Python cache files
make clean
```

## Code Architecture

**Current Structure:**
- `refresh.py`: Entry point containing a simple example that fetches Microsoft (MSFT) stock data for 1 week and prints calendar information
- The application uses yfinance_cache.Ticker to create ticker objects and retrieve historical data

**Key Components:**
- Single main() function that demonstrates basic stock data retrieval
- Uses yfinance_cache library for stock market data access with caching capabilities

## Coding Standards – Python

- Always include **type hints** for **all** function/method parameters and **explicit return types**.
- Use `from __future__ import annotations` (if <3.11) and prefer `X | None` for optionals.
- Avoid `Any` unless unavoidable; prefer `TypedDict`, `Protocol`, `Literal`, and `dataclass` where appropriate.
- Do not leave `*args`/`**kwargs` untyped.
- Public APIs must be fully typed; internal helpers should be too unless there’s a clear reason not to (document it).

### When generating code (Claude):
1) Add type hints everywhere and an explicit return type.
2) Include a brief docstring describing parameters and return values.
3) If a type is uncertain, choose the narrowest reasonable type and add a `# TODO: refine type` note.

## Testing

- **Test Framework**: pytest
- **Test Files**: `test_*.py` pattern
- **Test Runner**: Use `make test` or `make test-verbose`

### Testing Guidelines:
- **NO FILESYSTEM INTERACTION**: Unit tests must never interact with the filesystem
- For functions that interact with the filesystem, split them into inner/outer functions where the inner function can be tested with mock data
- Test files should be named `test_<module>.py`
- Use descriptive test function names: `test_function_name_scenario()`
- Test the inner, pure functions that take data structures as input