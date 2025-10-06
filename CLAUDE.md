# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview



## Development Setup

This project uses **uv** for fast, modern Python package management. All Python commands should use `uv run` instead of direct Python invocation.

```bash
# Install dependencies (use uv, not pip)
uv sync

# Install with development dependencies
uv sync --all-extras

# Install pre-commit hooks (IMPORTANT: run this once after cloning)
uv run pre-commit install
```

## Common Commands

### Using uv
**IMPORTANT: Always use `uv run` to execute Python commands.** This ensures the correct environment and dependencies.

```bash
# Run tests
uv run pytest

```

### Building Locally

```bash
# Build platform-specific wheel for local testing
uv build

```

**Note:** See "Release Process" section below for publishing to PyPI.

### Code Quality
```bash
# Format and lint with Ruff (replaces black, isort, flake8)
uv run ruff format .
uv run ruff check .
uv run ruff check --fix .

# Run pre-commit hooks (runs automatically on commit, or manually)
uv run pre-commit run --all-files

```

## Build System

### Modern Tooling Stack (2025)
- **Package manager**: uv (fast, modern alternative to pip/poetry)
- **Build backend**: Hatchling (PEP 517 compliant, replaces setuptools)
- **Linter/formatter**: Ruff (replaces black, isort, flake8, pyupgrade)
- **Python version**: 3.11+ (specified in pyproject.toml)


## Important Notes

- Always use `uv run` for Python commands to ensure correct environment
- Pre-commit hooks enforce code quality (runs Ruff automatically)
- The C extensions must compile successfully for extraction to work
- Do not commit changes without asking unless you are sure this is intended. NEVER push until asked explicitly.
