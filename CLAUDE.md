# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This package contains the central slit-decomposition algorithm from PyReduce (see [Piskunov, Wehrhahn & Marquart 2021](https://doi.org/10.1051/0004-6361/202038293)), extracted for further development. It extracts 1D spectra from 2D detector frames captured by echelle spectrographs.

### Progression Beyond Polynomial Slit Shapes

The code here extends the original polynomial-based slit shape model (described in the paper) to support **array-based slit descriptions**. Instead of polynomial coefficients, the slit is now described by an array of `delta_x` values—pixel shifts from vertical—that specify how the slit curves from row to row in each spectral order.

### Workflow

1. **Generate test data**: `make_testdata.py` creates synthetic detector frames
2. **Measure slit deltas**: `make_slitdeltas.py` measures the `delta_x` values from the data
3. **Extract spectra**: Run `uv run py.test` to perform extraction using both the data and measured deltas


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
uv run py.test

```

### Building Locally

```bash
# Build platform-specific wheel for local testing
uv build

# For development: rebuild C/C++ extensions after code changes
uv sync --reinstall-package charslit
```

**IMPORTANT:** The package is installed in editable mode by `uv sync`. However:
- **Python code changes** are picked up automatically (no rebuild needed)
- **C/C++ code changes** require `uv sync --reinstall-package charslit` to recompile the extension

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
- After modifying C/C++ code, use `uv sync --reinstall-package charslit` to rebuild
- Do not commit changes without asking unless you are sure this is intended. NEVER push until asked explicitly.