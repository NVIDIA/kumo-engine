#!/usr/bin/env bash
set -euo pipefail
uv sync --extra dev
uv run ruff check src tests recipes
uv run ruff format --check src tests recipes
uv run pyright src/tfm
uv run pytest tests/unit tests/contract tests/recipe_smoke -q
