#!/usr/bin/env bash
set -euo pipefail
uv run pytest tests/recipe_smoke -q
