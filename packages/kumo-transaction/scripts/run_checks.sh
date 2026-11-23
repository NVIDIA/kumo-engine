#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
uv sync --extra dev
uv run ruff check src tests recipes
uv run ruff format --check src tests recipes
uv run pyright src/tfm
uv run pytest tests/unit tests/contract tests/recipe_smoke -q
