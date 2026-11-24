# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pretraining windows must fit their row, and the row the model's context."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from tfm.config import RecipeConfig, load_recipe_config

FULL_SIZE = Path("recipes/blueprint_v1/config.yaml")


def _with_windows(**windows: Any) -> dict[str, Any]:
    raw = load_recipe_config(FULL_SIZE).model_dump(mode="json")
    raw["data"]["pretrain_windows"].update(windows)
    return raw


def test_full_size_windows_fit() -> None:
    windows = RecipeConfig.model_validate(_with_windows()).data.pretrain_windows
    assert windows is not None
    assert (windows.events, windows.max_tokens) == (341, 4096)


def test_window_wider_than_its_row_is_rejected() -> None:
    # 12 * 342 + 1 = 4105 tokens in a 4096-token row.
    with pytest.raises(ValidationError, match="4105 tokens"):
        RecipeConfig.model_validate(_with_windows(events=342))


def test_unset_row_width_falls_back_to_max_tokens() -> None:
    with pytest.raises(ValidationError, match="wider than its row of 1537"):
        RecipeConfig.model_validate(_with_windows(max_tokens=None))


def test_row_wider_than_context_is_rejected() -> None:
    raw = _with_windows()
    raw["model"]["composition"]["granularity"]["decoder"]["max_seq_len"] = 1537
    with pytest.raises(ValidationError, match="wider than the model's context of 1537"):
        RecipeConfig.model_validate(raw)


def test_fields_set_the_tokens_per_event() -> None:
    raw = _with_windows()
    raw["features"]["fields"] = None  # all 12 fields: 13 * 341 + 1 = 4434 tokens
    with pytest.raises(ValidationError, match="4434 tokens"):
        RecipeConfig.model_validate(raw)


def test_packing_cuts_its_own_rows() -> None:
    raw = _with_windows(events=342)
    raw["training"]["packing"] = True
    RecipeConfig.model_validate(raw)
