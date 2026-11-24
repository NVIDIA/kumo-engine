# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`data.sequence_length` and `data.max_tokens` reach the key-value-time tokenizer.

The tokenizer's own default keeps each user's last 32 events, so a runtime that
does not pass the config through silently trains on 32 events whatever the
recipe says.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tfm.config import load_recipe_config
from tfm.runtime.data import PreparedData, prepare_data

CONFIG = Path("recipes/blueprint_v2/config.smoke.yaml")


def _prepare(tmp_path: Path, *, sequence_length: int, events: int) -> PreparedData:
    config = load_recipe_config(CONFIG)
    data = config.data.model_copy(
        update={
            "sequence_length": sequence_length,
            "events_per_entity": events,
            "max_tokens": 12,
        }
    )
    return prepare_data(config.model_copy(update={"data": data}), fixture_dir=tmp_path / "book")


@pytest.mark.parametrize("sequence_length", [3, 48])
def test_tokenizer_caps_follow_the_config(tmp_path: Path, sequence_length: int) -> None:
    prepared = _prepare(tmp_path, sequence_length=sequence_length, events=9)
    manifest = prepared.tokenizer.manifest()
    assert manifest["max_events_per_user"] == sequence_length
    assert manifest["max_event_tokens"] == 12
    for batch in prepared.batches:
        assert batch.key_ids.shape[1] <= sequence_length
        assert batch.key_ids.shape[2] <= 12


def test_short_sequence_length_keeps_the_latest_events(tmp_path: Path) -> None:
    train = _prepare(tmp_path, sequence_length=3, events=9).batches[0]
    assert train.key_ids.shape[1] == 3
    assert int(train.event_mask.sum(dim=1).max()) == 3
