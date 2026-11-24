# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A probe is never scored on a row it was fit on (ADR 0008).

This is a repository invariant, not a recipe detail: an overlapping probe
split silently produced ROC-AUC and PR-AUC of exactly 1.0 for `default_12m`,
including for the five-feature raw baseline that has no model in it at all.
"""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tfm.contracts.errors import InsufficientLabelSupportError
from tfm.data.adapters.pragmatiq import (
    LabelRow,
    generate_pragmatiq_fixture,
    load_pragmatiq_dataset,
)
from tfm.data.sequence_builders.entity_sequences import build_sequences_at_label_cutoffs
from tfm.data.splits.entity import MIN_ENTITIES_PER_CLASS, split_entities
from tfm.runtime.common import parse_duration
from tfm.runtime.evaluate import split_label_pairs as _split_label_pairs

SEEDS = (0, 1, 7, 42, 1234)


def _pairs(n_pos: int, n_neg: int) -> list[tuple[str, LabelRow]]:
    ts = datetime(2024, 3, 1, tzinfo=UTC)

    def row(entity_id: str, label: int) -> tuple[str, LabelRow]:
        return (
            f"seq_{entity_id}",
            LabelRow(
                entity_id=entity_id,
                timestamp=ts,
                label=label,
                task="default_12m",
                extra={},
            ),
        )

    return [row(f"p{i}", 1) for i in range(n_pos)] + [row(f"n{i}", 0) for i in range(n_neg)]


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize(
    ("n_pos", "n_neg"),
    [(2, 2), (2, 30), (5, 25), (30, 2), (15, 15), (2, 3), (3, 2)],
)
def test_probe_splits_never_share_an_entity(n_pos: int, n_neg: int, seed: int) -> None:
    train, evaluation = _split_label_pairs(_pairs(n_pos, n_neg), seed=seed)

    train_ids = {row.entity_id for _seq, row in train}
    eval_ids = {row.entity_id for _seq, row in evaluation}
    assert train_ids
    assert eval_ids
    assert not train_ids & eval_ids


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize(("n_pos", "n_neg"), [(2, 2), (2, 30), (5, 25), (30, 2)])
def test_both_sides_keep_every_represented_class(n_pos: int, n_neg: int, seed: int) -> None:
    train, evaluation = _split_label_pairs(_pairs(n_pos, n_neg), seed=seed)

    for side in (train, evaluation):
        labels = {int(row.label) for _seq, row in side}
        assert labels == {0, 1}


@pytest.mark.parametrize(("n_pos", "n_neg"), [(1, 30), (30, 1), (1, 1), (0, 30)])
def test_a_class_too_small_to_split_is_refused_not_overlapped(n_pos: int, n_neg: int) -> None:
    with pytest.raises(InsufficientLabelSupportError):
        _split_label_pairs(_pairs(n_pos, n_neg), seed=42)


def test_the_refusal_names_the_threshold_and_the_reason() -> None:
    with pytest.raises(InsufficientLabelSupportError) as excinfo:
        split_entities([("only_positive", 1), ("a", 0), ("b", 0)], seed=42)

    message = str(excinfo.value)
    assert str(MIN_ENTITIES_PER_CLASS) in message
    assert "training rows" in message


def test_a_skipped_probe_is_absent_from_metrics_rather_than_perfect(tmp_path: Path) -> None:
    """Five fixture users hold exactly one positive entity.

    Before ADR 0008 that produced `default_12m_*` rows reporting 1.0. Now the
    task is skipped with a warning, so no unearned number reaches the report.
    """
    from tfm.config import load_recipe_config
    from tfm.runtime.data import prepare_data
    from tfm.runtime.evaluate import forecast_task

    config = load_recipe_config(Path("recipes/pragma_like/config.smoke.yaml"))
    prepared = prepare_data(config, fixture_dir=tmp_path / "prepared")
    # The adapter generates at least eight users; five are written here directly.
    root = generate_pragmatiq_fixture(tmp_path / "book", n_users=5, events_per_user=9, seed=42)
    dataset = load_pragmatiq_dataset(root)
    rows = dataset.labels["default_12m"]
    assert sum(int(row.label) for row in rows) == 1

    with pytest.warns(UserWarning, match="Skipping the 'default_12m' probe"):
        probe = forecast_task(prepared, config, "default_12m", rows, seed=config.seed)
    assert probe is None


def test_forecast_rows_before_the_pretraining_cutoff_are_refused(tmp_path: Path) -> None:
    """ADR 0013 decision 4: the encoder must not have seen the label horizon."""
    from tfm.config import load_recipe_config
    from tfm.runtime.data import prepare_data
    from tfm.runtime.evaluate import forecast_task

    config = load_recipe_config(Path("recipes/pragma_like/config.smoke.yaml"))
    prepared = prepare_data(config, fixture_dir=tmp_path / "book")
    early = [
        LabelRow(
            entity_id=f"u{i:04d}",
            timestamp=datetime(2023, 12, 1, tzinfo=UTC),
            label=i % 2,
            task="default_12m",
        )
        for i in range(8)
    ]
    with pytest.warns(UserWarning, match="before the pretraining cutoff"):
        assert forecast_task(prepared, config, "default_12m", early, seed=0) is None


def test_the_eval_ts_cutoff_itself_stays_clean() -> None:
    """The cutoff path was the other leakage suspect; it was not the cause."""
    from tfm.config import load_recipe_config

    config = load_recipe_config(Path("recipes/pragma_like/config.synth_1m.yaml"))
    with tempfile.TemporaryDirectory() as tmp:
        root = generate_pragmatiq_fixture(
            Path(tmp), n_users=32, events_per_user=6, seed=config.seed
        )
        dataset = load_pragmatiq_dataset(root)
        pairs = build_sequences_at_label_cutoffs(
            dataset.events,
            dataset.labels["default_12m"],
            max_events=config.data.sequence_length,
            observation_window=parse_duration(config.data.observation_window),
        )

    assert pairs
    late = [
        (row.entity_id, event.event_timestamp, row.timestamp)
        for sequence, row in pairs
        for event in sequence.events
        if event.event_timestamp > row.timestamp
    ]
    assert late == []
