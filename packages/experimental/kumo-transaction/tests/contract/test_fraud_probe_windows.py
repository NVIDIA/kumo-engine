# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The fraud probe is entity-disjoint and labels come from each side's period (probe split rules)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import torch

from tfm.config import RecipeConfig, load_recipe_config
from tfm.contracts.errors import SchemaError
from tfm.contracts.schema import EventLabel, TransactionEvent, TransactionSequence
from tfm.data.collate import sequence_label
from tfm.data.labels.horizon import available_label_value
from tfm.data.labels.windows import LabelWindow
from tfm.data.sequence_builders.event_rows import build_event_history_rows
from tfm.runtime.data import PreparedData, prepare_data
from tfm.runtime.evaluate import (
    event_entity_split,
    fraud_entity_split,
    fraud_task,
    history_window_tasks,
)

CONFIGS = (
    Path("recipes/blueprint_v2/config.smoke.yaml"),
    Path("tests/fixtures/configs/flat_decoder.yaml"),
)
T0 = datetime(2024, 1, 1, tzinfo=UTC)


def _event(i: int, *, fraud: float | None) -> TransactionEvent:
    ts = T0 + timedelta(days=i)
    labels = {}
    if fraud is not None:
        labels["fraud"] = EventLabel(name="fraud", value=fraud, available_at=ts, level="event")
    return TransactionEvent(
        entity_id="e0",
        event_id=f"ev{i}",
        event_timestamp=ts,
        timezone="UTC",
        numeric={"amount": 10.0},
        labels=labels,
    )


def _prepared(path: Path, tmp_path: Path) -> PreparedData:
    return prepare_data(load_recipe_config(path), fixture_dir=tmp_path / "book")


def _counted(data: PreparedData, side: str) -> set[tuple[str, str]]:
    sequences = data.train_sequences if side == "train" else data.test_sequences
    window = data.cutoffs.windows[side]
    counted: set[tuple[str, str]] = set()
    for seq in sequences:
        for event in seq.events:
            value = available_label_value(event, "fraud", seq.cutoff_time)
            if window.contains(event.event_timestamp) and value is not None and value >= 0.5:
                counted.add((seq.entity_id, event.event_id))
    return counted


def test_a_window_excludes_earlier_fraud_from_the_label() -> None:
    sequence = TransactionSequence(
        entity_id="e0",
        events=(_event(0, fraud=1.0), _event(5, fraud=0.0)),
        cutoff_time=T0 + timedelta(days=6),
        observation_window=None,
    )
    recent = LabelWindow(T0 + timedelta(days=2), T0 + timedelta(days=6))
    assert sequence_label(sequence) == 1.0
    assert sequence_label(sequence, recent) == 0.0


def test_a_window_is_start_exclusive_and_end_inclusive() -> None:
    window = LabelWindow(T0, T0 + timedelta(days=1))
    assert not window.contains(T0)
    assert window.contains(T0 + timedelta(days=1))
    with pytest.raises(ValueError, match="precede"):
        LabelWindow(T0, T0)


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.parent.name)
def test_label_windows_do_not_overlap(path: Path, tmp_path: Path) -> None:
    windows = _prepared(path, tmp_path).cutoffs.windows
    assert windows["train"].end <= windows["test"].start  # type: ignore[operator]
    assert windows["validation"].end == windows["test"].start


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.parent.name)
def test_no_entity_is_on_both_fraud_probe_sides(path: Path, tmp_path: Path) -> None:
    data = _prepared(path, tmp_path)
    task = fraud_task(data, fraud_entity_split(data, seed=42))
    assert not set(task.train_batch.entity_ids) & set(task.eval_batch.entity_ids)
    assert task.y_eval.sum() >= 1 and task.y_train.sum() >= 1


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.parent.name)
def test_no_labelled_event_is_counted_by_both_sides(path: Path, tmp_path: Path) -> None:
    data = _prepared(path, tmp_path)
    assert not _counted(data, "train") & _counted(data, "test")


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.parent.name)
def test_every_counted_label_is_inside_its_window(path: Path, tmp_path: Path) -> None:
    data = _prepared(path, tmp_path)
    for side, batch_index, sequences in (
        ("train", 0, data.train_sequences),
        ("test", 2, data.test_sequences),
    ):
        window = data.cutoffs.windows[side]
        labels = data.batches[batch_index].fraud_labels.tolist()
        for seq, label in zip(sequences, labels, strict=True):
            assert label == sequence_label(seq, window)


def test_history_rows_end_at_their_event_and_carry_its_label() -> None:
    events = [_event(i, fraud=float(i == 3)) for i in range(6)]
    rows = build_event_history_rows(
        events,
        entities=["e0"],
        window=LabelWindow(T0 + timedelta(days=1), T0 + timedelta(days=5)),
        label_cutoff=T0 + timedelta(days=10),
        max_events=3,
        observation_window=None,
        max_rows=100,
        seed=0,
    )
    assert [r.history.events[-1].event_id for r in rows] == ["ev2", "ev3", "ev4", "ev5"]
    assert [r.label for r in rows] == [0.0, 1.0, 0.0, 0.0]
    for row in rows:
        assert len(row.history.events) <= 3
        assert row.event_only.events == (row.history.events[-1],)
        assert all(e.event_timestamp <= row.history.cutoff_time for e in row.history.events)


def _many(n: int, positives: set[int]) -> list[TransactionEvent]:
    return [_event(i, fraud=float(i in positives)) for i in range(n)]


def test_a_negatives_ratio_keeps_every_positive_and_samples_negatives() -> None:
    events = _many(60, {5, 17, 40})
    kwargs = {
        "entities": ["e0"],
        "window": LabelWindow(None, T0 + timedelta(days=100)),
        "label_cutoff": T0 + timedelta(days=100),
        "max_events": 4,
        "observation_window": None,
        "seed": 0,
    }
    rows = build_event_history_rows(events, max_rows=100, negatives_per_positive=2.0, **kwargs)
    assert sorted(r.history.events[-1].event_id for r in rows if r.label) == [
        "ev17",
        "ev40",
        "ev5",
    ]
    assert sum(1 for r in rows if not r.label) == 6
    ends = [r.history.events[-1].event_timestamp for r in rows]
    assert ends == sorted(ends)
    capped = build_event_history_rows(events, max_rows=5, negatives_per_positive=2.0, **kwargs)
    assert sum(r.label for r in capped) == 3 and len(capped) == 5
    uniform = build_event_history_rows(events, max_rows=100, **kwargs)
    assert len(uniform) == 60
    every = build_event_history_rows(events, max_rows=None, negatives_per_positive=50.0, **kwargs)
    assert len(every) == 60


def test_chunked_probe_collation_matches_one_call(monkeypatch: pytest.MonkeyPatch) -> None:
    from tfm.adaptation.batching import concat_rows, select_rows
    from tfm.data.collate import collate_batch
    from tfm.data.sequence_builders.entity_sequences import build_entity_sequences
    from tfm.data.synthetic import generate_synthetic_events
    from tfm.features.tokenizers.financial import TransactionFieldTokenizer
    from tfm.runtime import evaluate

    events = generate_synthetic_events(seed=0, n_entities=7, events_per_entity=9)
    histories = build_entity_sequences(events, T0 + timedelta(days=3650), max_events=6)
    singles = [
        TransactionSequence(
            entity_id=s.entity_id,
            events=s.events[-1:],
            cutoff_time=s.cutoff_time,
            observation_window=None,
        )
        for s in histories
    ]
    tokenizer = TransactionFieldTokenizer(fields=("AMT", "MCC", "HOUR"))
    monkeypatch.setattr(evaluate, "COLLATE_ROWS", 3)

    whole = collate_batch(histories, tokenizer, max_tokens=40)
    chunked = evaluate._collate_rows(histories, tokenizer, max_tokens=40, amount_key="amount")
    width = chunked.input_ids.shape[1]
    assert chunked.labels is None
    assert chunked.input_ids.tolist() == whole.input_ids[:, :width].tolist()
    assert int(whole.attention_mask[:, width:].sum()) == 0
    assert chunked.fraud_labels.tolist() == whole.fraud_labels.tolist()
    assert chunked.entity_ids == whole.entity_ids

    narrow = evaluate._collate_rows(singles, tokenizer, max_tokens=40, amount_key="amount")
    assert narrow.input_ids.shape[1] == 1 + 3 + 1 < 40
    assert (
        narrow.input_ids.tolist()
        == collate_batch(singles, tokenizer, max_tokens=40)
        .input_ids[:, : narrow.input_ids.shape[1]]
        .tolist()
    )

    halves = [select_rows(whole, range(0, 4)), select_rows(whole, range(4, 7))]
    assert concat_rows(halves).input_ids.tolist() == whole.input_ids.tolist()


def test_history_rows_respect_the_observation_window_and_the_event_cap() -> None:
    events = _many(30, {29})
    rows = build_event_history_rows(
        events,
        entities=["e0"],
        window=LabelWindow(T0 + timedelta(days=28), T0 + timedelta(days=29)),
        label_cutoff=T0 + timedelta(days=40),
        max_events=10,
        observation_window=timedelta(days=3),
        max_rows=10,
        seed=0,
    )
    assert [e.event_id for e in rows[0].history.events] == ["ev26", "ev27", "ev28", "ev29"]


def test_the_single_event_control_is_a_view_on_the_history_task_only() -> None:
    import numpy as np

    from tfm.runtime.evaluate import ProbeTask, _embedding_views

    y = np.zeros(2, dtype=np.float32)
    history = ProbeTask(None, None, y, y)
    control = ProbeTask(
        None, None, y, y, train_vectors=np.ones((2, 3)), eval_vectors=np.ones((2, 3))
    )
    tasks = {"history_window": history, "single_event": control}
    views = _embedding_views("history_window", tasks)
    assert list(views) == ["single_event"]
    assert views["single_event"][0] is control.train_vectors
    assert _embedding_views("single_event", tasks) == {}
    assert _embedding_views("history_window", {"history_window": history}) == {}


def test_the_surprise_view_joins_each_rows_embedding_to_its_own_surprise() -> None:
    import numpy as np

    from tfm.runtime.evaluate import ProbeTask, _embedding_views

    y = np.zeros(2, dtype=np.float32)
    task = ProbeTask(
        None,
        None,
        y,
        y,
        train_vectors=np.array([[1.0, 2.0], [3.0, 4.0]]),
        eval_vectors=np.array([[5.0, 6.0], [7.0, 8.0]]),
        train_surprise=np.array([[0.1], [0.2]]),
        eval_surprise=np.array([[0.3], [0.4]]),
    )
    train, evaluation = _embedding_views("churn_6m", {"churn_6m": task})["surprise"]
    np.testing.assert_array_equal(train, [[1.0, 2.0, 0.1], [3.0, 4.0, 0.2]])
    np.testing.assert_array_equal(evaluation, [[5.0, 6.0, 0.3], [7.0, 8.0, 0.4]])
    without = ProbeTask(None, None, y, y, train_vectors=task.train_vectors)
    assert _embedding_views("churn_6m", {"churn_6m": without}) == {}


def test_the_event_split_counts_fraud_the_truncated_sequence_misses(tmp_path: Path) -> None:
    from tfm.runtime.evaluate import event_entity_split

    path = Path("tests/fixtures/configs/flat_decoder.yaml")
    data = _prepared(path, tmp_path)
    window = data.cutoffs.windows["test"]
    positive = {
        e.entity_id
        for e in data.events
        if window.contains(e.event_timestamp)
        and (available_label_value(e, "fraud", data.cutoffs.test) or 0.0) >= 0.5
    }
    split = event_entity_split(data, seed=42)
    sides = set(split.train) | set(split.evaluation)
    assert positive <= sides
    assert positive & set(split.evaluation) and positive & set(split.train)
    assert not set(split.train) & set(split.evaluation)


def test_collate_reads_the_named_label_and_amount_field() -> None:
    from datetime import UTC, datetime, timedelta

    from tfm.data.collate import raw_feature_vector

    start = datetime(2030, 1, 1, tzinfo=UTC)
    events = tuple(
        TransactionEvent(
            event_id=f"e{i}",
            entity_id="u",
            event_timestamp=start + timedelta(days=i),
            numeric={"amount": 1.0, "balance": 10.0 * (i + 1)},
            labels={
                "fraud": EventLabel(name="fraud", value=0.0, available_at=start),
                "chargeback": EventLabel(name="chargeback", value=float(i), available_at=start),
            },
        )
        for i in range(2)
    )
    sequence = TransactionSequence(
        entity_id="u",
        events=events,
        cutoff_time=start + timedelta(days=5),
        observation_window="30d",
    )
    assert sequence_label(sequence) == 0.0
    assert sequence_label(sequence, label="chargeback") == 1.0
    assert raw_feature_vector(sequence)[:2] == [1.0, 1.0]
    assert raw_feature_vector(sequence, amount_key="balance")[:2] == [15.0, 20.0]


def test_every_raw_feature_reads_data_amount_field() -> None:
    def prepared(amount_field: str) -> tuple[RecipeConfig, PreparedData]:
        config = load_recipe_config(Path("tests/fixtures/configs/flat_decoder.yaml"))
        data = config.data.model_copy(update={"amount_field": amount_field})
        config = config.model_copy(update={"data": data})
        return config, prepare_data(config)

    # `zip` is constant per entity, so its mean equals its maximum; `amount` varies.
    config, data = prepared("zip")
    probes = history_window_tasks(data, config, event_entity_split(data, config.seed))
    batches = (*data.batches, *(p.train_batch for p in probes.values()))
    assert len(batches) == 5
    for batch in batches:
        features = batch.numeric_features
        assert torch.equal(features[:, 0], features[:, 1])
    _, default = prepared("amount")
    features = default.batches[0].numeric_features
    assert not torch.equal(features[:, 0], features[:, 1])
    with pytest.raises(SchemaError, match="data.amount_field"):
        prepared("balance")


def test_concat_rows_joins_named_arrays_in_any_order() -> None:
    import dataclasses

    from tfm.adaptation.batching import concat_rows

    @dataclasses.dataclass
    class Rows:
        x: torch.Tensor
        events: dict[str, torch.Tensor]

        @property
        def batch_size(self) -> int:
            return len(self.x)

    a = Rows(torch.zeros(2), {"hour": torch.zeros(2), "dow": torch.ones(2)})
    b = Rows(torch.zeros(1), {"dow": torch.ones(1), "hour": torch.zeros(1)})
    joined = concat_rows([a, b]).events
    assert joined["hour"].tolist() == [0, 0, 0] and joined["dow"].tolist() == [1, 1, 1]
    with pytest.raises(ValueError, match="differs"):
        concat_rows([a, Rows(torch.zeros(1), {"hour": torch.zeros(1)})])
    # 3 + 1 named rows add up to the 4 batch rows, but neither batch has one row per row.
    with pytest.raises(ValueError, match="differs"):
        concat_rows(
            [
                Rows(torch.zeros(2), {"hour": torch.zeros(3), "dow": torch.ones(3)}),
                Rows(torch.zeros(2), {"hour": torch.zeros(1), "dow": torch.ones(1)}),
            ]
        )


def test_batching_keeps_a_dict_that_is_not_per_row_arrays() -> None:
    import dataclasses

    from tfm.adaptation.batching import concat_rows, select_rows

    @dataclasses.dataclass
    class Rows:
        x: torch.Tensor
        metadata: dict[str, str]

        @property
        def batch_size(self) -> int:
            return len(self.x)

    batch = Rows(torch.arange(3.0), {"split": "train"})
    assert select_rows(batch, [2, 0]).metadata == {"split": "train"}
    assert concat_rows([batch, batch]).metadata == {"split": "train"}
