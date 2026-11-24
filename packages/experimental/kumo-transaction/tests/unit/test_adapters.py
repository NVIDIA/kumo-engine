# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from tfm.config import DataConfig, RecipeConfig, TemporalSplitConfig
from tfm.contracts.errors import SchemaError
from tfm.data.adapters.paths import resolve_data_path
from tfm.data.adapters.registry import advertised_adapters, load_adapter_events
from tfm.data.adapters.tabformer import write_tabformer_fixture
from tfm.data.splits.temporal import assign_temporal_split
from tfm.features.tokenizers import build_tokenizer
from tfm.runtime import parse_timestamp


def _synthetic_config(**overrides: object) -> RecipeConfig:
    base = {
        "seed": 0,
        "data": {
            "adapter": "synthetic",
            "n_entities": 8,
            "events_per_entity": 8,
            "temporal_split": {
                "train_end": "2024-01-04T00:00:00Z",
                "validation_end": "2024-01-07T00:00:00Z",
                "label_delay": "12h",
            },
        },
    }
    base.update(overrides)
    return RecipeConfig.model_validate(base)


def test_advertised_adapters_include_synthetic() -> None:
    names = advertised_adapters()
    assert "synthetic" in names
    assert "tabformer" in names
    assert "pragmatiq" in names


def test_synthetic_adapter_via_registry() -> None:
    result = load_adapter_events(_synthetic_config())
    assert result.events
    assert result.splits is None
    assert result.metadata["adapter"] == "synthetic"


def test_tabformer_precomputed_fixture(tmp_path) -> None:
    root = write_tabformer_fixture(tmp_path / "tabformer", n_users=6, events_per_user=4, seed=1)
    config = _synthetic_config(
        data={
            "adapter": "tabformer",
            "path": str(root),
            "n_entities": 4,
            "max_entities": 4,
            "split_protocol": "precomputed",
            "observation_window": "3650d",
            "sequence_length": 8,
            "temporal_split": {
                "train_end": "2016-01-01T00:00:00Z",
                "validation_end": "2018-01-01T00:00:00Z",
                "label_delay": "0h",
            },
        }
    )
    result = load_adapter_events(config)
    assert result.splits is not None
    assert result.splits.train
    assert result.splits.validation
    assert result.splits.test
    assert all(event.labels["fraud"].name == "fraud" for event in result.events)
    tok = build_tokenizer("transaction_field_tokenizer")
    tokens = tok.event_tokens(result.events[0])
    assert tokens[0].startswith("AMT_")
    assert len(tokens) == 12
    sampled = {event.entity_id for event in result.events}
    assert any(
        event.entity_id in sampled and event.labels["fraud"].value >= 1.0 for event in result.events
    )


def test_precomputed_sequences_ignore_stale_yaml_train_end(tmp_path) -> None:
    """TabFormer train parquet can extend past YAML train_end; use partition max."""
    from tfm.runtime import prepare_data

    root = write_tabformer_fixture(tmp_path / "tabformer", n_users=6, events_per_user=4, seed=1)
    config = _synthetic_config(
        data={
            "adapter": "tabformer",
            "path": str(root),
            "n_entities": 4,
            "max_entities": 4,
            "split_protocol": "precomputed",
            "observation_window": "3650d",
            "sequence_length": 8,
            "temporal_split": {
                "train_end": "2000-01-01T00:00:00Z",
                "validation_end": "2001-01-01T00:00:00Z",
                "label_delay": "0h",
            },
        }
    )
    train_batch, _val_batch, test_batch = prepare_data(config).batches
    assert train_batch.batch_size > 0
    assert float(test_batch.fraud_labels.sum()) >= 1.0


def test_tabformer_missing_path_errors() -> None:
    config = DataConfig(
        adapter="tabformer",
        temporal_split=TemporalSplitConfig(
            train_end="2016-01-01T00:00:00Z",
            validation_end="2018-01-01T00:00:00Z",
        ),
    )
    try:
        resolve_data_path(config)
    except SchemaError as exc:
        assert "TFM_TABFORMER_ROOT" in str(exc)
    else:
        raise AssertionError("expected SchemaError")


def test_resolve_data_path_from_env(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TFM_TABFORMER_ROOT", str(tmp_path))
    config = DataConfig(
        adapter="tabformer",
        temporal_split=TemporalSplitConfig(
            train_end="2016-01-01T00:00:00Z",
            validation_end="2018-01-01T00:00:00Z",
        ),
    )
    assert resolve_data_path(config) == tmp_path.resolve()


def test_pragmatiq_entity_subsample(tmp_path) -> None:
    from tfm.data.adapters.pragmatiq import generate_pragmatiq_fixture, load_pragmatiq_dataset

    root = generate_pragmatiq_fixture(tmp_path / "book", n_users=8, events_per_user=4, seed=3)
    full = load_pragmatiq_dataset(root)
    positives = {
        event.entity_id
        for event in full.events
        if event.labels.get("fraud") is not None and event.labels["fraud"].value >= 1.0
    }
    sampled = load_pragmatiq_dataset(root, max_entities=3, sample_seed=0)
    users = {event.entity_id for event in sampled.events}
    assert len(users) <= 3
    assert sampled.events
    if positives:
        assert users & positives


def test_timestamp_split_still_works_on_tabformer_fixture(tmp_path) -> None:
    root = write_tabformer_fixture(tmp_path / "tabformer", n_users=6, events_per_user=4, seed=2)
    config = _synthetic_config(
        data={
            "adapter": "tabformer",
            "path": str(root),
            "n_entities": 6,
            "split_protocol": "timestamp",
            "temporal_split": {
                "train_end": "2010-01-01T00:00:00Z",
                "validation_end": "2017-01-01T00:00:00Z",
                "label_delay": "0h",
            },
        }
    )
    result = load_adapter_events(config)
    spec_train = parse_timestamp(config.data.temporal_split.train_end)
    spec_val = parse_timestamp(config.data.temporal_split.validation_end)
    from datetime import timedelta

    from tfm.data.splits.temporal import TemporalSplitSpec

    splits = assign_temporal_split(
        list(result.events),
        TemporalSplitSpec(train_end=spec_train, validation_end=spec_val, label_delay=timedelta(0)),
    )
    assert splits.train and splits.validation and splits.test


def _tabformer_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "User": 0,
        "Card": 0,
        "Year": 2016,
        "Month": 9,
        "Day": 1,
        "Time": "06:15",
        "Amount": "$25.00",
        "Use Chip": "Online Transaction",
        "Merchant Name": 3527213246127876953,
        "Merchant City": "ONLINE",
        "Merchant State": None,
        "Zip": None,
        "MCC": 4900,
        "Is Fraud?": "No",
    }
    row.update(overrides)
    return row


def test_missing_merchant_state_tokenizes_as_xx() -> None:
    """Online TabFormer rows carry no state; the Blueprint corpus uses STATE_XX."""
    from datetime import timedelta

    from tfm.data.adapters.tabformer import _row_to_event

    event = _row_to_event(_tabformer_row(), index=0, label_delay=timedelta(0))
    assert event.categorical["merchant_state"] == "XX"

    tokenizer = build_tokenizer("transaction_field_tokenizer")
    assert "STATE_XX" in tokenizer.event_tokens(event)


def test_present_merchant_state_is_preserved() -> None:
    from datetime import timedelta

    from tfm.data.adapters.tabformer import _row_to_event

    event = _row_to_event(
        _tabformer_row(**{"Merchant State": "CA"}), index=0, label_delay=timedelta(0)
    )
    assert event.categorical["merchant_state"] == "CA"

    tokenizer = build_tokenizer("transaction_field_tokenizer")
    assert "STATE_CA" in tokenizer.event_tokens(event)
