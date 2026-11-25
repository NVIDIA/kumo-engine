# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A TabFormer flat-field token cache equals the object path (token-cache guide).

The book below holds what could make the two disagree: tied timestamps whose
`event_id` indices sort differently as strings ("-10" before "-9"), rows out
of time order, online rows without a state or ZIP, fraud labels on both sides
of the cutoff's label delay, user ids that sort differently as strings, and a
sample smaller than the book.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import timedelta
from pathlib import Path
from random import Random

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

import tfm.runtime.data as runtime_data
from tfm.config import DataConfig, RecipeConfig
from tfm.contracts.errors import SchemaError, UnsupportedCombinationError
from tfm.data.adapters.tabformer import load_tabformer_adapter, precomputed_sample
from tfm.data.collate import collate_batch, minibatches, raw_feature_vector, shuffled_rows
from tfm.data.sequence_builders.entity_sequences import build_entity_windows
from tfm.data.token_cache.flat import (
    CachedFlatBatches,
    FlatCacheRequest,
    FlatRows,
    FlatTokenCache,
    build_flat_cache,
)
from tfm.features.tokenizers.financial import TransactionFieldTokenizer
from tfm.runtime.data import PretrainingData, tokenizer_for
from tfm.runtime.pipeline import RunArtifacts, run_train, setup_run, training_batches
from tfm.runtime.token_cache import build_recipe_cache

FIELDS = ("AMT", "MERCH", "CAT", "MCC", "HOUR", "DOW", "MONTH", "CARD", "CHIP", "ZIP3", "STATE")
SCHEMA = pa.schema(
    [
        pa.field("User", pa.int64()),
        pa.field("Card", pa.int64()),
        pa.field("Year", pa.int64()),
        pa.field("Month", pa.int64()),
        pa.field("Day", pa.int64()),
        pa.field("Time", pa.string()),
        pa.field("Amount", pa.string()),
        pa.field("Use Chip", pa.string()),
        pa.field("Merchant Name", pa.int64()),
        pa.field("Merchant City", pa.string()),
        pa.field("Merchant State", pa.string()),
        pa.field("Zip", pa.float64()),
        pa.field("MCC", pa.int64()),
        pa.field("Is Fraud?", pa.string()),
    ]
)


def _rows(rng: Random, users: range, years: tuple[int, ...], n: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for user in users:
        for _ in range(n):
            online = rng.random() < 0.2
            rows.append(
                {
                    "User": user,
                    "Card": rng.randrange(4),
                    "Year": rng.choice(years),
                    "Month": rng.randint(1, 12),
                    # Few days and hours, so many events share a timestamp.
                    "Day": rng.randint(1, 3),
                    "Time": f"{rng.choice((8, 9, 23)):02d}:{rng.randrange(60):02d}",
                    "Amount": f"${rng.uniform(-20, 3000):,.2f}",
                    "Use Chip": "Online Transaction" if online else "Chip Transaction",
                    "Merchant Name": rng.choice((-727612092139916043, 3527213246127876953, 12)),
                    "Merchant City": "ONLINE" if online else "La Verne",
                    "Merchant State": None if online else rng.choice(("CA", "NY", "Italy")),
                    "Zip": None if online else float(rng.choice((91750, 1001, 501))),
                    "MCC": rng.choice((5411, 5541, 4829, 1234)),
                    "Is Fraud?": "Yes" if rng.random() < 0.15 else "No",
                }
            )
    rng.shuffle(rows)  # rows out of time and user order
    return rows


def _write_book(root: Path) -> Path:
    rng = Random(7)
    split_dir = root / "temporal_split"
    split_dir.mkdir(parents=True)
    for name, years, n in (
        ("train.parquet", (2010, 2011), 40),
        ("val.parquet", (2016,), 5),
        ("test.parquet", (2018,), 5),
    ):
        rows = _rows(rng, range(14), years, n)
        table = pa.table({f.name: pa.array([r[f.name] for r in rows], f.type) for f in SCHEMA})
        pq.write_table(table, split_dir / name)
    return root


def _data_config(root: Path, *, max_entities: int = 9) -> DataConfig:
    return DataConfig.model_validate(
        {
            "adapter": "tabformer",
            "path": str(root),
            "split_protocol": "precomputed",
            "observation_window": "300d",
            "n_entities": max_entities,
            "max_entities": max_entities,
            "temporal_split": {
                "train_end": "2016-01-01T00:00:00Z",
                "validation_end": "2018-01-01T00:00:00Z",
                "label_delay": "30d",
            },
        }
    )


def _request(root: Path, data: DataConfig, seed: int = 3) -> FlatCacheRequest:
    users = precomputed_sample(root, cap=int(data.max_entities or data.n_entities), seed=seed)
    return FlatCacheRequest(
        root=root,
        users=tuple(users),
        label_delay=timedelta(days=30),
        tokenizer=TransactionFieldTokenizer(fields=FIELDS),
        amount_field="amount",
    )


@pytest.fixture(scope="module")
def book(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, FlatTokenCache]:
    root = _write_book(tmp_path_factory.mktemp("tabformer"))
    out = root / "cache"
    build_flat_cache(
        _request(root, _data_config(root)), out, workers=1, rows_per_block=97, progress=False
    )
    return root, FlatTokenCache(out)


def _assert_batches_equal(a: object, b: object) -> None:
    for name in (
        "input_ids",
        "attention_mask",
        "labels",
        "fraud_labels",
        "numeric_features",
    ):
        left, right = getattr(a, name), getattr(b, name)
        assert left.dtype == right.dtype, name
        assert torch.equal(left, right), name
    for name in ("entity_ids", "sequence_ids", "event_ids"):
        assert getattr(a, name) == getattr(b, name), name


@pytest.mark.parametrize(
    ("window_events", "stride", "max_tokens", "batch_size", "shuffle"),
    [
        (5, 3, 12 * 5 + 1, 4, 7),  # overlapping windows, rows fit
        (6, None, 40, 3, None),  # tiling windows, rows truncated to whole events
        (500, None, 12 * 500 + 1, 2, 0),  # one window per user, trailing partial windows
    ],
)
def test_cached_minibatches_equal_the_object_path(
    book: tuple[Path, FlatTokenCache],
    window_events: int,
    stride: int | None,
    max_tokens: int,
    batch_size: int,
    shuffle: int | None,
) -> None:
    root, cache = book
    tokenizer = TransactionFieldTokenizer(fields=FIELDS)
    train = load_tabformer_adapter(root, _data_config(root), seed=3).splits
    assert train is not None
    cutoff = max(e.event_timestamp for e in train.train)
    lookback = timedelta(days=300)
    windows = build_entity_windows(
        train.train, cutoff, window_events=window_events, stride=stride, observation_window=lookback
    )
    batch = collate_batch(windows, tokenizer, max_tokens=max_tokens)
    if shuffle is not None:
        batch = shuffled_rows(batch, shuffle)
    expected = minibatches(batch, batch_size)

    assert cache.train_cutoff() == cutoff
    rows = cache.train_windows(
        cutoff=cutoff, lookback=lookback, window_events=window_events, stride=stride
    )
    cached = CachedFlatBatches(
        cache, rows, tokenizer, batch_size=batch_size, max_length=max_tokens, shuffle_seed=shuffle
    )
    assert len(rows) == len(windows)
    assert len(cached) == len(expected)
    for got, want in zip(cached, expected, strict=True):
        _assert_batches_equal(got, want)


def test_the_book_exercises_ties_truncation_and_both_label_sides(
    book: tuple[Path, FlatTokenCache],
) -> None:
    root, cache = book
    train = load_tabformer_adapter(root, _data_config(root), seed=3).splits
    assert train is not None
    stamps = [(e.entity_id, e.event_timestamp) for e in train.train]
    assert len(set(stamps)) < len(stamps), "the book needs tied timestamps"
    indices = {int(e.event_id.rsplit("-", 1)[1]) for e in train.train}
    assert min(indices) < 10 <= max(indices), "event ids must sort differently as strings"
    cutoff = max(e.event_timestamp for e in train.train)
    available = [e.labels["fraud"].available_at <= cutoff for e in train.train]
    assert any(available) and not all(available)
    assert len(cache.user_ids) == 9 < 14, "the cache holds the sample, not the book"


def test_a_cache_for_another_sample_or_tokenizer_is_refused(
    book: tuple[Path, FlatTokenCache],
) -> None:
    root, cache = book
    cache.check_identity(_request(root, _data_config(root)).identity())
    with pytest.raises(SchemaError, match="population"):
        cache.check_identity(_request(root, _data_config(root, max_entities=5)).identity())
    other = FlatCacheRequest(
        root=root,
        users=_request(root, _data_config(root)).users,
        label_delay=timedelta(days=30),
        tokenizer=TransactionFieldTokenizer(fields=FIELDS[:-1]),
        amount_field="amount",
    )
    with pytest.raises(SchemaError, match="tokenizer"):
        cache.check_identity(other.identity())


def test_another_seed_that_samples_the_same_users_reads_the_cache(
    book: tuple[Path, FlatTokenCache],
) -> None:
    # A cap above the book's size samples every user whatever the seed, as
    # blueprint_v1/config.yaml's cap of 2,000 does on TabFormer; a seed sweep must not
    # need one cache per seed.
    root, _cache = book
    everyone = _data_config(root, max_entities=50)
    first, second = _request(root, everyone, seed=3), _request(root, everyone, seed=99)
    assert first.identity() == second.identity()
    assert first.identity() != _request(root, _data_config(root), seed=3).identity()


def test_workers_build_the_same_cache(book: tuple[Path, FlatTokenCache], tmp_path: Path) -> None:
    root, cache = book
    out = tmp_path / "cache"
    build_flat_cache(
        _request(root, _data_config(root)), out, workers=2, rows_per_block=61, progress=False
    )
    parallel = FlatTokenCache(out)
    for name in ("tokens", "ts_us", "amount", "fraud", "fraud_available_us", "event_id"):
        assert (getattr(parallel, name) == getattr(cache, name)).all(), name
    assert parallel.user_ids == cache.user_ids


# The recipe lifecycle with `data.token_cache` --------------------------------


def test_a_book_without_the_amount_field_is_refused(
    book: tuple[Path, FlatTokenCache], tmp_path: Path
) -> None:
    root, _ = book
    request = dataclasses.replace(_request(root, _data_config(root)), amount_field="balance")
    with pytest.raises(SchemaError, match="data.amount_field"):
        build_flat_cache(request, tmp_path / "cache", workers=1, progress=False)


def _recipe_config(root: Path, *, token_cache: Path | None, **training: object) -> RecipeConfig:
    return RecipeConfig.model_validate(
        {
            "seed": 3,
            "data": {
                **_data_config(root).model_dump(mode="json"),
                "sequence_length": 4,
                "max_tokens": 49,
                "pretrain_windows": {"events": 5, "max_tokens": 61},
                "token_cache": None if token_cache is None else str(token_cache),
            },
            "features": {"tokenizer": "transaction_field_tokenizer", "fields": list(FIELDS)},
            "model": {
                "backbone": "eager",
                "composition": {
                    "position": "nope",
                    "granularity": {
                        "decoder": {
                            "depth": 1,
                            "hidden_size": 16,
                            "num_heads": 2,
                            "max_seq_len": 61,
                            "dropout": 0.0,
                            "mask": "causal",
                        }
                    },
                },
            },
            "training": {"max_steps": 3, "batch_size": 4, "shuffle": True, **training},
            "runtime": {"device": "cpu"},
        }
    )


@pytest.fixture(scope="module")
def recipe_cache(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = _write_book(tmp_path_factory.mktemp("tabformer_recipe"))
    out = root / "recipe_cache"
    config = _recipe_config(root, token_cache=out)
    build_recipe_cache(config, tokenizer_for(config), workers=1)
    return root, out


def test_the_config_names_what_a_flat_cache_cannot_serve(tmp_path: Path) -> None:
    _recipe_config(tmp_path, token_cache=tmp_path / "cache")
    with pytest.raises(ValueError, match=r"training\.packing"):
        _recipe_config(tmp_path, token_cache=tmp_path / "cache", packing=True)
    raw = _recipe_config(tmp_path, token_cache=tmp_path / "cache").model_dump(mode="json")
    raw["data"]["split_protocol"] = "timestamp"
    raw["data"]["pretrain_windows"] = None
    with pytest.raises(ValueError, match="precomputed") as error:
        RecipeConfig.model_validate(raw)
    assert "set data.pretrain_windows" in str(error.value)
    raw = _recipe_config(tmp_path, token_cache=tmp_path / "cache").model_dump(mode="json")
    raw["data"]["batch_builder"] = "tests.fixtures.user_tokenizer:placement_batches"
    with pytest.raises(ValueError, match="data.batch_builder reads columnar token caches"):
        RecipeConfig.model_validate(raw)


def _all_batches(artifacts: RunArtifacts) -> list[object]:
    batches, _stats = training_batches(artifacts)
    return [batches[i] for i in range(len(batches))]


def test_setup_reads_the_object_paths_pretraining_batches_from_the_cache(
    recipe_cache: tuple[Path, Path], tmp_path: Path
) -> None:
    root, out = recipe_cache
    reference = setup_run(config=_recipe_config(root, token_cache=None), run_dir=tmp_path / "a")
    cached = setup_run(config=_recipe_config(root, token_cache=out), run_dir=tmp_path / "b")
    assert isinstance(cached.data.pretrain_rows, FlatRows)
    pretraining_only = setup_run(
        config=_recipe_config(root, token_cache=out),
        run_dir=tmp_path / "c",
        pretraining_only=True,
    )
    expected = _all_batches(reference)
    for got in (_all_batches(cached), _all_batches(pretraining_only)):
        assert len(got) == len(expected)
        for left, right in zip(got, expected, strict=True):
            _assert_batches_equal(left, right)


def test_pretraining_only_loads_no_events_and_trains(
    recipe_cache: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, out = recipe_cache

    def refuse(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("pretraining from a flat cache must not load the book")

    monkeypatch.setattr(runtime_data, "load_adapter_events", refuse)
    artifacts = setup_run(
        config=_recipe_config(root, token_cache=out),
        run_dir=tmp_path / "run",
        pretraining_only=True,
    )
    assert isinstance(artifacts.data, PretrainingData)
    with pytest.raises(UnsupportedCombinationError, match="pretrain only"):
        _ = artifacts.train_batch
    stats = run_train(artifacts)
    assert stats["steps"] == 3.0
    assert (tmp_path / "run" / "model.pt").exists()
    split = json.loads((tmp_path / "run" / "split_manifest.json").read_text(encoding="utf-8"))
    assert split["pretraining_only"] is True
    assert split["train_cutoff"] == artifacts.data.train_cutoff.isoformat()


def test_a_run_refuses_a_cache_built_for_another_sample(
    recipe_cache: tuple[Path, Path], tmp_path: Path
) -> None:
    root, out = recipe_cache
    raw = _recipe_config(root, token_cache=out).model_dump(mode="json")
    raw["data"]["max_entities"] = raw["data"]["n_entities"] = 5
    with pytest.raises(SchemaError, match="population"):
        setup_run(config=RecipeConfig.model_validate(raw), run_dir=tmp_path, pretraining_only=True)


def test_flat_cache_population_identity_preserves_delimiters_and_set_semantics(book):
    root, _ = book
    request = _request(root, _data_config(root))
    a = dataclasses.replace(request, users=("a", "b\nc"))
    b = dataclasses.replace(request, users=("a\nb", "c"))
    assert a.identity()["population"] != b.identity()["population"]
    assert a.identity() == dataclasses.replace(a, users=("b\nc", "a", "a")).identity()


def test_cached_raw_features_mask_nonfinite_values_without_mutating_arrays(book, monkeypatch):
    root, cache = book
    train = load_tabformer_adapter(root, _data_config(root), seed=3).splits
    assert train is not None
    amounts = np.array(cache.amount, copy=True)
    amounts[::5], amounts[1::5], amounts[2::5] = np.nan, np.inf, -np.inf
    before = amounts.copy()
    amounts.setflags(write=False)
    monkeypatch.setattr(cache, "amount", amounts)
    by_id = {
        bytes(key).decode("ascii"): value
        for key, value in zip(cache.event_id, amounts.tolist(), strict=True)
    }
    events = [
        dataclasses.replace(event, numeric={**event.numeric, "amount": by_id[event.event_id]})
        for event in train.train
    ]
    cutoff = max(event.event_timestamp for event in events)
    lookback = timedelta(days=300)
    windows = build_entity_windows(
        events, cutoff, window_events=5, stride=3, observation_window=lookback
    )
    rows = cache.train_windows(cutoff=cutoff, lookback=lookback, window_events=5, stride=3)
    batches = CachedFlatBatches(
        cache,
        rows,
        TransactionFieldTokenizer(fields=FIELDS),
        batch_size=4,
        max_length=65,
        shuffle_seed=None,
    )
    actual = torch.cat([batch.numeric_features for batch in batches])
    expected = torch.tensor([raw_feature_vector(window) for window in windows])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.isfinite(actual).all()
    np.testing.assert_array_equal(amounts, before)
    assert not amounts.flags.writeable
