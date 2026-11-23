# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The token cache reproduces the object path exactly (ADR 0016).

Every comparison here is against `load_pragmatiq_dataset`, `build_entity_*`,
`KeyValueTimeTokenizer.fit`, and `collate_batch`: the cache is a faster route
to the same vocabulary and the same batches, never a different one.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from random import Random
from typing import Any

import pyarrow.parquet as pq
import pytest
import torch

from tfm.config import RecipeConfig, load_recipe_config
from tfm.contracts.errors import SchemaError, UnsupportedCombinationError
from tfm.data.adapters.pragmatiq import (
    events_table_from_records,
    load_pragmatiq_dataset,
    profiles_table_from_records,
)
from tfm.data.collate import collate_batch, minibatches, shuffled_rows
from tfm.data.sequence_builders.entity_sequences import (
    build_entity_sequences,
    build_entity_windows,
)
from tfm.data.splits.temporal import TemporalSplitSpec, assign_temporal_split
from tfm.data.token_cache import (
    CachedBatches,
    CacheRequest,
    FitSpec,
    TokenCache,
    build_token_cache,
)
from tfm.features.tokenizers.key_value_time import KeyValueTimeConfig, KeyValueTimeTokenizer
from tfm.training.distributed import shard_batches

LAB_CONFIG = Path(__file__).parents[2] / "benchmarks" / "ablations" / "blueprint_v2_lab.yaml"
TRAIN_END = datetime(2024, 3, 1, tzinfo=UTC)
VALIDATION_END = datetime(2024, 4, 1, tzinfo=UTC)
FIT_WINDOW = timedelta(days=45)
SEQUENCE_LENGTH = 8
CONFIG = KeyValueTimeConfig(
    max_events_per_user=6,
    max_event_tokens=12,
    max_profile_tokens=5,
    categorical_threshold=6,
    numeric_min_cardinality=4,
    categorical_keys=("mcc",),
)
MERCHANTS = tuple(f"SHOP {i}" for i in range(12))
"""More distinct values than `categorical_threshold`, so `merchant` becomes text."""


def _book(root: Path, *, seed: int = 0, late_value: str = "x") -> Path:
    """A synthetic book with every case the column-wise reader must get right."""
    rng = Random(seed)
    start = datetime(2024, 1, 1, tzinfo=UTC)
    rows: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []
    for u in range(9):
        user = f"u{u:02d}"
        mine: list[dict[str, Any]] = []
        for i in range(24 + u):
            ts = start + timedelta(hours=rng.randrange(0, 120 * 24), microseconds=rng.randrange(7))
            if rng.random() < 0.35:
                source = "app"
                fields: dict[str, Any] = {
                    "screen": rng.choice(("home", "pay", "cards")),
                    "app_version": rng.choice(("10.1", "10.10", "9.2", "11")),
                }
            else:
                source = "transaction"
                fields = {
                    "amount": rng.choice(("12.50", "7", "10.10", "3.3", "250", "0", "41.99")),
                    "mcc": rng.choice(("5411", "4111", "5812")),
                    "merchant": rng.choice(MERCHANTS),
                    "channel": rng.choice(("chip", "online")),
                }
                if i % 5 == 0:
                    fields["memo"] = "m" * 70 + str(i)
                if i % 7 == 0:
                    fields["channel"] = None
            if ts >= TRAIN_END and i % 4 == 0:
                fields["late_key"] = late_value
            mine.append({"user_id": user, "ts": ts, "source": source, "fields": fields})
        if u == 3:
            # First in the history, so the tied ids run 0..11 and "10" sorts before "2".
            tied = start - timedelta(days=1)
            for i in range(12):
                mine.append(
                    {
                        "user_id": user,
                        "ts": tied,
                        "source": "transaction",
                        "fields": {"amount": str(i), "merchant": MERCHANTS[i], "mcc": "5411"},
                    }
                )
        if u == 4:
            mine.append(
                {"user_id": user, "ts": TRAIN_END, "source": "app", "fields": {"screen": "edge"}}
            )
        rng.shuffle(mine)
        rows.extend(mine)
        if u != 5:
            profiles.append(
                {
                    "user_id": user,
                    "as_of": datetime(2024, 6, 1, tzinfo=UTC),
                    "attributes": {"plan": rng.choice(("plus", "basic")), "country": "GB"},
                    "lifelong": [
                        ("account_opened", datetime(2023, 5, 1, tzinfo=UTC)),
                        ("card_upgraded", TRAIN_END + timedelta(days=3)),
                    ],
                }
            )
    root.mkdir(parents=True)
    table = events_table_from_records(
        [{**r, "fields": {k: v for k, v in r["fields"].items()}} for r in rows]
    )
    pq.write_table(table, root / "events.parquet", row_group_size=7)
    pq.write_table(profiles_table_from_records(profiles), root / "profiles.parquet")
    return root


def _legacy(
    root: Path, users: set[str] | None = None, config: KeyValueTimeConfig = CONFIG
) -> dict[str, Any]:
    dataset = load_pragmatiq_dataset(root, attach_fraud_labels=False)
    events = [e for e in dataset.events if users is None or e.entity_id in users]
    splits = assign_temporal_split(
        events, TemporalSplitSpec(TRAIN_END, VALIDATION_END, timedelta(0))
    )
    tokenizer = KeyValueTimeTokenizer(config, {p.entity_id: p for p in dataset.profiles})
    train_seq = build_entity_sequences(
        splits.train, TRAIN_END, max_events=SEQUENCE_LENGTH, observation_window=FIT_WINDOW
    )
    tokenizer.fit(train_seq)
    return {"events": events, "splits": splits, "tokenizer": tokenizer, "train_seq": train_seq}


def _build(
    root: Path,
    out: Path,
    users: tuple[str, ...] | None = None,
    config: KeyValueTimeConfig = CONFIG,
) -> TokenCache:
    request = CacheRequest(
        root=root,
        users=users,
        fit=FitSpec(TRAIN_END, FIT_WINDOW, SEQUENCE_LENGTH),
        tokenizer=config,
        population={"users": "all" if users is None else len(users)},
    )
    build_token_cache(request, out, workers=1, rows_per_block=40, progress=False)
    return TokenCache(out)


def _assert_batches_equal(got: Any, want: Any) -> None:
    for field in dataclasses.fields(want):
        a, b = getattr(got, field.name), getattr(want, field.name)
        if isinstance(b, torch.Tensor):
            assert isinstance(a, torch.Tensor), field.name
            assert a.shape == b.shape, (field.name, a.shape, b.shape)
            assert a.dtype == b.dtype, field.name
            assert torch.equal(a, b), field.name
        else:
            assert a == b, field.name


@pytest.fixture(scope="module")
def book(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _book(tmp_path_factory.mktemp("book") / "book")


@pytest.fixture(scope="module")
def cache(book: Path, tmp_path_factory: pytest.TempPathFactory) -> TokenCache:
    return _build(book, tmp_path_factory.mktemp("cache") / "cache")


@pytest.fixture(scope="module")
def legacy(book: Path) -> dict[str, Any]:
    return _legacy(book)


def test_the_fixture_exercises_the_hard_cases(legacy: dict[str, Any]) -> None:
    tokenizer = legacy["tokenizer"]
    assert tokenizer.field_kind["merchant"] == "text"
    assert "memo" in tokenizer.field_kind
    assert any("memo" in e.text for e in legacy["events"])
    assert tokenizer.field_kind["mcc"] == "categorical"
    assert tokenizer.field_kind["amount"] == "numeric"
    assert "late_key" not in tokenizer.field_kind
    assert "10.10" not in tokenizer.cat_vocab["app_version"]
    tied = [e for e in legacy["events"] if e.entity_id == "u03"]
    assert len({e.event_timestamp for e in tied}) < len(tied)


def test_fit_from_cache_statistics_matches_fit_on_objects(
    cache: TokenCache, legacy: dict[str, Any]
) -> None:
    assert cache.tokenizer_state == legacy["tokenizer"].state_dict()
    assert cache.manifest["n_events"] == len(legacy["events"])


def test_worker_processes_build_the_same_cache(
    book: Path, cache: TokenCache, tmp_path: Path
) -> None:
    request = CacheRequest(
        root=book,
        users=None,
        fit=FitSpec(TRAIN_END, FIT_WINDOW, SEQUENCE_LENGTH),
        tokenizer=CONFIG,
        population={"users": "all"},
    )
    build_token_cache(request, tmp_path / "cache", workers=2, rows_per_block=40, progress=False)
    parallel = TokenCache(tmp_path / "cache")
    assert parallel.tokenizer_state == cache.tokenizer_state
    for name in ("ts_us", "id_us", "by_loader", "token_offsets", "key_ids", "value_ids"):
        assert (getattr(parallel, name) == getattr(cache, name)).all(), name


def test_fit_on_a_user_subset_matches(book: Path, tmp_path: Path) -> None:
    users = ("u01", "u03", "u04", "u07")
    cache = _build(book, tmp_path / "cache", users)
    assert cache.user_ids == users
    assert cache.tokenizer_state == _legacy(book, set(users))["tokenizer"].state_dict()


def test_events_after_train_end_never_change_the_vocabulary(tmp_path: Path) -> None:
    a = _build(_book(tmp_path / "a", late_value="x"), tmp_path / "ca")
    b = _build(_book(tmp_path / "b", late_value="y" * 90), tmp_path / "cb")
    assert a.tokenizer_state == b.tokenizer_state
    assert "late_key" not in a.tokenizer_state["key_vocab"]


def test_train_rows_collate_like_collate_batch(cache: TokenCache, legacy: dict[str, Any]) -> None:
    tokenizer = cache.tokenizer(legacy["tokenizer"]._profiles)
    rows = cache.train_rows(train_end=TRAIN_END, lookback=FIT_WINDOW, max_events=SEQUENCE_LENGTH)
    want = collate_batch(legacy["train_seq"], legacy["tokenizer"], max_tokens=10)
    _assert_batches_equal(cache.collate(rows, tokenizer, max_length=10), want)


def test_a_hashed_key_caches_like_the_object_path(book: Path, tmp_path: Path) -> None:
    config = dataclasses.replace(CONFIG, hashed_keys=("merchant",), hash_buckets=16)
    cache = _build(book, tmp_path / "cache", config=config)
    legacy = _legacy(book, config=config)
    assert legacy["tokenizer"].field_kind["merchant"] == "hashed"
    assert cache.tokenizer_state == legacy["tokenizer"].state_dict()
    assert cache.manifest["tokenizer_config"]["hashed_keys"] == ["merchant"]
    tokenizer = cache.tokenizer(legacy["tokenizer"]._profiles)
    rows = cache.train_rows(train_end=TRAIN_END, lookback=FIT_WINDOW, max_events=SEQUENCE_LENGTH)
    want = collate_batch(legacy["train_seq"], legacy["tokenizer"], max_tokens=10)
    _assert_batches_equal(cache.collate(rows, tokenizer, max_length=10), want)


def test_an_unhashed_cache_identity_has_no_hash_settings(cache: TokenCache) -> None:
    assert "hashed_keys" not in cache.manifest["tokenizer_config"]
    assert "hash_buckets" not in cache.manifest["tokenizer_config"]


@pytest.mark.parametrize(("window_events", "stride", "lookback"), [(5, 3, 60), (4, None, None)])
def test_pretraining_windows_match_shuffled_minibatches(
    cache: TokenCache,
    legacy: dict[str, Any],
    window_events: int,
    stride: int | None,
    lookback: int | None,
) -> None:
    look = None if lookback is None else timedelta(days=lookback)
    corpus = build_entity_windows(
        legacy["splits"].train,
        TRAIN_END,
        window_events=window_events,
        stride=stride,
        observation_window=look,
    )
    want = minibatches(
        shuffled_rows(collate_batch(corpus, legacy["tokenizer"], max_tokens=9), seed=7), 4
    )
    rows = cache.train_windows(
        train_end=TRAIN_END, lookback=look, window_events=window_events, stride=stride
    )
    got = CachedBatches(
        cache,
        rows,
        cache.tokenizer(legacy["tokenizer"]._profiles),
        batch_size=4,
        max_length=9,
        shuffle_seed=7,
    )
    assert len(rows) == len(corpus)
    assert len(got) == len(want)
    for g, w in zip(got, want, strict=True):
        _assert_batches_equal(g, w)


def test_probe_encode_reads_tokens_by_event_id(cache: TokenCache, legacy: dict[str, Any]) -> None:
    tokenizer = cache.tokenizer(legacy["tokenizer"]._profiles)
    last = max(e.event_timestamp for e in legacy["events"])
    for cutoff in (VALIDATION_END, last):
        for seq in build_entity_sequences(legacy["events"], cutoff, max_events=SEQUENCE_LENGTH):
            assert tokenizer.encode(seq) == legacy["tokenizer"].encode(seq)


def test_a_copied_event_needs_an_alias(cache: TokenCache, legacy: dict[str, Any]) -> None:
    tokenizer = cache.tokenizer(legacy["tokenizer"]._profiles)
    seq = build_entity_sequences(legacy["events"], VALIDATION_END, max_events=3)[0]
    donor = seq.events[-1]
    copy = dataclasses.replace(donor, entity_id="u08", event_id="u08:1:theft0")
    spliced = dataclasses.replace(seq, events=(*seq.events[:-1], copy))
    with pytest.raises(SchemaError, match="not in token cache"):
        tokenizer.encode(spliced)
    tokenizer.add_aliases({copy.event_id: donor.event_id})
    assert tokenizer.encode(spliced).key_ids == legacy["tokenizer"].encode(seq).key_ids


def test_a_cached_tokenizer_is_not_refit(cache: TokenCache, legacy: dict[str, Any]) -> None:
    with pytest.raises(UnsupportedCombinationError):
        cache.tokenizer().fit(legacy["train_seq"])


def test_a_cache_checks_what_it_was_built_from(cache: TokenCache) -> None:
    cache.check_identity({"fit": cache.manifest["fit"]})
    with pytest.raises(SchemaError, match="Rebuild"):
        cache.check_identity({"fit": {**cache.manifest["fit"], "max_events": 99}})


def test_sharding_reads_no_batch_until_it_is_indexed() -> None:
    class Counting(Sequence[int]):
        reads = 0

        def __len__(self) -> int:
            return 10

        def __getitem__(self, i: Any) -> Any:
            Counting.reads += 1
            return i

    shard = shard_batches(Counting(), world_size=4, rank=1)
    assert Counting.reads == 0
    assert list(shard) == [1, 5]


@pytest.mark.parametrize(
    ("change", "problem"),
    [
        ({"fraud_labels": "native"}, "own fraud labels"),
        ({"split_protocol": "precomputed"}, "timestamp split"),
    ],
)
def test_a_config_refuses_what_a_cache_cannot_reproduce(
    change: dict[str, str], problem: str
) -> None:
    raw = load_recipe_config(LAB_CONFIG).model_dump(mode="json")
    raw["data"]["token_cache"] = "/nonexistent/cache"
    RecipeConfig.model_validate(raw)
    raw["data"].update(change)
    with pytest.raises(ValueError, match=problem):
        RecipeConfig.model_validate(raw)


def test_an_unsorted_book_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "book"
    root.mkdir()
    ts = datetime(2024, 1, 2, tzinfo=UTC)
    rows = [
        {"user_id": u, "ts": ts, "source": "app", "fields": {"screen": "home"}}
        for u in ("b", "a", "b")
    ]
    pq.write_table(events_table_from_records(rows), root / "events.parquet")
    with pytest.raises(SchemaError, match="not sorted"):
        _build(root, tmp_path / "cache")
