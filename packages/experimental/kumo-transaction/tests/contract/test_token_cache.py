# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The token cache reproduces the object path exactly (token-cache guide).

Every comparison here is against `load_pragmatiq_dataset`, `build_entity_*`,
`KeyValueTimeTokenizer.fit`, and `collate_batch`: the cache is a faster route
to the same vocabulary and the same batches, never a different one.
"""

from __future__ import annotations

import ast
import contextlib
import copy
import dataclasses
import functools
import gc
import io
import json
import os
import pickle
import shutil
import subprocess
import sys
import warnings
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from random import Random
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import yaml

import tfm.data.token_cache.build as build_module
import tfm.data.token_cache.store as store_module
from tfm.config import RecipeConfig, load_recipe_config
from tfm.contracts.errors import (
    IncompatibleCompositionError,
    SchemaError,
    TfmError,
    UnsupportedCombinationError,
)
from tfm.contracts.tokenizers import (
    FIELD_CATEGORICAL,
    FIELD_NUMERIC,
    EncodedBlock,
    EventBlock,
)
from tfm.data.adapters.pragmatiq import (
    events_table_from_records,
    load_pragmatiq_dataset,
    profiles_table_from_records,
)
from tfm.data.adapters.pragmatiq_columns import PragmatiqColumns, plan_blocks, read_block
from tfm.data.collate import collate_batch, minibatches, raw_feature_vector, shuffled_rows
from tfm.data.sequence_builders.entity_sequences import (
    build_entity_sequences,
    build_entity_windows,
)
from tfm.data.splits.temporal import TemporalSplitSpec, assign_temporal_split, parse_timestamp
from tfm.data.synthetic.lab_book import generate_lab_book
from tfm.data.token_cache import (
    CachedBatches,
    CacheRequest,
    CacheRows,
    DeviceBatches,
    FitSpec,
    TokenCache,
    build_token_cache,
    fit_statistics_block,
)
from tfm.data.token_cache.build import encode_part
from tfm.data.token_cache.builders import DEFAULT_BATCH_BUILDER, batch_builder
from tfm.data.token_cache.columnar import cache_request
from tfm.data.token_cache.profile import (
    fit_block_seconds,
    format_profile,
    makespan,
    profile_token_cache,
)
from tfm.features.tokenizers import build_tokenizer, tokenizer_class
from tfm.features.tokenizers.key_value_time import KeyValueTimeConfig, KeyValueTimeTokenizer
from tfm.features.tokenizers.vocabulary import UNK, KeyValueStats
from tfm.runtime.data import prepare_data, tokenizer_for
from tfm.runtime.pipeline import setup_run, training_batches
from tfm.runtime.token_cache import build_recipe_cache, open_recipe_cache
from tfm.testing.conformance import ConformanceError, check_columnar_tokenizer
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


def _book(root: Path, *, seed: int = 0, late_value: str = "x", time_sorted: bool = False) -> Path:
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
                    "amount": rng.choice(
                        ("12.50", "7", "10.10", "3.3", "250", "0", "-0.0", "41.99")
                    ),
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
        if time_sorted:
            mine.sort(key=lambda r: r["ts"])
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
    profiles = {p.entity_id: p for p in dataset.profiles}
    tokenizer = KeyValueTimeTokenizer(config, profiles)
    train_seq = build_entity_sequences(
        splits.train, TRAIN_END, max_events=SEQUENCE_LENGTH, observation_window=FIT_WINDOW
    )
    tokenizer.fit(train_seq)
    return {
        "events": events,
        "splits": splits,
        "tokenizer": tokenizer,
        "train_seq": train_seq,
        "profiles": profiles,
    }


def _build(
    root: Path,
    out: Path,
    users: tuple[str, ...] | None = None,
    config: KeyValueTimeConfig = CONFIG,
    rows_per_block: int = 40,
) -> TokenCache:
    build_token_cache(
        _request(root, users, config), out, workers=1, rows_per_block=rows_per_block, progress=False
    )
    return TokenCache(out)


def _request(
    root: Path, users: tuple[str, ...] | None = None, config: KeyValueTimeConfig = CONFIG
) -> CacheRequest:
    return CacheRequest(
        source=PragmatiqColumns(root),
        users=users,
        fit=FitSpec(TRAIN_END, FIT_WINDOW, SEQUENCE_LENGTH),
        tokenizer=KeyValueTimeTokenizer(config),
        population={"users": "all" if users is None else len(users)},
        amount_field="amount",
    )


def _assert_tensors_equal(a: Any, b: torch.Tensor, name: str) -> None:
    assert isinstance(a, torch.Tensor), name
    assert a.shape == b.shape, (name, a.shape, b.shape)
    assert a.dtype == b.dtype, name
    assert torch.equal(a, b), name


def _assert_batches_equal(got: Any, want: Any) -> None:
    for field in dataclasses.fields(want):
        a, b = getattr(got, field.name), getattr(want, field.name)
        if isinstance(b, torch.Tensor):
            _assert_tensors_equal(a, b, field.name)
        elif isinstance(b, dict):
            assert list(a) == list(b), field.name
            for name in b:
                _assert_tensors_equal(a[name], b[name], f"{field.name}.{name}")
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
    assert tokenizer.vocabulary.field_kind["merchant"] == "text"
    assert "memo" in tokenizer.vocabulary.field_kind
    assert any("memo" in e.text for e in legacy["events"])
    assert tokenizer.vocabulary.field_kind["mcc"] == "categorical"
    assert tokenizer.vocabulary.field_kind["amount"] == "numeric"
    assert "late_key" not in tokenizer.vocabulary.field_kind
    assert "10.10" not in tokenizer.vocabulary.cat_vocab["app_version"]
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
        source=PragmatiqColumns(book),
        users=None,
        fit=FitSpec(TRAIN_END, FIT_WINDOW, SEQUENCE_LENGTH),
        tokenizer=KeyValueTimeTokenizer(CONFIG),
        population={"users": "all"},
        amount_field="amount",
    )
    build_token_cache(request, tmp_path / "cache", workers=2, rows_per_block=40, progress=False)
    parallel = TokenCache(tmp_path / "cache")
    assert parallel.tokenizer_state == cache.tokenizer_state
    for part in sorted((cache.path / "parts").iterdir()):
        assert (tmp_path / "cache" / "parts" / part.name).read_bytes() == part.read_bytes()
    for name in ("ts_us", "id_us", "by_loader", "token_offsets", "key_ids", "value_ids"):
        assert np.array_equal(getattr(parallel, name), getattr(cache, name)), name


@dataclasses.dataclass(frozen=True)
class _HeldBlock:
    index: int
    n_rows: int


@dataclasses.dataclass(frozen=True)
class _HeldPlan:
    blocks: tuple[_HeldBlock, ...]
    n_users: int
    n_rows: int
    longest_user_id: int


@dataclasses.dataclass(frozen=True)
class HeldBlocks:
    """A `ColumnarSource` other than the pragmatiq reader: blocks read once, then held."""

    book: Path
    held: tuple[EventBlock, ...]

    def fingerprint(self) -> dict[str, Any]:
        return PragmatiqColumns(self.book).fingerprint()

    def plan(self, *, rows_per_block: int, users: Any = None) -> _HeldPlan:
        assert users is None
        return _HeldPlan(
            tuple(_HeldBlock(b.index, len(b.ts_us)) for b in self.held),
            sum(len(b.user_ids) for b in self.held),
            sum(len(b.ts_us) for b in self.held),
            max(len(u.encode()) for b in self.held for u in b.user_ids),
        )

    def read_block(self, block: _HeldBlock, *, users: Any = None) -> EventBlock:
        return self.held[block.index]

    def write_profiles(self, users: Any, out: Path) -> bool:
        shutil.copyfile(self.book / "profiles.parquet", out)
        return True


def test_any_columnar_source_builds_the_same_cache(
    book: Path, cache: TokenCache, tmp_path: Path
) -> None:
    held = HeldBlocks(
        book, tuple(read_block(b) for b in plan_blocks(book, rows_per_block=40).blocks)
    )
    request = dataclasses.replace(_request(book), source=held)
    manifest = build_token_cache(
        request, tmp_path / "cache", workers=2, rows_per_block=40, progress=False
    )
    assert manifest["book"] == held.fingerprint()
    profiles = pq.read_table(tmp_path / "cache" / "profiles.parquet")
    assert profiles.equals(pq.read_table(book / "profiles.parquet"))
    for name in ("manifest.json", "tokenizer.json"):
        got = json.loads((tmp_path / "cache" / name).read_text())
        want = json.loads((cache.path / name).read_text())
        got.pop("build", None), want.pop("build", None)
        assert got == want, name
    for part in sorted((cache.path / "parts").iterdir()):
        assert (tmp_path / "cache" / "parts" / part.name).read_bytes() == part.read_bytes()


def test_fit_on_a_user_subset_matches(book: Path, tmp_path: Path) -> None:
    users = ("u01", "u03", "u04", "u07")
    cache = _build(book, tmp_path / "cache", users)
    assert tuple(cache.user_ids()) == users
    assert cache.tokenizer_state == _legacy(book, set(users))["tokenizer"].state_dict()


def test_a_float_value_labelled_categorical_fits_as_numeric(book: Path) -> None:
    """`collect` counts every value that parses as a float as a reading, whatever the
    source labels it; the block fit must choose the same kinds."""
    block = read_block(plan_blocks(book, rows_per_block=10**9).blocks[0])
    kind = np.where(block.pair_kind == FIELD_NUMERIC, FIELD_CATEGORICAL, block.pair_kind)
    flipped = dataclasses.replace(block, pair_kind=kind.astype(block.pair_kind.dtype))

    def kinds(b: EventBlock) -> dict[str, str]:
        tokenizer = KeyValueTimeTokenizer(CONFIG)
        return tokenizer.fit_stats(tokenizer.fit_block(b, {})).vocabulary.field_kind

    assert kinds(flipped) == kinds(block)


def test_profiles_in_any_row_order_count_the_same(book: Path, tmp_path: Path) -> None:
    profiles = pq.read_table(book / "profiles.parquet")
    shuffled = profiles.take(Random(1).sample(range(profiles.num_rows), profiles.num_rows))
    pq.write_table(shuffled, tmp_path / "profiles.parquet", row_group_size=2)
    fit = FitSpec(TRAIN_END, FIT_WINDOW, SEQUENCE_LENGTH)
    blocks = plan_blocks(book, rows_per_block=40).blocks

    tokenizer = KeyValueTimeTokenizer(CONFIG)

    def counts(path: Path) -> dict[str, tuple[dict[str, int], int]]:
        parts = [
            fit_statistics_block(read_block(b), tokenizer, fit, path, "amount").statistics
            for b in blocks
        ]
        keys = functools.reduce(tokenizer.merge, parts).fit_statistics().keys
        return {k: (dict(v.counts), v.n_samples) for k, v in keys.items()}

    want = counts(book / "profiles.parquet")
    assert want["plan"][1] > 0
    assert counts(tmp_path / "profiles.parquet") == want


def test_a_build_refuses_profiles_not_sorted_by_user(book: Path, tmp_path: Path) -> None:
    root = tmp_path / "book"
    root.mkdir()
    (root / "events.parquet").write_bytes((book / "events.parquet").read_bytes())
    profiles = pq.read_table(book / "profiles.parquet")
    pq.write_table(profiles.take(list(range(profiles.num_rows))[::-1]), root / "profiles.parquet")
    with pytest.raises(SchemaError, match="not sorted by user_id. Sort it"):
        _build(root, tmp_path / "cache")


@pytest.mark.parametrize("budget", [2, 64])
def test_a_cache_whose_blocks_share_files_reads_the_same(
    book: Path, cache: TokenCache, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, budget: int
) -> None:
    monkeypatch.setattr(build_module, "MAX_PARTS", 3)
    monkeypatch.setattr(store_module, "open_file_budget", lambda: budget)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        many = _build(book, tmp_path / "cache", rows_per_block=39)
    assert many.manifest["parts"] == 3 < len(many.manifest["blocks"])
    assert len(list((tmp_path / "cache" / "parts").iterdir())) == 3
    assert any("ulimit -n" in str(w.message) for w in caught) == (budget < 3)
    order = np.random.default_rng(0).permutation(cache.manifest["n_tokens"])
    assert np.array_equal(many.key_ids[order], np.asarray(cache.key_ids)[order])
    for name in ("ts_us", "id_us", "by_loader", "token_offsets", "key_ids", "value_ids"):
        assert np.array_equal(getattr(many, name), getattr(cache, name)), name
    copy = pickle.loads(pickle.dumps(many.key_ids))
    assert np.array_equal(copy[order], many.key_ids[order])


def test_a_failed_header_read_closes_its_descriptor(
    cache: TokenCache, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = store_module.CacheFiles(cache.key_ids.files.paths, cache.manifest["build_id"])

    def fails(*_: Any) -> bytes:
        raise OSError("read failed")

    monkeypatch.setattr(os, "pread", fails)
    before = len(os.listdir("/dev/fd"))
    for _ in range(5):
        with pytest.raises(OSError, match="read failed"):
            files.fd(0)
    assert len(os.listdir("/dev/fd")) == before


def test_a_reopened_part_is_checked_again(
    book: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A descriptor closed for the open-file budget and reopened rechecks the header."""
    monkeypatch.setattr(build_module, "MAX_PARTS", 2)
    monkeypatch.setattr(store_module, "open_file_budget", lambda: 1)
    a = _build(book, tmp_path / "a", rows_per_block=39)
    b = _build(book, tmp_path / "b", rows_per_block=40)
    files = store_module.CacheFiles(a.key_ids.files.paths, a.manifest["build_id"])
    files.fd(0)
    files.fd(1)  # the budget of one closes part 0
    shutil.copyfile(b.key_ids.files.paths[0], files.paths[0])
    with pytest.raises(SchemaError, match="not part 0"):
        files.fd(0)


def test_max_events_below_one_keeps_every_event(cache: TokenCache, legacy: dict[str, Any]) -> None:
    """As `build_entity_sequences` does, so a cached run matches the sequence path."""
    rows = cache.train_rows(train_end=TRAIN_END, lookback=None, max_events=0)
    assert np.array_equal(cache.capped(rows, 0), rows.lo)
    assert np.array_equal(cache.capped(rows, None), rows.lo)
    assert (cache.capped(rows, 1) == rows.hi - 1).all()
    seq = max(legacy["train_seq"], key=lambda s: len(s.events))
    assert len(seq.events) > 3
    want = cache.tokenizer(legacy["profiles"], max_events=None).encode(seq)
    for cap in (0, -3):
        assert cache.tokenizer(legacy["profiles"], max_events=cap).encode(seq) == want, cap


@pytest.mark.parametrize("budget", [1, 64])
def test_rows_that_end_a_block_collate_like_one_block(
    book: Path, legacy: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, budget: int
) -> None:
    """A row's last event is a non-final block's last: its end offset is that block's
    terminal offset, stored but outside the block's logical shard."""
    monkeypatch.setattr(store_module, "open_file_budget", lambda: budget)
    many = _build(book, tmp_path / "many")
    whole = _build(book, tmp_path / "whole", rows_per_block=10**9)
    rows = CacheRows(np.arange(many.n_users), many.user_start, many.user_stop, TRAIN_END)
    assert np.isin(rows.hi, np.cumsum([b["events"] for b in many.manifest["blocks"]])[:-1]).any()
    tokenizer = many.tokenizer(legacy["profiles"], max_events=CONFIG.max_events_per_user)
    _assert_batches_equal(
        many.collate(rows, tokenizer, max_length=10), whole.collate(rows, tokenizer, max_length=10)
    )


@pytest.mark.parametrize("budget", [1, 64])
def test_a_part_file_from_another_cache_is_refused(
    book: Path, cache: TokenCache, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, budget: int
) -> None:
    monkeypatch.setattr(store_module, "open_file_budget", lambda: budget)
    other = _build(book, tmp_path / "other", rows_per_block=39)
    mixed = tmp_path / "mixed"
    shutil.copytree(cache.path, mixed)
    shutil.copy(other.path / "parts" / "00000.bin", mixed / "parts" / "00000.bin")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        swapped = TokenCache(mixed)
    with pytest.raises(SchemaError, match="00000.bin is not part 0 .*header differs"):
        np.asarray(swapped.key_ids)


def test_a_profile_sampling_every_block_projects_the_cache_built(
    book: Path, tmp_path: Path
) -> None:
    rows_per_block = plan_blocks(book, rows_per_block=10**9).n_rows // 2 + 40
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    report = profile_token_cache(
        _request(book), workers=3, rows_per_block=rows_per_block, scratch=scratch
    )
    assert not list(scratch.iterdir())
    built = build_token_cache(
        _request(book), tmp_path / "cache", workers=1, rows_per_block=rows_per_block, progress=False
    )
    assert report["sample"]["blocks"] == list(range(built["build"]["blocks"]))
    assert report["book"]["events"] == report["sample"]["events"] == built["n_events"]
    parts = sum(p.stat().st_size for p in (tmp_path / "cache" / "parts").iterdir())
    assert report["cache"]["bytes"] == parts
    assert report["cache"]["sample_vocab_size"] == built["vocab_size"]
    assert report["user_ids"]["longest_bytes"] == built["user_id_bytes"]
    assert 0 < report["memory"]["worker_peak_bytes"] < report["memory"]["available_bytes"]
    assert "suggestion:" in format_profile(report)
    decoded = [read_block(b) for b in plan_blocks(book, rows_per_block=rows_per_block).blocks]
    distinct = sum(len(v.encode()) + 12 for d in decoded for v in d.values)
    rows = sum(11 * len(d.pair_event) + 16 * d.n_events for d in decoded)
    assert report["spill"]["dictionary_bytes"] == distinct > 0
    assert report["spill"]["bytes_needed"] == rows + distinct


def test_a_value_only_layout_reuses_the_block_encode(book: Path) -> None:
    tokenizer = _legacy(book)["tokenizer"]
    cap = CONFIG.max_event_tokens
    for planned in plan_blocks(book, rows_per_block=40).blocks:
        block = read_block(planned)
        offsets, keys, values, positions, cut = tokenizer.vocabulary.encode_events(block, cap)
        want = tokenizer.pairs.encode_block(block, cap)
        labels = np.asarray([tokenizer.pairs.key_vocab.get(k, UNK) for k in block.keys])
        assert labels[keys].tobytes() == want[1].tobytes()
        for got, expected in zip((offsets, values, positions), (want[0], *want[2:4]), strict=True):
            assert got.tobytes() == expected.tobytes()
        assert cut == want[4]


def test_a_profile_schedules_blocks_as_the_pool_hands_them_out() -> None:
    assert makespan([1.0] * 45, 4) == 12.0  # 45 blocks take 12 rounds on 4 workers
    assert makespan([1.0] * 44 + [0.07], 4) == pytest.approx(11.07)  # a short last block
    assert makespan([3.0, 1.0, 1.0, 1.0], 2) == 3.0  # the long task overlaps the short ones
    assert makespan([2.0, 5.0], 8) == 5.0


def test_a_profile_separates_fixed_cost_from_cost_per_row() -> None:
    rows = [1_000_000, 1_000_000, 1_000_000, 70_000]
    fixed, per_row = fit_block_seconds(rows, [0.2 + 1e-6 * r for r in rows])
    assert fixed == pytest.approx(0.2) and per_row == pytest.approx(1e-6)
    assert fit_block_seconds([10, 10], [1.0, 3.0]) == (0.0, 0.2)  # one size: no fixed cost
    assert fit_block_seconds([10, 20], [1.0, 3.0]) == (0.0, pytest.approx(4 / 30))  # fixed < 0
    assert fit_block_seconds([10, 20], [3.0, 2.5]) == (0.0, pytest.approx(5.5 / 30))  # per row < 0


def test_cached_batches_survive_pickling(cache: TokenCache, legacy: dict[str, Any]) -> None:
    rows = cache.train_rows(train_end=TRAIN_END, lookback=FIT_WINDOW, max_events=SEQUENCE_LENGTH)
    tokenizer = cache.tokenizer(legacy["profiles"], max_events=CONFIG.max_events_per_user)
    batches = CachedBatches(cache, rows, tokenizer, batch_size=4, max_length=9, shuffle_seed=7)
    copy = pickle.loads(pickle.dumps(batches))
    _assert_batches_equal(copy[1], batches[1])
    del copy
    gc.collect()
    batches[2]  # the original's files are still open


def test_a_book_out_of_time_order_warns(book: Path, tmp_path: Path) -> None:
    with pytest.warns(UserWarning, match=r"not sorted by \(user_id, ts\)"):
        _build(book, tmp_path / "cache")


def _split(book: Path, out: Path, at: Sequence[int], names: Sequence[str]) -> Path:
    """`book` with its events cut at rows `at` into files `names`, 7 rows per row group."""
    table = pq.read_table(book / "events.parquet")
    (out / "events.parquet").mkdir(parents=True)
    bounds = [0, *at, table.num_rows]
    for name, a, b in zip(names, bounds[:-1], bounds[1:], strict=True):
        pq.write_table(table.slice(a, b - a), out / "events.parquet" / name, row_group_size=7)
    (out / "profiles.parquet").write_bytes((book / "profiles.parquet").read_bytes())
    return out


def test_a_multi_file_book_builds_the_same_cache(
    book: Path, cache: TokenCache, tmp_path: Path
) -> None:
    ids = pq.read_table(book / "events.parquet", columns=["user_id"]).column(0).to_pylist()
    at = [30, 31, 31, 100]
    assert ids[29] == ids[30] == ids[31]  # one user spans three files; part-12 is empty
    names = [f"part-{i}.parquet" for i in (10, 11, 12, 13, 9)]  # read in string order
    multi = _build(_split(book, tmp_path / "book", at, names), tmp_path / "cache")
    assert multi.tokenizer_state == cache.tokenizer_state
    for name in ("ts_us", "id_us", "by_loader", "token_offsets", "key_ids", "value_ids"):
        assert np.array_equal(getattr(multi, name), getattr(cache, name)), name
    assert [n for n, _ in multi.manifest["book"]["events_files"]] == names


def test_event_files_out_of_user_order_are_refused(book: Path, tmp_path: Path) -> None:
    root = _split(book, tmp_path / "book", [100], ["part-1.parquet", "part-0.parquet"])
    with pytest.raises(SchemaError, match="not sorted by user_id"):
        _build(root, tmp_path / "cache")


def test_event_files_with_different_schemas_are_refused(book: Path, tmp_path: Path) -> None:
    root = _split(book, tmp_path / "book", [100], ["part-0.parquet", "part-1.parquet"])
    part = root / "events.parquet" / "part-1.parquet"
    pq.write_table(pq.read_table(part).drop_columns(["source"]), part)
    with pytest.raises(SchemaError, match="another schema"):
        _build(root, tmp_path / "cache")


def test_requested_users_missing_from_a_multi_file_book_are_refused(
    book: Path, tmp_path: Path
) -> None:
    ids = pq.read_table(book / "events.parquet", columns=["user_id"]).column(0).to_pylist()
    root = _split(book, tmp_path / "book", [30], ["part-0.parquet", "part-1.parquet"])
    assert ids[29] == ids[30]  # counted once, though in both files
    with pytest.raises(SchemaError, match="^1 requested users have no events"):
        plan_blocks(root, rows_per_block=10**9, users=[ids[30], "nobody"])


@pytest.mark.parametrize("change", ["rewrite", "touch", "relayout"])
def test_an_event_file_changed_after_planning_is_refused(
    book: Path, tmp_path: Path, change: str
) -> None:
    root = _split(book, tmp_path / "book", [100], ["part-0.parquet", "part-1.parquet"])
    block = plan_blocks(root, rows_per_block=10**9).blocks[0]
    changed = root / "events.parquet" / "part-1.parquet"
    if change == "touch":
        stat = changed.stat()
        os.utime(changed, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
    else:
        pq.write_table(pq.read_table(changed), changed, row_group_size=3)
    if change == "relayout":  # same size and time as planned, as after `cp -p`
        stat = changed.stat()
        block = dataclasses.replace(
            block,
            groups=tuple(
                dataclasses.replace(g, size=stat.st_size, mtime_ns=stat.st_mtime_ns)
                if Path(g.path) == changed
                else g
                for g in block.groups
            ),
        )
    with pytest.raises(SchemaError, match="changed during the build"):
        read_block(block)


def test_an_outlier_user_id_warns_and_is_recorded(book: Path, tmp_path: Path) -> None:
    root = tmp_path / "book"
    root.mkdir()
    events = pq.read_table(book / "events.parquet")
    ids = events.column("user_id").to_pylist()
    long_id = ids[-1] + "x" * 60  # still sorts last
    renamed = [long_id if u == ids[-1] else u for u in ids]
    column = events.schema.get_field_index("user_id")
    events = events.set_column(column, "user_id", pa.array(renamed, pa.string()))
    pq.write_table(events, root / "events.parquet", row_group_size=7)
    (root / "profiles.parquet").write_bytes((book / "profiles.parquet").read_bytes())
    with pytest.warns(UserWarning, match="over four times the mean"):
        cache = _build(root, tmp_path / "cache")
    assert cache.manifest["user_id_bytes"] == len(long_id)


def test_a_user_bigger_than_a_block_is_refused(book: Path, tmp_path: Path) -> None:
    with pytest.raises(UnsupportedCombinationError, match="Raise rows_per_block to at least 39"):
        _build(book, tmp_path / "cache", rows_per_block=38)


def test_a_time_sorted_book_builds_the_same_cache(cache: TokenCache, tmp_path: Path) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ordered = _build(_book(tmp_path / "book", time_sorted=True), tmp_path / "cache")
    assert ordered.tokenizer_state == cache.tokenizer_state
    for name in ("ts_us", "id_us", "by_loader", "token_offsets", "key_ids", "value_ids"):
        assert np.array_equal(getattr(ordered, name), getattr(cache, name)), name


def test_block_values_ascend_including_sources(tmp_path: Path) -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    records = [
        {
            "user_id": f"u{u}",
            "ts": start + timedelta(hours=i),
            "source": f"s{(11 - u - i) % 12:02d}",
            "fields": {"note": "zzz", "amount": str(i)},
        }
        for u in range(4)
        for i in range(6)
    ]
    root = tmp_path / "book"
    root.mkdir()
    pq.write_table(events_table_from_records(records), root / "events.parquet")
    for block in plan_blocks(root, rows_per_block=6).blocks:
        values = list(read_block(block).values)
        assert values == sorted(set(values))


def test_fit_statistics_merge_the_same_in_any_order_and_any_block_split(book: Path) -> None:
    fit = FitSpec(TRAIN_END, FIT_WINDOW, SEQUENCE_LENGTH)
    profiles = book / "profiles.parquet"

    tokenizer = KeyValueTimeTokenizer(CONFIG)

    def parts(rows_per_block: int) -> list[KeyValueStats]:
        plan = plan_blocks(book, rows_per_block=rows_per_block)
        return [
            fit_statistics_block(read_block(b), tokenizer, fit, profiles, "amount").statistics
            for b in plan.blocks
        ]

    def merged(ps: list[KeyValueStats], *, tree: bool = False) -> list[tuple[Any, ...]]:
        ps = copy.deepcopy(ps)  # merge reuses its first argument
        while tree and len(ps) > 1:
            ps = [
                tokenizer.merge(*ps[i : i + 2]) if i + 1 < len(ps) else ps[i]
                for i in range(0, len(ps), 2)
            ]
        keys = functools.reduce(tokenizer.merge, ps).fit_statistics().keys
        return [
            (
                k,
                list(v.counts.items()),
                v.n_samples,
                np.asarray(v.numeric).tobytes(),
                np.asarray(v.numeric_counts).tobytes(),
            )
            for k, v in keys.items()
        ]

    small = parts(40)
    seen = set().union(*(p.keys["merchant"].values for p in small))
    assert (
        CONFIG.vocabulary().fit_value_cap("merchant") == 7 < len(seen)
    )  # the merge must drop some
    want = merged(small)
    for seed in range(5):
        shuffled = small[:]
        Random(seed).shuffle(shuffled)
        assert merged(shuffled) == want, seed
        assert merged(shuffled, tree=True) == want, seed
    assert merged(parts(10**9)) == want


def test_both_pair_numberings_build_the_same_cache(
    book: Path, cache: TokenCache, tmp_path: Path
) -> None:
    def table_fits(rows_per_block: int) -> list[bool]:
        # encode_block numbers (key, value) pairs with a presence table of keys × values
        # when that fits in the block's field count, else with np.unique.
        plan = plan_blocks(book, rows_per_block=rows_per_block)
        blocks = (read_block(b) for b in plan.blocks)
        return [len(b.keys) * len(b.values) <= len(b.pair_key) for b in blocks]

    assert not any(table_fits(40)) and all(table_fits(10**9))
    whole = _build(book, tmp_path / "cache", rows_per_block=10**9)
    assert whole.tokenizer_state == cache.tokenizer_state
    for name in ("ts_us", "id_us", "by_loader", "token_offsets", "key_ids", "value_ids"):
        assert np.array_equal(getattr(whole, name), getattr(cache, name)), name


def test_events_after_train_end_never_change_the_vocabulary(tmp_path: Path) -> None:
    a = _build(_book(tmp_path / "a", late_value="x"), tmp_path / "ca")
    b = _build(_book(tmp_path / "b", late_value="y" * 90), tmp_path / "cb")
    assert a.tokenizer_state == b.tokenizer_state
    assert "late_key" not in a.tokenizer_state["key_vocab"]


def test_the_fit_pass_sees_no_field_or_value_from_after_train_end(tmp_path: Path) -> None:
    """Whatever a tokenizer reads of a block, pairs or the key and value tables, holds
    nothing that occurs only after train_end."""
    book = _book(tmp_path / "book", late_value="only-late")
    fit = FitSpec(TRAIN_END, FIT_WINDOW, SEQUENCE_LENGTH)

    class Seen:
        uses_profiles = False

        def fit_block(self, block: EventBlock, profiles: Any) -> set[str]:
            for value, number in zip(block.values, block.numbers.tolist(), strict=True):
                assert np.isnan(number) or number == float(value), (value, number)
            return {*block.keys, *block.values}

    blocks = plan_blocks(book, rows_per_block=40).blocks
    parts = [fit_statistics_block(read_block(b), Seen(), fit, None, "amount") for b in blocks]  # type: ignore[arg-type]
    seen = set().union(*(p.statistics for p in parts))
    assert "amount" in seen
    assert not {"late_key", "only-late"} & seen


def test_train_rows_collate_like_collate_batch(cache: TokenCache, legacy: dict[str, Any]) -> None:
    tokenizer = cache.tokenizer(legacy["profiles"], max_events=CONFIG.max_events_per_user)
    rows = cache.train_rows(train_end=TRAIN_END, lookback=FIT_WINDOW, max_events=SEQUENCE_LENGTH)
    want = collate_batch(legacy["train_seq"], legacy["tokenizer"], max_tokens=10)
    _assert_batches_equal(cache.collate(rows, tokenizer, max_length=10), want)


def test_a_hashed_key_caches_like_the_object_path(book: Path, tmp_path: Path) -> None:
    config = dataclasses.replace(CONFIG, hashed_keys=("merchant",), hash_buckets=16)
    cache = _build(book, tmp_path / "cache", config=config)
    legacy = _legacy(book, config=config)
    assert legacy["tokenizer"].vocabulary.field_kind["merchant"] == "hashed"
    assert cache.tokenizer_state == legacy["tokenizer"].state_dict()
    assert cache.manifest["tokenizer_settings"]["hashed_keys"] == ["merchant"]
    tokenizer = cache.tokenizer(legacy["profiles"], max_events=CONFIG.max_events_per_user)
    rows = cache.train_rows(train_end=TRAIN_END, lookback=FIT_WINDOW, max_events=SEQUENCE_LENGTH)
    want = collate_batch(legacy["train_seq"], legacy["tokenizer"], max_tokens=10)
    _assert_batches_equal(cache.collate(rows, tokenizer, max_length=10), want)


def test_excluded_keys_cache_like_the_object_path(book: Path, tmp_path: Path) -> None:
    config = dataclasses.replace(CONFIG, excluded_keys=("channel", "source"))
    cache = _build(book, tmp_path / "cache", config=config)
    legacy = _legacy(book, config=config)
    assert "channel" not in legacy["tokenizer"].pairs.key_vocab
    assert cache.tokenizer_state == legacy["tokenizer"].state_dict()
    tokenizer = cache.tokenizer(legacy["profiles"], max_events=CONFIG.max_events_per_user)
    rows = cache.train_rows(train_end=TRAIN_END, lookback=FIT_WINDOW, max_events=SEQUENCE_LENGTH)
    want = collate_batch(legacy["train_seq"], legacy["tokenizer"], max_tokens=10)
    _assert_batches_equal(cache.collate(rows, tokenizer, max_length=10), want)
    cache.manifest["tokenizer_settings"]["excluded_keys"] = []
    with pytest.raises(SchemaError, match="Rebuild it with `tfm cache-tokens`"):
        cache.check_identity(_request(book, config=config).identity())


def test_an_excluded_amount_field_still_feeds_the_raw_features(
    book: Path, cache: TokenCache, tmp_path: Path
) -> None:
    config = dataclasses.replace(CONFIG, excluded_keys=("amount",))
    excluded = _build(book, tmp_path / "cache", config=config)
    assert "amount" not in excluded.tokenizer_state["key_vocab"]
    assert np.asarray(excluded.amount).tobytes() == np.asarray(cache.amount).tobytes()


def test_a_cache_stores_its_arrays_in_the_format_4_layout(cache: TokenCache) -> None:
    assert cache.manifest["arrays"] == {
        "ts_us": "<i8",
        "id_us": "<i8",
        "hour": "|u1",
        "dow": "|u1",
        "dom": "|u1",
        "amount": "<f8",
        "by_loader": "<i4",
        "token_offsets": "<i8",
        "key_ids": "<u2",
        "value_ids": "<u2",
        "positions": "|u1",
    }
    assert list(cache.manifest["arrays"]) == list(cache.manifest["blocks"][0]["offsets"])


_SWAPPED_I8 = np.dtype(np.int64).newbyteorder()
_SWAPPED_U8 = np.dtype(np.uint64).newbyteorder()


def _week(block: EventBlock, dtype: Any = np.int8) -> dict[str, np.ndarray]:
    return {"week": np.zeros(block.n_events, dtype)}


@pytest.mark.parametrize(
    ("extra", "error"),
    [
        (_week, None),
        (lambda b: {"fractional": np.full(b.n_events, 0.5, np.float32)}, "integer dtypes"),
        (lambda b: {"fractional": np.full(b.n_events, 0.5, np.float64)}, "integer dtypes"),
        (lambda b: {"pairs": np.zeros((b.n_events, 2), np.int64)}, "one value per event"),
        (lambda b: {"swapped": np.zeros(b.n_events, _SWAPPED_I8)}, "native byte order"),
        (lambda b: {"swapped": np.zeros(b.n_events, _SWAPPED_U8)}, "uint64"),
        (lambda b: {"amount": np.zeros(b.n_events)}, "names the cache keeps"),
        (lambda b: _week(b, np.int8 if b.index == 0 else np.int16), "differ from the first"),
    ],
)
def test_a_part_stores_the_tokenizers_per_event_arrays(
    book: Path, cache: TokenCache, tmp_path: Path, extra: Any, error: str | None
) -> None:
    class Extra(KeyValueTimeTokenizer):
        def encode_block(self, block: EventBlock) -> EncodedBlock:
            encoded = super().encode_block(block)
            return dataclasses.replace(encoded, events={**encoded.events, **extra(block)})

    (tmp_path / "parts").mkdir()
    (tmp_path / "users").mkdir()
    blocks = plan_blocks(book, rows_per_block=40).blocks[:2]
    args = (
        PragmatiqColumns(book),
        blocks,
        Extra,
        cache.tokenizer_state,
        None,
        str(tmp_path),
        0,
        "0" * 32,
        "amount",
    )
    if error is not None:
        with pytest.raises(SchemaError, match=error):
            encode_part(*args)
        return
    stats, dtypes = encode_part(*args)
    assert list(dtypes) == [
        *list(cache.manifest["arrays"])[:5],
        "week",
        *list(cache.manifest["arrays"])[5:],
    ]
    assert dtypes["week"] == "|i1" and len(stats) == 2


def test_a_cache_built_before_tokenizer_settings_is_refused(book: Path, cache: TokenCache) -> None:
    old = TokenCache(cache.path)
    del old.manifest["tokenizer"], old.manifest["tokenizer_settings"]
    with pytest.raises(SchemaError, match="was built with tokenizer None.*`tfm cache-tokens`"):
        old.check_identity(_request(book).identity())


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
        cache.tokenizer(legacy["profiles"], max_events=CONFIG.max_events_per_user),
        batch_size=4,
        max_length=9,
        shuffle_seed=7,
    )
    assert len(rows) == len(corpus)
    assert len(got) == len(want)
    for g, w in zip(got, want, strict=True):
        _assert_batches_equal(g, w)


DEVICES = ["cpu", *(["cuda"] if torch.cuda.is_available() else [])]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("max_length", [3, 9])
@pytest.mark.parametrize("capped", [False, True])
def test_device_batches_equal_cached_batches(
    cache: TokenCache, device: str, max_length: int, capped: bool
) -> None:
    assert cache.profiles()
    tokenizer = cache.tokenizer(max_events=CONFIG.max_events_per_user)
    if capped:
        rows = cache.train_rows(
            train_end=TRAIN_END, lookback=FIT_WINDOW, max_events=SEQUENCE_LENGTH
        )
        assert CONFIG.max_events_per_user is not None
        assert (rows.hi - rows.lo).max() > CONFIG.max_events_per_user
    else:
        rows = cache.train_windows(train_end=TRAIN_END, lookback=None, window_events=5, stride=3)
    kwargs: dict[str, Any] = {"batch_size": 4, "max_length": max_length, "shuffle_seed": 7}
    want = CachedBatches(cache, rows, tokenizer, **kwargs)
    got = DeviceBatches(cache, rows, tokenizer, device=torch.device(device), **kwargs)
    assert len(got) == len(want)
    for g, w in zip(got, want, strict=True):
        assert g.key_ids.device.type == device
        _assert_batches_equal(g.to(torch.device("cpu")), w)


def test_batches_load_and_carry_only_the_arrays_the_model_reads(book: Path, tmp_path: Path) -> None:
    calendar = tokenizer_class("tests.fixtures.user_tokenizer:CalendarKeyTokenizer")
    request = dataclasses.replace(_request(book), tokenizer=calendar())
    build_token_cache(request, tmp_path / "cache", workers=1, rows_per_block=40, progress=False)
    cache = TokenCache(tmp_path / "cache")
    assert sorted(cache.events) == ["dom", "dow", "hour", "minute", "n_keys"]
    tokenizer = cache.tokenizer(max_events=SEQUENCE_LENGTH)
    rows = cache.train_rows(train_end=TRAIN_END, lookback=FIT_WINDOW, max_events=SEQUENCE_LENGTH)
    kwargs: dict[str, Any] = {"batch_size": 4, "max_length": 9, "shuffle_seed": 7}
    every = CachedBatches(cache, rows, tokenizer, **kwargs)
    host = CachedBatches(cache, rows, tokenizer, event_arrays=("minute", "hour"), **kwargs)
    on_device = DeviceBatches(
        cache,
        rows,
        tokenizer,
        device=torch.device("cpu"),
        event_arrays=("minute", "hour"),
        **kwargs,
    )
    assert sorted(on_device.arrays) == [
        "hour",
        "id_us",
        "key_ids",
        "minute",
        "positions",
        "token_offsets",
        "value_ids",
    ]
    for h, d, e in zip(host, on_device, every, strict=True):
        assert list(h.events) == list(d.events) == ["hour", "minute"]
        assert all(torch.equal(h.events[n], e.events[n]) for n in h.events)
        _assert_batches_equal(d, h)
    with pytest.raises(UnsupportedCombinationError, match=r"\['weekend'\]"):
        CachedBatches(cache, rows, tokenizer, event_arrays=("weekend",), **kwargs)


def test_host_batches_pad_an_empty_row_after_the_last_nonempty_one(cache: TokenCache) -> None:
    rows = cache.train_rows(train_end=TRAIN_END, lookback=FIT_WINDOW, max_events=SEQUENCE_LENGTH)
    rows = CacheRows(rows.user[:2], rows.lo[:2], np.array([rows.hi[0], rows.lo[1]]), rows.cutoff)
    tokenizer = cache.tokenizer(max_events=CONFIG.max_events_per_user)
    kwargs: dict[str, Any] = {"batch_size": 2, "max_length": 9, "shuffle_seed": None}
    batch = CachedBatches(cache, rows, tokenizer, **kwargs)[0]  # [0]: iterating hides IndexError
    alone = CachedBatches(cache, rows.take([0]), tokenizer, **kwargs)[0]
    assert batch.event_mask[1].sum() == 0
    assert torch.equal(batch.time_log[:1], alone.time_log)
    assert torch.equal(batch.elapsed_log[:1], alone.elapsed_log)


def test_a_built_in_format_resolves_after_the_other_was_imported() -> None:
    script = (
        "import tfm.data.token_cache.columnar; "
        "from tfm.data.token_cache.formats import token_cache_format; "
        "token_cache_format('flat_fields')"
    )
    subprocess.check_call([sys.executable, "-c", script])  # a fresh registry


def test_device_batches_refuse_an_empty_row(cache: TokenCache) -> None:
    rows = cache.train_rows(train_end=TRAIN_END, lookback=FIT_WINDOW, max_events=SEQUENCE_LENGTH)
    empty = CacheRows(rows.user, rows.lo, rows.lo, rows.cutoff)
    with pytest.raises(UnsupportedCombinationError, match="at least one event"):
        DeviceBatches(
            cache,
            empty,
            cache.tokenizer(max_events=CONFIG.max_events_per_user),
            batch_size=4,
            max_length=9,
            shuffle_seed=None,
            device=torch.device("cpu"),
        )


@pytest.fixture(scope="module")
def lab_cache(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("lab")
    raw = load_recipe_config(LAB_CONFIG).model_dump(mode="json")
    raw["data"].update(
        path=str(generate_lab_book(root / "book", 40)),
        token_cache=str(root / "cache"),
        n_entities=40,
        max_entities=40,
        sequence_length=16,
        events_per_entity=16,
        pretrain_windows={"events": 16},
    )
    raw["data"]["identity_theft"]["donors"] = "random"
    config = RecipeConfig.model_validate(raw)
    build_recipe_cache(config, tokenizer_for(config), workers=1)
    return raw


@pytest.mark.parametrize("device", DEVICES)
def test_setup_builds_batches_on_a_gpu_unless_switched_off(
    lab_cache: dict[str, Any], device: str, tmp_path: Path
) -> None:
    def first_batch(on_device: bool) -> Any:
        raw = {
            **lab_cache,
            "data": {**lab_cache["data"], "token_cache_on_device": on_device},
            "runtime": {**lab_cache["runtime"], "device": device},
        }
        artifacts = setup_run(
            config=RecipeConfig.model_validate(raw), run_dir=tmp_path / str(on_device)
        )
        batches, stats = training_batches(artifacts)
        expected = on_device and device == "cuda"
        assert isinstance(batches, DeviceBatches) is expected
        assert stats["token_cache_on_device"] is expected
        return batches[0]

    _assert_batches_equal(first_batch(True).to(torch.device("cpu")), first_batch(False))


def test_probe_encode_reads_tokens_by_event_id(cache: TokenCache, legacy: dict[str, Any]) -> None:
    tokenizer = cache.tokenizer(legacy["profiles"], max_events=CONFIG.max_events_per_user)
    last = max(e.event_timestamp for e in legacy["events"])
    for cutoff in (VALIDATION_END, last):
        for seq in build_entity_sequences(legacy["events"], cutoff, max_events=SEQUENCE_LENGTH):
            assert tokenizer.encode(seq) == legacy["tokenizer"].encode(seq)


def test_a_copied_event_needs_an_alias(cache: TokenCache, legacy: dict[str, Any]) -> None:
    tokenizer = cache.tokenizer(legacy["profiles"], max_events=CONFIG.max_events_per_user)
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
        cache.tokenizer(max_events=CONFIG.max_events_per_user).fit(legacy["train_seq"])


def test_cache_arrays_index_across_shards_as_one_array(cache: TokenCache) -> None:
    assert len(cache.manifest["blocks"]) > 1
    assert cache.token_offsets[-1] == cache.manifest["n_tokens"]
    for name in ("ts_us", "token_offsets", "key_ids"):
        array = getattr(cache, name)
        whole = np.asarray(array)
        backwards = np.arange(len(whole))[::-1]
        assert np.array_equal(array[backwards], whole[backwards]), name
        for cut in (slice(1, -1), slice(None, None, 3), slice(5, 5)):
            assert np.array_equal(array[cut], whole[cut]), (name, cut)
        assert array[-1] == whole[-1], name
        with pytest.raises(IndexError):
            array[len(whole)]
        if np.lib.NumpyVersion(np.__version__) >= "2.0.0":
            with pytest.raises(ValueError, match="cannot avoid a copy"):
                np.asarray(array, copy=False)


def test_a_format_1_cache_is_refused_with_the_rebuild_command(
    cache: TokenCache, tmp_path: Path
) -> None:
    old = tmp_path / "cache"
    old.mkdir()
    manifest = {**cache.manifest, "format_version": 1}
    (old / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SchemaError, match="has format 1.*tfm cache-tokens --config"):
        TokenCache(old)


def test_users_are_found_by_id_and_nothing_else(cache: TokenCache) -> None:
    ids = cache.user_ids()
    assert ids == sorted(ids) and len(ids) == cache.n_users
    # "u0" is a prefix of every id; "u030" extends one.
    asked = [ids[3], "u0", ids[0], "u030", ids[-1]]
    assert cache.user_index(asked).tolist() == [3, -1, 0, -1, cache.n_users - 1]
    assert cache.user_ids([3, 0]) == [ids[3], ids[0]]


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


def test_a_config_refuses_what_a_cache_cannot_reproduce() -> None:
    raw = load_recipe_config(LAB_CONFIG).model_dump(mode="json")
    raw["data"].update(token_cache="/nonexistent/cache", fraud_labels="native")
    RecipeConfig.model_validate(raw)
    raw["data"]["split_protocol"] = "precomputed"
    with pytest.raises(ValueError, match="timestamp split"):
        RecipeConfig.model_validate(raw)


def test_an_event_id_of_another_form_names_the_cache_limit(cache: TokenCache) -> None:
    with pytest.raises(SchemaError, match="other event ids cannot use data.token_cache"):
        cache.event_position("t-123")


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


def _renamed_amount(book: Path, out: Path, name: str) -> Path:
    rows = pq.read_table(book / "events.parquet").to_pylist()
    for row in rows:
        row["fields"] = [(name if k == "amount" else k, v) for k, v in row["fields"]]
    out.mkdir()
    pq.write_table(events_table_from_records(rows), out / "events.parquet", row_group_size=7)
    shutil.copy(book / "profiles.parquet", out / "profiles.parquet")
    return out


def test_the_amount_is_read_from_data_amount_field(
    book: Path, cache: TokenCache, legacy: dict[str, Any], tmp_path: Path
) -> None:
    renamed = _renamed_amount(book, tmp_path / "book", "txn_amt")
    request = dataclasses.replace(_request(renamed), amount_field="txn_amt")
    build_token_cache(request, tmp_path / "cache", workers=1, rows_per_block=40, progress=False)
    mapped = TokenCache(tmp_path / "cache")
    assert np.asarray(mapped.amount).tobytes() == np.asarray(cache.amount).tobytes()
    rows = cache.train_rows(train_end=TRAIN_END, lookback=FIT_WINDOW, max_events=SEQUENCE_LENGTH)
    assert torch.equal(mapped.raw_features(rows.lo, rows.hi), cache.raw_features(rows.lo, rows.hi))
    renamed_seq = _legacy(renamed)["train_seq"]
    assert [raw_feature_vector(s, amount_key="txn_amt") for s in renamed_seq] == [
        raw_feature_vector(s) for s in legacy["train_seq"]
    ]
    with pytest.raises(SchemaError, match="data.amount_field"):
        build_token_cache(
            _request(renamed), tmp_path / "unmapped", workers=1, rows_per_block=40, progress=False
        )


@pytest.mark.parametrize("field", ["late_key", "memo"])
def test_an_amount_field_with_no_train_side_number_is_refused(tmp_path: Path, field: str) -> None:
    # `late_key` is numeric only after TRAIN_END; `memo` is train-side text.
    book = _book(tmp_path / "book", late_value="7")
    request = dataclasses.replace(_request(book), amount_field=field)
    with pytest.raises(SchemaError, match="data.amount_field"):
        build_token_cache(request, tmp_path / "cache", workers=1, rows_per_block=40, progress=False)


def test_raw_features_sum_and_maximise_as_python_does(tmp_path: Path) -> None:
    # Python's float sum compensates: 1e16 + 1 - 1e16 is 1.0 there and 0.0 added in order.
    # Twenty 3-event rows are summed together and the two longer rows one by one.
    amounts = {"u0": ["1e16", "1", "-1e16"], "u1": ["-0.0"], "u2": ["2.5", "7", "7", "0.1"]}
    amounts |= {f"s{i:02d}": [f"{i}.1", "0.2", f"-{i}"] for i in range(20)}
    amounts["u3"] = ["0.5"] * 9 + ["1e16", "1", "-1e16"]
    records = [
        {
            "user_id": user,
            "ts": TRAIN_END - timedelta(days=len(values) - i),
            "source": "transaction",
            "fields": {"amount": value},
        }
        for user, values in amounts.items()
        for i, value in enumerate(values)
    ]
    records += [
        {
            "user_id": user,
            "ts": later + timedelta(days=1),
            "source": "transaction",
            "fields": {"amount": "5"},
        }
        for user in amounts
        for later in (TRAIN_END, VALIDATION_END)
    ]
    records.sort(key=lambda r: (r["user_id"], r["ts"]))
    root = tmp_path / "book"
    root.mkdir()
    pq.write_table(events_table_from_records(records), root / "events.parquet")
    cache = _build(root, tmp_path / "cache")
    rows = cache.train_rows(train_end=TRAIN_END, lookback=FIT_WINDOW, max_events=SEQUENCE_LENGTH)
    sequences = _legacy(root)["train_seq"]
    want = torch.tensor([raw_feature_vector(s) for s in sequences], dtype=torch.float32)
    got = cache.raw_features(rows.lo, rows.hi)
    assert got.numpy().tobytes() == want.numpy().tobytes()
    users = [s.entity_id for s in sequences]
    assert got[users.index("u0"), 0].item() == pytest.approx(1 / 3)
    assert got[users.index("u3"), 0].item() == pytest.approx(
        3.5 / 8
    )  # its last 8 events: 5 × 0.5 + 1


def test_raw_features_mask_numeric_nonfinite_cached_rows_without_changing_bytes(
    cache: TokenCache, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Inject numbers after adapter parsing: string "nan"/"inf" fields can be
    # filtered before caching and therefore do not exercise the reader guard.
    amounts = np.array(cache.amount, dtype=np.float64, copy=True)
    amounts[:19] = [
        np.nan,
        np.inf,
        -np.inf,
        np.nan,
        np.inf,
        -np.inf,
        -3.0,
        9.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        6.0,
        np.nan,
        -4.0,
        -2.0,
        np.nan,
    ]
    header = build_module.part_header(cache.manifest["build_id"], 0)
    part = tmp_path / "numeric-amounts.part"
    part.write_bytes(header + amounts.tobytes())
    before = part.read_bytes()
    files = store_module.CacheFiles([part], cache.manifest["build_id"])
    numeric = store_module.ShardedArray(files, [(0, len(header))], [len(amounts)], np.float64)
    monkeypatch.setattr(cache, "amount", numeric)
    assert not files.map(0).flags.writeable
    assert np.isnan(numeric[0]) and np.isposinf(numeric[1]) and np.isneginf(numeric[2])

    # Many three-event rows use batched accumulation; the twelve-event row
    # uses the long-row path. Include a singleton and negative finite values.
    lo = np.array([0] * 20 + [3, 15, 16], dtype=np.int64)
    hi = np.array([3] * 20 + [15, 16, 19], dtype=np.int64)
    got = cache.raw_features(lo, hi)
    expected = torch.tensor(
        [[0.0, 0.0, 3.0]] * 20
        + [
            [1.0, 9.0, 12.0],
            [0.0, 0.0, 1.0],
            [-2.0, 0.0, 3.0],
        ]
    )
    assert torch.isfinite(got).all()
    torch.testing.assert_close(got[:, :3], expected, rtol=0, atol=0)
    assert np.asarray(numeric).tobytes() == amounts.tobytes()
    assert part.read_bytes() == before
    assert not files.map(0).flags.writeable


USER_TOKENIZER = "tests.fixtures.user_tokenizer:KeyTokenizer"


CONFORMING = {
    "key_value_time": KeyValueTimeTokenizer(CONFIG),
    **{
        f"key_value_time {name}": KeyValueTimeTokenizer(
            dataclasses.replace(CONFIG, **{name: value})
        )
        for name, value in {
            "n_buckets": 3,
            "categorical_threshold": 3,
            "numeric_min_cardinality": 9,
            "max_event_tokens": 3,
            "max_profile_tokens": None,
            "max_events_per_user": 2,
            "seed": 7,
            "categorical_keys": ("mcc", "amount"),
            "hashed_keys": ("merchant",),
            "hash_buckets": 5,
            "excluded_keys": ("channel", "source"),
        }.items()
    },
    "user": tokenizer_class(USER_TOKENIZER)(min_count=2),
}
"""The built-in under each config field it fits or encodes with, and the user tokenizer."""


@pytest.mark.parametrize("tokenizer", CONFORMING.values(), ids=CONFORMING.keys())
def test_columnar_tokenizers_pass_the_conformance_check(book: Path, tokenizer: Any) -> None:
    check_columnar_tokenizer(
        tokenizer,
        PragmatiqColumns(book),
        fit=FitSpec(TRAIN_END, FIT_WINDOW, SEQUENCE_LENGTH),
        rows_per_block=40,
    )


@pytest.fixture(scope="module")
def user_cache(tmp_path_factory: pytest.TempPathFactory) -> RecipeConfig:
    root = tmp_path_factory.mktemp("user")
    raw = load_recipe_config(LAB_CONFIG).model_dump(mode="json")
    raw["data"].update(
        path=str(generate_lab_book(root / "book", 40)),
        token_cache=str(root / "cache"),
        n_entities=40,
        max_entities=40,
    )
    raw["features"].update(
        tokenizer=USER_TOKENIZER, tokenizer_options={"min_count": 2}, categorical_keys=[]
    )
    config = RecipeConfig.model_validate(raw)
    build_recipe_cache(config, tokenizer_for(config), workers=2, rows_per_block=5000)
    return config


def test_a_tokenizer_named_by_import_path_builds_a_cache_in_workers(
    user_cache: RecipeConfig,
) -> None:
    path = Path(str(user_cache.data.token_cache))
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    state = json.loads((path / "tokenizer.json").read_text(encoding="utf-8"))
    assert manifest["tokenizer"] == USER_TOKENIZER
    assert manifest["tokenizer_settings"] == {"state_version": 1, "min_count": 2}
    assert manifest["build"]["blocks"] > 1
    assert manifest["arrays"]["n_keys"] == "<i2"
    assert manifest["vocab_size"] == len(state["vocab"]) + 2 > 2
    cache = open_recipe_cache(user_cache, tokenizer_for(user_cache))
    assert isinstance(cache, TokenCache) and list(cache.events) == ["n_keys"]
    with pytest.raises(UnsupportedCombinationError, match=r"arrays \['hour'\], which token"):
        cache.check_event_arrays(("n_keys", "hour"))


def test_a_user_tokenizer_cache_reads_back_through_the_sdk_reader(
    user_cache: RecipeConfig,
) -> None:
    cache = open_recipe_cache(user_cache, tokenizer_for(user_cache))
    assert isinstance(cache, TokenCache)
    train_end = parse_timestamp(user_cache.data.temporal_split.train_end)
    tokenizer = cache.tokenizer(max_events=4)
    assert tokenizer.name == USER_TOKENIZER and tokenizer.field_names() == {}
    assert tokenizer.encode_profile("anyone", train_end) == ([], [], [], [0], [])
    rows = cache.train_rows(train_end=train_end, lookback=None, max_events=4)
    batch = cache.collate(rows.take(range(3)), tokenizer, max_length=16)
    n_keys = batch.event_array("n_keys")
    assert torch.equal(n_keys, batch.token_mask.sum(-1))
    assert (n_keys[batch.event_mask.bool()] > 0).all()
    with pytest.raises(IncompatibleCompositionError, match="'hour'"):
        batch.event_array("hour")


def test_a_user_tokenizer_class_builds_itself_from_the_recipe(user_cache: RecipeConfig) -> None:
    raw = user_cache.model_dump(mode="json")
    raw["features"].update(
        tokenizer="tests.fixtures.user_tokenizer:ConfiguredKeyTokenizer",
        tokenizer_options=None,
        exclude_keys=["a", "b"],
    )
    tokenizer = tokenizer_for(RecipeConfig.model_validate(raw))
    assert type(tokenizer).__name__ == "ConfiguredKeyTokenizer"
    assert tokenizer.min_count == 3  # type: ignore[attr-defined]
    raw["features"]["categorical_keys"] = ["mcc"]
    with pytest.raises(UnsupportedCombinationError, match=r"categorical_keys does not apply"):
        tokenizer_for(RecipeConfig.model_validate(raw))


def test_a_tokenizer_names_its_cache_format_by_import_path(tmp_path: Path) -> None:
    from tests.fixtures.user_tokenizer import recording_format

    raw = load_recipe_config(LAB_CONFIG).model_dump(mode="json")
    raw["data"].update(
        path=str(generate_lab_book(tmp_path / "book", 40)),
        token_cache=str(tmp_path / "cache"),
        n_entities=40,
        max_entities=40,
        sequence_length=16,
        events_per_entity=16,
    )
    raw["data"]["identity_theft"]["donors"] = "random"
    raw["features"].update(
        tokenizer="tests.fixtures.user_tokenizer:FormattedKeyTokenizer",
        tokenizer_options={"min_count": 2},
        categorical_keys=[],
    )
    recording_format.calls.clear()
    config = RecipeConfig.model_validate(raw)
    build_recipe_cache(config, tokenizer_for(config), workers=1)
    data = prepare_data(config)
    assert data.cache_format is recording_format
    assert recording_format.calls == [
        "config_problems",
        "build",
        "open",
        "run_tokenizer",
        "pretrain_rows",
    ]
    assert data.pretrain_rows is not None and len(data.pretrain_rows) > 0
    assert data.tokenizer.name == "tests.fixtures.user_tokenizer:FormattedKeyTokenizer"


def test_a_feature_key_set_to_an_empty_value_is_refused_where_it_does_not_apply(
    user_cache: RecipeConfig,
) -> None:
    raw = user_cache.model_dump(mode="json")
    raw["features"].update(
        tokenizer="tests.fixtures.user_tokenizer:ConfiguredKeyTokenizer",
        tokenizer_options=None,
        fields=[],
    )
    with pytest.raises(UnsupportedCombinationError, match=r"features\.fields does not apply"):
        tokenizer_for(RecipeConfig.model_validate(raw))


def test_a_batch_builder_is_chosen_by_name_or_import_path(cache: TokenCache) -> None:
    rows = cache.train_rows(train_end=TRAIN_END, lookback=FIT_WINDOW, max_events=SEQUENCE_LENGTH)
    tokenizer = cache.tokenizer(max_events=CONFIG.max_events_per_user)
    kwargs: dict[str, Any] = {
        "batch_size": 4,
        "max_length": 9,
        "shuffle_seed": 7,
        "event_arrays": ("hour", "dow", "dom"),
    }
    cpu = torch.device("cpu")
    batches = batch_builder(DEFAULT_BATCH_BUILDER)(cache, rows, tokenizer, device=cpu, **kwargs)
    assert type(batches) is CachedBatches
    _assert_batches_equal(batches[0], CachedBatches(cache, rows, tokenizer, **kwargs)[0])
    user = batch_builder("tests.fixtures.user_tokenizer:placement_batches")
    assert list(user(cache, rows, tokenizer, device=cpu, **kwargs)) == [cpu]
    with pytest.raises(UnsupportedCombinationError, match="Unknown batch builder 'nope'"):
        batch_builder("nope")


def test_a_cache_is_refused_under_other_tokenizer_settings(book: Path, cache: TokenCache) -> None:
    other = _request(book, config=dataclasses.replace(CONFIG, seed=1)).identity()
    with pytest.raises(SchemaError, match="tokenizer_settings .*'seed': 0.*'seed': 1"):
        cache.check_identity(other)


def test_tokenizer_options_and_built_in_feature_keys_are_not_mixed(
    user_cache: RecipeConfig,
) -> None:
    raw = user_cache.model_dump(mode="json")
    raw["features"]["categorical_keys"] = ["mcc"]
    with pytest.raises(UnsupportedCombinationError, match="categorical_keys are both set"):
        tokenizer_for(RecipeConfig.model_validate(raw))
    raw["features"].update(categorical_keys=[], tokenizer_options={"min_cnt": 2})
    with pytest.raises(UnsupportedCombinationError, match=r"do not fit KeyTokenizer\(min_count"):
        tokenizer_for(RecipeConfig.model_validate(raw))


class _LosesVocab(tokenizer_class(USER_TOKENIZER)):  # type: ignore[misc]
    @classmethod
    def from_state(cls, state: Any) -> Any:
        return cls(state["min_count"])


class _FirstWins(tokenizer_class(USER_TOKENIZER)):  # type: ignore[misc]
    """Keeps the first block's statistics, and records how many fields it fit on."""

    seen = 0

    def merge(self, a: Any, b: Any) -> Any:
        return a

    def fit_stats(self, stats: Any) -> Any:
        self.seen = sum(stats.values())
        return super().fit_stats(stats)

    def state_dict(self) -> dict[str, Any]:
        return {**super().state_dict(), "seen": self.seen}

    @classmethod
    def from_state(cls, state: Any) -> Any:
        tokenizer = super().from_state(state)
        tokenizer.seen = state["seen"]
        return tokenizer


class _PresencePerBlock(tokenizer_class(USER_TOKENIZER)):  # type: ignore[misc]
    """Counts each key once per block, so more blocks count it more often."""

    def fit_block(self, block: EventBlock, profiles: Any) -> Any:
        return Counter(dict.fromkeys(super().fit_block(block, profiles), 1))


class _Uint64Events(tokenizer_class(USER_TOKENIZER)):  # type: ignore[misc]
    def encode_block(self, block: EventBlock) -> EncodedBlock:
        encoded = super().encode_block(block)
        n_keys = encoded.events["n_keys"].astype(np.uint64)
        return dataclasses.replace(encoded, events={"n_keys": n_keys})


@pytest.mark.parametrize(
    ("tokenizer", "workers", "error"),
    [
        (_LosesVocab(), 1, "from_state"),
        (_Uint64Events(), 1, "uint64, which torch cannot index"),
        (tokenizer_class(USER_TOKENIZER)(), 2, None),
        (KeyValueTimeTokenizer(dataclasses.replace(CONFIG, calendar_tz="Europe/Berlin")), 1, "UTC"),
        (_FirstWins(), 1, "merge order changes the fitted state"),
        (_PresencePerBlock(min_count=2), 1, "in two halves gives another state"),
    ],
)
def test_a_tokenizer_that_cannot_build_fails_before_the_fit_pass(
    book: Path, tmp_path: Path, tokenizer: Any, workers: int, error: str | None
) -> None:
    class Local(type(tokenizer)):  # type: ignore[misc]
        """Defined in a function, so worker processes cannot unpickle it."""

    if error is None:
        tokenizer, error = Local(), "Cannot import tokenizer.*Local"
    request = dataclasses.replace(_request(book), tokenizer=tokenizer)
    log = io.StringIO()
    with contextlib.redirect_stdout(log), pytest.raises(TfmError, match=error):
        build_token_cache(request, tmp_path / "cache", workers=workers, rows_per_block=120)
    assert "fit statistics collected" not in log.getvalue()
    assert "preflight passed" not in log.getvalue()


class _NoTokens(tokenizer_class(USER_TOKENIZER)):  # type: ignore[misc]
    def encode_block(self, block: EventBlock) -> EncodedBlock:
        encoded = super().encode_block(block)
        return dataclasses.replace(
            encoded,
            token_offsets=np.zeros_like(encoded.token_offsets),
            key_ids=encoded.key_ids[:0],
            value_ids=encoded.value_ids[:0],
            positions=encoded.positions[:0],
        )


def test_a_cache_with_no_token_is_not_published(book: Path, tmp_path: Path) -> None:
    request = dataclasses.replace(_request(book), tokenizer=_NoTokens())
    with pytest.raises(SchemaError, match="_NoTokens encoded no token for any event"):
        build_token_cache(request, tmp_path / "cache", workers=1, rows_per_block=120)
    assert not (tmp_path / "cache").exists() and not (tmp_path / "cache.partial").exists()


class _BlockLocal(tokenizer_class(USER_TOKENIZER)):  # type: ignore[misc]
    def encode_block(self, block: EventBlock) -> EncodedBlock:
        encoded = super().encode_block(block)
        return dataclasses.replace(encoded, value_ids=encoded.value_ids + block.index)


class _EventsInAnotherOrder(KeyValueTimeTokenizer):
    def encode_block(self, block: EventBlock) -> EncodedBlock:
        encoded = super().encode_block(block)
        return dataclasses.replace(encoded, events=dict(reversed(encoded.events.items())))


class _BlockOnlyBuckets(KeyValueTimeTokenizer):
    """Caps value ids in `encode_block` only, the same way in every block."""

    def encode_block(self, block: EventBlock) -> EncodedBlock:
        encoded = super().encode_block(block)
        return dataclasses.replace(encoded, value_ids=np.minimum(encoded.value_ids, np.uint16(5)))


def test_the_built_in_encodes_blocks_as_it_encodes_sequences(book: Path) -> None:
    config = dataclasses.replace(CONFIG, max_events_per_user=SEQUENCE_LENGTH)
    check_columnar_tokenizer(
        KeyValueTimeTokenizer(config),
        PragmatiqColumns(book),
        fit=FitSpec(TRAIN_END, FIT_WINDOW, SEQUENCE_LENGTH),
        rows_per_block=40,
        sequences=_legacy(book, config=config)["train_seq"],
        max_length=CONFIG.max_event_tokens,
    )
    check_columnar_tokenizer(  # per-event arrays are read by name, in any order
        _EventsInAnotherOrder(config),
        PragmatiqColumns(book),
        fit=FitSpec(TRAIN_END, FIT_WINDOW, SEQUENCE_LENGTH),
        rows_per_block=40,
        sequences=_legacy(book, config=config)["train_seq"],
        max_length=CONFIG.max_event_tokens,
    )
    with pytest.raises(ConformanceError, match="encode_block and encode: value_ids differ"):
        check_columnar_tokenizer(
            _BlockOnlyBuckets(config),
            PragmatiqColumns(book),
            fit=FitSpec(TRAIN_END, FIT_WINDOW, SEQUENCE_LENGTH),
            rows_per_block=40,
            sequences=_legacy(book, config=config)["train_seq"],
            max_length=CONFIG.max_event_tokens,
        )


def test_the_conformance_check_survives_python_O() -> None:
    import tfm.testing.conformance as conformance

    tree = ast.parse(Path(conformance.__file__).read_text(encoding="utf-8"))
    assert not [node.lineno for node in ast.walk(tree) if isinstance(node, ast.Assert)]


class _PerToken(tokenizer_class(USER_TOKENIZER)):  # type: ignore[misc]
    def encode_block(self, block: EventBlock) -> EncodedBlock:
        encoded = super().encode_block(block)
        return dataclasses.replace(encoded, events={"n_keys": encoded.positions})


class _Fractional(tokenizer_class(USER_TOKENIZER)):
    def encode_block(self, block: EventBlock) -> EncodedBlock:
        encoded = super().encode_block(block)
        return dataclasses.replace(
            encoded, events={"fractional": np.full(block.n_events, 0.5, np.float32)}
        )


class _Swapped(tokenizer_class(USER_TOKENIZER)):  # type: ignore[misc]
    def encode_block(self, block: EventBlock) -> EncodedBlock:
        encoded = super().encode_block(block)
        return dataclasses.replace(
            encoded, events={"swapped": np.zeros(block.n_events, _SWAPPED_I8)}
        )


class _Labels(tokenizer_class(USER_TOKENIZER)):  # type: ignore[misc]
    def encode_block(self, block: EventBlock) -> EncodedBlock:
        encoded = super().encode_block(block)
        names = np.array(["event"] * (len(encoded.token_offsets) - 1))
        return dataclasses.replace(encoded, events={"name": names})


@pytest.mark.parametrize(
    ("tokenizer", "error"),
    [
        (_BlockLocal(), "value_ids depends on block size"),
        (_FirstWins(), "merge order"),
        (_PerToken(), "one value per event"),
        (_Labels(), "not a number"),
        (_Fractional(), "integer dtype"),
        (_Swapped(), "native byte order"),
    ],
)
def test_the_conformance_check_catches_a_tokenizer_that_breaks_the_protocol(
    book: Path, tokenizer: Any, error: str
) -> None:
    with pytest.raises(ConformanceError, match=error):
        check_columnar_tokenizer(
            tokenizer,
            PragmatiqColumns(book),
            fit=FitSpec(TRAIN_END, FIT_WINDOW, SEQUENCE_LENGTH),
            rows_per_block=40,
        )


def test_a_newer_tokenizer_state_is_refused(cache: TokenCache) -> None:
    with pytest.raises(SchemaError, match="version 2.*rebuild the token cache"):
        KeyValueTimeTokenizer.from_state({**cache.tokenizer_state, "version": 2})


def test_a_tokenizer_from_an_interactive_session_is_refused(
    book: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Notebook(KeyValueTimeTokenizer):
        pass

    monkeypatch.setattr(Notebook, "__module__", "__main__")
    monkeypatch.delattr(sys.modules["__main__"], "__file__", raising=False)
    request = dataclasses.replace(_request(book), tokenizer=Notebook(CONFIG))
    with pytest.raises(UnsupportedCombinationError, match="importable module"):
        build_token_cache(request, tmp_path / "cache", workers=1, progress=False)


@pytest.mark.parametrize("name", ["transaction_field_tokenizer", "event_fields", "flat_key_value"])
def test_a_tokenizer_without_the_columnar_methods_cannot_build_a_cache(
    user_cache: RecipeConfig, name: str
) -> None:
    with pytest.raises(UnsupportedCombinationError, match="does not implement.*ColumnarTokenizer"):
        cache_request(user_cache, build_tokenizer(name))


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
        ),
    ],
)
def test_a_user_tokenizer_and_adapter_run_a_cache_end_to_end_on_native_labels(
    tmp_path: Path, device: str
) -> None:
    from tfm.runtime import run_smoke

    raw = load_recipe_config(LAB_CONFIG).model_dump(mode="json")
    raw["data"].update(
        adapter="tests.fixtures.user_adapter:PRAGMATIQ_BOOK",
        fraud_labels="native",
        path=str(generate_lab_book(tmp_path / "book", 40)),
        token_cache=str(tmp_path / "cache"),
        n_entities=40,
        max_entities=40,
        sequence_length=16,
        events_per_entity=16,
    )
    raw["features"].update(
        tokenizer="tests.fixtures.user_tokenizer:CalendarKeyTokenizer",
        tokenizer_options={"min_count": 2},
        categorical_keys=[],
    )
    granularity = raw["model"]["composition"]["granularity"]
    granularity.update(profile_depth=1, event_depth=1)
    granularity["time_features"]["minute"] = 60
    granularity["decoder"].update(depth=1, hidden_size=16, num_heads=2)
    raw["training"].update(max_steps=2, batch_size=8, precision="fp32")
    raw["fraud_task"]["finetune"]["enabled"] = False
    raw["evaluation"]["history_window"]["enabled"] = False
    raw["evaluation"]["tabular_baselines"]["enabled"] = False
    config = RecipeConfig.model_validate(raw)
    build_recipe_cache(config, tokenizer_for(config), workers=2, rows_per_block=5000)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config.model_dump(mode="json")), encoding="utf-8")

    out = run_smoke(run_dir=tmp_path / "run", config_path=path, device=device)

    batches = json.loads((out / "packing_manifest.json").read_text(encoding="utf-8"))
    assert batches["event_arrays"] == ["hour", "dow", "dom", "minute"]
    assert batches["token_cache_on_device"] is (device == "cuda")
    assert batches["batch_builder_default"] is True
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert "roc_auc" in metrics["churn_6m_embedding"]  # the lab book has no fraud labels


def test_device_batches_count_profiles_against_free_memory(
    cache: TokenCache, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tfm.data.token_cache.device as device_module

    eight_gib = torch.empty((2**20, 2**10), dtype=torch.long, device="meta")
    monkeypatch.setattr(device_module, "profile_tensors", lambda *_: {"prof_key_ids": eight_gib})
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _: (4 * 2**30, 8 * 2**30))
    rows = cache.train_rows(train_end=TRAIN_END, lookback=FIT_WINDOW, max_events=SEQUENCE_LENGTH)
    with pytest.raises(UnsupportedCombinationError, match="token_cache_on_device: false"):
        DeviceBatches(
            cache,
            rows,
            cache.tokenizer(max_events=CONFIG.max_events_per_user),
            batch_size=4,
            max_length=9,
            shuffle_seed=None,
            device=torch.device("cuda"),
        )


def test_identity_records_which_users_a_subset_cache_holds(book: Path) -> None:
    one, other = _request(book, users=("u01",)), _request(book, users=("u02",))
    assert one.population == other.population
    assert one.identity()["population"] != other.identity()["population"]
    unordered = _request(book, users=("u02", "u01", "u01")).identity()["population"]
    assert (
        unordered["users_sha256"]
        == _request(book, users=("u01", "u02")).identity()["population"]["users_sha256"]
    )
    assert _request(book).identity()["population"] == {"users": "all"}


def test_a_tokenizer_its_cache_cannot_restore_is_refused_with_one_worker(
    book: Path, tmp_path: Path
) -> None:
    class Local(KeyValueTimeTokenizer):
        pass

    request = dataclasses.replace(_request(book), tokenizer=Local(CONFIG))
    with pytest.raises(UnsupportedCombinationError, match="importable"):
        build_token_cache(request, tmp_path / "cache", workers=1, progress=False)
    assert not (tmp_path / "cache").exists()


class _NoSourceColumns(PragmatiqColumns):
    """A custom source whose events omit the optional source field."""

    def fingerprint(self):
        return {**super().fingerprint(), "source_fields": "omitted"}

    def read_block(self, block, *, users=None):
        value = super().read_block(block, users=users)
        if "source" not in value.keys:
            return value
        key = value.keys.index("source")
        keep = value.pair_key != key
        return dataclasses.replace(
            value,
            keys=tuple(name for name in value.keys if name != "source"),
            pair_event=value.pair_event[keep],
            pair_key=value.pair_key[keep] - (value.pair_key[keep] > key),
            pair_value=value.pair_value[keep],
            pair_kind=value.pair_kind[keep],
            pair_map_position=value.pair_map_position[keep],
        )


@pytest.mark.parametrize("cap", [1, 9])
def test_source_less_custom_columns_match_object_tokenization(book, cap):
    config = dataclasses.replace(CONFIG, max_events_per_user=SEQUENCE_LENGTH, max_event_tokens=cap)
    sequences = [
        dataclasses.replace(
            sequence,
            events=tuple(
                dataclasses.replace(
                    event, categorical={k: v for k, v in event.categorical.items() if k != "source"}
                )
                for event in sequence.events
            ),
        )
        for sequence in _legacy(book, config=config)["train_seq"]
    ]
    check_columnar_tokenizer(
        KeyValueTimeTokenizer(config),
        _NoSourceColumns(book),
        fit=FitSpec(TRAIN_END, FIT_WINDOW, SEQUENCE_LENGTH),
        rows_per_block=40,
        sequences=sequences,
        max_length=cap,
    )


def test_missing_source_uses_a_known_source_label_before_truncation(book):
    tokenizer = _legacy(book)["tokenizer"]
    source = _NoSourceColumns(book)
    block = source.read_block(source.plan(rows_per_block=40).blocks[0])
    offsets, keys, values, positions, _ = tokenizer.pairs.encode_block(block, 1)
    want_keys, want_values, want_positions = tokenizer.pairs.field_tokens("source", "transaction")
    assert len(want_keys) == 1
    np.testing.assert_array_equal(np.diff(offsets), 1)
    np.testing.assert_array_equal(keys, np.full(block.n_events, want_keys[0]))
    np.testing.assert_array_equal(values, np.full(block.n_events, want_values[0]))
    np.testing.assert_array_equal(positions, np.full(block.n_events, want_positions[0]))


def test_subset_cache_identity_keeps_user_id_delimiters_distinct(book):
    a = _request(book, users=("a", "b\nc"))
    b = _request(book, users=("a\nb", "c"))
    assert a.identity()["population"] != b.identity()["population"]


@pytest.mark.parametrize("dtype", ["<f4", "<f8", "<u8", _SWAPPED_I8.str, _SWAPPED_U8.str])
def test_opening_a_cache_rejects_non_indexable_event_arrays(cache, tmp_path, dtype):
    path = tmp_path / "bad_cache"
    shutil.copytree(cache.path, path)
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["arrays"]["hour"] = dtype
    (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(SchemaError, match="indexable integer dtypes"):
        TokenCache(path)


def test_cache_identity_rejects_pre_source_default_encoding(book, cache, tmp_path):
    path = tmp_path / "old_encoding"
    shutil.copytree(cache.path, path)
    manifest = json.loads((path / "manifest.json").read_text())
    del manifest["tokenizer_settings"]["event_encoding"]
    (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(SchemaError, match="tokenizer_settings.*implicit-source-v1"):
        TokenCache(path).check_identity(_request(book).identity())


class _PartialSourceColumns(PragmatiqColumns):
    """A custom source that omits the source field on events whose source is `drop`."""

    drop = "app"

    def fingerprint(self):
        return {**super().fingerprint(), "source_fields": f"omitted for {self.drop}"}

    def read_block(self, block, *, users=None):
        value = super().read_block(block, users=users)
        source = value.pair_key == value.keys.index("source")
        names = np.asarray(value.values, object)[value.pair_value]
        keep = ~(source & (names == self.drop))
        return dataclasses.replace(
            value,
            pair_event=value.pair_event[keep],
            pair_key=value.pair_key[keep],
            pair_value=value.pair_value[keep],
            pair_kind=value.pair_kind[keep],
            pair_map_position=value.pair_map_position[keep],
        )


class _LabelColumns(PragmatiqColumns):
    """A custom source that names field `field` `label`."""

    field, label = "amount", "is_fraud"

    def read_block(self, block, *, users=None):
        value = super().read_block(block, users=users)
        return dataclasses.replace(
            value, keys=tuple(self.label if k == self.field else k for k in value.keys)
        )


@pytest.mark.parametrize(
    ("field", "label"), [("amount", "is_fraud"), ("late_key", "Label"), ("memo", " fraud ")]
)
def test_a_label_like_input_field_fails_the_build_before_publishing(book, tmp_path, field, label):
    source = type("Labelled", (_LabelColumns,), {"field": field, "label": label})(book)
    request = dataclasses.replace(_request(book), source=source)
    with pytest.raises(SchemaError, match=f"field '{label}' looks like a label"):
        build_token_cache(request, tmp_path / "cache", workers=1, progress=False)
    assert not (tmp_path / "cache").exists()


@pytest.mark.parametrize("cap", [1, 9])
def test_partly_source_less_columns_match_object_tokenization(book, cap):
    source = _PartialSourceColumns(book)
    blocks = [source.read_block(b) for b in source.plan(rows_per_block=40).blocks]
    assert any(
        0 < len(set(b.pair_event[b.pair_key == b.keys.index("source")].tolist())) < b.n_events
        for b in blocks
    )
    config = dataclasses.replace(CONFIG, max_events_per_user=SEQUENCE_LENGTH, max_event_tokens=cap)
    sequences = [
        dataclasses.replace(
            sequence,
            events=tuple(
                dataclasses.replace(
                    event,
                    categorical={
                        k: v
                        for k, v in event.categorical.items()
                        if not (k == "source" and v == source.drop)
                    },
                )
                for event in sequence.events
            ),
        )
        for sequence in _legacy(book, config=config)["train_seq"]
    ]
    check_columnar_tokenizer(
        KeyValueTimeTokenizer(config),
        source,
        fit=FitSpec(TRAIN_END, FIT_WINDOW, SEQUENCE_LENGTH),
        rows_per_block=40,
        sequences=sequences,
        max_length=cap,
    )
