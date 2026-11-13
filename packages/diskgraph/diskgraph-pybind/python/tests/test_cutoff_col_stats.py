# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import collections
import datetime as dt
import decimal
import json
import pathlib
import random
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import diskgraph

T0 = 1_700_000_000
CUTOFF = T0 + 2_500
QUANTILES = [0.1, 0.5, 0.9]
REQUEST = {
    "cutoffs": [CUTOFF, None],
    "dirs": ["at-cutoff", "every-row"],
    "columns": {
        "users": {"tier": ["categorical"], "age": ["categorical", "numerical"],
                  "raw": ["categorical"], "seen": ["categorical"], "born": ["categorical"],
                  "price": ["categorical", "numerical"], "big": ["categorical"]},
        "items": {"name": ["categorical"]},
    },
    "quantiles": QUANTILES,
}


@pytest.fixture(scope="module")
def data(tmp_path_factory) -> pathlib.Path:
    rng = random.Random(0)
    root = tmp_path_factory.mktemp("cutoff_col_stats")
    n = 3_000
    pq.write_table(pa.table({
        "id": pa.array(range(n), pa.int64()),
        "signup": pa.array([(T0 + rng.randrange(-2_000, 7_000)) * 1_000 for _ in range(n)]),
        "tier": pa.array([None if i % 17 == 0 else rng.choice(["gold", "", "é"])
                          for i in range(n)]),
        "age": pa.array([rng.randrange(90) for _ in range(n)], pa.int32()),
        "raw": pa.array([bytes([i % 7]) for i in range(n)], pa.binary()),
        "seen": pa.array([(T0 + i % 13) * 10**6 for i in range(n)], pa.timestamp("us")),
        "born": pa.array([19_000 + i % 11 for i in range(n)], pa.date32()),
        "price": pa.array([decimal.Decimal(i % 1000) / 100 for i in range(n)], pa.decimal128(10, 2)),
        "big": pa.array([(5, 2**63, 2**64 - 1)[i % 3] for i in range(n)], pa.uint64()),
    }), root / "users.parquet", row_group_size=300)
    pq.write_table(pa.table({
        "item_id": pa.array(range(500), pa.int64()),
        "name": pa.array([f"item{i % 37}" for i in range(500)]),
    }), root / "items.parquet")
    return root


def _ingest(data: pathlib.Path, out: pathlib.Path, col_stats=False, **options) -> str:
    def table(name, key, columns, create_time=None):
        spec = {"name": name, "primary_key": key,
                "columns": [{"name": c, "ingested_dtype": d, **dict(*extra)} for c, d, *extra in columns],
                "source": {"type": "parquet", "paths": [str(data / f"{name}.parquet")]}}
        return {**spec, "create_time": create_time} if create_time else spec

    diskgraph.ingest(
        config_json=json.dumps({"dataset_name": "cutoffs", "timestamp_unit": "seconds", "tables": [
            table("users", "id", [("id", "int64"), ("signup", "timestamp", {"timestamp_unit": "millis"}),
                                  ("tier", "string"),
                                  ("age", "int32"), ("raw", "bytes"), ("seen", "timestamp"),
                                  ("born", "date"), ("price", "decimal128", {"precision": 10, "scale": 2}),
                                  ("big", "u_int64")],
                  "signup"),
            table("items", "item_id", [("item_id", "int64"), ("name", "string")]),
        ]}),
        output_dir=str(out),
        col_stats=col_stats,
        **options,
    )
    return str(out)


def _stats(index: str) -> dict:
    root = pathlib.Path(index) / "col_stats"
    return {d: {t: json.loads((root / d / f"{t}.col_stats.json").read_text())[t] for t in REQUEST["columns"]}
            for d in REQUEST["dirs"]}


@pytest.fixture(scope="module")
def ingested(data, tmp_path_factory) -> str:
    return _ingest(data, tmp_path_factory.mktemp("ingested") / "index",
                   cutoff_col_stats=json.dumps(REQUEST))


def test_ingest_counts_the_rows_created_at_or_before_each_cutoff(data, ingested):
    table = pq.read_table(data / "users.parquet")
    users = {**table.to_pydict(), "seen": table["seen"].cast(pa.int64()).to_pylist()}
    kept = [i for i, ms in enumerate(users["signup"]) if ms <= CUTOFF * 1_000]
    for directory, rows in zip(REQUEST["dirs"], (kept, range(len(users["id"])))):
        stats = _stats(ingested)[directory]
        tiers = collections.Counter(users["tier"][i] for i in rows if users["tier"][i] is not None)
        assert dict(map(tuple, stats["users"]["tier"]["category_counts"])) == tiers
        assert not stats["users"]["tier"]["category_counts_capped"]
        ages = np.array([users["age"][i] for i in rows])
        age = stats["users"]["age"]
        assert age["mean"] == pytest.approx(ages.mean(), rel=1e-12)
        assert age["std"] == pytest.approx(ages.std(ddof=1), rel=1e-12)
        ranks = [np.mean(ages <= q) for q in age["quantiles"]]
        assert ranks == pytest.approx(QUANTILES, abs=0.02)
        assert {int(k): v for k, v in age["category_counts"]} == collections.Counter(ages.tolist())
        raw = collections.Counter(users["raw"][i].hex() for i in rows)
        assert dict(map(tuple, stats["users"]["raw"]["category_counts"])) == raw
        seen = collections.Counter(users["seen"][i] // 10**6 for i in rows)
        assert {int(k): v for k, v in stats["users"]["seen"]["category_counts"]} == seen
        born = collections.Counter((users["born"][i] - dt.date(1970, 1, 1)).days for i in rows)
        assert {int(k): v for k, v in stats["users"]["born"]["category_counts"]} == born
        prices = np.array([float(users["price"][i]) for i in rows])
        assert stats["users"]["price"]["mean"] == pytest.approx(prices.mean(), rel=1e-12)
        decimals = collections.Counter(str(users["price"][i]) for i in rows)
        assert dict(map(tuple, stats["users"]["price"]["category_counts"])) == decimals
        # A table without a create time counts every row at every cutoff.
        assert sum(v for _, v in stats["items"]["name"]["category_counts"]) == 500


def test_recompute_from_the_index_matches_ingest(data, ingested, tmp_path):
    index = tmp_path / "index"
    shutil.copytree(ingested, index, ignore=shutil.ignore_patterns("col_stats"))
    users = {**REQUEST["columns"]["users"], "age": ["categorical"]}
    diskgraph.cutoff_col_stats(str(index), json.dumps({**REQUEST, "columns": {**REQUEST["columns"], "users": users}}),
                               num_threads=4)
    # Statistics added to a column join its entry; its other keys and the other columns stay.
    diskgraph.cutoff_col_stats(str(index), json.dumps({**REQUEST, "columns": {"users": {"age": ["numerical"]}}}))
    assert _stats(str(index)) == _stats(ingested)
    counts = _stats(ingested)["every-row"]["users"]["big"]["category_counts"]
    assert {int(key) for key, _ in counts} == {5, -(2**63), -1}


def test_rejects_unknown_columns_and_exclusive_options(data, ingested, tmp_path):
    with pytest.raises(Exception, match="has no column missing"):
        diskgraph.cutoff_col_stats(ingested, json.dumps(
            {**REQUEST, "columns": {"users": {"missing": ["categorical"]}}}))
    with pytest.raises(Exception, match="more than one kind of category counts"):
        diskgraph.cutoff_col_stats(ingested, json.dumps(
            {**REQUEST, "columns": {"users": {"tier": ["categorical", "id"]}}}))
    with pytest.raises(Exception, match="its own directory"):
        diskgraph.cutoff_col_stats(ingested, json.dumps({**REQUEST, "dirs": ["same", "same"]}))
    with pytest.raises(Exception, match="exclusive"):
        _ingest(data, tmp_path / "both", True, cutoff_col_stats=json.dumps(REQUEST))


def test_rows_the_index_does_not_store_count_on_neither_route(tmp_path):
    times = [dt.datetime.fromtimestamp(T0 + i, dt.UTC).strftime("%Y-%m-%d %H:%M:%S") for i in range(60)]
    times[3], times[4] = "garbage", None
    tiers = [f"t{i % 5}" for i in range(60)]
    tiers[7] = "x" * (65 << 20)
    pq.write_table(pa.table({"id": range(60), "at": times, "tier": pa.array(tiers, pa.large_string())}),
                   tmp_path / "events.parquet")
    request = {"cutoffs": [T0 + 29, None], "dirs": ["at-cutoff", "every-row"],
               "columns": {"events": {"tier": ["categorical"]}}}
    index = tmp_path / "index"
    diskgraph.ingest(
        config_json=json.dumps({"dataset_name": "dropped", "tables": [{
            "name": "events", "primary_key": "id", "create_time": "at",
            "columns": [{"name": "id", "ingested_dtype": "int64"},
                        {"name": "at", "ingested_dtype": "timestamp", "timestamp_format": "%Y-%m-%d %H:%M:%S"},
                        {"name": "tier", "ingested_dtype": "string"}],
            "source": {"type": "parquet", "paths": [str(tmp_path / "events.parquet")]}}]}),
        output_dir=str(index), col_stats=False, cutoff_col_stats=json.dumps(request),
    )

    def counts():
        return [dict(map(tuple, json.loads((index / "col_stats" / d / "events.col_stats.json").read_text())
                         ["events"]["tier"]["category_counts"])) for d in request["dirs"]]

    ingested = counts()
    shutil.rmtree(index / "col_stats")
    diskgraph.cutoff_col_stats(str(index), json.dumps(request))
    stored = [i for i in range(60) if i not in (3, 4, 7)]
    assert counts() == ingested == [collections.Counter(tiers[i] for i in stored if i <= 29),
                                    collections.Counter(tiers[i] for i in stored)]


def _counts_at(tmp_path, unit: str, times_ns: list[int], cutoff_ns: int) -> dict:
    tiers = [f"row{i}" for i in range(len(times_ns))]
    pq.write_table(pa.table({"id": range(len(tiers)), "at": pa.array(times_ns, pa.timestamp("ns")), "tier": tiers}),
                   tmp_path / "events.parquet")
    request = {"cutoffs": [diskgraph.convert_ticks(cutoff_ns, "nanoseconds", unit)], "dirs": ["cut"],
               "columns": {"events": {"tier": ["categorical"]}}}
    diskgraph.ingest(
        config_json=json.dumps({"dataset_name": "clock", "timestamp_unit": unit, "tables": [{
            "name": "events", "primary_key": "id", "create_time": "at",
            "columns": [{"name": "id", "ingested_dtype": "int64"},
                        {"name": "at", "ingested_dtype": "timestamp", "timestamp_unit": "nanos"},
                        {"name": "tier", "ingested_dtype": "string"}],
            "source": {"type": "parquet", "paths": [str(tmp_path / "events.parquet")]}}]}),
        output_dir=str(tmp_path / "index"), col_stats=False, cutoff_col_stats=json.dumps(request),
    )
    stats = json.loads((tmp_path / "index" / "col_stats" / "cut" / "events.col_stats.json").read_text())
    return dict(map(tuple, stats["events"]["tier"]["category_counts"]))


@pytest.mark.parametrize("unit, tick", [("milliseconds", 10**6), ("days", 86_400 * 10**9)])
def test_a_cutoff_truncates_to_its_tick_so_later_rows_in_that_tick_are_excluded(tmp_path, unit, tick):
    start = T0 // 86_400 * 86_400 * 10**9
    assert _counts_at(tmp_path, unit, [start, start + tick // 4, start + tick], start + tick // 2) == {"row0": 1}


def test_a_sub_second_cutoff_on_a_nanosecond_index_is_exact(tmp_path):
    cutoff = T0 * 10**9 + 123_456_789
    assert _counts_at(tmp_path, "nanoseconds", [cutoff, cutoff + 1], cutoff) == {"row0": 1}
