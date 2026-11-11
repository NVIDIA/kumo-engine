# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Column statistics at a cutoff: ingest, a rescan of the index and Arrow agree byte for byte."""

import collections
import fractions
import json
import math
import pathlib
import random
import shutil
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import diskgraph

T0 = 1_700_000_000
CUTOFF = T0 + 2_500
DAY_CUTOFF = (T0 // 86_400 + 3) * 86_400
CUTOFFS = [CUTOFF, DAY_CUTOFF, None]
NUMERIC = ["counts", "moments", "bins"]
REQUESTS = {
    "users": {
        "signup": ["counts"],
        "age": NUMERIC,
        "score": NUMERIC,
        "flag": ["counts"],
        "tier": ["counts"],
        "big": NUMERIC,
        "seen": ["counts"],
        "born": ["counts"],
    },
    "orders": {"placed": ["counts"], "amount": NUMERIC},
    "items": {"name": ["counts"], "price": NUMERIC},
}
REQUEST_LIST = [[t, c, s] for t, cols in REQUESTS.items() for c, stats in cols.items()
                for s in stats]
CREATE_TIME = {"users": "signup", "orders": "placed", "items": None}


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory) -> pathlib.Path:
    rng = random.Random(0)
    root = tmp_path_factory.mktemp("cutoff_stats_data")
    n = 3_000
    ids = list(range(n)) + [7, 8]  # duplicate primary keys are stored too
    signup_ms = [(T0 + rng.randrange(-2_000, 7_000)) * 1_000 + rng.randrange(1_000)
                 for _ in ids]
    signup_ms[:5] = [CUTOFF * 1_000, CUTOFF * 1_000 + 999, (CUTOFF + 1) * 1_000, -1_500, None]
    specials = [float("nan"), float("inf"), float("-inf"), -0.0, 0.0, None, 1e300, -1e-300]
    pq.write_table(pa.table({
        "user_id": pa.array(ids, pa.int64()),
        "signup": pa.array(signup_ms, pa.timestamp("ms")),
        "age": pa.array([None if i % 11 == 0 else rng.randrange(-5, 90) for i in ids],
                        pa.int32()),
        "score": pa.array([specials[i] if i < len(specials) else rng.gauss(3.0, 2.0) * 1e6
                           for i in range(len(ids))], pa.float64()),
        "flag": pa.array([None if i % 13 == 0 else i % 3 == 0 for i in ids], pa.bool_()),
        "tier": pa.array([None if i % 17 == 0 else rng.choice(["gold", "silver", "", "é"])
                          for i in ids], pa.string()),
        "big": pa.array([2**64 - 1 - rng.randrange(3) if i % 2 else rng.randrange(10)
                         for i in ids], pa.uint64()),
        "seen": pa.array([None if i % 5 == 0 else (T0 + rng.randrange(100)) * 10**6 + 17
                          for i in ids], pa.timestamp("us")),
        "born": pa.array([None if i % 7 == 0 else 19_000 + rng.randrange(50) for i in ids],
                         pa.date32()),
    }), root / "users.parquet", row_group_size=300)
    pq.write_table(pa.table({
        "order_id": pa.array(range(2_000), pa.int64()),
        "user_id": pa.array([rng.randrange(n) for _ in range(2_000)], pa.int64()),
        "placed": pa.array([T0 // 86_400 + rng.randrange(-10, 10) for _ in range(2_000)],
                           pa.date32()),
        "amount": pa.array([None if i % 9 == 0 else rng.uniform(-1, 1000) for i in range(2_000)],
                           pa.float32()),
    }), root / "orders.parquet", row_group_size=300)
    pq.write_table(pa.table({
        "item_id": pa.array(range(500), pa.int64()),
        "name": pa.array([f"item{i % 37}" for i in range(500)], pa.string()),
        "price": pa.array([i * 0.25 - 10 for i in range(500)], pa.float64()),
    }), root / "items.parquet", row_group_size=100)
    return root


def _config(data_dir: pathlib.Path, num_shards: int) -> str:
    def column(name, dtype):
        return {"name": name, "ingested_dtype": dtype}

    def source(name):
        return {"type": "parquet", "paths": [str(data_dir / f"{name}.parquet")]}

    return json.dumps({
        "dataset_name": "cutoff_stats",
        "num_shards": num_shards,
        "timestamp_unit": "seconds",
        "tables": [
            {"name": "users", "primary_key": "user_id", "create_time": "signup",
             "columns": [column("user_id", "int64"), column("signup", "timestamp"),
                         column("age", "int32"), column("score", "float64"),
                         column("flag", "bool"), column("tier", "string"),
                         column("big", "u_int64"), column("seen", "timestamp"),
                         column("born", "date")],
             "source": source("users")},
            {"name": "orders", "primary_key": "order_id", "create_time": "placed",
             "columns": [column("order_id", "int64"), column("user_id", "int64"),
                         column("placed", "date"), column("amount", "float32")],
             "fkeys": [{"src_column": "user_id", "dst_table": "users"}],
             "source": source("orders")},
            {"name": "items", "primary_key": "item_id",
             "columns": [column("item_id", "int64"), column("name", "string"),
                         column("price", "float64")],
             "source": source("items")},
        ],
    })


def _ingest(data_dir, out, threads, num_shards, requests=True) -> str:
    diskgraph.ingest(
        config_json=_config(data_dir, num_shards),
        output_dir=str(out),
        encode_threads=threads,
        index_threads=threads,
        col_stats_requests=json.dumps({"cutoffs": CUTOFFS, "requests": REQUEST_LIST})
        if requests else None,
    )
    return str(out)


def _files(index: str) -> dict:
    root = pathlib.Path(index) / "col_stats"
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*"))
            if p.is_file()}


def _ensure_all(index: str) -> dict:
    return {cutoff: diskgraph.ensure_col_stats(index, cutoff, REQUEST_LIST, num_threads=4)
            for cutoff in CUTOFFS}


@pytest.fixture(scope="module")
def builds(data_dir, tmp_path_factory) -> dict:
    """Per (encode/index threads, shards): the index, its ingested files, its rescans."""
    out = {}
    for threads, shards in [(1, 1), (1, 3), (4, 1), (4, 3)]:
        index = _ingest(data_dir, tmp_path_factory.mktemp(f"index_t{threads}_s{shards}"),
                        threads, shards)
        ingested = _files(index)
        shutil.rmtree(pathlib.Path(index) / "col_stats")
        out[threads, shards] = index, ingested, _ensure_all(index)
    return out


@pytest.fixture(scope="module")
def built(builds):
    return builds[1, 1]


def _data(files: dict) -> dict:
    out = {}
    for path, body in files.items():
        stored = json.loads(body)
        if path.endswith("manifest.json"):
            continue
        key = (stored["cutoff"]["value"], stored["table"], stored["column"], stored["stat"])
        out[key] = stored["data"]
    return out


def test_ingest_and_recompute_write_identical_files(builds):
    for index, ingested, ensured in builds.values():
        assert _files(index) == ingested
        assert len(_data(ingested)) == len(CUTOFFS) * len(REQUEST_LIST)
        for cutoff, result in ensured.items():
            assert {f["source"] for f in result["files"].values()} == {"recompute"}
            stored = json.loads(pathlib.Path(result["files"]["users/score.moments"]["path"])
                                .read_text())
            assert (stored["cutoff_id"], stored["index_id"]) == (result["cutoff_id"],
                                                                 result["index_id"])
            assert stored["cutoff"] == {"value": cutoff, "unit": "seconds", "inclusive": True}
            again = diskgraph.ensure_col_stats(index, cutoff, REQUEST_LIST)
            assert {f["source"] for f in again["files"].values()} == {"existing"}
        assert result["index_id"] == diskgraph.col_stats_index_id(index)


def test_thread_and_shard_counts_do_not_change_data(builds):
    data = [_data(ingested) for _, ingested, _ in builds.values()]
    assert all(d == data[0] for d in data[1:])


def _arrow_tables(data_dir):
    return {t: pq.read_table(data_dir / f"{t}.parquet") for t in REQUESTS}


def test_arrow_matches_files(data_dir, built):
    tables = _arrow_tables(data_dir)
    files = _data(built[1])
    for cutoff in CUTOFFS:
        for table, columns in REQUESTS.items():
            ct = CREATE_TIME[table]
            for column, stats in columns.items():
                got = diskgraph.col_stats_from_arrow(
                    tables[table][column].combine_chunks(),
                    tables[table][ct].combine_chunks() if ct else None,
                    cutoff, stats)
                assert got == {s: files[(cutoff, table, column, s)] for s in stats}


def test_statistics_are_exact(data_dir, built):
    table = pq.read_table(data_dir / "users.parquet")
    users = table.to_pydict()
    files = _data(built[1])
    # Ingest rounds Arrow timestamps up to the clock, so only an exact second meets the cutoff.
    signup_ms = table["signup"].cast(pa.int64()).to_pylist()
    rows = [i for i, ms in enumerate(signup_ms) if ms is not None and -(-ms // 1000) <= CUTOFF]
    assert 0 in rows and 1 not in rows and 3 in rows and 4 not in rows

    scores = [users["score"][i] for i in rows]
    finite = [x for x in scores if x is not None and math.isfinite(x)]
    n = len(finite)
    exact_sum = sum(map(fractions.Fraction, finite))
    exact_squares = sum(fractions.Fraction(x) ** 2 for x in finite)
    moments = files[(CUTOFF, "users", "score", "moments")]
    assert moments == {
        "count": n,
        "nulls": scores.count(None),
        "non_finite": len(scores) - n - scores.count(None),
        "mean": float(exact_sum / n),
        "m2": float((n * exact_squares - exact_sum ** 2) / n),
    }

    tiers = collections.Counter(users["tier"][i] for i in rows)
    nulls = tiers.pop(None)
    counts = files[(CUTOFF, "users", "tier", "counts")]
    assert counts == {
        "count": sum(tiers.values()), "nulls": nulls, "non_finite": 0, "truncated": False,
        "values": [list(kv) for kv in sorted(tiers.items(), key=lambda kv: (-kv[1], kv[0]))],
    }
    big = files[(None, "users", "big", "bins")]
    assert (big["min"], big["max"], big["bins_per_octave"]) == (0, 2**64 - 1, 128)


def test_concurrent_ensure_publishes_valid_identical_files(built, tmp_path):
    index = tmp_path / "index"
    shutil.copytree(built[0], index, ignore=shutil.ignore_patterns("col_stats"))
    script = ("import diskgraph, json, sys; "
              "diskgraph.ensure_col_stats(sys.argv[1], None, json.loads(sys.argv[2]))")
    procs = [subprocess.Popen([sys.executable, "-c", script, str(index),
                               json.dumps(REQUEST_LIST)]) for _ in range(2)]
    assert [p.wait() for p in procs] == [0, 0]
    result = diskgraph.ensure_col_stats(str(index), None, REQUEST_LIST)
    assert {f["source"] for f in result["files"].values()} == {"existing"}
    reference = {k: v for k, v in _files(built[0]).items()
                 if k.startswith(result["cutoff_id"])}
    assert _files(str(index)) == reference


def test_index_without_id_uses_fallback(data_dir, built, tmp_path):
    index = _ingest(data_dir, tmp_path / "plain", 1, 1, requests=False)
    assert not (pathlib.Path(index) / "col_stats").exists()
    metadata_path = pathlib.Path(index) / "index_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    recorded = metadata.pop("index_id")
    assert diskgraph.col_stats_index_id(index) == recorded
    metadata_path.write_text(json.dumps(metadata))
    fallback = diskgraph.col_stats_index_id(index)
    assert len(fallback) == 64 and fallback != recorded
    result = diskgraph.ensure_col_stats(index, CUTOFF, REQUEST_LIST)
    assert result["index_id"] == fallback
    assert _data(_files(index)) == {k: v for k, v in _data(built[1]).items() if k[0] == CUTOFF}


def test_rejects_unsupported_requests(built):
    with pytest.raises(Exception, match="does not support moments"):
        diskgraph.ensure_col_stats(built[0], None, [["users", "tier", "moments"]])
    with pytest.raises(Exception, match="no column"):
        diskgraph.ensure_col_stats(built[0], None, [["users", "missing", "counts"]])
    with pytest.raises(Exception, match="unknown column statistic"):
        diskgraph.col_stats_from_arrow(pa.array([1]), None, None, ["median"])
    assert diskgraph.col_stats_from_arrow(pa.array([1]), None, None, []) == {}


DAY_START = 19_700 * 86_400
CLOCK_CUTOFF = DAY_START + 10 * 3_600  # 10:00 on day 19700
# Milliseconds from the cutoff: 00:00 and 09:00 that day, four just before or at it, three just
# after, then 00:00 and 00:00:01 the next day.
OFFSETS_MS = [-36_000_000, -3_600_000, -500, -1, 0, 1, 500, 1_000, 50_400_000, 50_401_000]


@pytest.mark.parametrize("unit, ticks, kept", [
    ("milliseconds", CLOCK_CUTOFF * 1_000, 5),
    ("seconds", CLOCK_CUTOFF, 5),
    # A coarser clock rounds the cutoff up to the end of its day, like the rows' times.
    ("days", 19_701, 9),
])
def test_cutoff_converts_like_row_times(tmp_path, unit, ticks, kept):
    assert diskgraph.cutoff_ticks(CLOCK_CUTOFF, unit) == ticks
    pq.write_table(pa.table({
        "id": pa.array(range(len(OFFSETS_MS)), pa.int64()),
        "ts": pa.array([CLOCK_CUTOFF * 1_000 + o for o in OFFSETS_MS], pa.timestamp("ms")),
    }), tmp_path / "events.parquet")
    requests = [["events", "id", "counts"]]
    diskgraph.ingest(
        config_json=json.dumps({
            "dataset_name": "clocks", "timestamp_unit": unit,
            "tables": [{"name": "events", "primary_key": "id", "create_time": "ts",
                        "columns": [{"name": "id", "ingested_dtype": "int64"},
                                    {"name": "ts", "ingested_dtype": "timestamp"}],
                        "source": {"type": "parquet",
                                   "paths": [str(tmp_path / "events.parquet")]}}],
        }),
        output_dir=str(tmp_path / "index"),
        col_stats_requests=json.dumps({"cutoffs": [ticks], "requests": requests}),
    )
    index = str(tmp_path / "index")
    ingested = _files(index)
    shutil.rmtree(tmp_path / "index" / "col_stats")
    result = diskgraph.ensure_col_stats(index, ticks, requests)
    assert result["unit"] == unit and _files(index) == ingested
    counts = json.loads(pathlib.Path(result["files"]["events/id.counts"]["path"]).read_text())
    assert sorted(v for v, _ in counts["data"]["values"]) == list(range(kept))


def test_cutoff_ticks_rejects_what_the_clock_cannot_hold():
    with pytest.raises(Exception, match="nanoseconds|Nanoseconds"):
        diskgraph.cutoff_ticks(2**62, "nanoseconds")
    with pytest.raises(Exception, match="unsupported timestamp_unit"):
        diskgraph.cutoff_ticks(0, "minutes")
