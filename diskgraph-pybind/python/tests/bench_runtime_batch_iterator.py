"""Simple end-to-end throughput bench for RuntimeBatchIterator.

Not a pytest test — run directly:

  cd diskgraph-pybind
  python3 python/tests/bench_runtime_batch_iterator.py [--n-users N] [--orders-per-user O] \
      [--n-seeds S] [--batch-size B] [--n-iter K]

Reports wall time, batches/sec, batches' avg edge count, and peak RSS
for fused vs non-fused dispatch. Establishes a baseline number we can
defend (and compare against PR 269's TaskBatchIterator if/when we
resurrect it for a head-to-head — see plan doc §9 Follow-up C).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import resource
import sys
import tempfile
import time
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

import diskgraph
from diskgraph import RuntimeIndex


def make_runtime_batch_iterator(dataset_ref, **kwargs):
    """One-shot open-and-iterate, for tests that use an index exactly once.

    Deliberately NOT part of the public API. Production callers hold a
    :class:`diskgraph.RuntimeIndex` and call ``iter()`` per epoch -- that
    separation is the whole point, since opening the index is ~1.6 s and
    choosing rows is microseconds. These tests genuinely do open once and
    iterate once, so they keep a local convenience instead of repeating the
    two-step at every call site.
    """
    from diskgraph import RuntimeIndex

    index = RuntimeIndex(
        dataset_ref,
        rnd_seed=kwargs.get("rnd_seed"),
        num_threads=kwargs.pop("num_threads", None),
    )
    return index.iter(**kwargs)



def _peak_rss_mb() -> float:
    # ru_maxrss is KB on Linux, bytes on Darwin.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1024.0 if sys.platform == "linux" else rss / (1024.0 * 1024.0)


def _build_fixture(
    root: pathlib.Path,
    n_users: int,
    orders_per_user: int,
    n_seeds: int,
) -> dict[str, Any]:
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    parent_root = root / "parent"

    pq.write_table(
        pa.table({"user_id": pa.array(list(range(n_users)), type=pa.int64())}),
        data_dir / "users.parquet",
    )
    order_ids, order_uids, order_times = [], [], []
    for u in range(n_users):
        for k in range(orders_per_user):
            order_ids.append(u * orders_per_user + k)
            order_uids.append(u)
            order_times.append([100, 200, 2000, 3000][k % 4])
    pq.write_table(
        pa.table({
            "order_id": pa.array(order_ids, type=pa.int64()),
            "user_id": pa.array(order_uids, type=pa.int64()),
            "order_time": pa.array(order_times, type=pa.timestamp("s")),
        }),
        data_dir / "orders.parquet",
    )
    diskgraph.ingest(
        config_json=json.dumps({
            "dataset_name": "bench_parent",
            "num_shards": 1,
            "tables": [
                {
                    "name": "users",
                    "columns": [{"name": "user_id", "ingested_dtype": "int64"}],
                    "primary_key": "user_id",
                    "source": {"type": "parquet", "paths": [str(data_dir / "users.parquet")]},
                },
                {
                    "name": "orders",
                    "columns": [
                        {"name": "order_id", "ingested_dtype": "int64"},
                        {"name": "user_id", "ingested_dtype": "int64"},
                        {"name": "order_time", "ingested_dtype": "timestamp"},
                    ],
                    "primary_key": "order_id",
                    "create_time": "order_time",
                    "fkeys": [{
                        "src_column": "user_id",
                        "dst_table": "users",
                        "forward": {},
                        "reverse": {
                            "time": {
                                "table": "orders", "column": "order_time",
                                "bitwidth": 32, "unit": "s",
                            },
                        },
                    }],
                    "source": {"type": "parquet", "paths": [str(data_dir / "orders.parquet")]},
                },
            ],
        }),
        output_dir=str(parent_root),
        col_stats=False,
    )

    pq.write_table(
        pa.table({
            "ENTITY1": pa.array([i % n_users for i in range(n_seeds)], type=pa.int64()),
            "TIMESTAMP": pa.array(
                [1000 + i for i in range(n_seeds)], type=pa.timestamp("s")
            ),
            "TARGET1": pa.array([i * 0.1 for i in range(n_seeds)], type=pa.float64()),
        }),
        data_dir / "seeds.parquet",
    )
    tt_id = "tt_bench"
    seed_output = parent_root / "train_tables" / tt_id
    diskgraph.ingest(
        config_json=json.dumps({
            "dataset_name": "bench_tt",
            "num_shards": 1,
            "tables": [{
                "name": "train_seeds",
                "columns": [
                    {"name": "ENTITY1", "ingested_dtype": "int64"},
                    {"name": "TIMESTAMP", "ingested_dtype": "timestamp"},
                    {"name": "TARGET1", "ingested_dtype": "float64"},
                ],
                "primary_key": None,
                "create_time": "TIMESTAMP",
                "fkeys": [{
                    "src_column": "ENTITY1", "dst_table": "users",
                    "forward": {
                        "time": {
                            "table": "train_seeds", "column": "TIMESTAMP",
                            "bitwidth": 32, "unit": "s",
                        },
                    },
                    "external_parent": True,
                    "bidirectional": False,
                }],
                "source": {"type": "parquet", "paths": [str(data_dir / "seeds.parquet")]},
            }],
        }),
        output_dir=str(seed_output),
        col_stats=True,
        parent_index_path=str(parent_root),
    )

    manifest = {
        "kind": "diskgraph_task_dataset",
        "graph": {"index_path": str(parent_root)},
        "stages": {
            "train": {
                "table_name": "train_seeds",
                "index_path": str(seed_output),
                "row_count": n_seeds,
            }
        },
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return {"manifest_path": str(manifest_path), "n_seeds": n_seeds}


def _run_one(fixture: dict, batch_size: int, n_iter: int, fused: bool) -> dict:
    plan = {
        "layers": [
            {"fanouts": {}, "default_fanout": {
                "fanout": 8, "mode": "uniform", "absent_key_policy": "uniform_fallback",
            }},
            {"fanouts": {}, "default_fanout": {
                "fanout": 8, "mode": "uniform", "absent_key_policy": "uniform_fallback",
            }},
        ],
    }

    # Warm-up batch — page-in mmapped data + JIT-warm the iterator state.
    iterator, _ = make_runtime_batch_iterator(
        dataset_ref=fixture["manifest_path"],
        stage="train",
        sampling_plan=plan,
        num_workers=1, worker_id=0,
        shuffle=False, rnd_seed=None, epoch=0,
        batch_size=batch_size, drop_last=False,
        fused=fused,
    )
    _ = next(iter(iterator))

    # Bench loop: fresh iterator, take up to n_iter batches (or until exhausted).
    iterator, _ = make_runtime_batch_iterator(
        dataset_ref=fixture["manifest_path"],
        stage="train",
        sampling_plan=plan,
        num_workers=1, worker_id=0,
        shuffle=False, rnd_seed=None, epoch=0,
        batch_size=batch_size, drop_last=False,
        fused=fused,
    )

    edge_counts: list[int] = []
    t0 = time.perf_counter()
    n = 0
    for batch in iterator:
        edges = batch["graphs"]["main"]["edge_index_dict"]
        edge_counts.append(sum(len(e["src"]) for e in edges))
        n += 1
        if n >= n_iter:
            break
    wall = time.perf_counter() - t0

    return {
        "fused": fused,
        "batches": n,
        "wall_s": wall,
        "batches_per_s": n / wall if wall > 0 else float("inf"),
        "avg_edges_per_batch": (sum(edge_counts) / len(edge_counts)) if edge_counts else 0,
        "peak_rss_mb": _peak_rss_mb(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-users", type=int, default=10_000)
    parser.add_argument("--orders-per-user", type=int, default=8)
    parser.add_argument("--n-seeds", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-iter", type=int, default=100)
    args = parser.parse_args()

    print(
        f"# fixture: n_users={args.n_users}, orders_per_user={args.orders_per_user}, "
        f"n_seeds={args.n_seeds} (= {args.n_users * args.orders_per_user} orders)",
    )
    print(f"# bench: batch_size={args.batch_size}, n_iter={args.n_iter}")
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        t_ingest = time.perf_counter()
        fixture = _build_fixture(
            root,
            n_users=args.n_users,
            orders_per_user=args.orders_per_user,
            n_seeds=args.n_seeds,
        )
        ingest_s = time.perf_counter() - t_ingest
        print(f"# ingest: {ingest_s:.2f}s ({args.n_users + args.n_users * args.orders_per_user + args.n_seeds:,} rows total)")

        for mode in (False, True):
            stats = _run_one(fixture, args.batch_size, args.n_iter, fused=mode)
            label = "fused" if stats["fused"] else "non-fused"
            print(
                f"  {label:>10}: batches={stats['batches']:>4}, "
                f"wall={stats['wall_s']:>6.2f}s, "
                f"rate={stats['batches_per_s']:>6.1f} batches/s, "
                f"avg_edges/batch={stats['avg_edges_per_batch']:>7.1f}, "
                f"peak_rss={stats['peak_rss_mb']:>6.0f} MB"
            )


if __name__ == "__main__":
    main()
