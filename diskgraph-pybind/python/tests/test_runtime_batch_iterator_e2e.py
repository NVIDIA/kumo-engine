"""End-to-end test for the RuntimeBatchIterator pybind class.

Pins the Phase 3 acceptance criterion from plan doc §2.5: build a parent
DG index + a seed-table extension under ``train_tables/<tt_id>/``, open
the merged reader via ``RuntimeBatchIterator``, iterate, and assert the
emitted PyDict structurally matches PR 28765's ``runtime_batch_to_hetero``
contract.

This is the test we lacked when we declared "Phase 3 done" at the Rust
component level — it exercises:
  * parent index ingest from Python (via ``diskgraph.ingest``)
  * seed extension ingest with ``parent_index_path`` so the external
    pkmap loader resolves cross-boundary FKs at ingest time
  * a flat ``kind=diskgraph_task_dataset`` runtime manifest
  * ``make_runtime_batch_iterator`` → multi-worker offset sharding +
    shuffle determinism + Arrow round-trip → PyDict iteration
  * structural assertions on the dict against the documented contract
"""
from __future__ import annotations

import inspect
import json
import pathlib
import struct
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import diskgraph
from diskgraph import make_runtime_batch_iterator, task_stage_size


# ---------------------------------------------------------------------------
# Fixture: users + orders parent (bidirectional FK with order_time sort_key)
# + a seed extension at train_tables/tt_e2e/ pointing at parent.users.
# Returns the runtime manifest path that make_runtime_batch_iterator consumes.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def seed_table_fixture(tmp_path_factory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("seed_e2e")
    data_dir = root / "data"
    data_dir.mkdir()
    parent_root = root / "parent_idx"

    n_users = 8
    orders_per_user = 4

    # 1. Parent parquet — users + orders.
    users_pq = data_dir / "users.parquet"
    pq.write_table(
        pa.table({"user_id": pa.array(list(range(n_users)), type=pa.int64())}),
        users_pq,
    )

    order_ids: list[int] = []
    order_user_ids: list[int] = []
    # order_time bucketed so we have something to filter on at hop 1.
    order_times: list[int] = []
    for u in range(n_users):
        for k in range(orders_per_user):
            order_ids.append(u * orders_per_user + k)
            order_user_ids.append(u)
            order_times.append([100, 200, 2000, 3000][k])
    orders_pq = data_dir / "orders.parquet"
    pq.write_table(
        pa.table({
            "order_id": pa.array(order_ids, type=pa.int64()),
            "user_id": pa.array(order_user_ids, type=pa.int64()),
            "order_time": pa.array(order_times, type=pa.timestamp("s")),
        }),
        orders_pq,
    )

    parent_config = {
        "dataset_name": "seed_e2e_parent",
        "num_shards": 1,
        "tables": [
            {
                "name": "users",
                "columns": [{"name": "user_id", "ingested_dtype": "int64"}],
                "primary_key": "user_id",
                "source": {"type": "parquet", "paths": [str(users_pq)]},
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
                        # Time-encoded reverse direction so the merged
                        # users→orders edge gets a dest_sort_key, enabling
                        # TemporalMostRecent at hop 1 if anyone uses it.
                        "time": {
                            "table": "orders",
                            "column": "order_time",
                            "bitwidth": 32,
                            "unit": "s",
                        },
                    },
                }],
                "source": {"type": "parquet", "paths": [str(orders_pq)]},
            },
        ],
    }
    diskgraph.ingest(
        config_json=json.dumps(parent_config),
        output_dir=str(parent_root),
        col_stats=False,
    )

    # 2. Seed extension at <parent>/train_tables/tt_e2e/.
    n_seeds = 6
    seed_pq = data_dir / "seeds.parquet"
    # Resolution mode: ENTITY1 is pre-resolved to entity_node_id (users node-id
    # == user pk == ENTITY1 in this single-shard fixture).
    pq.write_table(
        pa.table({
            "ENTITY1": pa.array(list(range(n_seeds)), type=pa.int64()),
            "TIMESTAMP": pa.array(
                [1000 + i for i in range(n_seeds)], type=pa.timestamp("s")
            ),
            "TARGET1": pa.array([i * 0.1 for i in range(n_seeds)], type=pa.float64()),
            "entity_node_id": pa.array(list(range(n_seeds)), type=pa.uint64()),
        }),
        seed_pq,
    )

    seed_config = {
        "dataset_name": "seed_e2e_tt",
        "num_shards": 1,
        "tables": [{
            "name": "train_seeds",
            "columns": [
                {"name": "ENTITY1", "ingested_dtype": "int64"},
                {"name": "TIMESTAMP", "ingested_dtype": "timestamp"},
                {"name": "TARGET1", "ingested_dtype": "float64"},
                {"name": "entity_node_id", "ingested_dtype": "u_int64"},
            ],
            "primary_key": None,
            "create_time": "TIMESTAMP",
            "fkeys": [],
            "source": {"type": "parquet", "paths": [str(seed_pq)]},
        }],
    }
    tt_id = "tt_e2e"
    seed_output = parent_root / "train_tables" / tt_id
    diskgraph.ingest(
        config_json=json.dumps(seed_config),
        output_dir=str(seed_output),
        col_stats=True,
        parent_index_path=str(parent_root),
    )

    # 3. Flat runtime manifest (the shape make_runtime_batch_iterator
    # consumes — matches what execute_tt_plan writes in production).
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

    return {
        "manifest_path": str(manifest_path),
        "parent_root": str(parent_root),
        "tt_id": tt_id,
        "n_seeds": n_seeds,
        "n_users": n_users,
    }


def _sampling_plan() -> dict[str, Any]:
    # Depth-2: hop 0 = users (resolved seeds), hop 1 = users → orders.
    fan = {"fanout": 4, "mode": "uniform", "absent_key_policy": "uniform_fallback"}
    return {
        "layers": [
            {"fanouts": {}, "default_fanout": fan},
            {"fanouts": {}, "default_fanout": fan},
        ],
    }


# ---------------------------------------------------------------------------
# Contract assertions — what PR 28765's runtime_batch_to_hetero consumes.
# ---------------------------------------------------------------------------


def _assert_runtime_batch_dict_contract(batch: dict, batch_size: int, n_users: int) -> None:
    assert "task" in batch, f"batch missing 'task'; keys: {list(batch.keys())}"
    assert "graphs" in batch, f"batch missing 'graphs'; keys: {list(batch.keys())}"
    assert "raw_features" in batch, f"batch missing 'raw_features'; keys: {list(batch.keys())}"

    task = batch["task"]
    for key in ("input_ids", "tt_stage_node_ids", "seed_node_ids"):
        assert key in task, f"task missing {key!r}; got: {list(task.keys())}"
        assert len(task[key]) == batch_size, (
            f"task.{key} length {len(task[key])} != batch_size {batch_size}"
        )

    # ENTITY1 = i → parent users pk = i → row i. seed_node_ids should be < n_users.
    for sid in task["seed_node_ids"].to_pylist():
        assert sid < n_users, f"seed_node_id {sid} not a valid users row (n_users={n_users})"

    # seed_timestamps should be populated since the fixture has TIMESTAMP.
    assert "seed_timestamps" in task, "task missing seed_timestamps despite TIMESTAMP column"
    assert len(task["seed_timestamps"]) == batch_size

    # task.target is a RecordBatch with at least TARGET1.
    assert "target" in task, "task missing target"
    target = task["target"]
    target_cols = {f.name for f in target.schema}
    assert "TARGET1" in target_cols, f"target missing TARGET1; cols: {target_cols}"
    assert target.num_rows == batch_size

    # graphs.main is the canonical channel.
    assert "main" in batch["graphs"], f"graphs missing 'main'; keys: {list(batch['graphs'].keys())}"
    main = batch["graphs"]["main"]
    for key in ("node_ids", "batch_dict", "edge_index_dict"):
        assert key in main, f"graphs.main missing {key!r}; got: {list(main.keys())}"

    # node_ids[users] should include seed_node_ids.
    assert "users" in main["node_ids"], (
        f"graphs.main.node_ids missing 'users'; got: {list(main['node_ids'].keys())}"
    )
    users_in_subgraph = set(main["node_ids"]["users"].to_pylist())
    for sid in task["seed_node_ids"].to_pylist():
        assert sid in users_in_subgraph, (
            f"seed_node_id {sid} not in graphs.main.node_ids[users]={users_in_subgraph}"
        )

    # The seed namespace must NOT appear in graphs.main.node_ids.
    seed_ns = "tt_e2e/train_seeds"
    assert seed_ns not in main["node_ids"], (
        f"seed namespace {seed_ns} appears as topology in node_ids — should be task data only"
    )

    # Every edge's bounds must be valid against its declared edge_type.
    for entry in main["edge_index_dict"]:
        et = entry["edge_type"]
        src_t, _rel, dst_t = et["src"], et["rel"], et["dst"]
        assert src_t != seed_ns, f"seed-src edge leaked: {entry}"
        assert src_t in main["node_ids"], f"edge_type.src={src_t} not in node_ids"
        assert dst_t in main["node_ids"], f"edge_type.dst={dst_t} not in node_ids"
        for s in entry["src"].to_pylist():
            assert s < len(main["node_ids"][src_t])
        for d in entry["dst"].to_pylist():
            assert d < len(main["node_ids"][dst_t])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_iterator_yields_dict_matching_contract(seed_table_fixture):
    """Single-worker iteration: every batch matches PR 28765's contract."""
    iterator, num_samples = make_runtime_batch_iterator(
        dataset_ref=seed_table_fixture["manifest_path"],
        stage="train",
        entity_node_type="users",
        sampling_plan=_sampling_plan(),
        num_workers=1,
        worker_id=0,
        shuffle=False,
        rnd_seed=None,
        epoch=0,
        batch_size=3,
        drop_last=False,
    )
    assert num_samples == seed_table_fixture["n_seeds"], (
        f"single worker should see all {seed_table_fixture['n_seeds']} rows; got {num_samples}"
    )

    batches = list(iterator)
    # ceil(6 / 3) = 2 batches.
    assert len(batches) == 2, f"expected 2 batches of size 3; got {len(batches)}"
    for batch in batches:
        _assert_runtime_batch_dict_contract(
            batch,
            batch_size=3,
            n_users=seed_table_fixture["n_users"],
        )


def test_static_task_manifest_suppresses_seed_timestamps(
        seed_table_fixture, tmp_path):
    """Static task datasets may carry pseudo TIMESTAMP columns, but those
    columns must not become temporal sampler anchors."""
    manifest = json.loads(pathlib.Path(
        seed_table_fixture["manifest_path"]).read_text())
    manifest["task"] = {
        "task_type": "multiclass_classification",
        "is_temporal": False,
    }
    static_manifest = tmp_path / "static_manifest.json"
    static_manifest.write_text(json.dumps(manifest))

    iterator, _ = make_runtime_batch_iterator(
        dataset_ref=str(static_manifest),
        stage="train",
        entity_node_type="users",
        sampling_plan=_sampling_plan(),
        num_workers=1,
        worker_id=0,
        shuffle=False,
        rnd_seed=None,
        epoch=0,
        batch_size=3,
        drop_last=False,
    )

    batch = next(iter(iterator))
    assert "seed_timestamps" not in batch["task"]


def test_iterator_selected_stage_node_ids_preserve_order(seed_table_fixture):
    selected = [4, 1, 3]
    iterator, num_samples = make_runtime_batch_iterator(
        dataset_ref=seed_table_fixture["manifest_path"],
        stage="train",
        entity_node_type="users",
        sampling_plan=_sampling_plan(),
        num_workers=1,
        worker_id=0,
        shuffle=False,
        rnd_seed=None,
        epoch=0,
        batch_size=2,
        drop_last=False,
        selected_stage_node_ids=selected,
    )

    assert num_samples == len(selected)
    batches = list(iterator)
    assert [b["task"]["tt_stage_node_ids"].to_pylist() for b in batches] == [[4, 1], [3]]
    assert [b["task"]["input_ids"].to_pylist() for b in batches] == [[0, 1], [0]]

    seen: list[int] = []
    for batch in batches:
        batch_size = len(batch["task"]["input_ids"])
        _assert_runtime_batch_dict_contract(
            batch,
            batch_size=batch_size,
            n_users=seed_table_fixture["n_users"],
        )
        seen.extend(int(v) for v in batch["task"]["tt_stage_node_ids"].to_pylist())
    assert seen == selected


def test_iterator_selected_stage_node_ids_partitions_across_workers(
    seed_table_fixture,
):
    selected = [5, 0, 3, 1, 4, 2]
    num_workers = 8
    batch_size = 2
    seen_by_worker: list[list[int]] = []

    for worker_id in range(num_workers):
        iterator, num_samples = make_runtime_batch_iterator(
            dataset_ref=seed_table_fixture["manifest_path"],
            stage="train",
            entity_node_type="users",
            sampling_plan=_sampling_plan(),
            num_workers=num_workers,
            worker_id=worker_id,
            shuffle=False,
            rnd_seed=None,
            epoch=0,
            batch_size=batch_size,
            drop_last=False,
            selected_stage_node_ids=selected,
        )

        expected = [
            node_id
            for pos, node_id in enumerate(selected)
            if pos % num_workers == worker_id
        ]
        assert num_samples == len(expected)

        batches = list(iterator)
        assert len(batches) == (len(expected) + batch_size - 1) // batch_size

        seen: list[int] = []
        for batch in batches:
            batch_size_actual = len(batch["task"]["input_ids"])
            _assert_runtime_batch_dict_contract(
                batch,
                batch_size=batch_size_actual,
                n_users=seed_table_fixture["n_users"],
            )
            seen.extend(int(v) for v in batch["task"]["tt_stage_node_ids"].to_pylist())
        assert seen == expected
        seen_by_worker.append(seen)

    reassembled = [
        seen_by_worker[pos % num_workers][pos // num_workers]
        for pos in range(len(selected))
    ]
    assert reassembled == selected


def test_iterator_selected_stage_node_ids_allows_empty_selection(seed_table_fixture):
    iterator, num_samples = make_runtime_batch_iterator(
        dataset_ref=seed_table_fixture["manifest_path"],
        stage="train",
        entity_node_type="users",
        sampling_plan=_sampling_plan(),
        num_workers=1,
        worker_id=0,
        shuffle=False,
        rnd_seed=None,
        epoch=0,
        batch_size=2,
        selected_stage_node_ids=[],
    )

    assert num_samples == 0
    assert list(iterator) == []


def test_iterator_selected_stage_node_ids_rejects_invalid_id(seed_table_fixture):
    with pytest.raises(diskgraph.DiskgraphDataError, match="selected_stage_node_ids"):
        make_runtime_batch_iterator(
            dataset_ref=seed_table_fixture["manifest_path"],
            stage="train",
            entity_node_type="users",
            sampling_plan=_sampling_plan(),
            num_workers=1,
            worker_id=0,
            shuffle=False,
            rnd_seed=None,
            epoch=0,
            batch_size=2,
            selected_stage_node_ids=[0, 999],
        )


def test_iterator_weighted_seed_sampling_draws_with_replacement(
    seed_table_fixture,
    tmp_path,
):
    sidecar_dir = tmp_path / "weighted"
    sidecar_dir.mkdir()
    false_node_ids = sidecar_dir / "false_node_ids.bin"
    true_node_ids = sidecar_dir / "true_node_ids.bin"
    false_node_ids.write_bytes((0).to_bytes(8, "little") + (1).to_bytes(8, "little"))
    true_node_ids.write_bytes((2).to_bytes(8, "little"))

    manifest = json.loads(pathlib.Path(
        seed_table_fixture["manifest_path"]).read_text())
    stage = manifest["stages"]["train"]
    stage["binary_label_index"] = {
        "false_node_ids": str(false_node_ids),
        "false_count": 2,
        "true_node_ids": str(true_node_ids),
        "true_count": 1,
    }
    manifest_path = tmp_path / "weighted_manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    iterator, num_samples = make_runtime_batch_iterator(
        dataset_ref=str(manifest_path),
        stage="train",
        entity_node_type="users",
        sampling_plan=_sampling_plan(),
        num_workers=1,
        worker_id=0,
        shuffle=False,
        rnd_seed=11,
        epoch=0,
        batch_size=2,
        weighted_seed_sampling={
            "false_weight": 1.0,
            "true_weight": 10.0,
            "num_samples": 6,
        },
    )

    assert num_samples == 6
    seen: list[int] = []
    for batch in iterator:
        seen.extend(int(v) for v in batch["task"]["tt_stage_node_ids"].to_pylist())
    assert len(seen) == 6
    assert set(seen) <= {0, 1, 2}
    assert seen.count(2) > seen.count(0)
    assert seen.count(2) > seen.count(1)


def test_iterator_weighted_seed_sampling_partitions_across_workers(
    seed_table_fixture,
    tmp_path,
):
    sidecar_dir = tmp_path / "weighted_workers"
    sidecar_dir.mkdir()
    false_node_ids = sidecar_dir / "false_node_ids.bin"
    true_node_ids = sidecar_dir / "true_node_ids.bin"
    false_values = [0, 1, 2]
    true_values = [3, 4, 5]
    false_node_ids.write_bytes(
        b"".join(value.to_bytes(8, "little") for value in false_values)
    )
    true_node_ids.write_bytes(
        b"".join(value.to_bytes(8, "little") for value in true_values)
    )

    manifest = json.loads(pathlib.Path(
        seed_table_fixture["manifest_path"]).read_text())
    stage = manifest["stages"]["train"]
    stage["binary_label_index"] = {
        "false_node_ids": str(false_node_ids),
        "false_count": len(false_values),
        "true_node_ids": str(true_node_ids),
        "true_count": len(true_values),
    }
    manifest_path = tmp_path / "weighted_workers_manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    sampling_config = {
        "false_weight": 1.0,
        "true_weight": 2.0,
        "num_samples": 17,
    }

    def collect(worker_id: int, num_workers: int) -> tuple[list[int], int]:
        iterator, num_samples = make_runtime_batch_iterator(
            dataset_ref=str(manifest_path),
            stage="train",
            entity_node_type="users",
            sampling_plan=_sampling_plan(),
            num_workers=num_workers,
            worker_id=worker_id,
            shuffle=False,
            rnd_seed=37,
            epoch=2,
            batch_size=3,
            weighted_seed_sampling=sampling_config,
        )
        seen: list[int] = []
        for batch in iterator:
            seen.extend(
                int(v) for v in batch["task"]["tt_stage_node_ids"].to_pylist()
            )
        return seen, num_samples

    global_seen, global_num_samples = collect(worker_id=0, num_workers=1)
    assert global_num_samples == sampling_config["num_samples"]

    num_workers = 4
    seen_by_worker: list[list[int]] = []
    for worker_id in range(num_workers):
        seen, num_samples = collect(worker_id=worker_id, num_workers=num_workers)
        expected = [
            node_id
            for pos, node_id in enumerate(global_seen)
            if pos % num_workers == worker_id
        ]
        assert num_samples == len(expected)
        assert seen == expected
        seen_by_worker.append(seen)

    reassembled: list[int] = []
    cursors = [0] * num_workers
    for pos in range(len(global_seen)):
        worker_id = pos % num_workers
        reassembled.append(seen_by_worker[worker_id][cursors[worker_id]])
        cursors[worker_id] += 1
    assert reassembled == global_seen


def test_iterator_row_weight_seed_sampling_uses_cdf(
    seed_table_fixture,
    tmp_path,
):
    sidecar_dir = tmp_path / "row_weights"
    sidecar_dir.mkdir()
    node_ids = sidecar_dir / "node_ids.bin"
    cdf = sidecar_dir / "cdf.bin"
    node_ids.write_bytes(b"".join(value.to_bytes(8, "little") for value in [0, 1, 2]))
    weights = [0.1, 0.1, 10.0]
    total = 0.0
    cdf_bytes = bytearray()
    for weight in weights:
        total += weight
        cdf_bytes.extend(struct.pack("<d", total))
    cdf.write_bytes(cdf_bytes)

    manifest = json.loads(pathlib.Path(
        seed_table_fixture["manifest_path"]).read_text())
    manifest["stages"]["train"]["sample_weight_index"] = {
        "all": {
            "node_ids": str(node_ids),
            "cdf": str(cdf),
            "count": 3,
            "total_weight": total,
        },
    }
    manifest_path = tmp_path / "row_weight_manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    iterator, num_samples = make_runtime_batch_iterator(
        dataset_ref=str(manifest_path),
        stage="train",
        entity_node_type="users",
        sampling_plan=_sampling_plan(),
        num_workers=1,
        worker_id=0,
        shuffle=False,
        rnd_seed=19,
        epoch=0,
        batch_size=3,
        weighted_seed_sampling={
            "use_sample_weights": True,
            "num_samples": 12,
        },
    )

    assert num_samples == 12
    seen: list[int] = []
    for batch in iterator:
        seen.extend(int(v) for v in batch["task"]["tt_stage_node_ids"].to_pylist())
    assert len(seen) == 12
    assert set(seen) <= {0, 1, 2}
    assert seen.count(2) > seen.count(0)
    assert seen.count(2) > seen.count(1)


@pytest.mark.parametrize(
    "overrides",
    [
        {"shuffle": True, "rnd_seed": 7},
        {"drop_last": True},
        {"group_by_timestamp": True},
        {"rhs_seed_node_type": "users", "rhs_seed_time": 1000},
    ],
)
def test_iterator_selected_stage_node_ids_rejects_ambiguous_modes(
    seed_table_fixture,
    overrides,
):
    kwargs = dict(
        dataset_ref=seed_table_fixture["manifest_path"],
        stage="train",
        entity_node_type="users",
        sampling_plan=_sampling_plan(),
        num_workers=1,
        worker_id=0,
        shuffle=False,
        rnd_seed=None,
        epoch=0,
        batch_size=2,
        drop_last=False,
        selected_stage_node_ids=[0, 1],
    )
    kwargs.update(overrides)

    with pytest.raises(diskgraph.DiskgraphConfigError, match="selected_stage_node_ids"):
        make_runtime_batch_iterator(**kwargs)


def test_iterator_accepts_sampler_flag_kwargs(seed_table_fixture):
    iterator, _ = make_runtime_batch_iterator(
        dataset_ref=seed_table_fixture["manifest_path"],
        stage="train",
        entity_node_type="users",
        sampling_plan=_sampling_plan(),
        num_workers=1,
        worker_id=0,
        shuffle=False,
        rnd_seed=None,
        epoch=0,
        batch_size=3,
        replace=True,
        disjoint=False,
        adaptive=True,
    )

    assert list(iterator)


def test_make_runtime_batch_iterator_exposes_num_threads_kwarg():
    """The wrapper signature must surface `num_threads` so the TT-NG datamodule
    can pass a per-worker rayon thread budget through to the Rust ctor."""
    params = inspect.signature(make_runtime_batch_iterator).parameters
    assert "num_threads" in params, (
        f"make_runtime_batch_iterator must accept num_threads; got {list(params)}"
    )


def test_iterator_accepts_num_threads_kwarg(seed_table_fixture):
    """num_threads reaches the Rust ctor: an explicit value builds + yields a
    batch, and the default (None) path still works."""
    it_threads, _ = make_runtime_batch_iterator(
        dataset_ref=seed_table_fixture["manifest_path"],
        stage="train",
        entity_node_type="users",
        sampling_plan=_sampling_plan(),
        num_workers=1,
        worker_id=0,
        shuffle=False,
        rnd_seed=None,
        epoch=0,
        batch_size=3,
        num_threads=2,
    )
    batches = list(it_threads)
    assert batches, "iterator built with num_threads=2 must yield at least one batch"
    _assert_runtime_batch_dict_contract(
        batches[0], batch_size=3, n_users=seed_table_fixture["n_users"]
    )

    # Default (None) path: engine default, unchanged behavior.
    it_default, _ = make_runtime_batch_iterator(
        dataset_ref=seed_table_fixture["manifest_path"],
        stage="train",
        entity_node_type="users",
        sampling_plan=_sampling_plan(),
        num_workers=1,
        worker_id=0,
        shuffle=False,
        rnd_seed=None,
        epoch=0,
        batch_size=3,
        num_threads=None,
    )
    assert list(it_default)


def test_iterator_multi_worker_offsets_are_disjoint(seed_table_fixture):
    """num_workers=2: each worker sees a disjoint shard of seed offsets."""
    iters = []
    for worker_id in (0, 1):
        it, _ = make_runtime_batch_iterator(
            dataset_ref=seed_table_fixture["manifest_path"],
            stage="train",
            entity_node_type="users",
            sampling_plan=_sampling_plan(),
            num_workers=2,
            worker_id=worker_id,
            shuffle=False,
            rnd_seed=None,
            epoch=0,
            batch_size=2,
            drop_last=False,
        )
        iters.append(it)

    worker_stage_ids: list[set[int]] = []
    for it in iters:
        seen: set[int] = set()
        for batch in it:
            for sid in batch["task"]["tt_stage_node_ids"].to_pylist():
                seen.add(int(sid))
        worker_stage_ids.append(seen)

    # Round-robin sharding on offsets 0..5 with 2 workers, no shuffle:
    # worker 0 gets {0, 2, 4}; worker 1 gets {1, 3, 5}.
    assert worker_stage_ids[0] == {0, 2, 4}, worker_stage_ids[0]
    assert worker_stage_ids[1] == {1, 3, 5}, worker_stage_ids[1]
    assert not worker_stage_ids[0] & worker_stage_ids[1], "worker shards must be disjoint"


def test_iterator_shuffle_is_deterministic_per_epoch(seed_table_fixture):
    """Two invariants on the shuffle:
      1. Same (rnd_seed, epoch) → byte-identical iteration order on replay.
      2. Same rnd_seed, different epochs → orderings differ (otherwise the
         epoch knob is wired but ineffective — silent bug class).
    Supersedes the pytest-skip placeholder from PR 268.
    """
    def collect_stage_ids(epoch: int) -> list[list[int]]:
        it, _ = make_runtime_batch_iterator(
            dataset_ref=seed_table_fixture["manifest_path"],
            stage="train",
            entity_node_type="users",
            sampling_plan=_sampling_plan(),
            num_workers=1,
            worker_id=0,
            shuffle=True,
            rnd_seed=42,
            epoch=epoch,
            batch_size=3,
            drop_last=False,
        )
        return [batch["task"]["tt_stage_node_ids"].to_pylist() for batch in it]

    # Invariant 1: replay determinism.
    run_a = collect_stage_ids(epoch=3)
    run_b = collect_stage_ids(epoch=3)
    assert run_a == run_b, (
        f"shuffle must be deterministic for (rnd_seed, epoch); got {run_a} vs {run_b}"
    )

    # Invariant 2: epoch actually varies the ordering.
    run_e0 = collect_stage_ids(epoch=0)
    run_e1 = collect_stage_ids(epoch=1)
    assert run_e0 != run_e1, (
        f"different epoch at same rnd_seed must change ordering; got identical {run_e0}. "
        "epoch knob is ineffective"
    )
    # And those alternate epochs each replay deterministically too.
    assert collect_stage_ids(epoch=0) == run_e0
    assert collect_stage_ids(epoch=1) == run_e1


def test_iterator_drop_last_drops_partial_final_batch(seed_table_fixture):
    """6 rows / batch_size=4 / drop_last=True → 1 full batch, partial dropped."""
    it, _ = make_runtime_batch_iterator(
        dataset_ref=seed_table_fixture["manifest_path"],
        stage="train",
        entity_node_type="users",
        sampling_plan=_sampling_plan(),
        num_workers=1,
        worker_id=0,
        shuffle=False,
        rnd_seed=None,
        epoch=0,
        batch_size=4,
        drop_last=True,
    )
    batches = list(it)
    assert len(batches) == 1
    assert len(batches[0]["task"]["input_ids"]) == 4


def test_task_stage_size_reads_manifest(seed_table_fixture):
    assert task_stage_size(seed_table_fixture["manifest_path"], "train") == 6


def test_execute_tt_plan_end_to_end(tmp_path):
    """The production manifest path: build a parent index → run
    `execute_tt_plan` (the real TT-NG ingest entry point Ben's PR 269
    introduced) → take its returned task_dataset manifest → iterate via
    `make_runtime_batch_iterator`.

    This is the test that closes the §2.5 Phase-3 acceptance gap. The
    other e2e tests in this file ingest the seed extension directly via
    ``diskgraph.ingest()``, bypassing TT-NG's pass-A/B / demux pipeline.
    This one runs the same pipeline kumo's production trainer will trip.

    Catches: manifest-shape drift between ``execute_tt_plan``'s output
    and what ``make_runtime_batch_iterator`` consumes; surface drift in
    ``ingest_stage_parquets_into_extension``; split.bin requirements.
    """
    from diskgraph import DiskGraphTTBuilder, EdgeRef, execute_tt_plan

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    parent_root = tmp_path / "parent_idx"

    # Parent: users + orders with bidirectional FK + order_time sort key.
    # Same shape as the seed_table_fixture but timestamps span [0, 5000]
    # so the TimeRangeSplit below produces well-distributed train/val/test.
    n_users = 12
    orders_per_user = 6
    pq.write_table(
        pa.table({"user_id": pa.array(list(range(n_users)), type=pa.int64())}),
        data_dir / "users.parquet",
    )
    order_ids: list[int] = []
    order_uids: list[int] = []
    order_times: list[int] = []
    # Time-distributed so the TimeRangeSplit has rows in each window.
    time_buckets = [500, 1500, 2500, 3500, 4500]
    for u in range(n_users):
        for k in range(orders_per_user):
            order_ids.append(u * orders_per_user + k)
            order_uids.append(u)
            order_times.append(time_buckets[k % len(time_buckets)])
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
            "dataset_name": "execute_tt_parent",
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

    # Build a TT-NG plan via the same builder kumo's compiler uses.
    # Anchored node-prediction at 3 timestamps, count orders per user,
    # TimeRangeSplit over the [0, 5000] window to produce train/val/test
    # buckets (and a non-empty split.bin which execute_tt_plan requires).
    builder = DiskGraphTTBuilder()
    builder.temporal([1500, 2500, 3500], forecast_length=1, unit="seconds")
    builder.node_prediction()
    builder.for_each("users", "user_id")
    builder.set_target_stype("numerical")
    builder.aggregate(
        "cnt",
        EdgeRef("users", "user_id", "orders", "user_id"),
        "count",
    )
    builder.split_ratio(
        None,  # ratio is positional; TimeRangeSplit doesn't use it
        kind="TimeRangeSplit",
        ranges=[(0, 1999), (2000, 2999), (3000, 5000)],
        is_temporal=True,
        range_unit="seconds",
        offset_unit="seconds",
        unit="seconds",
    )
    plan_json = builder.build()

    # Run execute_tt_plan against the real parent. Writes
    # `<parent_root>/train_tables/tt_real/` containing train/val/test
    # seed sub-tables.
    tt_id = "tt_real"
    result = execute_tt_plan(plan_json, str(parent_root), tt_id)
    assert "task_dataset" in result, f"execute_tt_plan return missing task_dataset; keys: {list(result.keys())}"

    task_dataset = result["task_dataset"]
    # Document the shape we get vs. the shape our iterator expects.
    # If these diverge in the future, this test breaks loudly.
    assert "stages" in task_dataset, task_dataset.keys()
    train_stage = task_dataset["stages"]["train"]
    assert "table_name" in train_stage and "index_path" in train_stage and "row_count" in train_stage

    if train_stage["row_count"] == 0:
        pytest.skip("TimeRangeSplit produced zero training rows on this fixture")

    # Translate execute_tt_plan's task_dataset into the flat-manifest shape
    # our wrapper expects (`kind: diskgraph_task_dataset` + `graph.index_path`).
    # In production kumo's runtime manifest writer does this translation;
    # this test pins the contract our wrapper has against that production
    # shape.
    flat_manifest = {
        "kind": "diskgraph_task_dataset",
        "graph": {"index_path": task_dataset["graph_index_path"]},
        "stages": task_dataset["stages"],
    }
    manifest_path = tmp_path / "runtime_manifest.json"
    manifest_path.write_text(json.dumps(flat_manifest))

    # Iterate. Assert the dict contract holds against this real
    # execute_tt_plan-produced extension.
    iterator, num_samples = make_runtime_batch_iterator(
        dataset_ref=str(manifest_path),
        stage="train",
        entity_node_type="users",
        sampling_plan=_sampling_plan(),
        num_workers=1, worker_id=0,
        shuffle=False, rnd_seed=None, epoch=0,
        batch_size=min(train_stage["row_count"], 4),
        drop_last=False,
    )
    assert num_samples == train_stage["row_count"]

    seen = 0
    for batch in iterator:
        seen += len(batch["task"]["input_ids"])
        # task fields populated
        for key in ("input_ids", "tt_stage_node_ids", "seed_node_ids"):
            assert key in batch["task"]
        # parent entity must appear in graphs.main.node_ids
        assert "users" in batch["graphs"]["main"]["node_ids"], (
            f"users missing from emitted dict for execute_tt_plan-produced extension; "
            f"got: {list(batch['graphs']['main']['node_ids'].keys())}"
        )
        # seed namespace must NOT be in topology
        seed_ns = f"{tt_id}/train_seeds"
        assert seed_ns not in batch["graphs"]["main"]["node_ids"], (
            f"seed namespace {seed_ns} leaked into topology"
        )
    assert seen == train_stage["row_count"]


def test_fused_and_nonfused_iterators_emit_same_dict_shape(seed_table_fixture):
    """Both sampler dispatch paths produce the same dict structure."""
    def collect_keys(fused: bool):
        it, _ = make_runtime_batch_iterator(
            dataset_ref=seed_table_fixture["manifest_path"],
            stage="train",
            entity_node_type="users",
            sampling_plan=_sampling_plan(),
            num_workers=1,
            worker_id=0,
            shuffle=False,
            rnd_seed=None,
            epoch=0,
            batch_size=6,
            drop_last=False,
            fused=fused,
        )
        batch = next(iter(it))
        return {
            "task": sorted(batch["task"].keys()),
            "graphs.main": sorted(batch["graphs"]["main"].keys()),
            "raw_features": sorted(batch["raw_features"].keys()),
            "task.input_ids.len": len(batch["task"]["input_ids"]),
            "task.seed_node_ids.len": len(batch["task"]["seed_node_ids"]),
        }

    assert collect_keys(fused=True) == collect_keys(fused=False)

def test_execute_tt_plan_end_to_end_multi_fk_custom_tt(tmp_path):
    import json
    import pyarrow as pa
    import pyarrow.parquet as pq
    import diskgraph
    from diskgraph._c import build_train_table_artifacts_for_existing_table
    from diskgraph import execute_tt_ingest, make_runtime_batch_iterator

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    parent_root = tmp_path / "parent_idx"
    tt_root = tmp_path / "tt_artifacts"
    tt_root.mkdir()

    # 1. Build a simple parent graph with Users and Items
    n_users = 10
    n_items = 10
    pq.write_table(
        pa.table({
            "user_id": pa.array(list(range(n_users)), type=pa.int64()),
            "user_key": pa.array(list(range(n_users)), type=pa.int64()),
        }),
        data_dir / "users.parquet",
    )
    pq.write_table(
        pa.table({"item_id": pa.array(list(range(n_items)), type=pa.int64())}),
        data_dir / "items.parquet",
    )

    diskgraph.ingest(
        config_json=json.dumps({
            "dataset_name": "multi_fk_parent",
            "num_shards": 1,
            "tables": [
                {
                    "name": "users",
                    "columns": [
                        {"name": "user_id", "ingested_dtype": "int64"},
                        {"name": "user_key", "ingested_dtype": "int64"},
                    ],
                    "primary_key": "user_id",
                    "source": {"type": "parquet", "paths": [str(data_dir / "users.parquet")]},
                },
                {
                    "name": "items",
                    "columns": [{"name": "item_id", "ingested_dtype": "int64"}],
                    "primary_key": "item_id",
                    "source": {"type": "parquet", "paths": [str(data_dir / "items.parquet")]},
                },
            ],
        }),
        output_dir=str(parent_root),
        col_stats=False,
    )

    # 2. Build an LP custom TT parquet. TT-NG LP CTT uses array-target format:
    # one row per lhs/timestamp with TARGET1 as a list of RHS entities.
    tt_pq_path = data_dir / "custom_tt.parquet"
    pq.write_table(
        pa.table({
            "ENTITY1": pa.array([0, 1, 2], type=pa.int64()),
            "TIMESTAMP": pa.array([1000, 2000, 3000], type=pa.int64()),
            "TARGET1": pa.array([[5], [6], [7]], type=pa.list_(pa.int64())),
        }),
        tt_pq_path,
    )

    # 3. GEN step: build the durable staging bundle. The gen-only entry does
    # NOT publish an extension — it writes `tt.parquet` + `tt_ingest_spec.json`
    # (+ seed parquets) into a kept staging dir and returns bundle refs
    # (`staging_dir`, `tt_parquet_path`, ...); NO `extension_path`/`task_dataset`.
    tt_id = "custom_tt"
    gen_result = build_train_table_artifacts_for_existing_table(
        output_path=str(tt_pq_path),
        index_path=str(parent_root),
        tt_id=tt_id,
        entity_table_name="users",
        entity_pk_column="user_id",
        target_stype="categorical",
        input_timestamp_unit="seconds",
        split_spec_json=None,
        weight_col=None,
        target_entity_table="items",
    )

    # The extension is NOT published by gen.
    target_artifacts_dir = parent_root / "train_tables" / tt_id
    assert not target_artifacts_dir.exists()
    assert "extension_path" not in gen_result
    assert "task_dataset" not in gen_result

    # INGEST step: publish the extension from the durable staging bundle, exactly
    # how kumo drives the real gen->ingest flow (gen persists the bundle, the
    # train pod calls `execute_tt_ingest`). This is what makes the samplable
    # `train_tables/<tt_id>/` extension exist.
    ingest_result = execute_tt_ingest(
        staging_dir=gen_result["staging_dir"],
        index_path=str(parent_root),
        tt_id=tt_id,
        predict_stage_name="predict",
    )
    assert target_artifacts_dir.exists()
    assert ingest_result["tt_extension_path"] == str(target_artifacts_dir)

    # Pull the published stage ref (table_name == `<predict>_seeds`, index_path
    # == the train_tables/<tt_id> extension) from the INGEST result's
    # task_dataset. The runtime manifest maps logical `train` -> the no-split
    # `predict` stage.
    predict_stage = ingest_result["task_dataset"]["stages"]["predict"]
    task_manifest = {
        "kind": "diskgraph_task_dataset",
        "graph": {"index_path": str(parent_root)},
        "stages": {
            "train": {
                "table_name": predict_stage["table_name"],
                "index_path": predict_stage["index_path"],
                "row_count": predict_stage["row_count"],
            }
        },
    }
    task_manifest_path = tmp_path / "task_dataset_manifest.json"
    task_manifest_path.write_text(json.dumps(task_manifest))

    # 4. Create the iterator! This triggers the Stage 3 sample_runtime_batch logic.
    iterator, num_samples = make_runtime_batch_iterator(
        dataset_ref=str(task_manifest_path),
        stage="train",
        sampling_plan={"layers": []},
        num_workers=1,
        worker_id=0,
        shuffle=False,
        rnd_seed=None,
        epoch=0,
        batch_size=2,
        entity_node_type="users",
        target_node_type="items",
    )

    # We expect 2 batches (total 3 rows, batch_size=2)
    batches = list(iterator)
    assert len(batches) == 2

    # Check the first batch. FK-free LP resolution roots lhs entities and emits
    # RHS positives as ragged labels from `target_node_ids`.
    b0 = batches[0]
    node_ids = b0["graphs"]["main"]["node_ids"]
    batch_dict = b0["graphs"]["main"]["batch_dict"]

    # By contract, the node batch contains lhs user roots. RHS items are labels,
    # not roots, unless negative sampling or RHS NodeRange mode is enabled.
    assert "users" in node_ids, "LHS user nodes missing from output"

    # We didn't do any sampling (default plan is empty layers), so every
    # returned user node is a root. Parallel Encode deliberately does not
    # preserve Parquet row order, so validate the stable business keys across
    # both batches instead of assuming that key N has internal node ID N.
    users_nodes = node_ids["users"].to_pylist()
    user_key_by_node_id = _feature_value_by_node_id(
        parent_root, "users", "user_key", n_users
    )
    all_user_keys = [
        user_key_by_node_id[node_id]
        for batch in batches
        for node_id in batch["graphs"]["main"]["node_ids"]["users"].to_pylist()
    ]
    assert sorted(all_user_keys) == [0, 1, 2]
    assert len(users_nodes) == 2

    # Check that seed_index matches local IDs for the lhs roots.
    users_batch = batch_dict["users"].to_pylist()
    assert users_batch == [0, 1]

    task = b0["task"]
    assert task["num_positives"] == 2
    assert task["lp_pos_counts"].to_pylist() == [1, 1]
    assert task["lp_pos_mask"].to_pylist() == [False, False]
    assert task["lp_pos_weight"].to_pylist() == [1.0, 1.0]


# ---------------------------------------------------------------------------
# Global negative RHS sampling — FFI/iterator stage.
#
# These tests pin the FFI contract for engine-side global negatives:
#   * `task.amount_m` / `task.num_positives` / `task.rhs_seed_type` surface
#     the negative-sampling shape through the pybind dict.
#   * The rhs node set in graphs.main.node_ids is `[B positives, B*M negatives]`
#     anchor-major (positives first, in seed order).
#   * `task.seed_node_ids` length stays B — negatives are extra *sampling*
#     seeds, never task inputs.
#   * The temporal-RHS case exercises the live `is_node_temporally_valid`
#     decode path (a negative may only be drawn from RHS rows created at or
#     before the positive anchor).
#
# The fixtures here build a 2-FK LP seed table (ENTITY1 -> lhs entity,
# TARGET1 -> rhs/target entity) directly via `diskgraph.ingest`, mirroring
# `test_execute_tt_plan_end_to_end_multi_fk_custom_tt` above.
# ---------------------------------------------------------------------------


def _feature_value_by_node_id(
    index_path: pathlib.Path,
    node_type: str,
    feature_name: str,
    node_count: int,
) -> dict[int, int]:
    """Read a duplicate business-key feature for every dense internal ID."""
    engine = diskgraph.SamplerEngine(str(index_path))
    request = [{
        "node_type": node_type,
        "node_ids": list(range(node_count)),
        "feature_names": [feature_name],
    }]
    batches = [
        event["batch"] for event in engine.fetch_features(request)
        if event["type"] == "feature"
    ]
    assert batches, f"no {node_type} feature batches"
    table = pa.Table.from_batches(batches)
    node_id_column = (
        "node_id" if "node_id" in table.schema.names else table.schema.names[0]
    )
    return dict(
        zip(
            table.column(node_id_column).to_pylist(),
            table.column(feature_name).to_pylist(),
            strict=True,
        ))


def _expected_target_node_ids(fx: dict[str, Any], task: dict[str, Any]) -> list[int]:
    """Return expected RHS internal IDs in the task's emitted seed-row order."""
    return [
        fx["target_node_id_by_entity_node_id"][entity_node_id]
        for entity_node_id in task["seed_node_ids"].to_pylist()
    ]


def _ingest_lp_fixture(
    tmp_path: pathlib.Path,
    *,
    item_create_times: list[int] | None = None,
    seed_timestamps: list[int] | None = None,
    n_items: int = 10,
    seed_targets: list[int] | None = None,
    seed_entities: list[int] | None = None,
    n_users: int | None = None,
) -> dict[str, Any]:
    """Build a parent (users + items) + a 2-FK LP seed extension.

    ENTITY1 -> users (lhs), TARGET1 -> items (rhs/target). When
    ``item_create_times`` is provided, the ``items`` table carries a
    ``create_time`` column so the engine's temporal-validity path is live;
    otherwise items is a plain non-temporal node table.

    ``seed_entities`` / ``n_users`` default to the small 3-seed / 10-user shape;
    callers that need a larger batch (to actually fan the engine's parallel
    negative draw across rayon threads) pass explicit lists. ``n_users``
    auto-grows to cover the supplied ``seed_entities`` when not given.

    Returns the runtime-manifest dict + bookkeeping the tests assert on.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    parent_root = tmp_path / "parent_idx"

    if seed_entities is None:
        seed_entities = [0, 1, 2]  # lhs users picked by each seed row
    if n_users is None:
        n_users = max(10, max(seed_entities) + 1)
    if seed_targets is None:
        seed_targets = [5, 6, 7]   # rhs items = the positives, one per seed row
    if seed_timestamps is None:
        seed_timestamps = [1000, 2000, 3000]

    user_ids = list(range(n_users))
    pq.write_table(
        pa.table({
            "user_id": pa.array(user_ids, type=pa.int64()),
            "user_key": pa.array(user_ids, type=pa.int64()),
        }),
        data_dir / "users.parquet",
    )

    items_cols: dict[str, Any] = {
        "item_id": pa.array(list(range(n_items)), type=pa.int64()),
        "item_key": pa.array(list(range(n_items)), type=pa.int64()),
    }
    item_table_def: dict[str, Any] = {
        "name": "items",
        "columns": [
            {"name": "item_id", "ingested_dtype": "int64"},
            {"name": "item_key", "ingested_dtype": "int64"},
        ],
        "primary_key": "item_id",
    }
    if item_create_times is not None:
        items_cols["item_create"] = pa.array(item_create_times, type=pa.timestamp("s"))
        item_table_def["columns"].append(
            {"name": "item_create", "ingested_dtype": "timestamp"}
        )
        item_table_def["create_time"] = "item_create"
    pq.write_table(pa.table(items_cols), data_dir / "items.parquet")
    item_table_def["source"] = {
        "type": "parquet",
        "paths": [str(data_dir / "items.parquet")],
    }

    diskgraph.ingest(
        config_json=json.dumps({
            "dataset_name": "lp_neg_parent",
            "num_shards": 1,
            "tables": [
                {
                    "name": "users",
                    "columns": [
                        {"name": "user_id", "ingested_dtype": "int64"},
                        {"name": "user_key", "ingested_dtype": "int64"},
                    ],
                    "primary_key": "user_id",
                    "source": {"type": "parquet", "paths": [str(data_dir / "users.parquet")]},
                },
                item_table_def,
            ],
        }),
        output_dir=str(parent_root),
        col_stats=False,
    )

    # Parallel Encode deliberately does not preserve Parquet row order. Resolve
    # the source business keys through duplicate features before constructing
    # the already-resolved stage fixture.
    user_key_by_node_id = _feature_value_by_node_id(
        parent_root, "users", "user_key", n_users
    )
    item_key_by_node_id = _feature_value_by_node_id(
        parent_root, "items", "item_key", n_items
    )
    user_node_id_by_key = {
        key: node_id for node_id, key in user_key_by_node_id.items()
    }
    item_node_id_by_key = {
        key: node_id for node_id, key in item_key_by_node_id.items()
    }
    entity_node_ids = [user_node_id_by_key[key] for key in seed_entities]
    target_node_ids = [item_node_id_by_key[key] for key in seed_targets]

    tt_pq_path = data_dir / "lp_seeds.parquet"
    # Resolution mode: ENTITY1/TARGET1 are pre-resolved to parent node IDs.
    pq.write_table(
        pa.table({
            "ENTITY1": pa.array(seed_entities, type=pa.int64()),
            "TIMESTAMP": pa.array(seed_timestamps, type=pa.timestamp("s")),
            "TARGET1": pa.array(seed_targets, type=pa.int64()),
            "entity_node_id": pa.array(entity_node_ids, type=pa.uint64()),
            "target_node_ids": pa.array(
                [[node_id] for node_id in target_node_ids],
                type=pa.list_(pa.int64()),
            ),
        }),
        tt_pq_path,
    )

    tt_id = "lp_neg_tt"
    seed_output = parent_root / "train_tables" / tt_id
    seed_output.parent.mkdir(parents=True, exist_ok=True)
    diskgraph.ingest(
        config_json=json.dumps({
            "dataset_name": "lp_neg_seed",
            "num_shards": 1,
            "tables": [{
                "name": "lp_seeds",
                "columns": [
                    {"name": "ENTITY1", "ingested_dtype": "int64"},
                    {"name": "TIMESTAMP", "ingested_dtype": "timestamp"},
                    {"name": "TARGET1", "ingested_dtype": "int64"},
                    {"name": "entity_node_id", "ingested_dtype": "u_int64"},
                    {"name": "target_node_ids", "ingested_dtype": "int64_array"},
                ],
                "primary_key": None,
                "create_time": "TIMESTAMP",
                "fkeys": [],
                "source": {"type": "parquet", "paths": [str(tt_pq_path)]},
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
                "table_name": "lp_seeds",
                "index_path": str(seed_output),
                "row_count": len(seed_entities),
            }
        },
    }
    manifest_path = tmp_path / "lp_manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    return {
        "manifest_path": str(manifest_path),
        "tt_id": tt_id,
        "n_items": n_items,
        "seed_entities": seed_entities,
        "seed_targets": seed_targets,
        "item_key_by_node_id": item_key_by_node_id,
        "target_node_id_by_entity_node_id": dict(
            zip(entity_node_ids, target_node_ids, strict=True)
        ),
        "seed_timestamps": seed_timestamps,
        "n_seeds": len(seed_entities),
    }


def test_negative_sampling_present_nontemporal(tmp_path):
    """(a) With neg sampling on (non-temporal RHS):
      * task.amount_m == M, task.num_positives == B, task.rhs_seed_type == 'items'
      * graphs.main.node_ids['items'] = [B positives, B*M negatives] anchor-major
      * the first B ids equal the seed table's TARGET ids (positives)
      * every negative is a valid item id (in [0, n_items))
      * task.seed_node_ids length == B (negatives must NOT leak in)
    """
    fx = _ingest_lp_fixture(tmp_path)
    B = fx["n_seeds"]
    M = 3

    iterator, num_samples = make_runtime_batch_iterator(
        dataset_ref=fx["manifest_path"],
        stage="train",
        entity_node_type="users",
        target_node_type="items",
        sampling_plan={"layers": []},  # 0-hop: just the roots + negatives
        num_workers=1, worker_id=0,
        shuffle=False, rnd_seed=None, epoch=0,
        batch_size=B, drop_last=False,
        neg_rhs_node_type="items",
        neg_amount_m=M,
        neg_rnd_seed=12345,
    )
    assert num_samples == B

    batch = next(iter(iterator))
    task = batch["task"]

    assert task["amount_m"] == M, task["amount_m"]
    assert task["num_positives"] == B, task["num_positives"]
    assert task["rhs_seed_type"] == "items", task["rhs_seed_type"]

    # seed_node_ids stays positives-only (lhs users), length B.
    assert len(task["seed_node_ids"]) == B

    items = batch["graphs"]["main"]["node_ids"]["items"].to_pylist()
    # node set is [B positives, B*M negatives] anchor-major.
    assert len(items) >= B * (1 + M), (len(items), B, M)
    positives = items[:B]
    negatives = items[B:B + B * M]
    # Positives follow the emitted seed-row order and contain the resolved
    # TARGET node IDs, independent of parent/stage Encode ordering.
    expected_positives = _expected_target_node_ids(fx, task)
    assert positives == expected_positives, (positives, expected_positives)
    # negatives are valid item ids.
    for neg in negatives:
        assert 0 <= neg < fx["n_items"], neg


def test_negative_sampling_absent_regression(tmp_path):
    """(b) Without neg config: amount_m == 0, rhs_seed_type is None,
    seed_node_ids length == B. Positives are emitted as ragged LABELS, not
    rooted into the subgraph (leak-free label path).
    """
    fx = _ingest_lp_fixture(tmp_path)
    B = fx["n_seeds"]

    iterator, _ = make_runtime_batch_iterator(
        dataset_ref=fx["manifest_path"],
        stage="train",
        entity_node_type="users",
        target_node_type="items",
        sampling_plan={"layers": []},
        num_workers=1, worker_id=0,
        shuffle=False, rnd_seed=None, epoch=0,
        batch_size=B, drop_last=False,
    )
    batch = next(iter(iterator))
    task = batch["task"]

    assert task["amount_m"] == 0, task["amount_m"]
    assert task["rhs_seed_type"] is None, task["rhs_seed_type"]
    assert task["num_positives"] == B, task["num_positives"]
    assert len(task["seed_node_ids"]) == B

    # Positives are labels, not rooted — no items node set without negatives.
    assert "items" not in batch["graphs"]["main"]["node_ids"], (
        "positives must not be rooted (leak-free label path)"
    )
    expected_positives = _expected_target_node_ids(fx, task)
    assert task["lp_pos_rhs_global"].to_pylist() == expected_positives, (
        task["lp_pos_rhs_global"].to_pylist()
    )


def test_negative_sampling_temporal_rhs(tmp_path):
    """(c) Temporal RHS — exercises the LIVE is_node_temporally_valid path.

    The ``items`` table has a ``create_time`` column. We make the
    overwhelming majority of items "past" (created at t=10, valid for every
    anchor) and reserve a small clearly-identifiable block of "future" items
    (created at t=10_000_000, invalid for every anchor in [1000, 3000]).

    The engine must filter these future items out of the negative draw: a
    negative whose create_time > its row's anchor is temporally invalid.
    Because valid candidates dominate (40 of 50), the engine's bounded
    resample budget (5 retries) is effectively never exhausted on this
    fixture — so a future item appearing among the negatives would mean the
    decode/filter path did not run. This is precisely the coverage the
    engine-core stage's stubbed-closure tests could not give: here the engine
    decodes each candidate's create_time from the real items rowpack.

    Fixture: items 0..39 are "past" (create_time = 10); items 40..49 are
    "future" (create_time = 10_000_000). Positives are past items (0,1,2).
    Anchors [1000, 2000, 3000] are all >> 10 and all << 10_000_000, so the
    valid set for every row is exactly items 0..39.

    Guarantee: every drawn negative id is < 40 (a past item). We assert no
    future item (>= 40) is ever drawn. (Theoretical leak probability via an
    exhausted budget is (10/50)**6 ~= 6.4e-5 per slot; with a fixed seed this
    run is deterministic and lands zero leaks — the assert pins that.)
    """
    n_items = 50
    n_future = 10
    n_past = n_items - n_future  # 40 valid items, ids 0..39
    item_create_times = [10] * n_past + [10_000_000] * n_future
    anchors = [1000, 2000, 3000]
    fx = _ingest_lp_fixture(
        tmp_path,
        item_create_times=item_create_times,
        seed_timestamps=anchors,
        n_items=n_items,
        seed_targets=[0, 1, 2],  # positives are past items
    )
    B = fx["n_seeds"]
    M = 4

    iterator, _ = make_runtime_batch_iterator(
        dataset_ref=fx["manifest_path"],
        stage="train",
        entity_node_type="users",
        target_node_type="items",
        sampling_plan={"layers": []},
        num_workers=1, worker_id=0,
        shuffle=False, rnd_seed=None, epoch=0,
        batch_size=B, drop_last=False,
        neg_rhs_node_type="items",
        neg_amount_m=M,
        neg_rnd_seed=777,
    )
    batch = next(iter(iterator))
    task = batch["task"]
    assert task["amount_m"] == M
    assert task["rhs_seed_type"] == "items"

    items = batch["graphs"]["main"]["node_ids"]["items"].to_pylist()
    negatives = items[B:B + B * M]
    assert len(negatives) == B * M
    # Every negative must be a "past" item (id < n_past); no future item may be
    # drawn for any anchor. A future id here would prove the temporal decode
    # path did not filter.
    for i in range(B):
        anchor = anchors[i]
        row_negs = negatives[i * M:(i + 1) * M]
        for neg in row_negs:
            item_key = fx["item_key_by_node_id"][neg]
            assert item_key < n_past, (
                f"row {i} (anchor {anchor}) drew future item key {item_key} "
                f"(node ID {neg}) "
                f"(create_time=10_000_000 > anchor) — temporal filter not applied"
            )

    # neg_mask is present, length B*M, parallel to the negative slots. Here
    # valid candidates dominate so the budget is never exhausted → all true.
    neg_mask = task["neg_mask"].to_pylist()
    assert len(neg_mask) == B * M, (len(neg_mask), B, M)
    assert all(neg_mask), "no slot should be masked when valid candidates dominate"


def test_negative_sampling_neg_mask_present_nontemporal(tmp_path):
    """neg_mask surfaces through the FFI dict for a non-temporal RHS: it is
    present, length B*amount_m, and (since no temporal gating runs) all-true.
    """
    fx = _ingest_lp_fixture(tmp_path)
    B = fx["n_seeds"]
    M = 3

    iterator, _ = make_runtime_batch_iterator(
        dataset_ref=fx["manifest_path"],
        stage="train",
        entity_node_type="users",
        target_node_type="items",
        sampling_plan={"layers": []},
        num_workers=1, worker_id=0,
        shuffle=False, rnd_seed=None, epoch=0,
        batch_size=B, drop_last=False,
        neg_rhs_node_type="items",
        neg_amount_m=M,
        neg_rnd_seed=12345,
    )
    batch = next(iter(iterator))
    task = batch["task"]

    neg_mask = task["neg_mask"].to_pylist()
    assert len(neg_mask) == B * M, (len(neg_mask), B, M)
    assert all(neg_mask), "non-temporal RHS must never mask a slot"


def test_negative_sampling_temporal_exhaustion_masks(tmp_path):
    """Forced temporal exhaustion → masked slots, dense placeholders.

    Every item is "future" (create_time = 10_000_000) relative to every anchor
    (in [1000, 3000]), so NO candidate in the RHS draw range is temporally
    valid. The engine's bounded resample budget is exhausted on every slot, so
    each negative must be MASKED (neg_mask=false) while still keeping a dense
    placeholder id (a real, in-range item id) — the tensor stays dense so the
    [B, M] reshape is unaffected; the loss honors the mask.
    """
    n_items = 20
    # All items future-dated: invalid for every anchor below.
    item_create_times = [10_000_000] * n_items
    anchors = [1000, 2000, 3000]
    # Positives are just seeds (not subject to the mask); any item ids work.
    fx = _ingest_lp_fixture(
        tmp_path,
        item_create_times=item_create_times,
        seed_timestamps=anchors,
        n_items=n_items,
        seed_targets=[0, 1, 2],
    )
    B = fx["n_seeds"]
    M = 4

    iterator, _ = make_runtime_batch_iterator(
        dataset_ref=fx["manifest_path"],
        stage="train",
        entity_node_type="users",
        target_node_type="items",
        sampling_plan={"layers": []},
        num_workers=1, worker_id=0,
        shuffle=False, rnd_seed=None, epoch=0,
        batch_size=B, drop_last=False,
        neg_rhs_node_type="items",
        neg_amount_m=M,
        neg_rnd_seed=777,
    )
    batch = next(iter(iterator))
    task = batch["task"]
    assert task["amount_m"] == M

    neg_mask = task["neg_mask"].to_pylist()
    assert len(neg_mask) == B * M, (len(neg_mask), B, M)
    # Every slot exhausted the budget → all masked false.
    assert not any(neg_mask), (
        "all candidates are temporally invalid; every slot must be masked"
    )

    # Dense: the masked negatives are still present as real, in-range item ids.
    items = batch["graphs"]["main"]["node_ids"]["items"].to_pylist()
    negatives = items[B:B + B * M]
    assert len(negatives) == B * M
    for neg in negatives:
        assert 0 <= neg < n_items, neg


def test_custom_tt_weight_column_reaches_runtime_iterator(tmp_path):
    from diskgraph._c import build_train_table_artifacts_for_existing_table
    from diskgraph import execute_tt_ingest

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    parent_root = tmp_path / "parent_idx"

    users_pq = data_dir / "users.parquet"
    pq.write_table(
        pa.table({"user_id": pa.array([0, 1, 2], type=pa.int64())}),
        users_pq,
    )

    diskgraph.ingest(
        config_json=json.dumps({
            "dataset_name": "weighted_custom_parent",
            "num_shards": 1,
            "tables": [{
                "name": "users",
                "columns": [{"name": "user_id", "ingested_dtype": "int64"}],
                "primary_key": "user_id",
                "source": {"type": "parquet", "paths": [str(users_pq)]},
            }],
        }),
        output_dir=str(parent_root),
        col_stats=False,
    )

    tt_pq_path = data_dir / "weighted_custom_tt.parquet"
    pq.write_table(
        pa.table({
            "ENTITY1": pa.array([0, 1, 2], type=pa.int64()),
            "TIMESTAMP": pa.array([1000, 2000, 3000], type=pa.int64()),
            "TARGET1": pa.array([1, 0, 1], type=pa.int64()),
            "WEIGHT": pa.array([0.5, 2.0, 4.0], type=pa.float64()),
        }),
        tt_pq_path,
    )

    tt_id = "weighted_custom_tt"
    # GEN step: build the durable staging bundle (no extension published yet).
    gen_result = build_train_table_artifacts_for_existing_table(
        output_path=str(tt_pq_path),
        index_path=str(parent_root),
        tt_id=tt_id,
        entity_table_name="users",
        entity_pk_column="user_id",
        target_stype="categorical",
        input_timestamp_unit="seconds",
        split_spec_json=None,
        weight_col="WEIGHT",
    )

    stage_index_path = parent_root / "train_tables" / tt_id
    assert not stage_index_path.exists()
    assert "extension_path" not in gen_result

    # INGEST step: publish the extension from the durable staging bundle (the
    # real gen->ingest flow). The WEIGHT column travels with the bundle.
    ingest_result = execute_tt_ingest(
        staging_dir=gen_result["staging_dir"],
        index_path=str(parent_root),
        tt_id=tt_id,
        predict_stage_name="predict",
    )
    assert pathlib.Path(ingest_result["tt_extension_path"]) == stage_index_path
    assert stage_index_path.exists()

    # Pull the published predict-stage ref from the ingest task_dataset block.
    predict_stage = ingest_result["task_dataset"]["stages"]["predict"]
    task_manifest = {
        "kind": "diskgraph_task_dataset",
        "graph": {"index_path": str(parent_root)},
        "stages": {
            "predict": {
                "table_name": predict_stage["table_name"],
                "index_path": predict_stage["index_path"],
                "row_count": predict_stage["row_count"],
            },
        },
    }
    task_manifest_path = tmp_path / "weighted_task_dataset_manifest.json"
    task_manifest_path.write_text(json.dumps(task_manifest))

    iterator, _ = make_runtime_batch_iterator(
        dataset_ref=str(task_manifest_path),
        stage="predict",
        entity_node_type="users",
        sampling_plan={"layers": []},
        num_workers=1,
        worker_id=0,
        shuffle=False,
        rnd_seed=None,
        epoch=0,
        batch_size=2,
    )

    batches = list(iterator)

    assert batches[0]["task"]["weight"].to_pylist() == [0.5, 2.0]
    assert batches[1]["task"]["weight"].to_pylist() == [4.0]


# ---------------------------------------------------------------------------
# Global negative RHS sampling — INTEGRATION e2e at non-trivial scale.
#
# The tests above pin the FFI *contract* at the tiny 3-seed fixture. They do
# NOT exercise the property that matters once negative selection runs in
# parallel across positive rows on the engine's per-instance rayon pool
# (engine.rs `append_global_negatives` → `self.thread_pool.install(|| (0..b)
# .into_par_iter()…)`): that the real multi-threaded draw is **deterministic
# per seed** (replay-safe) and stays correct at a batch large enough to fan
# rows across multiple worker threads.
#
# These three tests close that gap by driving the FULL real path — the built
# wheel's `make_runtime_batch_iterator` → `RuntimeBatchIterator` → real
# `sample_runtime_batch` → real `self.thread_pool` → real
# `is_node_temporally_valid` — at B = 64 (vs. the 3 above):
#   1. determinism across two separate iterator builds with the same
#      `neg_rnd_seed` (and that a different seed changes the draws);
#   2. the `[B positives, B*M negatives]` split + id range at scale;
#   3. temporal validity of every non-masked negative through the real
#      parallel + real-rowpack-decode path.
#
# B = 64 is chosen deliberately: it is comfortably above rayon's per-task
# granularity so the row loop actually distributes across the pool's worker
# threads (so the parallel collect/ordering and per-thread validator scratch
# are genuinely exercised), while staying small enough to ingest quickly. The
# test passing at this B is also the no-deadlock/no-crash assertion (4).
# ---------------------------------------------------------------------------

_LARGE_B = 64


@pytest.fixture
def multithreaded_engine_pool(monkeypatch):
    """Force the engine's per-instance rayon pool to be genuinely
    multi-threaded for the duration of a test.

    ``make_runtime_batch_iterator`` does not forward ``num_threads``, so
    ``DiskGraphEngine::from_reader`` falls back to ``RAYON_NUM_THREADS`` (else
    1). Setting it >1 here makes the negative-draw row loop actually fan across
    worker threads — without this the parallel code path still *runs* but on a
    single thread, so the cross-thread scheduling that the determinism test
    guards against would never be exercised. Read at engine construction time,
    so each fresh iterator build inside the test picks it up.
    """
    monkeypatch.setenv("RAYON_NUM_THREADS", "8")


def _large_lp_fixture(tmp_path, *, temporal: bool):
    """A B=64 LP fixture. lhs users 0..63 (one per seed), rhs item universe of
    128 so the negative draw range is wide. Positives are items 0..63.

    When ``temporal`` is set, items 0..95 are "past" (create_time=10, valid for
    every anchor) and items 96..127 are "future" (create_time=10_000_000,
    invalid for every anchor in [1000, 1000+B)); positives stay in the past
    block. Anchors are distinct per row so the per-row anchor wiring is
    exercised across the parallel rows.
    """
    b = _LARGE_B
    n_items = 128
    seed_entities = list(range(b))               # lhs users 0..63
    seed_targets = list(range(b))                # rhs positives = items 0..63
    seed_timestamps = [1000 + i for i in range(b)]  # distinct anchors per row

    item_create_times = None
    n_past = None
    if temporal:
        n_past = 96  # items 0..95 valid for every anchor (<< 1000); 96..127 future
        item_create_times = [10] * n_past + [10_000_000] * (n_items - n_past)

    fx = _ingest_lp_fixture(
        tmp_path,
        item_create_times=item_create_times,
        seed_timestamps=seed_timestamps,
        n_items=n_items,
        seed_targets=seed_targets,
        seed_entities=seed_entities,
    )
    fx["B"] = b
    fx["n_past"] = n_past
    return fx


def _collect_negatives(manifest_path, *, B, M, neg_rnd_seed):
    """Build the iterator from scratch and return (rhs_negative_ids, neg_mask)
    for the single batch. Builds a *fresh* iterator each call so we exercise the
    real engine/pool from cold — this is what makes the cross-run determinism
    assertion meaningful (it is not reusing in-process RNG state)."""
    iterator, num_samples = make_runtime_batch_iterator(
        dataset_ref=manifest_path,
        stage="train",
        entity_node_type="users",
        target_node_type="items",
        sampling_plan={"layers": []},  # 0-hop: roots + negatives only
        num_workers=1, worker_id=0,
        shuffle=False, rnd_seed=None, epoch=0,
        batch_size=B, drop_last=False,
        neg_rhs_node_type="items",
        neg_amount_m=M,
        neg_rnd_seed=neg_rnd_seed,
    )
    assert num_samples == B
    batch = next(iter(iterator))
    items = batch["graphs"]["main"]["node_ids"]["items"].to_pylist()
    negatives = items[B:B + B * M]
    neg_mask = batch["task"]["neg_mask"].to_pylist()
    return negatives, neg_mask


def test_negative_sampling_parallel_deterministic_across_runs(tmp_path, multithreaded_engine_pool):
    """INTEGRATION: the real parallel negative draw is deterministic per seed.

    Build the iterator TWICE from cold with the SAME neg_rnd_seed over the same
    B=64 fixture; the rhs negative ids AND neg_mask must be byte-identical
    across the two runs (this is the replay-safety property the parallel rayon
    pool must hold — row order / thread scheduling must not perturb the draw).
    Then build with a DIFFERENT neg_rnd_seed and assert the negatives differ,
    proving the seed actually drives the draw (not a constant).
    """
    fx = _large_lp_fixture(tmp_path, temporal=False)
    B, M = fx["B"], 3

    negs_a, mask_a = _collect_negatives(fx["manifest_path"], B=B, M=M, neg_rnd_seed=2024)
    negs_b, mask_b = _collect_negatives(fx["manifest_path"], B=B, M=M, neg_rnd_seed=2024)

    assert len(negs_a) == B * M, (len(negs_a), B, M)
    assert negs_a == negs_b, (
        "same neg_rnd_seed must reproduce identical negatives across two cold "
        "iterator builds — the parallel pool must be replay-deterministic"
    )
    assert mask_a == mask_b, "neg_mask must also be identical across runs for a fixed seed"

    # A different seed must move the draw (sanity that the seed is wired through
    # the real engine path, not ignored).
    negs_c, _ = _collect_negatives(fx["manifest_path"], B=B, M=M, neg_rnd_seed=99999)
    assert negs_c != negs_a, (
        "a different neg_rnd_seed must change the negatives; identical draws "
        "would mean the seed is not actually driving the parallel sampler"
    )


def test_negative_sampling_none_seed_is_random_across_runs(tmp_path, multithreaded_engine_pool):
    """INTEGRATION: the training/prod path — ``neg_rnd_seed=None`` draws from
    ``thread_rng``, so two cold builds (very likely) produce DIFFERENT negatives.

    This pins the behavior that matters for training: no fixed seed ⇒ fresh
    negatives each epoch (better negative coverage), rather than the same draw
    every run. The wide rhs id-space makes a collision between two independent
    runs astronomically unlikely. Determinism remains available by passing an
    explicit seed (see the test above), which is what test/replay paths use.
    """
    fx = _large_lp_fixture(tmp_path, temporal=False)
    B, M = fx["B"], 3

    negs_a, _ = _collect_negatives(fx["manifest_path"], B=B, M=M, neg_rnd_seed=None)
    negs_b, _ = _collect_negatives(fx["manifest_path"], B=B, M=M, neg_rnd_seed=None)

    assert len(negs_a) == B * M, (len(negs_a), B, M)
    assert negs_a != negs_b, (
        "neg_rnd_seed=None must draw fresh (non-deterministic) negatives across "
        "cold builds — this is the random training/prod path"
    )


def test_negative_sampling_parallel_split_and_range_at_scale(tmp_path, multithreaded_engine_pool):
    """INTEGRATION: at B=64 (real parallel fan-out) the rhs node set is exactly
    [B positives, B*M negatives] anchor-major; positives equal the seed TARGET
    ids in seed order; every negative id is in range; neg_mask has length B*M.
    """
    fx = _large_lp_fixture(tmp_path, temporal=False)
    B, M = fx["B"], 4

    iterator, num_samples = make_runtime_batch_iterator(
        dataset_ref=fx["manifest_path"],
        stage="train",
        entity_node_type="users",
        target_node_type="items",
        sampling_plan={"layers": []},
        num_workers=1, worker_id=0,
        shuffle=False, rnd_seed=None, epoch=0,
        batch_size=B, drop_last=False,
        neg_rhs_node_type="items",
        neg_amount_m=M,
        neg_rnd_seed=4242,
    )
    assert num_samples == B
    batch = next(iter(iterator))
    task = batch["task"]

    assert task["amount_m"] == M
    assert task["num_positives"] == B
    assert task["rhs_seed_type"] == "items"
    # negatives are extra sampling seeds — they must NOT inflate seed_node_ids.
    assert len(task["seed_node_ids"]) == B

    items = batch["graphs"]["main"]["node_ids"]["items"].to_pylist()
    # Exactly [B positives, B*M negatives] — no dedup/merge may collapse the
    # anchor-major prefix the Python side slices on.
    assert len(items) == B + B * M, (len(items), B, M)
    positives = items[:B]
    negatives = items[B:B + B * M]
    assert positives == _expected_target_node_ids(fx, task), (
        "positives must be resolved TARGET node IDs in emitted seed order"
    )
    for neg in negatives:
        assert 0 <= neg < fx["n_items"], neg

    neg_mask = task["neg_mask"].to_pylist()
    assert len(neg_mask) == B * M, (len(neg_mask), B, M)
    # Non-temporal RHS → nothing is ever masked.
    assert all(neg_mask), "non-temporal RHS must never mask a slot, even at scale"


def test_negative_sampling_parallel_temporal_valid_at_scale(tmp_path, multithreaded_engine_pool):
    """INTEGRATION: temporal correctness through the real parallel + real
    rowpack-decode path at B=64.

    Items 0..95 are past (valid for every anchor); 96..127 are future (invalid).
    Valid candidates dominate (96 of 128) so the bounded resample budget is
    effectively never exhausted → every slot stays unmasked. Assert that every
    NON-masked negative is temporally valid (a past item < n_past, i.e. created
    <= its anchor) — proving the live `is_node_temporally_valid` decode ran on
    each parallel row — and that any masked slot (none expected here) still
    carries a dense in-range placeholder id.
    """
    fx = _large_lp_fixture(tmp_path, temporal=True)
    B, M, n_past = fx["B"], 4, fx["n_past"]

    iterator, _ = make_runtime_batch_iterator(
        dataset_ref=fx["manifest_path"],
        stage="train",
        entity_node_type="users",
        target_node_type="items",
        sampling_plan={"layers": []},
        num_workers=1, worker_id=0,
        shuffle=False, rnd_seed=None, epoch=0,
        batch_size=B, drop_last=False,
        neg_rhs_node_type="items",
        neg_amount_m=M,
        neg_rnd_seed=31337,
    )
    batch = next(iter(iterator))
    task = batch["task"]
    assert task["amount_m"] == M
    assert task["rhs_seed_type"] == "items"

    items = batch["graphs"]["main"]["node_ids"]["items"].to_pylist()
    negatives = items[B:B + B * M]
    neg_mask = task["neg_mask"].to_pylist()
    assert len(negatives) == B * M
    assert len(neg_mask) == B * M

    for slot, (neg, valid) in enumerate(zip(negatives, neg_mask)):
        # Every slot, masked or not, keeps a dense in-range placeholder id.
        assert 0 <= neg < fx["n_items"], (slot, neg)
        if valid:
            # A non-masked negative MUST be temporally valid: a past item, i.e.
            # create_time <= anchor. A future item here would prove the real
            # parallel temporal-decode path did not filter on this row.
            item_key = fx["item_key_by_node_id"][neg]
            assert item_key < n_past, (
                f"slot {slot}: non-masked negative node ID {neg} has future "
                f"item key {item_key} "
                f"(>= {n_past}); real parallel temporal filter did not run"
            )

    # Valid candidates dominate, so the budget should never be exhausted here.
    assert all(neg_mask), (
        "valid items dominate (96/128); no slot should be masked at this scale — "
        "any mask would imply the resample budget was unexpectedly exhausted"
    )
