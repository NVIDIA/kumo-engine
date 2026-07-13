"""TT-NG integration tests.

Two layers of verification:

  Layer A — pure diskgraph (no kumo imports):
    * execute_tt_plan → make_runtime_batch_iterator round-trip
    * batch dict contract (input_ids, tt_stage_node_ids, seed_node_ids, graphs)
    * wheel probe: 3-arg execute_tt_plan signature, make_runtime_batch_iterator present

  Layer B — kumo service layer (requires kumo in sys.path):
    * DiskgraphService.materialize_task_dataset placeholder path
    * Manifest structure correct
    * Dispatch: use_tt_ng() True with KUMO_DISKGRAPH_TT_NG_ENABLED=1

Run layer A only:
    pytest diskgraph-pybind/python/tests/test_ttng.py -m "not kumo_service"

Run both (needs kumo repo on PYTHONPATH):
    PYTHONPATH=/path/to/kumo pytest diskgraph-pybind/python/tests/test_ttng.py

Against the TT-NG compose stack (see kumo/docker/compose/docker-compose.ttng-override.yml):
    docker compose exec kumo-pool-worker python -m pytest \
        /opt/app/diskgraph-pybind/python/tests/test_ttng.py
"""
from __future__ import annotations

import importlib
import json
import os
import pathlib
import sys
import tempfile
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_parent_index(tmp_path: pathlib.Path) -> pathlib.Path:
    """Ingest a tiny users+orders parent DG index. Returns index path."""
    import diskgraph

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    parent_root = tmp_path / "parent_idx"

    n_users = 12
    orders_per_user = 6
    pq.write_table(
        pa.table({"user_id": pa.array(list(range(n_users)), type=pa.int64())}),
        data_dir / "users.parquet",
    )
    order_ids, order_uids, order_times = [], [], []
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
            "dataset_name": "ttng_stage1_parent",
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
                                "table": "orders",
                                "column": "order_time",
                                "bitwidth": 32,
                                "unit": "s",
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
    return parent_root


def _build_tt_plan() -> str:
    """Return PQL plan JSON for binary-classification node prediction on users."""
    from diskgraph import DiskGraphTTBuilder, EdgeRef

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
        None,
        kind="TimeRangeSplit",
        ranges=[(0, 1999), (2000, 2999), (3000, 5000)],
        is_temporal=True,
        range_unit="seconds",
        offset_unit="seconds",
        unit="seconds",
    )
    return builder.build()


def _build_forecasting_tt_plan() -> str:
    from diskgraph import DiskGraphTTBuilder, EdgeRef

    builder = DiskGraphTTBuilder()
    builder.temporal(
        [i * 1000 for i in range(11)],
        forecast_length=3,
        unit="seconds",
    )
    builder.forecasting_prediction()
    builder.for_each("users", "user_id")
    builder.set_target_stype("numerical")
    builder.aggregate(
        "target",
        EdgeRef("users", "user_id", "orders", "user_id"),
        "count",
    )
    target_op = builder.pop_last_op_json()
    builder.future_labels("future_labels", target_op, [0, 1000, 2000])
    builder.split_ratio(
        None,
        kind="TimeRangeSplit",
        ranges=[(0, 4000), (4001, 7000), (7001, 10000)],
        is_temporal=True,
        range_unit="seconds",
        offset_unit="seconds",
        unit="seconds",
    )
    return builder.build()


def _default_sampling_plan() -> dict:
    # Depth-2: hop-0 resolves seeds (no sampling), hop-1 walks users→orders.
    # Schema: layers[] each with fanouts dict + default_fanout.
    fan = {"fanout": 4, "mode": "uniform", "absent_key_policy": "uniform_fallback"}
    return {
        "layers": [
            {"fanouts": {}, "default_fanout": fan},
            {"fanouts": {}, "default_fanout": fan},
        ],
    }


# ---------------------------------------------------------------------------
# Layer A — pure diskgraph (no kumo)
# ---------------------------------------------------------------------------


def test_wheel_is_tt_ng():
    """Installed wheel must expose the 3-arg execute_tt_plan signature."""
    import inspect
    from diskgraph import execute_tt_plan

    sig = inspect.signature(execute_tt_plan)
    assert len(sig.parameters) >= 3, (
        f"execute_tt_plan has {len(sig.parameters)} params — looks like legacy wheel. "
        "Install a wheel built from kumo-diskgraph tt-ng branch."
    )


def test_wheel_has_runtime_batch_iterator():
    """TT-NG wheel must expose make_runtime_batch_iterator."""
    from diskgraph import make_runtime_batch_iterator  # noqa: F401


def test_wheel_lacks_legacy_symbols():
    """execute_tt_plan_with_artifacts was removed on tt-ng branch."""
    try:
        from diskgraph import execute_tt_plan_with_artifacts  # noqa: F401
        pytest.fail(
            "execute_tt_plan_with_artifacts present — this is the legacy wheel, not TT-NG."
        )
    except ImportError:
        pass  # expected


def test_execute_tt_plan_produces_manifest(tmp_path):
    """execute_tt_plan returns a task_dataset dict with train/val/test stages."""
    from diskgraph import execute_tt_plan

    parent_root = _build_parent_index(tmp_path)
    plan_json = _build_tt_plan()

    result = execute_tt_plan(plan_json, str(parent_root), "tt_stage1")

    assert "task_dataset" in result, f"missing task_dataset key; got: {list(result)}"
    td = result["task_dataset"]
    assert "stages" in td, f"task_dataset missing stages; keys: {list(td)}"
    assert "graph_index_path" in td, f"task_dataset missing graph_index_path; keys: {list(td)}"

    stages = td["stages"]
    assert set(stages) >= {"train"}, f"expected at least train stage; got: {list(stages)}"
    for stage_name, stage_info in stages.items():
        for field in ("table_name", "index_path", "row_count"):
            assert field in stage_info, (
                f"stage {stage_name} missing {field!r}; keys: {list(stage_info)}"
            )
    # Intentionally no return — assertions above cover the contract.


def test_forecasting_tt_gen_ingest_preserves_split_stats(tmp_path):
    from diskgraph import execute_tt_gen, execute_tt_ingest

    parent_root = _build_parent_index(tmp_path)
    plan_json = _build_forecasting_tt_plan()

    gen = execute_tt_gen(plan_json, str(parent_root), "tt_forecast_stats")
    staging = pathlib.Path(gen["staging_dir"])
    split_stats = staging / "split_stats.json"
    assert split_stats.exists()

    payload = json.loads(split_stats.read_text())
    train_stats = payload["train_table_stats"]
    train_stage = next(s for s in json.loads(
        (staging / "tt_ingest_spec.json").read_text()
    )["split_stages"] if s["key"] == "train")
    assert train_stats["num_train_rows"] == train_stage["row_count"]
    target_distribution = train_stats["target_distribution"]
    assert target_distribution["target_max"] >= target_distribution["target_min"]
    assert target_distribution["histogram"]["num_bins"] == 0
    assert target_distribution["histogram"]["counts"] == []

    execute_tt_ingest(
        staging_dir=str(staging),
        index_path=str(parent_root),
        tt_id="tt_forecast_stats",
    )
    assert (parent_root / "train_tables" / "tt_forecast_stats" /
            "split_stats.json").exists()


def test_compiler_forecasting_tt_gen_uses_aggregate_as_target(tmp_path):
    from kumoapi.pquery.AST import Aggregation, Column, DateOffsetRange, Join
    from kumoapi.pquery.validated_predictive_query import ValidatedPredictiveQuery
    from kumoapi.typing import AggregationType

    from diskgraph import DiskGraphPlanCompiler, TimeUnit, execute_tt_gen
    from diskgraph.compiler import AUTOREGRESSION_OUTPUT_ALIAS

    parent_root = _build_parent_index(tmp_path)
    target_ast = Aggregation(
        target=Column(fqn="orders.order_id"),
        aggr=AggregationType.AVG,
        aggr_time_range=DateOffsetRange(start=0, end=1000, unit="seconds"),
    )
    target_ast.target = Join(
        rhs_target=Column(fqn="orders.order_id"),
        lhs_key="users.user_id",
        rhs_key="orders.user_id",
    )
    query = ValidatedPredictiveQuery(
        entity_ast=Column(fqn="users.user_id"),
        target_ast=target_ast,
    )
    query._query_type = "TEMPORAL"

    plan_json = DiskGraphPlanCompiler().compile(
        query=query,
        timestamps=[1500],
        task_type="forecasting",
        input_timestamp_unit=TimeUnit.Seconds,
        target_stype="numerical",
        forecast_length=1,
        lag_timesteps=1,
        lag_offsets_secs=[-1000],
    )
    ops = json.loads(plan_json)["stages"][0]["ops"]
    assert ops[0]["autoregression"]["alias"] == AUTOREGRESSION_OUTPUT_ALIAS
    assert ops[-1]["column"]["alias"] == "entity_id_target"

    generated = execute_tt_gen(
        plan_json,
        str(parent_root),
        "tt_compiled_forecast_target",
    )
    table = pq.read_table(generated["tt_parquet_path"])

    assert table["TARGET1"].type == pa.float64()
    actual = dict(
        zip(
            table["ENTITY1"].to_pylist(),
            table["TARGET1"].to_pylist(),
        )
    )
    expected = {user_id: 6 * user_id + 2.0 for user_id in range(12)}
    assert actual == pytest.approx(expected)


def test_runtime_batch_iterator_dict_contract(tmp_path):
    """Full chain: execute_tt_plan → flat manifest → make_runtime_batch_iterator.

    Validates the dict shape against the PR #28765 runtime_batch_to_hetero
    adapter contract:
      batch["task"]["input_ids"]          — seed row ids
      batch["task"]["tt_stage_node_ids"]  — same as input_ids (alias)
      batch["task"]["seed_node_ids"]      — resolved parent-row ids
      batch["graphs"]["main"]["node_ids"] — parent entities (users, orders)
      seed namespace NOT in topology
    """
    from diskgraph import execute_tt_plan, make_runtime_batch_iterator

    parent_root = _build_parent_index(tmp_path)
    plan_json = _build_tt_plan()

    result = execute_tt_plan(plan_json, str(parent_root), "tt_batch_contract")
    td = result["task_dataset"]
    train_stage = td["stages"].get("train")
    if train_stage is None or train_stage["row_count"] == 0:
        pytest.skip("TimeRangeSplit produced no training rows on this fixture")

    flat_manifest = {
        "kind": "diskgraph_task_dataset",
        "graph": {"index_path": td["graph_index_path"]},
        "stages": td["stages"],
    }
    manifest_path = tmp_path / "runtime_manifest.json"
    manifest_path.write_text(json.dumps(flat_manifest))

    batch_size = min(train_stage["row_count"], 4)
    iterator, num_samples = make_runtime_batch_iterator(
        dataset_ref=str(manifest_path),
        stage="train",
        sampling_plan=_default_sampling_plan(),
        num_workers=1, worker_id=0,
        shuffle=False, rnd_seed=None, epoch=0,
        batch_size=batch_size,
        drop_last=False,
    )

    assert num_samples == train_stage["row_count"], (
        f"task_stage_size ({num_samples}) != execute_tt_plan row_count ({train_stage['row_count']})"
    )

    seen = 0
    for batch in iterator:
        seen += len(batch["task"]["input_ids"])

        # task bag: required keys
        for key in ("input_ids", "tt_stage_node_ids", "seed_node_ids"):
            assert key in batch["task"], f"batch['task'] missing {key!r}"

        # topology: parent entities present
        assert "users" in batch["graphs"]["main"]["node_ids"], (
            f"parent entity 'users' missing from topology; "
            f"present: {list(batch['graphs']['main']['node_ids'])}"
        )

        # seed namespace excluded from topology
        seed_ns = "tt_batch_contract/train_seeds"
        assert seed_ns not in batch["graphs"]["main"]["node_ids"], (
            f"seed namespace {seed_ns!r} leaked into topology — "
            "seed rows must be in task.*, not graphs.main"
        )

    assert seen == train_stage["row_count"], (
        f"iterator yielded {seen} rows but execute_tt_plan produced {train_stage['row_count']}"
    )


def test_multi_worker_yields_disjoint_shards(tmp_path):
    """Two workers must partition the stage without overlap."""
    from diskgraph import execute_tt_plan, make_runtime_batch_iterator

    parent_root = _build_parent_index(tmp_path)
    plan_json = _build_tt_plan()

    result = execute_tt_plan(plan_json, str(parent_root), "tt_shards")
    td = result["task_dataset"]
    train_stage = td["stages"].get("train")
    if train_stage is None or train_stage["row_count"] < 2:
        pytest.skip("Need at least 2 training rows for multi-worker test")

    flat_manifest = {
        "kind": "diskgraph_task_dataset",
        "graph": {"index_path": td["graph_index_path"]},
        "stages": td["stages"],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(flat_manifest))

    def collect_ids(worker_id: int) -> set:
        it, _ = make_runtime_batch_iterator(
            dataset_ref=str(manifest_path),
            stage="train",
            sampling_plan=_default_sampling_plan(),
            num_workers=2, worker_id=worker_id,
            shuffle=False, rnd_seed=None, epoch=0,
            batch_size=2,
            drop_last=False,
        )
        # tt_stage_node_ids = ingest row offset within stage table (0-indexed globally).
        # With no shuffle, worker 0 gets even offsets, worker 1 gets odd offsets.
        ids = set()
        for b in it:
            ids.update(int(x) for x in b["task"]["tt_stage_node_ids"].to_pylist())
        return ids

    ids0 = collect_ids(0)
    ids1 = collect_ids(1)
    assert ids0.isdisjoint(ids1), f"shard overlap: {ids0 & ids1}"
    assert ids0 | ids1 == set(range(train_stage["row_count"])) or len(ids0) + len(ids1) == train_stage["row_count"], (
        f"shards don't cover all rows: |ids0|={len(ids0)}, |ids1|={len(ids1)}, total={train_stage['row_count']}"
    )


# ---------------------------------------------------------------------------
# Layer B — kumo service layer
# ---------------------------------------------------------------------------

def _kumo_available() -> bool:
    """Check if kumo repo is importable via sys.path or installed."""
    try:
        import kumo.engine.diskgraph.service  # noqa: F401
        return True
    except ImportError:
        return False


kumo_service = pytest.mark.skipif(
    not _kumo_available(),
    reason="kumo not importable — add kumo repo to PYTHONPATH to run layer B",
)


@kumo_service
def test_dispatch_tt_ng_wheel_returns_true(monkeypatch):
    """use_tt_ng() returns True when env var is set and wheel is TT-NG."""
    monkeypatch.setenv("KUMO_DISKGRAPH_TT_NG_ENABLED", "1")
    from kumo.data.generic._tt_ng_dispatch import use_tt_ng
    assert use_tt_ng() is True


@kumo_service
def test_dispatch_mismatch_legacy_wheel_raises(monkeypatch):
    """use_tt_ng() raises if env var is on but wheel is legacy.

    We can't install the legacy wheel in the same process, but we can
    mock is_tt_ng_wheel() to simulate the mismatch.
    """
    import kumo.data.generic._tt_ng_dispatch as dispatch_mod

    monkeypatch.setenv("KUMO_DISKGRAPH_TT_NG_ENABLED", "1")
    monkeypatch.setattr(dispatch_mod, "is_tt_ng_wheel", lambda: False)
    with pytest.raises(RuntimeError, match="TT-NG entry point"):
        dispatch_mod.use_tt_ng()


@kumo_service
def test_materialize_task_dataset_placeholder_manifest(tmp_path):
    """DiskgraphService.materialize_task_dataset writes a placeholder manifest.

    spec.tt_manifest_path must point to a kumo TT artifact bundle (version=1,
    produced by the gen_tt activity) — not the flat runtime manifest from
    execute_tt_plan. We mock get_validated_artifact_bundle to bypass the file
    format check and focus on the placeholder-path logic in the service.
    """
    import os
    from unittest.mock import patch

    from kumo.engine._backend import DiskgraphHandle
    from kumo.engine.diskgraph._task_dataset_spec import TaskDatasetMaterializeSpec
    from kumo.engine.diskgraph.service import DiskgraphService

    # The TT artifact bundle (version=1) is produced by gen_tt and references
    # physical files (split assignment, stats json, etc.). We mock the validator
    # so this test exercises the placeholder path without needing a real gen_tt tree.
    fake_bundle = {
        "parsed_split": {"counts": {"train": 6, "val": 2, "test": 2}},
        "parsed_stats": {"provided_fields": []},
    }

    class _Config:
        data_root = str(tmp_path / "dg_data")

    os.makedirs(_Config.data_root, exist_ok=True)

    handle = DiskgraphHandle.for_test(
        graph_key="test-graph",
        graph_data_version="v1",
    )

    spec = TaskDatasetMaterializeSpec(
        tt_manifest_path=str(tmp_path / "fake_tt_bundle.json"),
        job_id="test-job-001",
        artifact_root=str(tmp_path / "artifacts"),
        task_family="BINARY_CLASSIFICATION",
        task_type="node_prediction",
        is_temporal=False,
        seed_node_type="users",
        train_table_name="tt_kumo_svc",
        entity_table_name="users",
        feature_schema={"users": {"feature_columns": ["user_id"], "mapper_key": "users"}},
        graph_node_types=("users", "orders"),
        graph_edge_types=(("users", "user_id", "orders"),),
    )

    svc = DiskgraphService(_Config())  # type: ignore[arg-type]
    with patch(
        "kumo.engine.diskgraph_native_tt_artifacts.get_validated_artifact_bundle",
        return_value=fake_bundle,
    ):
        result_mat = svc.materialize_task_dataset(handle, spec)

    assert result_mat.storage_kind == "placeholder", (
        f"Expected placeholder (Rust ingest not yet wired), got: {result_mat.storage_kind!r}. "
        "If this is 'real', the wheel bump + kumo wiring have landed — update this test."
    )
    assert os.path.exists(result_mat.manifest_path), (
        f"manifest_path {result_mat.manifest_path!r} does not exist on disk"
    )

    with open(result_mat.manifest_path) as f:
        manifest = json.load(f)

    assert "stages" in manifest or "stage_counts" in manifest or "ingested_stage_tables" in manifest, (
        f"manifest missing stage info; top-level keys: {list(manifest)}"
    )
    print(f"\nmanifest written to {result_mat.manifest_path}")
    print(json.dumps(manifest, indent=2))
