# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cross-graph TT-reuse contamination e2e (real wheel, unit-scale).

A train/predict bundle generated against graph G1 is ingested for a job
bound to G2 = the same schema minus 40% of the entity rows (synthetic
contamination). Before ingest-time membership verification, the orphan seed
FKs silently collided onto arbitrary G2 nodes and training completed on
mismatched labels with zero error signal; now ingest skips + counts them,
``ensure_stage_ingested`` warns with the skipped fraction, and the
``KUMO_TTNG_ORPHAN_FAIL_PCT`` knob hard-fails — including on a REUSED
already-published extension via the persisted manifest re-check.

All four bundle producers are driven through the ONE ingest chokepoint
(``ensure_stage_ingested`` → ``execute_tt_ingest``):

  1. PQL train table   — kumo compiler → ``execute_tt_gen`` (train shape)
  2. custom train table — ``build_train_table_artifacts_for_existing_table``
  3. predict table      — ``build_ttng_predict_plan`` → ``execute_tt_gen``,
     ingested with ``predict_stage_name='predict'`` (the predict path IS the
     train-table path plus that one extra argument)
  4. custom predict table —
     ``build_prediction_table_artifacts_for_existing_table``

Mirrors ``test_diskgraph_train_table_e2e.py``'s unit-scale harness: real
diskgraph wheel, tiny synthetic parquet, seconds per test, no Temporal dev
server. Skips cleanly when the installed wheel predates the
``key_failures`` counters (pre-membership-verify pin).
"""
from __future__ import annotations

import json
import logging
import os.path as osp
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from kumo_core.types import TaskType
from kumoapi.typing import TimeUnit

from kumo.connector.parquet_connector import ParquetConnector
from kumo.data.generic.diskgraph_staging import ensure_stage_ingested
from kumo.data.generic.diskgraph_train_table import (
    generate_train_table_via_diskgraph_ttng,
    persist_durable_ttng_bundle,
)
from kumo.graph.graph import Graph
from kumo.testing import get_validated_query

# 10 users in G1; G2 keeps only the first 6 → 4/10 seeds (40%) orphan.
N_USERS_G1 = 10
N_USERS_G2 = 6
EXPECTED_ORPHANS = N_USERS_G1 - N_USERS_G2
EXPECTED_PCT = 100.0 * EXPECTED_ORPHANS / N_USERS_G1

# Variable per-user order counts so the COUNT target is not constant.
ORDERS_PER_USER = [1, 3, 2, 5, 1, 4, 2, 3, 1, 2]

PQL = "PREDICT COUNT(orders.*, 0, 5000) FOR EACH users.user_id"

RANDOM_SPLIT_CONFIG = {
    "_target_": "kumo.task.split.RandomSplit",
    "ratio": [0.8, 0.1, 0.1],
}

# Config dicts the wheel's custom-TT wrappers require (entity metadata rides
# in split_stats_config; target_stype in train_table_col_stats_config).
CUSTOM_SPLIT_STATS_CONFIG = {
    "entity_table": "users",
    "entity_pk_column": "user_id",
}
CUSTOM_COL_STATS_CONFIG = {"target_stype": "numerical"}


def _write_source_parquets(data_dir: Path, n_users: int) -> None:
    pq.write_table(
        pa.table({
            "user_id": pa.array(list(range(n_users)), type=pa.int64()),
        }),
        str(data_dir / "users.parquet"),
    )
    user_ids, timestamps = [], []
    for uid in range(n_users):
        for k in range(ORDERS_PER_USER[uid]):
            user_ids.append(uid)
            timestamps.append(100 + 40 * k)
    pq.write_table(
        pa.table({
            "order_id":
            pa.array(list(range(len(user_ids))), type=pa.int64()),
            "user_id":
            pa.array(user_ids, type=pa.int64()),
            "ts":
            pa.array(timestamps, type=pa.int64()),
        }),
        str(data_dir / "orders.parquet"),
    )


def _ingest_index(data_dir: Path, index_dir: Path, dataset_name: str) -> str:
    """Build a diskgraph index over the users/orders parquets (real wheel)."""
    from diskgraph import ingest as rust_ingest

    config = {
        "dataset_name":
        dataset_name,
        "num_shards":
        1,
        "tables": [
            {
                "name": "users",
                "columns": [{
                    "name": "user_id",
                    "ingested_dtype": "int64"
                }],
                "primary_key": "user_id",
                "source": {
                    "type": "parquet",
                    "paths": [str(data_dir / "users.parquet")],
                },
            },
            {
                "name":
                "orders",
                "columns": [
                    {
                        "name": "order_id",
                        "ingested_dtype": "int64"
                    },
                    {
                        "name": "user_id",
                        "ingested_dtype": "int64"
                    },
                    {
                        "name": "ts",
                        "ingested_dtype": "int64"
                    },
                ],
                "primary_key":
                "order_id",
                "fkeys": [{
                    "src_column": "user_id",
                    "dst_table": "users",
                    "reverse": {
                        "time": {
                            "table": "orders",
                            "column": "ts",
                            "unit": "s",
                        },
                    },
                }],
                "source": {
                    "type": "parquet",
                    "paths": [str(data_dir / "orders.parquet")],
                },
            },
        ],
    }
    index_dir.mkdir(parents=True, exist_ok=True)
    rust_ingest(json.dumps(config), str(index_dir), col_stats=True,
                cat_count_strategy="hybrid")
    return str(index_dir)


def _build_kumo_graph(data_dir: Path) -> "Graph":
    graph = Graph(name="cross_graph_contamination_e2e")
    graph.connect(ParquetConnector("local", root_dir=str(data_dir)))
    graph.add("users", pkey="user_id")
    graph.add("orders", time_col="ts", pkey="order_id")
    graph.link(src_table="orders", src_key="user_id", dst_table="users")
    return graph.infer_metadata()


@pytest.fixture()
def contamination_env(tmp_path: Path):
    """G1 (10 users) + G2 (6 users, same schema) indexes + validated PQL."""
    g1_data = tmp_path / "g1_data"
    g2_data = tmp_path / "g2_data"
    g1_data.mkdir()
    g2_data.mkdir()
    _write_source_parquets(g1_data, N_USERS_G1)
    _write_source_parquets(g2_data, N_USERS_G2)
    g1_index = _ingest_index(g1_data, tmp_path / "g1_index", "cross_g1")
    g2_index = _ingest_index(g2_data, tmp_path / "g2_index", "cross_g2")
    graph = _build_kumo_graph(g1_data)
    validated_query = get_validated_query(graph, PQL)
    return {
        "g1_index": g1_index,
        "g2_index": g2_index,
        "graph": graph,
        "validated_query": validated_query,
        "tmp_path": tmp_path,
    }


def _manifest_skips(extension_path: str):
    """(skipped, total_rows) from the published manifest, or None when the
    wheel predates the counters (pre-membership-verify pin). ``total_rows`` is
    the PRE-drop seed total (``input_row_count``): orphan rows are dropped at
    ingest, so per-stage ``row_count`` is the surviving subset."""
    with open(osp.join(extension_path, "diskgraph_manifest.json")) as f:
        manifest = json.load(f)
    kf = manifest.get("key_failures")
    if kf is None:
        return None
    skipped = (int(kf.get("parse_failures") or 0) +
               int(kf.get("lookup_failures") or 0) +
               int(kf.get("membership_failures") or 0))
    task_dataset = manifest.get("task_dataset") or {}
    total = int(task_dataset.get("input_row_count") or 0)
    if total == 0:
        stages = task_dataset.get("stages") or {}
        total = sum(
            int((s or {}).get("row_count") or 0) for s in stages.values())
    return skipped, total


def _manifest_survivors(extension_path: str) -> int:
    """Sum of the surviving per-stage ``row_count`` (post encode-time drop)."""
    with open(osp.join(extension_path, "diskgraph_manifest.json")) as f:
        manifest = json.load(f)
    stages = (manifest.get("task_dataset") or {}).get("stages") or {}
    return sum(int((s or {}).get("row_count") or 0) for s in stages.values())


def _require_counter_wheel(extension_path: str):
    """Feature-detect the wheel; old pins publish no ``key_failures``."""
    result = _manifest_skips(extension_path)
    if result is None:
        pytest.skip("installed diskgraph wheel predates the ingest "
                    "key_failures counters (membership verification)")
    return result


def _bind_custom_bundle_to_attempt(staging_dir: str, tt_id: str) -> str:
    """Mirror the attempt-scoped scratch layout used by Kumo wrappers."""
    source = Path(staging_dir)
    destination = source.parent / f'.{tt_id}.staging'
    source.rename(destination)
    return str(destination)


def _skip_warnings(caplog) -> list:
    return [
        r.getMessage() for r in caplog.records
        if "seed FK reference(s)" in r.getMessage()
    ]


def _gen_pql_train_bundle(env, tt_id: str) -> str:
    """Produce the PQL train-table staging bundle against G1."""
    output_path = str(env["tmp_path"] / f"{tt_id}.parquet")
    out_path, _ = generate_train_table_via_diskgraph_ttng(
        query=env["validated_query"],
        graph=env["graph"],
        timestamps=[0],
        task_type=TaskType.REGRESSION.value.lower(),
        forecast_length=1,
        lag_timesteps=None,
        list_distinct_target_is_array=False,
        include_frequency=False,
        entity_time_col=None,
        index_path=env["g1_index"],
        output_path=output_path,
        tt_id=tt_id,
        split_config=RANDOM_SPLIT_CONFIG,
        split_family="RandomSplit",
        downsampling=None,
        input_timestamp_unit=TimeUnit.SECONDS,
        target_stype="numerical",
    )
    return osp.dirname(out_path)


def _assert_cross_graph_contamination(env, staging_dir: str, tt_id: str, caplog,
                                      monkeypatch,
                                      predict_stage_name=None) -> None:
    """The shared contract for every producer: same-graph ingest is silent
    with zero counters; G2 ingest skips exactly the orphan seeds, warns, and
    hard-fails via the knob on both the live and reused-extension paths."""
    kind = "predict" if predict_stage_name is not None else "train"
    durable_staging_dir = str(env["tmp_path"] / f"{tt_id}.ttng_staging")
    persist_durable_ttng_bundle(staging_dir, durable_staging_dir)
    staging_dir = durable_staging_dir

    # Same-graph control: zero skips, no warning.
    with caplog.at_level(logging.WARNING):
        ext_g1 = ensure_stage_ingested(index_path=env["g1_index"],
                                       target_tt_id=tt_id,
                                       staging_dir=staging_dir,
                                       predict_stage_name=predict_stage_name)
    skipped, total = _require_counter_wheel(ext_g1)
    assert (skipped, total) == (0, N_USERS_G1), (
        f"same-graph {kind} ingest must keep every seed: "
        f"skipped={skipped} total={total}")
    assert not _skip_warnings(caplog), "same-graph ingest must stay silent"
    caplog.clear()

    # Cross-graph: exactly the dropped users' seeds skip, with a warning.
    with caplog.at_level(logging.WARNING):
        ext_g2 = ensure_stage_ingested(index_path=env["g2_index"],
                                       target_tt_id=tt_id,
                                       staging_dir=staging_dir,
                                       predict_stage_name=predict_stage_name)
    skipped, total = _manifest_skips(ext_g2)
    assert (skipped, total) == (EXPECTED_ORPHANS, N_USERS_G1), (
        f"cross-graph {kind} ingest must skip exactly the contaminated "
        f"seeds: skipped={skipped} total={total}")
    # The orphan seeds are DROPPED at encode, not masked: the surviving stage
    # rows are exactly the members that exist in G2.
    assert _manifest_survivors(ext_g2) == N_USERS_G2, (
        f"cross-graph {kind} ingest must drop the orphan rows: survivors="
        f"{_manifest_survivors(ext_g2)} expected {N_USERS_G2}")
    warnings = _skip_warnings(caplog)
    assert warnings, f"cross-graph {kind} ingest must warn"
    assert f"TT-NG {kind}:" in warnings[0]
    assert f"~{EXPECTED_PCT:.1f}%" in warnings[0]
    caplog.clear()

    # Knob: at/above the threshold the materialization hard-fails — and it
    # must fire again on the REUSED already-published extension (via the
    # persisted manifest), not just on the ingest that published it.
    monkeypatch.setenv("KUMO_TTNG_ORPHAN_FAIL_PCT", str(EXPECTED_PCT - 10.0))
    with pytest.raises(RuntimeError, match="KUMO_TTNG_ORPHAN_FAIL_PCT"):
        ensure_stage_ingested(index_path=env["g2_index"], target_tt_id=tt_id,
                              staging_dir=staging_dir,
                              predict_stage_name=predict_stage_name)
    monkeypatch.delenv("KUMO_TTNG_ORPHAN_FAIL_PCT")


def test_pql_train_table_cross_graph_contamination(contamination_env, caplog,
                                                   monkeypatch):
    """PQL train table gen'd on G1, ingested for G2 (train shape)."""
    staging_dir = _gen_pql_train_bundle(contamination_env, "ttx-pql-train")
    _assert_cross_graph_contamination(contamination_env, staging_dir,
                                      "ttx-pql-train", caplog, monkeypatch)


def test_custom_train_table_cross_graph_contamination(contamination_env, caplog,
                                                      monkeypatch):
    """Customer-provided train table (custom TT) over G1 entities, ingested
    for G2 — identical chokepoint, custom producer."""
    from diskgraph import build_train_table_artifacts_for_existing_table
    from diskgraph.split_contract import SplitSpec

    env = contamination_env
    custom_pq = env["tmp_path"] / "custom_tt.parquet"
    pq.write_table(
        pa.table({
            "ENTITY1":
            pa.array(list(range(N_USERS_G1)), type=pa.int64()),
            "TIMESTAMP":
            pa.array([0] * N_USERS_G1, type=pa.timestamp("ns")),
            "TARGET1":
            pa.array([float(n) for n in ORDERS_PER_USER], type=pa.float64()),
        }),
        str(custom_pq),
    )
    result = build_train_table_artifacts_for_existing_table(
        output_path=str(custom_pq),
        split_config=SplitSpec(kind="RandomSplit", ratio=[0.8, 0.1, 0.1],
                               rnd_seed=42, is_temporal=False),
        input_timestamp_unit="nanoseconds",
        tt_id="ttx-custom-train",
        index_path=env["g1_index"],
        split_stats_config=CUSTOM_SPLIT_STATS_CONFIG,
        train_table_col_stats_config=CUSTOM_COL_STATS_CONFIG,
    )
    staging_dir = _bind_custom_bundle_to_attempt(result["staging_dir"],
                                                 "ttx-custom-train")
    _assert_cross_graph_contamination(env, staging_dir, "ttx-custom-train",
                                      caplog, monkeypatch)


def test_predict_table_cross_graph_contamination(contamination_env, caplog,
                                                 monkeypatch):
    """Plan-driven predict table gen'd on G1, ingested for G2. The predict
    path is the train-table path plus ``predict_stage_name='predict'``."""
    from diskgraph import execute_tt_gen

    from kumo.data.generic.diskgraph_batch_pred import build_ttng_predict_plan

    env = contamination_env
    plan_json = build_ttng_predict_plan(
        env["validated_query"],
        index_path=env["g1_index"],
        task_type=TaskType.REGRESSION,
        graph=env["graph"],
        timestamps=[0],
        list_distinct_target_is_array=False,
        input_timestamp_unit=TimeUnit.SECONDS,
        target_stype="numerical",
        entity_time_col=None,
    )
    gen = execute_tt_gen(plan_json, env["g1_index"], "ttx-pred")
    _assert_cross_graph_contamination(env, gen["staging_dir"], "ttx-pred",
                                      caplog, monkeypatch,
                                      predict_stage_name="predict")


def test_custom_predict_table_cross_graph_contamination(contamination_env,
                                                        caplog, monkeypatch):
    """Customer-provided predict table (custom pred) over G1 entities,
    ingested for G2 with ``predict_stage_name='predict'``."""
    from diskgraph import build_prediction_table_artifacts_for_existing_table

    env = contamination_env
    custom_pq = env["tmp_path"] / "custom_pred.parquet"
    pq.write_table(
        pa.table({
            "ENTITY1":
            pa.array(list(range(N_USERS_G1)), type=pa.int64()),
            "TIMESTAMP":
            pa.array([0] * N_USERS_G1, type=pa.timestamp("ns")),
        }),
        str(custom_pq),
    )
    result = build_prediction_table_artifacts_for_existing_table(
        output_path=str(custom_pq),
        input_timestamp_unit="nanoseconds",
        tt_id="ttx-custom-pred",
        index_path=env["g1_index"],
        split_stats_config=CUSTOM_SPLIT_STATS_CONFIG,
        train_table_col_stats_config=CUSTOM_COL_STATS_CONFIG,
    )
    staging_dir = _bind_custom_bundle_to_attempt(result["staging_dir"],
                                                 "ttx-custom-pred")
    _assert_cross_graph_contamination(env, staging_dir, "ttx-custom-pred",
                                      caplog, monkeypatch,
                                      predict_stage_name="predict")
