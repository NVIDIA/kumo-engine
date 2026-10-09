# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end harness for ``generate_batch_pred_table_via_diskgraph`` against
multicategorical sources and FK link-prediction targets.

Drives the kumo PQL → ``PredictiveQueryValidator`` → ``DiskGraphPlanCompiler``
(``is_prediction=True``) → ``execute_tt_plan`` chain through a real diskgraph
wheel against small synthetic parquet, parametrized over the source-column
on-disk shape and the link-prediction writer arm. Runs in seconds, no
Temporal infrastructure required.

Why this layer is needed:

- ``test_diskgraph_batch_pred.py`` mocks the wheel and ``DiskGraphPlanCompiler``
  — it never exercises a real BP plan against real parquet and cannot detect
  plan-shape regressions surfaced only at execution time.
- ``test/rest/integ/test_diskgraph_e2e.py`` runs BP through the real wheel
  but is Temporal-backed REST (5–15 min/run) tied to one hardcoded regression
  PQL with no multicategorical or LP-writer-arm coverage.
- BP-specific compile kwargs (``is_prediction=True``,
  ``lp_prediction_mode='LHS'``/``'RHS'``, ``lp_rhs_entity_table``,
  ``lp_rhs_entity_column``, ``target_stype``, ``list_distinct_target_is_array``)
  are not exercised by TT-gen tests; wire breaks in those kwargs surface
  only at BP execution time.

Coverage:

    +-------------+--------------------------+---------------------------------+
    | task        | source / target shape    | writer arm exercised            |
    +-------------+--------------------------+---------------------------------+
    | node-pred   | multicat utf8_delim      | _execute_node_prediction        |
    | node-pred   | multicat string_array    | _execute_node_prediction        |
    | node-pred   | multicat int64_array     | _execute_node_prediction        |
    | link-pred   | FK product_id, modern    | _execute_link_prediction (array)|
    | link-pred   | FK product_id, hstack    | _execute_link_prediction (legacy)|
    +-------------+--------------------------+---------------------------------+

The node-pred cases exercise the multicategorical per-token observation
path at BP plan time (the source-side feature read still observes per token
even though BP drops the TARGET column from the output parquet). The two
link-pred cases exercise both LP write arms reachable from kumo PQL — the
``list_distinct_target_is_array=True`` modern arm (LHS+RHS to separate
parquets) and the legacy HStack arm (zipped LHS×RHS into one parquet).

Fixture builders (canonical 3-user / 4-order multicat layout and 3-table FK
link-pred graph) are imported from the sibling TT-gen e2e file rather than
duplicated.

Run with::

    source ~/work/kumo/venv/bin/activate
    cd ~/work/kumo
    pytest test/data/generic/test_diskgraph_bp_table_e2e.py -v
"""
from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import pytest
from kumo_core.types import TaskType
from kumoapi.typing import TimeUnit

from kumo.data.generic.diskgraph_batch_pred import (
    generate_batch_pred_table_via_diskgraph_ttng,
)
from kumo.data.generic.diskgraph_staging import (
    predict_staging_tt_id,
    ttng_staging_dir,
)
from kumo.testing import get_validated_query

from .test_diskgraph_train_table_e2e import (
    USER_IDS,
    _build_kumo_graph,
    _build_linkpred_graph,
    _ingest_index_handrolled,
    _ingest_linkpred_index,
)

# NOTE on conftest's autouse ``run_before_and_after_tests`` fixture: let it
# run rather than opting out via ``auto_setup_kumo_backend``. The autouse
# fixture wires up the data backend ``persist_dir`` that ``kumo.Graph.add``
# needs at construction time (HybridBackend's ``__init_subclass__`` calls
# ``osp.join(get_persist_dir(), CACHE_DIR)``); skipping it leaves
# ``get_persist_dir()`` returning ``None`` and the graph build crashes with
# ``TypeError: expected str, bytes or os.PathLike object, not NoneType``.

# ---------------------------------------------------------------------------
# Node-prediction BP — multicat target across all three on-disk source shapes
# ---------------------------------------------------------------------------
#
# BP's node-pred writer drops the TARGET column (no ground truth at prediction
# time). The output parquet schema is {ENTITY, TIMESTAMP?, arange}. The
# multicategorical per-token observation still fires on the source side at
# plan-execution time — exercised here by the wheel reading
# ``column_semantic_type=multicategorical`` from ``diskgraph_config.json`` and
# dispatching the per-token observe op at the source feature read site.
#
# Assertions are limited to row count == #users (one prediction row per
# entity) and presence of the ENTITY column; per-token TARGET correctness is
# a TT-gen concern because BP does not emit TARGET for node-pred.

# The native-list ``string_array`` source column trips the PQL validator: the
# col-stats seed dtype (``stringlist``) mismatches the stype-inference dtype
# (``string``), so ``ArrayDtype(stringlist)`` is rejected at validation time
# (``ArrayDtype not supported for Dtype stringlist``) — before any diskgraph
# execution. Same pending production issue skipped in the sibling TT-gen file
# (``_STRING_ARRAY_DTYPE_MISMATCH_SKIP_REASON``).
_STRING_ARRAY_DTYPE_MISMATCH_SKIP_REASON = (
    "native-list parquet column: col-stats seed dtype (stringlist) mismatches "
    "stype-inference dtype (string), pending production fix")


@pytest.mark.parametrize("source_shape", [
    "utf8_delim",
    pytest.param(
        "string_array",
        marks=pytest.mark.skip(reason=_STRING_ARRAY_DTYPE_MISMATCH_SKIP_REASON),
    ),
    "int64_array",
])
def test_nodepred_bp_multilabel_classification(
    tmp_path: Path,
    source_shape: str,
):
    """Node-pred BP with multicat target through the real wheel.

    Drives the kumo PQL → validator →
    ``generate_batch_pred_table_via_diskgraph_ttng`` (GEN-only ``execute_tt_gen``
    into the kept predict staging dir) end-to-end against the canonical
    3-user / 4-order multicat fixture. Asserts the staged predict-seed parquet
    contains one seed row per entity with the expected ENTITY ids.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    index_dir = tmp_path / "index"
    output_root = tmp_path / "bp_out"
    output_root.mkdir()

    graph = _build_kumo_graph(data_dir, source_shape)
    index_path = _ingest_index_handrolled(data_dir, index_dir, source_shape)

    pql = "PREDICT LIST_DISTINCT(orders.tags, 0, 5000) CLASSIFY FOR EACH users.user_id"
    validated_query = get_validated_query(graph, pql)

    job_id = "bp-nodepred"
    total_rows, total_partitions = generate_batch_pred_table_via_diskgraph_ttng(
        query=validated_query,
        task_type=TaskType.MULTILABEL_CLASSIFICATION,
        graph=graph,
        timestamps=[5000],
        forecast_length=1,
        lag_timesteps=None,
        use_yoy_labels=False,
        list_distinct_target_is_array=False,
        index_path=index_path,
        output_root=str(output_root),
        input_timestamp_unit=TimeUnit.SECONDS,
        target_stype="multicategorical",
        entity_time_col=None,
        job_id=job_id,
    )
    assert total_rows == len(USER_IDS), (
        f"BP node-pred should emit one row per entity; got {total_rows} "
        f"for {len(USER_IDS)} users on shape={source_shape}")

    # TT-NG BP gens the predict seeds into the KEPT staging dir named by
    # ``predict_staging_tt_id(job_id)`` (materialize ingests it later).
    staging_dir = ttng_staging_dir(index_path, predict_staging_tt_id(job_id))
    output_path = Path(staging_dir) / "tt.parquet"
    assert output_path.exists(), f"BP staging parquet missing at {output_path}"

    table = pq.read_table(str(output_path))
    schema_names = set(table.schema.names)
    # TT-NG predict staging carries the raw producer ``ENTITY1`` seed column
    # (the legacy ``ENTITY``/``arange`` writer columns were an
    # ``execute_tt_plan_with_artifacts`` artifact). No TARGET at predict time.
    assert "ENTITY1" in schema_names, (
        f"BP predict seeds must have ENTITY1 column; got {schema_names}")
    assert "TARGET" not in schema_names and "TARGET1" not in schema_names, (
        f"BP node-pred must not emit a target column; got {schema_names}")

    entities = sorted(table.column("ENTITY1").to_pylist())
    assert entities == sorted(USER_IDS), (
        f"BP entity set mismatch on shape={source_shape}; got {entities}")

    # Gen ALSO materializes the customer-facing serving pred table at
    # ``output_root`` so GET /genpredtable_jobs/{id}/table_data returns a
    # readable path (ENTITY renamed, arange added; node-pred drops TARGET).
    serving = output_root / "part-00000.parquet"
    assert serving.exists(), f"serving pred parquet missing at {serving}"
    serving_cols = set(pq.read_table(str(serving)).schema.names)
    assert serving_cols == {"ENTITY", "TIMESTAMP", "arange"}, serving_cols


# ---------------------------------------------------------------------------
# Link-pred BP — modern array-output writer arm
# ---------------------------------------------------------------------------
#
# When ``list_distinct_target_is_array=True`` BP writes LHS (entities to
# predict for) into the kept predict staging dir. On TT-NG the RHS candidate
# set is FK-free and seeded at INGEST, so no separate RHS parquet and no HStack
# cross-product is materialized at gen time — only the LHS seed table is staged.


def test_linkpred_bp_fk_target_arrayout(tmp_path: Path):
    """Link-pred BP with FK target, TT-NG FK-free (array-target) arm.

    Asserts:
    - LHS predict seeds are staged (``tt.parquet`` under the kept predict
      staging dir); the RHS candidate pool is seeded at ingest, not here.
    - LHS staging schema carries ENTITY1 and no target column.
    - LHS ENTITY set covers all users in the fixture.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    index_dir = tmp_path / "index"
    output_root = tmp_path / "lhs_out"
    target_output_root = tmp_path / "rhs_out"
    output_root.mkdir()
    target_output_root.mkdir()

    graph = _build_linkpred_graph(data_dir)
    index_path = _ingest_linkpred_index(data_dir, index_dir)

    pql = ("PREDICT LIST_DISTINCT(orders.product_id, 0, 5000) RANK TOP 3 "
           "FOR EACH users.user_id")
    validated_query = get_validated_query(graph, pql)

    job_id = "bp-lp-arrayout"
    total_rows, _total_partitions = generate_batch_pred_table_via_diskgraph_ttng(
        query=validated_query,
        task_type=TaskType.TEMPORAL_LINK_PREDICTION,
        graph=graph,
        timestamps=[5000],
        forecast_length=1,
        lag_timesteps=None,
        use_yoy_labels=False,
        list_distinct_target_is_array=True,
        index_path=index_path,
        output_root=str(output_root),
        input_timestamp_unit=TimeUnit.SECONDS,
        target_stype="multicategorical",
        entity_time_col=None,
        target_output_root=str(target_output_root),
        job_id=job_id,
    )

    # TT-NG FK-free LP BP gens ONLY the LHS predict seeds into the kept staging
    # dir; the RHS candidate set is seeded at INGEST (no RHS parquet, no HStack
    # cross-product is materialized on TT-NG). One seed row per LHS entity.
    assert total_rows == len(USER_IDS), (
        f"LP BP should emit one predict seed per LHS entity; got {total_rows}")

    staging_dir = ttng_staging_dir(index_path, predict_staging_tt_id(job_id))
    lhs_path = Path(staging_dir) / "tt.parquet"
    assert lhs_path.exists(), f"LHS staging parquet missing at {lhs_path}"

    lhs_table = pq.read_table(str(lhs_path))
    lhs_schema = set(lhs_table.schema.names)
    assert "ENTITY1" in lhs_schema, (
        f"LP BP LHS seeds must have ENTITY1 column; got {lhs_schema}")
    # LHS predict seeds carry no target column (the RHS candidate pool is
    # seeded downstream at ingest, not joined here).
    assert "TARGET" not in lhs_schema and "TARGET1" not in lhs_schema, (
        f"LP BP LHS seeds must not carry a target column; got {lhs_schema}")
    lhs_entities = sorted(lhs_table.column("ENTITY1").to_pylist())
    assert lhs_entities == sorted(USER_IDS), (
        f"LHS entity set mismatch; got {lhs_entities}")


# ---------------------------------------------------------------------------
# Link-pred BP — legacy HStack writer arm is REMOVED under TT-NG
# ---------------------------------------------------------------------------
#
# The legacy HStack arm (``list_distinct_target_is_array=False``) zipped an
# LHS×RHS cross-product into one parquet. TT-NG LP is FK-free / array-target
# only: the RHS candidate set is seeded at ingest, never materialized as an
# HStack cross-product. The live TT-NG BP path therefore rejects the HStack
# layout with ``NotImplementedError`` rather than producing it. This test
# pins that removed-capability guard.


def test_linkpred_bp_fk_target_hstack(tmp_path: Path):
    """The legacy HStack LP-BP layout is unsupported on TT-NG.

    ``generate_batch_pred_table_via_diskgraph_ttng`` resolves a non-array LP
    request to the HStack layout and raises ``NotImplementedError`` — the
    FK-free / array-target (separate LHS/RHS) layout is the only supported LP
    BP arm on TT-NG.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    index_dir = tmp_path / "index"
    output_root = tmp_path / "hstack_out"
    output_root.mkdir()

    graph = _build_linkpred_graph(data_dir)
    index_path = _ingest_linkpred_index(data_dir, index_dir)

    pql = ("PREDICT LIST_DISTINCT(orders.product_id, 0, 5000) RANK TOP 3 "
           "FOR EACH users.user_id")
    validated_query = get_validated_query(graph, pql)

    with pytest.raises(NotImplementedError, match="separate|HStack|array"):
        generate_batch_pred_table_via_diskgraph_ttng(
            query=validated_query,
            task_type=TaskType.TEMPORAL_LINK_PREDICTION,
            graph=graph,
            timestamps=[5000],
            forecast_length=1,
            lag_timesteps=None,
            use_yoy_labels=False,
            list_distinct_target_is_array=False,
            index_path=index_path,
            output_root=str(output_root),
            input_timestamp_unit=TimeUnit.SECONDS,
            target_stype="multicategorical",
            entity_time_col=None,
            job_id="bp-lp-hstack",
        )
