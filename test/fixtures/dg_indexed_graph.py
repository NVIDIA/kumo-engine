# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real-FINANCIAL-backed kumo.Graph + DiskGraph index test fixture.

Bypasses :class:`DiskgraphService.ingest` (which requires a full
``data_persist_dir/ingest`` layout from ``Connector.export_table`` +
dataplane mirroring + deploy_env DG config — none of which is set up in
unit-scale tests) and calls the wheel's ``rust_ingest`` directly with a
hand-built ingest config, mirroring the hand-rolled ingest helper in the
train-table e2e test.

The kumo.Graph object is built via ``ParquetConnector`` +
``add_from_config('test/parquet_data/FINANCIAL/metadata.json')`` so the
in-memory schema matches the DG index produced by ``rust_ingest`` (same
metadata.json drives both).

Pass ``diskgraph_index_path=index_path`` to ``graph.materialize()`` to fire the
DG short-circuit (:func:`kumo.engine._backend._is_dg_short_circuit`).
"""
from __future__ import annotations

import json
import os
import os.path as osp
import uuid
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import pyarrow.parquet as pq
import pytest

from kumo.connector.parquet_connector import ParquetConnector
from kumo.graph.graph import Graph
from kumo.pquery.predictive_query import PredictiveQueryV2
from kumo.testing.data import (
    financial_csv_dir,
    financial_parquet_connector,
    financial_parquet_dir,
)

# Kumo metadata.json ``dtype`` → DG ``ingested_dtype``. Mirrors the mapping
# the production ``DiskgraphConfigConverter`` (in kumo-diskgraph) emits when
# fed a kumo GraphConfig — captured here as a small explicit table so the
# fixture does not depend on the converter's internal heuristics.
_KUMO_DTYPE_TO_INGESTED: Dict[str, str] = {
    "int": "int64",
    "float": "float64",
    "string": "string",
    "date": "int64",  # FINANCIAL parquet stores DATE as int64 epoch
    "timestamp": "int64",
    "bool": "bool",
}

# Override per (table, column) when the on-disk parquet dtype disagrees with
# metadata.json (e.g. TRANS.DATE is ``timestamp[us]`` on disk but declared
# ``date`` in metadata). Inspecting the parquet schema below resolves these
# automatically; this dict is the manual fallback if pyarrow inspection
# yields an unmappable physical type.
_PARQUET_PHYSICAL_TO_INGESTED: Dict[str, str] = {
    "int8": "int64",
    "int16": "int64",
    "int32": "int64",
    "int64": "int64",
    "uint8": "int64",
    "uint16": "int64",
    "uint32": "int64",
    "uint64": "int64",
    "float": "float64",
    "double": "float64",
    "string": "string",
    "large_string": "string",
    "bool": "bool",
}

# Element ``ingested_dtype`` (the value already mapped through
# ``_PARQUET_PHYSICAL_TO_INGESTED``) -> the DG ``*_array`` ``IngestedType``
# variant (snake_case, matching diskgraph-ingest/src/graph_config.rs:201
# ``IngestedType``). Embedding / list features (e.g.
# ``test_trainer_embedding_feature``'s float-list ``EMBEDDING`` and int-list
# ``EMBEDDING_INT`` columns) ingest through these array variants.
_ELEMENT_INGESTED_TO_ARRAY: Dict[str, str] = {
    "int64": "int64_array",
    "float64": "float32_array",  # DG stores float-list features as float32
    "string": "string_array",
    "bool": "int8_array",
}


def _physical_ingested_dtype(parquet_path: Path, col_name: str) -> str:
    """Inspect the parquet schema and return the DG ``ingested_dtype``."""
    schema = pq.read_schema(str(parquet_path))
    field = schema.field(col_name)
    pa_type = str(field.type)
    # Timestamp parquet columns ingest as int64 (epoch units) for DG.
    if pa_type.startswith("timestamp"):
        return "int64"
    if pa_type.startswith("date"):
        return "int64"
    # List / embedding columns: PyArrow renders these as
    # ``list<element: double>`` / ``list<item: int64>`` etc. Map the element
    # type through the scalar table, then to the DG ``*_array`` variant.
    if pa_type.startswith("list<") or pa_type.startswith("large_list<"):
        elem_field = field.type.value_field
        elem_pa_type = str(elem_field.type)
        elem_ingested = None
        if elem_pa_type in _PARQUET_PHYSICAL_TO_INGESTED:
            elem_ingested = _PARQUET_PHYSICAL_TO_INGESTED[elem_pa_type]
        if elem_ingested is None or elem_ingested not in (
                _ELEMENT_INGESTED_TO_ARRAY):
            raise ValueError(f"Unmappable list element dtype for "
                             f"{parquet_path.name}.{col_name}: {pa_type} "
                             f"(element {elem_pa_type}). "
                             f"Extend _PARQUET_PHYSICAL_TO_INGESTED / "
                             f"_ELEMENT_INGESTED_TO_ARRAY.")
        return _ELEMENT_INGESTED_TO_ARRAY[elem_ingested]
    if pa_type in _PARQUET_PHYSICAL_TO_INGESTED:
        return _PARQUET_PHYSICAL_TO_INGESTED[pa_type]
    raise ValueError(f"Unmappable physical parquet dtype for "
                     f"{parquet_path.name}.{col_name}: {pa_type}. "
                     f"Extend _PARQUET_PHYSICAL_TO_INGESTED.")


def _column_spec(parquet_path: Path, col_name: str) -> Dict[str, str]:
    return {
        "name": col_name,
        "ingested_dtype": _physical_ingested_dtype(parquet_path, col_name),
    }


def _time_col_name(time_col) -> "str | None":
    """Normalize a metadata ``time_col`` to its column-name string.

    ``time_col`` may be ``None``, a bare column-name string (FINANCIAL-shaped),
    or a full column-config dict (GraphConfig-native shape). Returns the
    column name or ``None``.
    """
    if time_col is None:
        return None
    if isinstance(time_col, str):
        return time_col
    if isinstance(time_col, dict):
        return time_col.get("name")
    raise TypeError(f"Unexpected time_col form: {type(time_col)!r}")


def _build_diskgraph_ingest_config(
    parquet_dir: str,
    metadata: dict,
) -> dict:
    """Build the DG ingest config JSON from FINANCIAL metadata.json.

    Iterates ``metadata['tables']`` and emits one DG table block per kumo
    table. Each column's ``ingested_dtype`` is read directly off the
    parquet schema (not metadata.json) so timestamp/date physical types
    map correctly.
    """
    tables_out: List[dict] = []
    for tbl in metadata["tables"]:
        name = tbl["table_name"]
        parquet_path = Path(parquet_dir) / f"{name}.parquet"

        columns: List[Dict[str, str]] = []
        # pkey first. A pkey is optional in the DG ingest config schema
        # (``graph_config.rs`` ``primary_key: Option<String>``); pure fact
        # tables (fkeys + time_col, no pkey) — e.g. the ``external____targets``
        # table built inline by ``test_underscore_names`` — carry ``None``.
        pkey = tbl.get("pkey")
        pkey_name = pkey["name"] if pkey is not None else None
        if pkey_name is not None:
            columns.append(_column_spec(parquet_path, pkey_name))
        # fkeys
        fkey_names: List[str] = []
        for fk in tbl.get("fkeys") or []:
            columns.append(_column_spec(parquet_path, fk["name"]))
            fkey_names.append(fk["name"])
        # regular cols
        for col in tbl.get("cols") or []:
            columns.append(_column_spec(parquet_path, col["name"]))

        # ``time_col`` may be either a bare column-name string (legacy
        # FINANCIAL-shaped fixtures) or a full column-config dict (the
        # GraphConfig-native shape, e.g. COUPON_LIST metadata.json). Normalize
        # to the column name for the DG ingest config either way. When it is a
        # dict the time column is NOT also listed under ``cols``, so add it to
        # the DG column list explicitly.
        time_col = _time_col_name(tbl.get("time_col"))
        if (isinstance(tbl.get("time_col"), dict) and time_col is not None
                and not any(c["name"] == time_col for c in columns)):
            columns.append(_column_spec(parquet_path, time_col))

        fkeys_out: List[dict] = []
        for fk in tbl.get("fkeys") or []:
            fk_entry: Dict[str, object] = {
                "src_column": fk["name"],
                "dst_table": fk["dst_table"],
            }
            # Expose a reverse-edge time so temporal samplers can apply time
            # windows on the (dst <- src) edge. Two shapes occur:
            #
            #  (a) SRC-fact shape (canonical node-pred): the table being
            #      iterated IS the temporal fact table (it carries the
            #      time_col) and its fkey points at the entity table. The
            #      reverse USERS<-ORDERS edge must carry the fact table's OWN
            #      time column as its sort key, e.g.
            #      ``reverse.time = {table: ORDERS, column: TIMESTAMP}``.
            #      This matches the hand-rolled e2e config in
            #      ``test/data/generic/test_diskgraph_train_table_e2e.py::
            #      _ingest_index_handrolled`` (orders.user_id -> users with
            #      reverse.time {table: orders, column: ts}).
            #
            #  (b) DST-fact fallback: the destination table carries the
            #      time_col (legacy FINANCIAL-shaped wiring). Kept for
            #      backward compatibility with fixtures that relied on it.
            #
            # (a) takes precedence when the src table has its own time_col.
            if time_col is not None:
                fk_entry["reverse"] = {
                    "time": {
                        "table": name,
                        "column": time_col,
                        "unit": "s",
                    }
                }
            else:
                dst_tbl = next(
                    (t for t in metadata["tables"]
                     if t["table_name"] == fk["dst_table"]),
                    None,
                )
                dst_time_col = (_time_col_name(dst_tbl.get("time_col"))
                                if dst_tbl is not None else None)
                if dst_time_col is not None:
                    fk_entry["reverse"] = {
                        "time": {
                            "table": name,
                            "column": dst_time_col,
                            "unit": "s",
                        }
                    }
            fkeys_out.append(fk_entry)

        table_block: Dict[str, object] = {
            "name": name,
            "columns": columns,
            "source": {
                "type": "parquet",
                "paths": [str(parquet_path)],
            },
        }
        if pkey_name is not None:
            table_block["primary_key"] = pkey_name
        if fkeys_out:
            table_block["fkeys"] = fkeys_out
        if time_col:
            table_block["time_column"] = time_col
        tables_out.append(table_block)

    return {
        "dataset_name": metadata["name"],
        "num_shards": 1,
        "tables": tables_out,
    }


def _training_classification_table_block(parquet_dir: str) -> dict:
    """Metadata-shaped block for the TRAINING_CLASSIFICATION parquet.

    The shipped ``metadata.json`` does not declare TRAINING_CLASSIFICATION
    (it is a per-test target table, not a graph node). The legacy v2.21
    test pattern attached it on the fly via
    ``graph.add('TRAINING_CLASSIFICATION', pkey='LOAN_ID').infer_metadata()``.
    To keep the DG ingest config and the kumo.Graph in lockstep (so the
    table is both indexed in DG and present in the in-memory graph), we
    inject one declarative block at fixture build time and reuse it from
    both code paths.

    Columns are read straight off the parquet schema (LOAN_ID/int64,
    DATE/int64, STATUS/string, arange/int64); LOAN_ID is the pkey. No
    fkeys, no time_col — this matches the legacy fixture shape (static
    node-pred target, materialized as a standalone node table).
    """
    return {
        "table_name":
        "TRAINING_CLASSIFICATION",
        "data_source":
        None,
        "data_version":
        None,
        "pkey": {
            "name": "LOAN_ID",
            "stype": None,
            "dtype": "int",
            "na_values": None,
            "metadata": {},
            "drop_duplicates": False,
        },
        "fkeys": [],
        "cols": [
            {
                "name": "DATE",
                "stype": None,
                "dtype": "int",
                "na_values": None,
                "metadata": {},
                "encoder": None,
                "stats": {},
            },
            {
                "name": "STATUS",
                "stype": None,
                "dtype": "string",
                "na_values": None,
                "metadata": {},
                "encoder": None,
                "stats": {},
            },
            {
                "name": "arange",
                "stype": None,
                "dtype": "int",
                "na_values": None,
                "metadata": {},
                "encoder": None,
                "stats": {},
            },
        ],
        "time_col":
        None,
        "source_name":
        None,
        "stats": {},
    }


def _build_financial_kumo_graph(augmented_metadata_path: str) -> Graph:
    """Build the kumo.Graph mirror of the DG-ingested FINANCIAL dataset.

    Uses the augmented metadata file (written by the fixture into the
    tmp index dir) so the in-memory kumo schema lines up 1:1 with the
    DG index — including the TRAINING_CLASSIFICATION target table.

    Note: this returns the graph with column ``dtype`` but **not**
    ``stype`` populated. Tests that need a PQL-validatable graph must
    call ``graph.infer_metadata()`` themselves — this requires the
    per-test autouse ``persist_dir`` setup which is not yet active when
    the session fixture builds.
    """
    graph = Graph(name="FINANCIAL")
    graph.connect(financial_parquet_connector())
    graph.add_from_config(augmented_metadata_path)
    return graph


@pytest.fixture(scope="session")
def dg_indexed_financial_graph(tmp_path_factory) -> Tuple[Graph, str]:
    """Real-FINANCIAL-backed kumo.Graph with a DiskGraph index built once
    per test session. Returns ``(graph, index_path)``.

    Calls ``from diskgraph import ingest as rust_ingest`` directly with a
    config built from ``test/parquet_data/FINANCIAL/metadata.json`` and
    the graph table specs. Bypasses :class:`DiskgraphService.ingest`
    because that code path requires a full data_persist_dir/ingest layout
    from ``Connector.export_table`` + dataplane mirroring + deploy_env DG
    config — none of which is set up in unit-scale tests.
    """
    from diskgraph import ingest as rust_ingest

    parquet_dir = financial_parquet_dir()
    metadata_path = osp.join(parquet_dir, "metadata.json")
    with open(metadata_path) as f:
        metadata = json.load(f)

    # Augment with TRAINING_CLASSIFICATION (target table, not in shipped
    # metadata.json — see _training_classification_table_block).
    metadata["tables"].append(_training_classification_table_block(parquet_dir))

    config = _build_diskgraph_ingest_config(parquet_dir, metadata)

    index_dir = tmp_path_factory.mktemp("dg_indexed_financial_graph_idx")
    rust_ingest(
        json.dumps(config),
        str(index_dir),
        col_stats=True,
        cat_count_strategy="hybrid",
        success_marker_content=json.dumps(
            {"source": "dg_indexed_financial_graph fixture"}),
    )

    # Persist augmented metadata next to the index so the kumo.Graph
    # build path consumes the same schema as DG.
    augmented_meta_path = osp.join(str(index_dir), "metadata_augmented.json")
    with open(augmented_meta_path, "w") as f:
        json.dump(metadata, f)

    graph = _build_financial_kumo_graph(augmented_meta_path)
    return graph, str(index_dir)


# ---------------------------------------------------------------------------
# STATIC synthetic node-pred graph (separate entity/target tables) + DiskGraph
# index.
# ---------------------------------------------------------------------------
#
# Why a second fixture: ~16 tests in
# ``test/pquery/train_table_generator/test_nodepred_train_generator.py`` build
# a BESPOKE SYNTHETIC node-pred graph (entity table + target/time table) -- NOT
# the FINANCIAL graph -- and drive ``generate_train_table()`` /
# ``Task.materialize()``, which now require a DG index. The FINANCIAL fixture
# above does not match those tests' table/column/target assertions.
#
# This is a STATIC node-pred graph (USERS has NO time_col). It is loosely based
# on the entity/target shape of the conftest helper
# ``static_node_pred_test_graph(mode="separate")``, but it deliberately DIVERGES:
# the conftest separate-mode graph gives USERS a temporal anchor (``time_col=DATE``),
# whereas this fixture keeps USERS static, mirroring the proven FINANCIAL
# fixture's static TRAINING_CLASSIFICATION entity. As a result it can only back
# STATIC node-pred assertions -- temporal-anchor / timestamp-tuple assertions
# (e.g. the original ``test_static_node_different_entity_and_target_table``
# TIMESTAMP check) cannot be restored here and need a TEMPORAL synthetic fixture.
#
# Logical graph (separate entity/target tables):
#   USERS         pkey=USER_ID, fkey USER_ID_FKEY -> USERS_TARGET,
#                 col PAYMENTS, NO time_col (static)
#   USERS_TARGET  pkey=USER_ID, cols STATUS / STATUS_BOOL / MULTI_STATUS
# Representative query: PREDICT USERS_TARGET.STATUS_BOOL FOR EACH USERS.USER_ID


def _synthetic_node_pred_tables() -> Dict[str, "object"]:
    """Return the {table_name: DataFrame} for the STATIC separate-entity/target
    synthetic node-pred graph (USER_ID 0..3, with STATUS/STATUS_BOOL/PAYMENTS
    values). NOTE: USERS is kept static (no DATE anchor) -- this diverges from
    ``static_node_pred_test_graph(mode='separate')`` in ``test/conftest.py``,
    which gives USERS a temporal anchor; only STATIC assertions are preserved.
    """
    import pandas as pd

    entity_df = pd.DataFrame({
        "USER_ID": [0, 1, 2, 3],
        "USER_ID_FKEY": [0, 1, 2, 3],
        "PAYMENTS": [10.0, 20.0, 40.0, 80.0],
        "DATE":
        pd.to_datetime(["1990-01-01", "1990-01-02", "1990-01-03",
                        "1990-01-04"]).astype("datetime64[ns]"),
    })
    target_df = pd.DataFrame({
        "USER_ID_PKEY": [0, 1, 3],
        "STATUS": ["A", "B", "B"],
        "STATUS_BOOL": [True, False, True],
        "MULTI_STATUS": ["A,B", "C,D", "A,B,C"],
    })
    return {"USERS": entity_df, "USERS_TARGET": target_df}


def _synthetic_node_pred_metadata(parquet_dir: str) -> dict:
    """Build the metadata.json-shaped dict for the synthetic graph.

    Same shape the shipped FINANCIAL metadata.json uses (``name`` /
    ``data_source_dict`` / ``tables``) so it can drive BOTH the DG ingest
    config (via :func:`_build_diskgraph_ingest_config`) and the in-memory
    kumo.Graph (via ``graph.add_from_config``) — keeping the two in lockstep,
    exactly like the FINANCIAL fixture.
    """
    def col(name: str, dtype: str) -> dict:
        return {
            "name": name,
            "stype": None,
            "dtype": dtype,
            "na_values": None,
            "metadata": {},
            "encoder": None,
            "stats": {},
        }

    users = {
        "table_name":
        "USERS",
        "data_source":
        None,
        "data_version":
        None,
        "pkey": {
            "name": "USER_ID",
            "stype": None,
            "dtype": "int",
            "na_values": None,
            "metadata": {},
            "drop_duplicates": False,
        },
        "fkeys": [{
            "name": "USER_ID_FKEY",
            "stype": None,
            "dtype": "int",
            "na_values": None,
            "metadata": {},
            "dst_table": "USERS_TARGET",
            "dst_key": None,
            "dst_data_source": None,
            "stats": {},
        }],
        "cols": [col("PAYMENTS", "float")],
        # Static entity table: no time_col. Mirrors the proven FINANCIAL
        # fixture's TRAINING_CLASSIFICATION entity (a static target/entity
        # table with no temporal anchor). A static node-pred query
        # (`PREDICT USERS_TARGET.STATUS_BOOL FOR EACH USERS.USER_ID`) does not
        # need entity anchor timestamps, so leaving USERS static lets the DG
        # producer materialize one training row per linked entity.
        "time_col":
        None,
        "source_name":
        None,
        "stats": {},
    }
    users_target = {
        "table_name":
        "USERS_TARGET",
        "data_source":
        None,
        "data_version":
        None,
        "pkey": {
            "name": "USER_ID_PKEY",
            "stype": None,
            "dtype": "int",
            "na_values": None,
            "metadata": {},
            "drop_duplicates": False,
        },
        "fkeys": [],
        "cols": [
            col("STATUS", "string"),
            col("STATUS_BOOL", "bool"),
            col("MULTI_STATUS", "string"),
        ],
        "time_col":
        None,
        "source_name":
        None,
        "stats": {},
    }
    return {
        "name": "SYNTHETIC_NODE_PRED",
        "data_source_dict": {},
        "tables": [users, users_target],
    }


@pytest.fixture(scope="session")
def dg_indexed_synthetic_node_pred_graph(tmp_path_factory) -> Tuple[Graph, str]:
    """Synthetic separate-entity/target node-pred kumo.Graph + DiskGraph index,
    built once per session. Returns ``(graph, index_path)`` in the SAME shape
    as :func:`dg_indexed_financial_graph`.

    Mirrors the FINANCIAL fixture's mechanics: write parquet for each table,
    build a DG ingest config from a metadata.json-shaped dict, call the wheel's
    ``rust_ingest`` directly, then build the matching in-memory kumo.Graph via
    ``add_from_config`` against the same metadata so schemas line up 1:1.
    """
    from diskgraph import ingest as rust_ingest

    parquet_dir = tmp_path_factory.mktemp("dg_synthetic_node_pred_parquet")
    tables = _synthetic_node_pred_tables()
    for name, df in tables.items():
        df.to_parquet(osp.join(str(parquet_dir), f"{name}.parquet"),
                      index=False)

    metadata = _synthetic_node_pred_metadata(str(parquet_dir))
    config = _build_diskgraph_ingest_config(str(parquet_dir), metadata)

    index_dir = tmp_path_factory.mktemp("dg_indexed_synthetic_node_pred_idx")
    rust_ingest(
        json.dumps(config),
        str(index_dir),
        col_stats=True,
        cat_count_strategy="hybrid",
        success_marker_content=json.dumps(
            {"source": "dg_indexed_synthetic_node_pred_graph fixture"}),
    )

    # Persist metadata next to the index and build the kumo.Graph mirror from
    # it (same schema as the DG index).
    meta_path = osp.join(str(index_dir), "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f)

    graph = Graph(name="SYNTHETIC_NODE_PRED")
    graph.connect(ParquetConnector("parquet", str(parquet_dir)))
    graph.add_from_config(meta_path, data_source="parquet")
    return graph, str(index_dir)


# ---------------------------------------------------------------------------
# TEMPORAL synthetic node-pred graph (entity + temporal fact table) + DiskGraph
# index.
# ---------------------------------------------------------------------------
#
# Why a third fixture: the STATIC synthetic fixture above (and the FINANCIAL
# fixture) have no usable temporal anchor on a fact table, so temporal node-pred
# pqueries -- ``PREDICT SUM(ORDERS.ORDER_SCORE, 0, N) FOR EACH USERS.USER_ID``
# and friends -- yield empty train tables there. Many node-pred generator tests
# in ``test_nodepred_train_generator.py`` build a BESPOKE synthetic TEMPORAL
# graph (entity table + a fact table carrying a real ``TIMESTAMP`` time_col) and
# assert temporal stats (anchor dates, per-timeframe min/max, SUM-window
# targets). They are blocked purely because no DG-indexed temporal synthetic
# block exists.
#
# This fixture mirrors the ESSENTIAL temporal shape of the conftest helper
# ``_small_temporal_graph_fixed`` / ``_temporal_graph_fixed`` (USERS entity +
# ORDERS temporal fact table). It deliberately drops the STORES table and the
# STORE_ID fkey, which the ``SUM(ORDERS.ORDER_SCORE) FOR EACH USERS.USER_ID``
# family of node-pred queries does not touch, while preserving the exact ORDERS
# column values (ORDER_SCORE / TIMESTAMP) so the parked tests' Kumo-owned
# ENTITY/TARGET assertions line up after rewiring onto the DG-native path. The
# fact table's ``time_col`` drives the temporal aggregation -- the entity table
# (USERS) needs no anchor of its own, exactly like the FINANCIAL
# ``COUNT(LOAN.AMOUNT, ...) FOR EACH ACCOUNT.ACCOUNT_ID`` shape.
#
# Logical graph:
#   USERS   pkey=USER_ID, col AGE, NO time_col (entity)
#   ORDERS  pkey=ORDER_ID, fkey USER_ID -> USERS, col ORDER_SCORE,
#           time_col TIMESTAMP (temporal fact table)
# Representative query:
#   PREDICT SUM(ORDERS.ORDER_SCORE, 0, 3) FOR EACH USERS.USER_ID


def _temporal_synthetic_node_pred_tables(
    num_users: int = 5,
    num_stores: int = 3,
    num_edges: int = 15,
    add_na_fkeys: bool = False,
    add_na_timestamps: bool = False,
) -> Dict[str, "object"]:
    """Return ``{table_name: DataFrame}`` for the TEMPORAL synthetic node-pred
    graph: a USERS entity table + an ORDERS temporal fact table.

    The ORDERS column values (ORDER_ID / USER_ID / ORDER_SCORE / TIMESTAMP)
    reproduce ``_temporal_graph_fixed`` (test/conftest.py) at its small default
    sizing (num_users=5, num_stores=3, num_edges=15) so a parked test rewired
    onto this fixture keeps the same Kumo-owned ENTITY/TARGET sequence. The
    STORES table and STORE_ID fkey are dropped (unused by the
    ``SUM(ORDERS.ORDER_SCORE) FOR EACH USERS.USER_ID`` query family).

    NA toggles (restore the legacy ``@parametrize('na_values', ...)`` axis the
    Item-11 migration dropped):

    * ``add_na_fkeys``: append a handful of ORDERS rows whose ``USER_ID`` fkey is
      ``pd.NA``.  DG drops fact rows that point at no entity, so the train table
      (shape + ENTITY/TARGET multisets) is UNCHANGED vs the non-NA build.  This
      is the same NA-key-drop contract the GREEN ``test_handle_nan_entity_keys``
      already proves for an entity-pkey column.
    * ``add_na_timestamps``: append a handful of ORDERS rows with ``NaT`` in the
      declared ``TIMESTAMP`` time_col.  A temporal fact row with no anchor cannot
      be placed in any timeframe, so DG is expected to drop it (output
      unchanged).  UNVERIFIED: no green test yet proves DG's Rust ingest accepts
      ``NaT`` in a declared time_col -- the NA-timestamp param is kept as a
      SEPARATE id so it can be skipped in isolation if a GPU run shows DG rejects
      ``NaT``, without losing the proven-good NA-fkey coverage.
    """
    import numpy as np
    import pandas as pd

    user_df = pd.DataFrame({
        "USER_ID": np.arange(num_users),
        "AGE": np.arange(num_users) % 20 + 20,
    })

    order_score_length_1 = int(num_edges / 3)
    order_score_length_2 = num_edges - 3 * order_score_length_1
    order_df = pd.DataFrame({
        "ORDER_ID":
        np.arange(0, num_edges),
        "USER_ID":
        np.repeat(np.arange(0, num_users), int(num_edges / num_users)),
        "ORDER_SCORE":
        np.concatenate((np.array([1, 2, 3] * order_score_length_1),
                        np.array([1] * order_score_length_2))).astype(float),
        # Post-2001 dates: the DiskGraph train-table manifest's per-split
        # SplitTrainTableStats.min_date/max_date are only decoded correctly by
        # the kumo consumer (diskgraph_native_tt_artifacts.py::_from_epoch) when
        # the timestamps are >= 1e18 ns (>= 2001-09-09). 1970-era dates clamp to
        # the Timestamp.max (2262-04-11) sentinel; a realistic 2023 origin flows
        # correctly with no product-code change. Day-granular spacing is kept
        # identical (only the absolute origin shifts 1970 -> 2023).
        "TIMESTAMP": (pd.Timestamp("2023-01-01") + pd.to_timedelta(
            np.arange(num_edges), unit="D")).astype("datetime64[ns]"),
    })
    order_df = _append_na_order_rows(order_df, num_edges, add_na_fkeys,
                                     add_na_timestamps)
    return {"USERS": user_df, "ORDERS": order_df}


def _append_na_order_rows(order_df, num_edges, add_na_fkeys, add_na_timestamps):
    """Append NA-fkey / NaT-timestamp ORDERS rows for the legacy ``na_values``
    coverage axis.  Mirrors how ``_invalid_static_node_pred_tables`` injects
    ``pd.NA`` into a key column (a few extra rows that DG is expected to drop).

    The appended rows reuse benign values for every non-NA column (a valid
    USER_ID, a numeric ORDER_SCORE, a real anchor) so the ONLY thing that can
    make DG drop them is the injected NA key / NaT anchor -- if DG did NOT drop
    them, the entity/target multiset would change and the test's
    UNCHANGED-vs-non-NA assertion would fail loudly.
    """
    import pandas as pd

    if not (add_na_fkeys or add_na_timestamps):
        return order_df

    extra_rows = []
    next_id = int(num_edges)
    if add_na_fkeys:
        # NA fkey: row points at no entity -> dropped, no new ENTITY appears.
        extra_rows.append({
            "ORDER_ID": next_id,
            "USER_ID": pd.NA,
            "ORDER_SCORE": 99.0,
            "TIMESTAMP": pd.Timestamp("2023-01-02"),
        })
        next_id += 1
    if add_na_timestamps:
        # NaT anchor on an otherwise-valid (existing) entity. Expected dropped
        # because a temporal fact row with no anchor cannot join any timeframe.
        # UNVERIFIED on DG ingest -- see _temporal_synthetic_node_pred_tables.
        extra_rows.append({
            "ORDER_ID": next_id,
            "USER_ID": 0,
            "ORDER_SCORE": 99.0,
            "TIMESTAMP": pd.NaT,
        })
        next_id += 1

    extra_df = pd.DataFrame(extra_rows)
    # Promote USER_ID to a nullable integer so pd.NA survives the parquet
    # round-trip as a null key (a plain int64 column cannot hold pd.NA).
    out = pd.concat([order_df, extra_df], ignore_index=True)
    out["USER_ID"] = out["USER_ID"].astype("Int64")
    out["ORDER_ID"] = out["ORDER_ID"].astype("int64")
    out["ORDER_SCORE"] = out["ORDER_SCORE"].astype(float)
    out["TIMESTAMP"] = pd.to_datetime(out["TIMESTAMP"])
    return out


def _temporal_synthetic_node_pred_metadata() -> dict:
    """metadata.json-shaped dict for the TEMPORAL synthetic graph.

    Same shape the FINANCIAL/static-synthetic fixtures use so it drives BOTH
    the DG ingest config (:func:`_build_diskgraph_ingest_config`) and the
    in-memory kumo.Graph (``graph.add_from_config``). ORDERS declares
    ``time_col=TIMESTAMP`` so DG ingest sets a ``time_column`` on the fact
    table block and the temporal sampler can resolve anchor dates.
    """
    def col(name: str, dtype: str) -> dict:
        return {
            "name": name,
            "stype": None,
            "dtype": dtype,
            "na_values": None,
            "metadata": {},
            "encoder": None,
            "stats": {},
        }

    users = {
        "table_name": "USERS",
        "data_source": None,
        "data_version": None,
        "pkey": {
            "name": "USER_ID",
            "stype": None,
            "dtype": "int",
            "na_values": None,
            "metadata": {},
            "drop_duplicates": False,
        },
        "fkeys": [],
        "cols": [col("AGE", "int")],
        "time_col": None,
        "source_name": None,
        "stats": {},
    }
    orders = {
        "table_name":
        "ORDERS",
        "data_source":
        None,
        "data_version":
        None,
        "pkey": {
            "name": "ORDER_ID",
            "stype": None,
            "dtype": "int",
            "na_values": None,
            "metadata": {},
            "drop_duplicates": False,
        },
        "fkeys": [{
            "name": "USER_ID",
            "stype": None,
            "dtype": "int",
            "na_values": None,
            "metadata": {},
            "dst_table": "USERS",
            "dst_key": None,
            "dst_data_source": None,
            "stats": {},
        }],
        "cols": [col("ORDER_SCORE", "float")],
        # GraphConfig (consumed by ``add_from_config``) requires ``time_col`` as
        # a column-config dict, not a bare string. ``_build_diskgraph_ingest_config``
        # / ``_time_col_name`` accept either form and emit the DG ``time_column``
        # off the dict, so this drives both the kumo.Graph and the DG index.
        "time_col":
        col("TIMESTAMP", "time"),
        "source_name":
        None,
        "stats": {},
    }
    return {
        "name": "TEMPORAL_SYNTHETIC_NODE_PRED",
        "data_source_dict": {},
        "tables": [users, orders],
    }


def _build_temporal_synthetic_node_pred_graph(
    tmp_path_factory,
    add_na_fkeys: bool = False,
    add_na_timestamps: bool = False,
) -> Tuple[Graph, str]:
    """Build + DG-index the TEMPORAL synthetic node-pred graph; return
    ``(graph, index_path)`` in the SAME shape as the other DG fixtures.

    Mirrors the FINANCIAL fixture's mechanics: write parquet per table, build a
    DG ingest config from a metadata.json-shaped dict, call the wheel's
    ``rust_ingest`` directly, then build the matching in-memory kumo.Graph via
    ``add_from_config`` against the same metadata so the schemas line up 1:1.

    ``add_na_fkeys`` / ``add_na_timestamps`` thread through to
    :func:`_temporal_synthetic_node_pred_tables` to restore the legacy
    ``na_values`` axis (the NA rows DG is expected to drop).
    """
    from diskgraph import ingest as rust_ingest

    parquet_dir = tmp_path_factory.mktemp("dg_temporal_node_pred_parquet")
    tables = _temporal_synthetic_node_pred_tables(
        add_na_fkeys=add_na_fkeys, add_na_timestamps=add_na_timestamps)
    for name, df in tables.items():
        df.to_parquet(osp.join(str(parquet_dir), f"{name}.parquet"),
                      index=False)

    metadata = _temporal_synthetic_node_pred_metadata()
    config = _build_diskgraph_ingest_config(str(parquet_dir), metadata)

    index_dir = tmp_path_factory.mktemp("dg_indexed_temporal_node_pred_idx")
    rust_ingest(
        json.dumps(config),
        str(index_dir),
        col_stats=True,
        cat_count_strategy="hybrid",
        success_marker_content=json.dumps(
            {"source":
             "dg_indexed_temporal_synthetic_node_pred_graph fixture"}),
    )

    meta_path = osp.join(str(index_dir), "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f)

    graph = Graph(name="TEMPORAL_SYNTHETIC_NODE_PRED")
    graph.connect(ParquetConnector("parquet", str(parquet_dir)))
    graph.add_from_config(meta_path, data_source="parquet")
    return graph, str(index_dir)


@pytest.fixture(scope="session")
def dg_indexed_temporal_synthetic_node_pred_graph(
        tmp_path_factory) -> Tuple[Graph, str]:
    """TEMPORAL synthetic node-pred kumo.Graph + DiskGraph index, built once per
    session. Returns ``(graph, index_path)`` in the SAME shape as
    :func:`dg_indexed_financial_graph` / :func:`dg_indexed_synthetic_node_pred_graph`.

    USERS entity table + ORDERS temporal fact table (real ``TIMESTAMP``
    time_col). Unlike the static synthetic fixture, ORDERS carries a real
    temporal anchor so SUM-window / time-bucketed / regression-over-time
    queries produce NON-EMPTY train tables.

    This is the NO-NA build (the common case for the ~dozen consumers that take
    the tuple directly). Tests that need to toggle NA rows for the legacy
    ``na_values`` coverage axis use the
    :func:`dg_indexed_temporal_synthetic_node_pred_graph_factory` factory.
    """
    return _build_temporal_synthetic_node_pred_graph(tmp_path_factory)


@pytest.fixture(scope="session")
def dg_indexed_temporal_synthetic_node_pred_graph_factory(tmp_path_factory):
    """FACTORY variant of
    :func:`dg_indexed_temporal_synthetic_node_pred_graph`: returns a callable so
    a caller can build the SAME USERS+ORDERS temporal graph WITH or WITHOUT
    injected NA-fkey / NaT-timestamp ORDERS rows, to restore the legacy
    ``@parametrize('na_values', [True, False])`` coverage the Item-11 migration
    dropped::

        graph, index_path = factory()                      # no NA rows
        graph, index_path = factory(add_na_fkeys=True)     # NA fkey, dropped
        graph, index_path = factory(add_na_timestamps=True)  # NaT, dropped

    Kept SEPARATE from the value fixture so the ~dozen tuple consumers are
    untouched.
    """
    def _factory(
        add_na_fkeys: bool = False,
        add_na_timestamps: bool = False,
    ) -> Tuple[Graph, str]:
        return _build_temporal_synthetic_node_pred_graph(
            tmp_path_factory,
            add_na_fkeys=add_na_fkeys,
            add_na_timestamps=add_na_timestamps,
        )

    return _factory


# ---------------------------------------------------------------------------
# SHORT-SPAN temporal synthetic node-pred graph (USERS + ORDERS, but ORDERS
# timestamps cluster into a tiny multi-day window) + DiskGraph index.
# ---------------------------------------------------------------------------
#
# Why a variant of the temporal synthetic fixture: a family of split-validation
# tests (test/task/test_split.py::test_irregular_date_offset_split and friends)
# assert the Kumo-owned INSUFFICIENT-TIMEFRAMES contract for a DateOffsetSplit
# whose val window matches the target aggregation window (so the pre-generation
# `validate_date_offset_split` "Split size is too small" guard is NOT tripped),
# yet the graph supplies fewer timeframes than the split needs. The default
# 15-day temporal synthetic fixture supplies plenty of timeframes for a
# 1-unit target window (~13), so it cannot exercise that path with a 1-unit
# DateOffsetSplit. This variant keeps the SAME USERS+ORDERS schema/metadata
# (so ORDER_SCORE / TIMESTAMP queries line up 1:1) but compresses every ORDERS
# timestamp into a `span_days`-day window, so a `SUM(ORDERS.ORDER_SCORE, 0, 1)`
# query over a `DateOffsetSplit([-1, 0])` (num_splits == 3) yields only 1
# generated timeframe, so the Kumo-owned pre-generation timeframe-vs-split
# validator ("needs at least N timeframes, but only M are available") fires.


def _short_span_temporal_node_pred_tables(
    num_users: int = 5,
    num_edges: int = 15,
    span_days: int = 3,
) -> Dict[str, "object"]:
    """Return ``{table_name: DataFrame}`` for the SHORT-SPAN temporal synthetic
    node-pred graph -- identical USERS+ORDERS schema to
    :func:`_temporal_synthetic_node_pred_tables`, but every ORDERS TIMESTAMP is
    folded into a ``span_days``-day window so the temporal sampler produces only
    a handful of timeframes."""
    import numpy as np
    import pandas as pd

    user_df = pd.DataFrame({
        "USER_ID": np.arange(num_users),
        "AGE": np.arange(num_users) % 20 + 20,
    })
    order_df = pd.DataFrame({
        "ORDER_ID":
        np.arange(0, num_edges),
        "USER_ID":
        np.repeat(np.arange(0, num_users), int(num_edges / num_users)),
        "ORDER_SCORE": (np.arange(num_edges) % 3 + 1).astype(float),
        # Timestamps fold into a span_days window (post-2001 origin per module
        # note) so the generated-timeframe count stays below the split's need.
        "TIMESTAMP":
        (pd.Timestamp("2023-01-01") +
         pd.to_timedelta(np.arange(num_edges) % span_days,
                         unit="D")).astype("datetime64[ns]"),
    })
    return {"USERS": user_df, "ORDERS": order_df}


@pytest.fixture(scope="session")
def dg_indexed_short_span_temporal_node_pred_graph(
        tmp_path_factory) -> Tuple[Graph, str]:
    """SHORT-SPAN temporal synthetic node-pred kumo.Graph + DiskGraph index.

    Same USERS+ORDERS schema as
    :func:`dg_indexed_temporal_synthetic_node_pred_graph` (reuses
    :func:`_temporal_synthetic_node_pred_metadata`), but ORDERS timestamps fold
    into a 3-day window so a 1-unit target window over a 3-way DateOffsetSplit
    generates too few timeframes -- exercising the Kumo-owned
    insufficient-timeframes-for-split validator. Returns ``(graph, index_path)``
    in the SAME shape as the other ``dg_indexed_*`` fixtures.
    """
    from diskgraph import ingest as rust_ingest

    parquet_dir = tmp_path_factory.mktemp("dg_short_span_temporal_parquet")
    tables = _short_span_temporal_node_pred_tables()
    for name, df in tables.items():
        df.to_parquet(osp.join(str(parquet_dir), f"{name}.parquet"),
                      index=False)

    metadata = _temporal_synthetic_node_pred_metadata()
    config = _build_diskgraph_ingest_config(str(parquet_dir), metadata)

    index_dir = tmp_path_factory.mktemp("dg_indexed_short_span_temporal_idx")
    rust_ingest(
        json.dumps(config),
        str(index_dir),
        col_stats=True,
        cat_count_strategy="hybrid",
        success_marker_content=json.dumps(
            {"source": "dg_indexed_short_span_temporal_node_pred_graph"}),
    )

    meta_path = osp.join(str(index_dir), "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f)

    graph = Graph(name="TEMPORAL_SYNTHETIC_NODE_PRED")
    graph.connect(ParquetConnector("parquet", str(parquet_dir)))
    graph.add_from_config(meta_path, data_source="parquet")
    return graph, str(index_dir)


# ---------------------------------------------------------------------------
# TEMPORAL link-pred graph (USERS entity + STORES RHS + ORDERS temporal edge/
# fact table) + DiskGraph index.
# ---------------------------------------------------------------------------
#
# Why a fourth fixture: ~13 temporal LINK-PREDICTION generator/stat tests in
# ``test_linkpred_train_generator.py`` (and rfm-parity / split / task / bp
# siblings) build a BESPOKE temporal LP graph -- USERS entity + STORES RHS +
# ORDERS temporal edge/fact table -- and drive ``generate_train_table()`` /
# ``Task.materialize()`` against a query of the form
# ``PREDICT LIST_DISTINCT(ORDERS.STORE_ID, 0, 2) RANK TOP 3 FOR EACH
# USERS.USER_ID`` with a ``DateOffsetSplit``. They are blocked purely because
# no DG-indexed temporal LINK-PRED block exists (the three node-pred fixtures
# above are single-fkey node-pred shapes, not a two-fkey LP edge table).
#
# This fixture mirrors the ESSENTIAL temporal-LP shape of the conftest helper
# ``_small_temporal_graph_fixed`` / ``_temporal_graph_fixed`` (USERS entity +
# STORES RHS + ORDERS edge table) at its small default sizing (num_users=5,
# num_stores=3, num_edges=15). Unlike the node-pred fixtures, ORDERS carries
# TWO fkeys (USER_ID -> USERS and STORE_ID -> STORES); the RANK-TOP-3 LP query
# ranks STORES (the RHS) for each USER. 15 edges over 5 users x 3 stores means
# every user touches all 3 stores, so RANK-TOP-3 targets and the DateOffset
# windows are non-empty.
#
# Logical graph (two-fkey LP edge table):
#   USERS   pkey=USER_ID, col AGE, NO time_col (entity)
#   STORES  pkey=STORE_ID, NO time_col (RHS / ranked target table)
#   ORDERS  pkey=ORDER_ID, fkey USER_ID -> USERS, fkey STORE_ID -> STORES,
#           cols ORDER_VALUE_3 (weight col) / ORDER_SCORE,
#           time_col TIMESTAMP (temporal edge/fact table)
# Representative query:
#   PREDICT LIST_DISTINCT(ORDERS.STORE_ID, 0, 2) RANK TOP 3
#   FOR EACH USERS.USER_ID
# Default LP split: DateOffsetSplit([-30, 0], DAYS, timeframe=30)


def _temporal_link_pred_tables(
    num_users: int = 5,
    num_stores: int = 3,
    num_edges: int = 15,
    with_rhs_feature: bool = False,
) -> Dict[str, "object"]:
    """Return ``{table_name: DataFrame}`` for the TEMPORAL link-pred graph:
    a USERS entity table + a STORES RHS table + an ORDERS temporal edge/fact
    table carrying TWO fkeys (USER_ID -> USERS, STORE_ID -> STORES).

    The ORDERS column values (ORDER_ID / USER_ID / STORE_ID / ORDER_VALUE_3 /
    ORDER_SCORE / TIMESTAMP) reproduce ``_temporal_graph_fixed``
    (test/conftest.py) at its small default sizing (num_users=5, num_stores=3,
    num_edges=15) so a parked LP test rewired onto this fixture keeps the same
    Kumo-owned ENTITY/TARGET sequence. 15 edges over 5 users x 3 stores means
    every user touches all 3 stores, so RANK-TOP-3 targets + DateOffset windows
    are non-empty.
    """
    import numpy as np
    import pandas as pd

    user_df = pd.DataFrame({
        "USER_ID": np.arange(num_users),
        "AGE": np.arange(num_users) % 20 + 20,
    })
    store_df = pd.DataFrame({"STORE_ID": np.arange(num_stores)})
    if with_rhs_feature:
        # Opt-in numerical RHS feature for the use_rhs_feature embedding modes;
        # default path stays primary-key-only.
        store_df["STORE_VALUE"] = (np.arange(num_stores) % 7).astype(float)

    length = num_edges - int(num_edges / num_stores) * num_stores
    length_2 = num_edges - int(num_edges / 8) * 4
    order_score_length_1 = int(num_edges / 3)
    order_score_length_2 = num_edges - 3 * order_score_length_1
    order_df = pd.DataFrame({
        "ORDER_ID":
        np.arange(0, num_edges),
        "USER_ID":
        np.repeat(np.arange(0, num_users), int(num_edges / num_users)),
        "STORE_ID":
        np.concatenate(
            (np.tile(np.arange(0, num_stores),
                     int(num_edges / num_stores)), np.arange(0, length))),
        # STORE_ID_NUM mirrors STORE_ID as a numerical column (matches
        # ``_temporal_graph_fixed``); it is the WHERE-filter column the RFM
        # parity "filtered" parked test predicates on (ORDERS.STORE_ID_NUM < 2).
        "STORE_ID_NUM":
        np.concatenate(
            (np.tile(np.arange(0, num_stores),
                     int(num_edges / num_stores)), np.arange(0, length))),
        # ORDER_VALUE_3 is the WEIGHT column referenced by the weight-col LP
        # tests; reproduce ``_temporal_graph_fixed``'s ORDER_VALUE_3 values.
        "ORDER_VALUE_3":
        np.concatenate((np.repeat(np.arange(int(num_edges / 8)),
                                  4), np.arange(length_2))),
        "ORDER_SCORE":
        np.concatenate((np.array([1, 2, 3] * order_score_length_1),
                        np.array([1] * order_score_length_2))).astype(float),
        # Post-2001 dates: the DiskGraph train-table manifest's per-split
        # SplitTrainTableStats.min_date/max_date are only decoded correctly by
        # the kumo consumer (diskgraph_native_tt_artifacts.py::_from_epoch) when
        # the timestamps are >= 1e18 ns (>= 2001-09-09). 1970-era dates clamp to
        # the Timestamp.max (2262-04-11) sentinel; a realistic 2023 origin flows
        # correctly with no product-code change. Day-granular spacing is kept
        # identical (only the absolute origin shifts 1970 -> 2023).
        "TIMESTAMP": (pd.Timestamp("2023-01-01") + pd.to_timedelta(
            np.arange(num_edges), unit="D")).astype("datetime64[ns]"),
    })
    return {"USERS": user_df, "STORES": store_df, "ORDERS": order_df}


def _temporal_link_pred_metadata(with_rhs_feature: bool = False) -> dict:
    """metadata.json-shaped dict for the TEMPORAL link-pred graph.

    Same shape the FINANCIAL / node-pred fixtures use so it drives BOTH the DG
    ingest config (:func:`_build_diskgraph_ingest_config`) and the in-memory
    kumo.Graph (``graph.add_from_config``). ORDERS declares TWO fkeys
    (USER_ID -> USERS, STORE_ID -> STORES) and ``time_col=TIMESTAMP``; the DG
    config builder iterates ``tbl['fkeys']`` so the two-fkey edge table emits
    one reverse-edge block per fkey, both carrying ORDERS' own TIMESTAMP as the
    reverse-edge sort key (the SRC-fact branch).
    """
    def col(name: str, dtype: str) -> dict:
        return {
            "name": name,
            "stype": None,
            "dtype": dtype,
            "na_values": None,
            "metadata": {},
            "encoder": None,
            "stats": {},
        }

    users = {
        "table_name": "USERS",
        "data_source": None,
        "data_version": None,
        "pkey": {
            "name": "USER_ID",
            "stype": None,
            "dtype": "int",
            "na_values": None,
            "metadata": {},
            "drop_duplicates": False,
        },
        "fkeys": [],
        "cols": [col("AGE", "int")],
        "time_col": None,
        "source_name": None,
        "stats": {},
    }
    stores = {
        "table_name": "STORES",
        "data_source": None,
        "data_version": None,
        "pkey": {
            "name": "STORE_ID",
            "stype": None,
            "dtype": "int",
            "na_values": None,
            "metadata": {},
            "drop_duplicates": False,
        },
        "fkeys": [],
        # Opt-in numerical RHS feature column; default stays primary-key-only.
        "cols": [col("STORE_VALUE", "float")] if with_rhs_feature else [],
        "time_col": None,
        "source_name": None,
        "stats": {},
    }
    orders = {
        "table_name":
        "ORDERS",
        "data_source":
        None,
        "data_version":
        None,
        "pkey": {
            "name": "ORDER_ID",
            "stype": None,
            "dtype": "int",
            "na_values": None,
            "metadata": {},
            "drop_duplicates": False,
        },
        "fkeys": [
            {
                "name": "USER_ID",
                "stype": None,
                "dtype": "int",
                "na_values": None,
                "metadata": {},
                "dst_table": "USERS",
                "dst_key": None,
                "dst_data_source": None,
                "stats": {},
            },
            {
                "name": "STORE_ID",
                "stype": None,
                "dtype": "int",
                "na_values": None,
                "metadata": {},
                "dst_table": "STORES",
                "dst_key": None,
                "dst_data_source": None,
                "stats": {},
            },
        ],
        "cols": [
            col("STORE_ID_NUM", "int"),
            col("ORDER_VALUE_3", "int"),
            col("ORDER_SCORE", "float"),
        ],
        # GraphConfig (consumed by ``add_from_config``) requires ``time_col`` as
        # a column-config dict, not a bare string. ``_build_diskgraph_ingest_config``
        # / ``_time_col_name`` accept either form and emit the DG ``time_column``
        # off the dict, so this drives both the kumo.Graph and the DG index.
        "time_col":
        col("TIMESTAMP", "time"),
        "source_name":
        None,
        "stats": {},
    }
    return {
        "name": "TEMPORAL_LINK_PRED",
        "data_source_dict": {},
        "tables": [users, stores, orders],
    }


def _build_temporal_link_pred_graph(
    tmp_path_factory,
    num_users: int = 5,
    num_stores: int = 3,
    num_edges: int = 15,
    with_rhs_feature: bool = False,
) -> Tuple[Graph, str]:
    """Build + DG-index the TEMPORAL link-pred graph; return ``(graph,
    index_path)`` in the SAME shape as the node-pred fixtures.

    Mirrors the node-pred fixtures' mechanics: write parquet per table, build a
    DG ingest config from a metadata.json-shaped dict, call the wheel's
    ``rust_ingest`` directly, then build the matching in-memory kumo.Graph via
    ``add_from_config`` against the same metadata so the schemas line up 1:1.
    """
    from diskgraph import ingest as rust_ingest

    parquet_dir = tmp_path_factory.mktemp("dg_temporal_link_pred_parquet")
    tables = _temporal_link_pred_tables(num_users, num_stores, num_edges,
                                        with_rhs_feature=with_rhs_feature)
    for name, df in tables.items():
        df.to_parquet(osp.join(str(parquet_dir), f"{name}.parquet"),
                      index=False)

    metadata = _temporal_link_pred_metadata(with_rhs_feature=with_rhs_feature)
    config = _build_diskgraph_ingest_config(str(parquet_dir), metadata)

    index_dir = tmp_path_factory.mktemp("dg_indexed_temporal_link_pred_idx")
    rust_ingest(
        json.dumps(config),
        str(index_dir),
        col_stats=True,
        cat_count_strategy="hybrid",
        success_marker_content=json.dumps(
            {"source": "dg_indexed_temporal_link_pred_graph fixture"}),
    )

    meta_path = osp.join(str(index_dir), "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f)

    graph = Graph(name="TEMPORAL_LINK_PRED")
    graph.connect(ParquetConnector("parquet", str(parquet_dir)))
    graph.add_from_config(meta_path, data_source="parquet")
    return graph, str(index_dir)


@pytest.fixture(scope="session")
def dg_indexed_temporal_link_pred_graph(tmp_path_factory):
    """FACTORY fixture: returns a callable that builds + DG-indexes a TEMPORAL
    link-pred kumo.Graph and yields ``(graph, index_path)`` in the SAME shape
    as :func:`dg_indexed_financial_graph` / the node-pred fixtures.

    Unlike the session-scoped node-pred fixtures (which return the built
    ``(graph, index_path)`` tuple directly), this is a FACTORY so callers can
    pass ``num_users`` / ``num_stores`` / ``num_edges`` for the inline
    weight-col / sizing variants that the parked LP tests need::

        def test_x(dg_indexed_temporal_link_pred_graph, tmp_path):
            graph, index_path = dg_indexed_temporal_link_pred_graph()

    USERS entity + STORES RHS + ORDERS two-fkey temporal edge/fact table
    (real ``TIMESTAMP`` time_col). The RANK-TOP-3 LP query ranks STORES for
    each USER; 15 edges / 5 users x 3 stores keeps targets + DateOffset windows
    non-empty.
    """
    def _factory(
        num_users: int = 5,
        num_stores: int = 3,
        num_edges: int = 15,
        with_rhs_feature: bool = False,
    ) -> Tuple[Graph, str]:
        return _build_temporal_link_pred_graph(
            tmp_path_factory, num_users, num_stores, num_edges,
            with_rhs_feature=with_rhs_feature)

    return _factory


# ---------------------------------------------------------------------------
# DG-indexed TEMPORAL node-pred ORDERS graph (large/small _temporal_graph_fixed)
# ---------------------------------------------------------------------------
#
# Temporal node-pred shape: USERS entity table (AGE) + a STORES RHS table + an
# ORDERS temporal fact table with TWO fkeys (USER_ID -> USERS, STORE_ID ->
# STORES), the numerical node-pred target columns ORDER_VALUE_2 / ORDER_VALUE_3
# / STORE_ID_NUM / ORDER_SCORE, the categorical multilabel-source ORDER_VALUE,
# and a real ``TIMESTAMP`` time_col. This is the sibling node-pred shape
# ``dg_indexed_temporal_link_pred_graph`` (which omits the node-pred target
# columns) does not provide.
#
# Scoping notes:
#   * Dates use a 2023 origin to avoid the pre-2001 ns-DATE clamp on the DG
#     ingest path. Day-granular spacing is identical, so per-split row counts /
#     num_entities / num_targets are preserved; only the asserted calendar dates
#     shift by a fixed offset.
#   * The IMAGE_URL image column is not modelled: it carried no node-pred target
#     or filter semantics, and the unit DG ingest config has no image-encoder
#     slot.


def _temporal_node_pred_orders_tables(
    num_users: int = 10,
    num_stores: int = 8,
    num_edges: int = 560,
    add_na_fkeys: bool = False,
    add_na_timestamps: bool = False,
) -> Dict[str, "object"]:
    """Return ``{table_name: DataFrame}`` for the TEMPORAL node-pred ORDERS
    graph -- a faithful re-build of ``_temporal_graph_fixed`` (test/conftest.py)
    at the requested sizing, carrying the full ORDERS column set.

    Every ORDERS column reproduces ``_temporal_graph_fixed``'s exact value
    formula (so a parked node-pred test rewired onto this fixture keeps the same
    Kumo-owned ENTITY/TARGET sequence and split sizing); ONLY the TIMESTAMP
    origin shifts 1970-01-01 -> 2023-01-01 (see the module note above).

    NA toggles (restore the legacy ``na_values`` axis):

    * ``add_na_fkeys``: append an ORDERS row whose ``USER_ID`` fkey is ``pd.NA``
      -> DG drops it (no entity), ENTITY/TARGET unchanged.
    * ``add_na_timestamps``: append an ORDERS row with ``NaT`` in the declared
      ``TIMESTAMP`` time_col -> expected dropped (no anchor). UNVERIFIED on DG
      ingest (kept as a SEPARATE param id; see
      :func:`_temporal_synthetic_node_pred_tables`).
    """
    import numpy as np
    import pandas as pd

    user_df = pd.DataFrame({
        "USER_ID": np.arange(num_users),
        "AGE": np.arange(num_users) % 20 + 20,
    })
    store_df = pd.DataFrame({"STORE_ID": np.arange(num_stores)})

    length = num_edges - int(num_edges / num_stores) * num_stores
    order_value_length_1 = int(num_edges / 3)
    order_value_length_2 = num_edges - 3 * order_value_length_1
    length_2 = num_edges - int(num_edges / 8) * 4
    order_score_length_1 = int(num_edges / 3)
    order_score_length_2 = num_edges - 3 * order_score_length_1

    order_df = pd.DataFrame({
        "ORDER_ID":
        np.arange(0, num_edges),
        "USER_ID":
        np.repeat(np.arange(0, num_users), int(num_edges / num_users)),
        "STORE_ID":
        np.concatenate(
            (np.tile(np.arange(0, num_stores),
                     int(num_edges / num_stores)), np.arange(0, length))),
        # STORE_ID_NUM mirrors STORE_ID as a numerical column (matches
        # ``_temporal_graph_fixed``); it is the WHERE-filter column some parked
        # node-pred rows predicate on (ORDERS.STORE_ID_NUM < 2).
        "STORE_ID_NUM":
        np.concatenate(
            (np.tile(np.arange(0, num_stores),
                     int(num_edges / num_stores)), np.arange(0, length))),
        # ORDER_VALUE_2 is the NUMERICAL node-pred target (PREDICT
        # ORDERS.ORDER_VALUE_2 / SUM(ORDERS.ORDER_VALUE_2, ...)). Reproduce
        # ``_temporal_graph_fixed``'s [1, 2, 1]-repeated + [2]-tail formula so the
        # num_targets == 2 (values {1, 2}) assertion holds.
        "ORDER_VALUE_2":
        np.concatenate((np.array([1, 2, 1] * order_value_length_1),
                        np.array([2] * order_value_length_2))),
        # ORDER_VALUE is the CATEGORICAL (string) column the LIST_DISTINCT
        # multilabel-classification rows aggregate over.
        "ORDER_VALUE":
        np.concatenate((np.repeat(np.arange(int(num_edges / 8)),
                                  4), np.arange(length_2))),
        # ORDER_VALUE_3 is the numerical WEIGHT column referenced by weight-col
        # variants; reproduce ``_temporal_graph_fixed``'s ORDER_VALUE_3 values.
        "ORDER_VALUE_3":
        np.concatenate((np.repeat(np.arange(int(num_edges / 8)),
                                  4), np.arange(length_2))),
        "ORDER_SCORE":
        np.concatenate((np.array([1, 2, 3] * order_score_length_1),
                        np.array([1] * order_score_length_2))).astype(float),
        # Post-2001 origin (see module note): identical day-granular spacing to
        # the legacy ``pd.to_datetime(arange(n), unit='D')``, shifted to 2023.
        "TIMESTAMP": (pd.Timestamp("2023-01-01") + pd.to_timedelta(
            np.arange(num_edges), unit="D")).astype("datetime64[ns]"),
    })
    if add_na_fkeys or add_na_timestamps:
        extra_rows = []
        next_id = int(num_edges)
        base = {
            "STORE_ID": 0,
            "STORE_ID_NUM": 0,
            "ORDER_VALUE_2": 1,
            "ORDER_VALUE": "0",
            "ORDER_VALUE_3": 0,
            "ORDER_SCORE": 99.0,
        }
        if add_na_fkeys:
            # NA fkey: dropped, no new ENTITY appears.
            extra_rows.append({
                "ORDER_ID": next_id,
                "USER_ID": pd.NA,
                "TIMESTAMP": pd.Timestamp("2023-01-02"),
                **base,
            })
            next_id += 1
        if add_na_timestamps:
            # NaT anchor on an existing entity: expected dropped (no timeframe).
            # UNVERIFIED on DG ingest.
            extra_rows.append({
                "ORDER_ID": next_id,
                "USER_ID": 0,
                "TIMESTAMP": pd.NaT,
                **base,
            })
            next_id += 1
        order_df = pd.concat([order_df, pd.DataFrame(extra_rows)],
                             ignore_index=True)
        order_df["ORDER_ID"] = order_df["ORDER_ID"].astype("int64")
        # nullable Int64 so pd.NA survives the parquet round-trip as a null key.
        order_df["USER_ID"] = order_df["USER_ID"].astype("Int64")
        order_df["STORE_ID"] = order_df["STORE_ID"].astype("int64")
        order_df["STORE_ID_NUM"] = order_df["STORE_ID_NUM"].astype("int64")
        order_df["ORDER_VALUE_2"] = order_df["ORDER_VALUE_2"].astype("int64")
        order_df["ORDER_VALUE"] = order_df["ORDER_VALUE"].astype(str)
        order_df["ORDER_VALUE_3"] = order_df["ORDER_VALUE_3"].astype("int64")
        order_df["ORDER_SCORE"] = order_df["ORDER_SCORE"].astype(float)
        order_df["TIMESTAMP"] = pd.to_datetime(order_df["TIMESTAMP"])
    return {"USERS": user_df, "STORES": store_df, "ORDERS": order_df}


def _temporal_node_pred_orders_metadata() -> dict:
    """metadata.json-shaped dict for the TEMPORAL node-pred ORDERS graph.

    Same shape the other fixtures use so it drives BOTH the DG ingest config
    (:func:`_build_diskgraph_ingest_config`) and the in-memory kumo.Graph
    (``graph.add_from_config``). ORDERS declares TWO fkeys (USER_ID -> USERS,
    STORE_ID -> STORES) and ``time_col=TIMESTAMP``; the DG config builder emits
    one reverse-edge block per fkey, both carrying ORDERS' own TIMESTAMP as the
    reverse-edge sort key (the SRC-fact branch).
    """
    def col(name: str, dtype: str) -> dict:
        return {
            "name": name,
            "stype": None,
            "dtype": dtype,
            "na_values": None,
            "metadata": {},
            "encoder": None,
            "stats": {},
        }

    users = {
        "table_name": "USERS",
        "data_source": None,
        "data_version": None,
        "pkey": {
            "name": "USER_ID",
            "stype": None,
            "dtype": "int",
            "na_values": None,
            "metadata": {},
            "drop_duplicates": False,
        },
        "fkeys": [],
        "cols": [col("AGE", "int")],
        "time_col": None,
        "source_name": None,
        "stats": {},
    }
    stores = {
        "table_name": "STORES",
        "data_source": None,
        "data_version": None,
        "pkey": {
            "name": "STORE_ID",
            "stype": None,
            "dtype": "int",
            "na_values": None,
            "metadata": {},
            "drop_duplicates": False,
        },
        "fkeys": [],
        "cols": [],
        "time_col": None,
        "source_name": None,
        "stats": {},
    }
    orders = {
        "table_name":
        "ORDERS",
        "data_source":
        None,
        "data_version":
        None,
        "pkey": {
            "name": "ORDER_ID",
            "stype": None,
            "dtype": "int",
            "na_values": None,
            "metadata": {},
            "drop_duplicates": False,
        },
        "fkeys": [
            {
                "name": "USER_ID",
                "stype": None,
                "dtype": "int",
                "na_values": None,
                "metadata": {},
                "dst_table": "USERS",
                "dst_key": None,
                "dst_data_source": None,
                "stats": {},
            },
            {
                "name": "STORE_ID",
                "stype": None,
                "dtype": "int",
                "na_values": None,
                "metadata": {},
                "dst_table": "STORES",
                "dst_key": None,
                "dst_data_source": None,
                "stats": {},
            },
        ],
        "cols": [
            col("STORE_ID_NUM", "int"),
            col("ORDER_VALUE_2", "int"),
            col("ORDER_VALUE", "string"),
            col("ORDER_VALUE_3", "int"),
            col("ORDER_SCORE", "float"),
        ],
        # GraphConfig requires ``time_col`` as a column-config dict;
        # ``_build_diskgraph_ingest_config`` / ``_time_col_name`` accept either
        # form and emit the DG ``time_column`` off the dict.
        "time_col":
        col("TIMESTAMP", "time"),
        "source_name":
        None,
        "stats": {},
    }
    return {
        "name": "TEMPORAL_NODE_PRED_ORDERS",
        "data_source_dict": {},
        "tables": [users, stores, orders],
    }


def _build_temporal_node_pred_orders_graph(
    tmp_path_factory,
    num_users: int = 10,
    num_stores: int = 8,
    num_edges: int = 560,
    add_na_fkeys: bool = False,
    add_na_timestamps: bool = False,
) -> Tuple[Graph, str]:
    """Build + DG-index the TEMPORAL node-pred ORDERS graph; return
    ``(graph, index_path)`` in the SAME shape as the other DG fixtures.

    Mirrors the node-pred / link-pred fixtures' mechanics: write parquet per
    table, build a DG ingest config from a metadata.json-shaped dict, call the
    wheel's ``rust_ingest`` directly, then build the matching in-memory
    kumo.Graph via ``add_from_config`` against the same metadata.

    ``add_na_fkeys`` / ``add_na_timestamps`` thread through to
    :func:`_temporal_node_pred_orders_tables` (the legacy ``na_values`` axis).
    """
    from diskgraph import ingest as rust_ingest

    parquet_dir = tmp_path_factory.mktemp(
        "dg_temporal_node_pred_orders_parquet")
    tables = _temporal_node_pred_orders_tables(
        num_users, num_stores, num_edges, add_na_fkeys=add_na_fkeys,
        add_na_timestamps=add_na_timestamps)
    for name, df in tables.items():
        df.to_parquet(osp.join(str(parquet_dir), f"{name}.parquet"),
                      index=False)

    metadata = _temporal_node_pred_orders_metadata()
    config = _build_diskgraph_ingest_config(str(parquet_dir), metadata)

    index_dir = tmp_path_factory.mktemp("dg_indexed_temporal_node_pred_orders")
    rust_ingest(
        json.dumps(config),
        str(index_dir),
        col_stats=True,
        cat_count_strategy="hybrid",
        success_marker_content=json.dumps(
            {"source": "dg_indexed_temporal_node_pred_orders_graph fixture"}),
    )

    meta_path = osp.join(str(index_dir), "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f)

    graph = Graph(name="TEMPORAL_NODE_PRED_ORDERS")
    graph.connect(ParquetConnector("parquet", str(parquet_dir)))
    graph.add_from_config(meta_path, data_source="parquet")
    return graph, str(index_dir)


@pytest.fixture(scope="session")
def dg_indexed_temporal_node_pred_orders_graph(tmp_path_factory):
    """FACTORY fixture: returns a callable that builds + DG-indexes a TEMPORAL
    node-pred kumo.Graph and yields ``(graph, index_path)`` in the SAME shape as
    the other ``dg_indexed_*`` fixtures.

    A faithful DG-native re-build of the legacy ``_temporal_graph_fixed``
    conftest factory; the callable accepts ``num_users`` / ``num_stores`` /
    ``num_edges`` so callers can pick the ``small_temporal_graph_fixed``
    (5/3/15) or ``large_temporal_graph_fixed`` (10/8/560) sizing the parked
    node-pred tests need::

        def test_x(dg_indexed_temporal_node_pred_orders_graph, tmp_path):
            graph, index_path = dg_indexed_temporal_node_pred_orders_graph(
                num_users=10, num_stores=8, num_edges=560)

    USERS entity (AGE) + STORES RHS + ORDERS two-fkey temporal fact table with
    the full node-pred column set (ORDER_VALUE_2 numeric target, ORDER_VALUE
    categorical multilabel source, STORE_ID_NUM / ORDER_VALUE_3 / ORDER_SCORE,
    real ``TIMESTAMP`` time_col).
    """
    def _factory(
        num_users: int = 10,
        num_stores: int = 8,
        num_edges: int = 560,
        add_na_fkeys: bool = False,
        add_na_timestamps: bool = False,
    ) -> Tuple[Graph, str]:
        return _build_temporal_node_pred_orders_graph(
            tmp_path_factory, num_users, num_stores, num_edges,
            add_na_fkeys=add_na_fkeys, add_na_timestamps=add_na_timestamps)

    return _factory


# ---------------------------------------------------------------------------
# DG-indexed DENSE-SMALL temporal node-pred graph (dense_small_temporal_graph_fixed)
# ---------------------------------------------------------------------------
#
# Dense-small temporal node-pred shape: USERS entity table (AGE) + a STORES RHS
# table + an ORDERS temporal fact table with a sparser edge rule and column set
# than the ORDERS node-pred fixture above:
#
#   * Edges iterate ``user_id x t(0..11) x store_id``, keeping only rows where
#     ``(user_id + t + store_id) % 2 == 0`` (12 timestamps per user). The default
#     5u/3s sizing yields exactly 90 ORDERS rows.
#   * The categorical target column ``ORDER_VALUE`` cycles through
#     ``value_0 .. value_4`` via ``f"value_{(user_id + t + store_id) % 5}"``; the
#     multilabel-ranking test asserts a per-label ``target_distribution`` over
#     these labels.
#   * No numerical ORDER_VALUE_2 / ORDER_VALUE_3 / ORDER_SCORE target column --
#     consumed only by the ``LIST_DISTINCT(ORDERS.ORDER_VALUE, 0, 2) RANK TOP 2``
#     multilabel-ranking query.
#
# Scoping notes:
#   * Dates use a 2023 origin to avoid the pre-2001 ns-DATE clamp on the DG
#     ingest path. Day-granular spacing is identical, so per-timeframe row counts
#     / num_entities / num_targets are preserved; only the asserted calendar
#     dates shift by a fixed offset.
#   * The ``add_na_pkeys`` / ``add_na_timestamps`` toggles are not modelled
#     (NA-pkey/timestamp DG ingest is a separate follow-up).


def _dense_small_temporal_node_pred_tables(
    num_users: int = 5,
    num_stores: int = 3,
) -> Dict[str, "object"]:
    """Return ``{table_name: DataFrame}`` for the DENSE-SMALL temporal node-pred
    graph at the requested sizing.

    The ORDERS edge set is generated by the deterministic sparse rule
    (``(user_id + t + store_id) % 2 == 0`` over 12 timestamps per user) and the
    ``ORDER_VALUE`` categorical cycles ``value_0 .. value_4``; the TIMESTAMP
    origin is 2023-01-01 (see the module note above).
    """
    import numpy as np
    import pandas as pd

    user_df = pd.DataFrame({
        "USER_ID": np.arange(num_users),
        "AGE": np.arange(num_users) % 20 + 20,
    })
    store_df = pd.DataFrame({"STORE_ID": np.arange(num_stores)})

    user_ids = []
    store_ids = []
    store_id_nums = []
    order_values = []
    timestamps = []
    for user_id in range(num_users):
        for t in range(12):  # 12 timestamps per user (legacy)
            for store_id in range(num_stores):
                if (user_id + t + store_id) % 2 == 0:
                    user_ids.append(user_id)
                    store_ids.append(store_id)
                    store_id_nums.append(store_id)
                    order_values.append(f"value_{(user_id + t + store_id) % 5}")
                    # Post-2001 origin (see module note): identical day-granular
                    # spacing to the legacy ``pd.to_datetime(t, unit='D')``,
                    # shifted to 2023.
                    timestamps.append(
                        pd.Timestamp("2023-01-01") +
                        pd.to_timedelta(t, unit="D"))

    order_df = pd.DataFrame({
        "ORDER_ID":
        np.arange(len(user_ids)),
        "USER_ID":
        user_ids,
        "STORE_ID":
        store_ids,
        "STORE_ID_NUM":
        store_id_nums,
        "ORDER_VALUE":
        order_values,
        "TIMESTAMP":
        pd.Series(timestamps).astype("datetime64[ns]"),
    })
    return {"USERS": user_df, "STORES": store_df, "ORDERS": order_df}


def _dense_small_temporal_node_pred_metadata() -> dict:
    """metadata.json-shaped dict for the DENSE-SMALL temporal node-pred graph.

    USERS entity (AGE) + STORES RHS + ORDERS two-fkey temporal fact table
    (USER_ID -> USERS, STORE_ID -> STORES) carrying STORE_ID_NUM (numerical),
    ORDER_VALUE (categorical multilabel source) and a ``time_col=TIMESTAMP``.
    Same column-config shape the other DG fixtures use so it drives BOTH the DG
    ingest config and the in-memory kumo.Graph (``add_from_config``).
    """
    def col(name: str, dtype: str) -> dict:
        return {
            "name": name,
            "stype": None,
            "dtype": dtype,
            "na_values": None,
            "metadata": {},
            "encoder": None,
            "stats": {},
        }

    users = {
        "table_name": "USERS",
        "data_source": None,
        "data_version": None,
        "pkey": {
            "name": "USER_ID",
            "stype": None,
            "dtype": "int",
            "na_values": None,
            "metadata": {},
            "drop_duplicates": False,
        },
        "fkeys": [],
        "cols": [col("AGE", "int")],
        "time_col": None,
        "source_name": None,
        "stats": {},
    }
    stores = {
        "table_name": "STORES",
        "data_source": None,
        "data_version": None,
        "pkey": {
            "name": "STORE_ID",
            "stype": None,
            "dtype": "int",
            "na_values": None,
            "metadata": {},
            "drop_duplicates": False,
        },
        "fkeys": [],
        "cols": [],
        "time_col": None,
        "source_name": None,
        "stats": {},
    }
    orders = {
        "table_name":
        "ORDERS",
        "data_source":
        None,
        "data_version":
        None,
        "pkey": {
            "name": "ORDER_ID",
            "stype": None,
            "dtype": "int",
            "na_values": None,
            "metadata": {},
            "drop_duplicates": False,
        },
        "fkeys": [
            {
                "name": "USER_ID",
                "stype": None,
                "dtype": "int",
                "na_values": None,
                "metadata": {},
                "dst_table": "USERS",
                "dst_key": None,
                "dst_data_source": None,
                "stats": {},
            },
            {
                "name": "STORE_ID",
                "stype": None,
                "dtype": "int",
                "na_values": None,
                "metadata": {},
                "dst_table": "STORES",
                "dst_key": None,
                "dst_data_source": None,
                "stats": {},
            },
        ],
        "cols": [
            col("STORE_ID_NUM", "int"),
            col("ORDER_VALUE", "string"),
        ],
        "time_col":
        col("TIMESTAMP", "time"),
        "source_name":
        None,
        "stats": {},
    }
    return {
        "name": "DENSE_SMALL_TEMPORAL_NODE_PRED",
        "data_source_dict": {},
        "tables": [users, stores, orders],
    }


def _build_dense_small_temporal_node_pred_graph(
    tmp_path_factory,
    num_users: int = 5,
    num_stores: int = 3,
) -> Tuple[Graph, str]:
    """Build + DG-index the DENSE-SMALL temporal node-pred graph; return
    ``(graph, index_path)`` in the SAME shape as the other DG fixtures.

    Mirrors the node-pred fixtures' mechanics: write parquet per table, build a
    DG ingest config from a metadata.json-shaped dict, call the wheel's
    ``rust_ingest`` directly, then build the matching in-memory kumo.Graph via
    ``add_from_config`` against the same metadata.
    """
    from diskgraph import ingest as rust_ingest

    parquet_dir = tmp_path_factory.mktemp("dg_dense_small_node_pred_parquet")
    tables = _dense_small_temporal_node_pred_tables(num_users, num_stores)
    for name, df in tables.items():
        df.to_parquet(osp.join(str(parquet_dir), f"{name}.parquet"),
                      index=False)

    metadata = _dense_small_temporal_node_pred_metadata()
    config = _build_diskgraph_ingest_config(str(parquet_dir), metadata)

    index_dir = tmp_path_factory.mktemp("dg_indexed_dense_small_node_pred")
    rust_ingest(
        json.dumps(config),
        str(index_dir),
        col_stats=True,
        cat_count_strategy="hybrid",
        success_marker_content=json.dumps({
            "source":
            "dg_indexed_dense_small_temporal_node_pred_graph "
            "fixture"
        }),
    )

    meta_path = osp.join(str(index_dir), "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f)

    graph = Graph(name="DENSE_SMALL_TEMPORAL_NODE_PRED")
    graph.connect(ParquetConnector("parquet", str(parquet_dir)))
    graph.add_from_config(meta_path, data_source="parquet")
    return graph, str(index_dir)


@pytest.fixture(scope="session")
def dg_indexed_dense_small_temporal_node_pred_graph(tmp_path_factory):
    """FACTORY fixture: returns a callable that builds + DG-indexes the
    DENSE-SMALL temporal node-pred kumo.Graph and yields ``(graph, index_path)``
    in the SAME shape as the other ``dg_indexed_*`` fixtures.

    A faithful DG-native re-build of the legacy
    ``_dense_small_temporal_graph_fixed`` conftest factory; the callable accepts
    ``num_users`` / ``num_stores`` so callers can pick the legacy default
    (5u/3s -> 90 ORDERS rows over 12 timestamps each)::

        def test_x(dg_indexed_dense_small_temporal_node_pred_graph, tmp_path):
            graph, index_path = dg_indexed_dense_small_temporal_node_pred_graph(
                num_users=5, num_stores=3)

    USERS entity (AGE) + STORES RHS + ORDERS two-fkey temporal fact table with
    the dense_small column set (STORE_ID_NUM numeric, ORDER_VALUE ``value_N``
    categorical multilabel source, real ``TIMESTAMP`` time_col); the sparse
    ``(user+t+store) % 2 == 0`` edge rule reproduces the legacy table verbatim.
    """
    def _factory(
        num_users: int = 5,
        num_stores: int = 3,
    ) -> Tuple[Graph, str]:
        return _build_dense_small_temporal_node_pred_graph(
            tmp_path_factory, num_users, num_stores)

    return _factory


# ---------------------------------------------------------------------------
# DG-indexed entity-dates graph (custom-pred-time + forecasting bp)
# ---------------------------------------------------------------------------
#
# USERS entity table carrying its own start-time column (JOINED), a PRODUCTS RHS
# table (CREATED start-time) and a TRANS temporal fact table (DATE time_col, one
# $1 transaction per day over a 30-day window x 3 users). Drives the
# custom-prediction-time node-pred and the forecasting (autoregression /
# lag-timestep / YOY) batch-prediction rows.
#
# Scoping notes:
#   * Dates use a 2023 origin; any pre-2001 ns timestamp clamps to the
#     Timestamp.max 2262 sentinel through the DG ingest path. Day-granular
#     spacing is preserved.
#   * END-TIME columns are NOT modelled: the unit DG ingest config emits only
#     `time_column` and has no end-time slot, so only START-time toggles are
#     exposed and end-time-dependent rows stay parked.


def _entity_dates_tables(
    set_user_time: bool = True,
    set_product_time: bool = False,
    add_year_ago_data: bool = False,
) -> Dict[str, "object"]:
    """Return ``{table_name: DataFrame}`` for the entity-dates graph.

    Reproduces the legacy ``_graph_with_entity_dates`` column values at a 2023
    origin (no end-time columns -- see the module note above):

      * USERS:    USER_ID range(3), AGE [17, 25, 40], JOINED start-dates.
      * PRODUCTS: PRODUCT_ID range(5, 8), BRAND, CREATED start-dates.
      * TRANS:    one $1.0 transaction per day for 30 days x 3 users, fkeys
                  USER_ID -> USERS and PRODUCT_ID -> PRODUCTS (cycled 5/6/7),
                  DATE time_col. ``add_year_ago_data`` duplicates every row
                  364 days earlier (the YOY-labels path).

    The DataFrames are time-column agnostic (they always carry the date
    columns); whether USERS/PRODUCTS are treated as temporal is decided by the
    metadata's ``time_col`` slots, toggled by ``set_user_time`` /
    ``set_product_time`` in :func:`_entity_dates_metadata`.
    """
    import pandas as pd

    entity_df = pd.DataFrame({
        "USER_ID":
        range(3),
        "AGE": [17, 25, 40],
        # Legacy JOINED: 1/1/2016, 1/21/2018, 1/1/2018 -> shifted to 2023 so
        # USER 1 joins mid-window (2023-01-21) and USERS 0/2 at the window
        # origin (2023-01-01). AGE < 30 keeps USERS {0, 1}.
        "JOINED":
        pd.to_datetime(["2023-01-01", "2023-01-21", "2023-01-01"]),
    })
    product_df = pd.DataFrame({
        "PRODUCT_ID":
        range(5, 8),
        "BRAND": ["AA", "BB", "CC"],
        "CREATED":
        pd.to_datetime(["2023-01-01", "2023-01-10", "2023-01-01"]),
    })
    dates = pd.Series(pd.date_range(start="2023-01-01", end="2023-01-30"))
    dates = pd.concat([dates, dates, dates], axis=0, ignore_index=True)
    target_df = pd.DataFrame({
        "TRANS_ID": range(len(dates)),
        "USER_ID": [0] * 30 + [1] * 30 + [2] * 30,
        "AMOUNT": [1.0] * len(dates),
        "PRODUCT_ID": [5, 6, 7] * 30,
        "DATE": dates,
    })
    if add_year_ago_data:
        year_ago = target_df.copy()
        year_ago["DATE"] = year_ago["DATE"] - pd.Timedelta(days=364)
        year_ago["TRANS_ID"] = year_ago["TRANS_ID"] + len(dates)
        target_df = pd.concat([target_df, year_ago], axis=0, ignore_index=True)
    return {"USERS": entity_df, "PRODUCTS": product_df, "TRANS": target_df}


def _entity_dates_metadata(
    set_user_time: bool = True,
    set_product_time: bool = False,
) -> dict:
    """metadata.json-shaped dict for the entity-dates graph.

    Same shape the other DG fixtures use so it drives BOTH the DG ingest config
    (:func:`_build_diskgraph_ingest_config`) and the in-memory kumo.Graph
    (``graph.add_from_config``). TRANS always declares ``time_col=DATE`` and the
    two fkeys; USERS/PRODUCTS carry a ``time_col`` only when the corresponding
    flag is set (mirroring the legacy ``with_start_date_users`` /
    ``with_start_date_products`` toggles). No end-time slot exists -- see the
    module note.
    """
    def col(name: str, dtype: str) -> dict:
        return {
            "name": name,
            "stype": None,
            "dtype": dtype,
            "na_values": None,
            "metadata": {},
            "encoder": None,
            "stats": {},
        }

    users = {
        "table_name": "USERS",
        "data_source": None,
        "data_version": None,
        "pkey": {
            "name": "USER_ID",
            "stype": None,
            "dtype": "int",
            "na_values": None,
            "metadata": {},
            "drop_duplicates": False,
        },
        "fkeys": [],
        "cols": [col("AGE", "int")],
        "time_col": col("JOINED", "time") if set_user_time else None,
        "source_name": None,
        "stats": {},
    }
    products = {
        "table_name": "PRODUCTS",
        "data_source": None,
        "data_version": None,
        "pkey": {
            "name": "PRODUCT_ID",
            "stype": None,
            "dtype": "int",
            "na_values": None,
            "metadata": {},
            "drop_duplicates": False,
        },
        "fkeys": [],
        "cols": [col("BRAND", "string")],
        "time_col": col("CREATED", "time") if set_product_time else None,
        "source_name": None,
        "stats": {},
    }
    trans = {
        "table_name":
        "TRANS",
        "data_source":
        None,
        "data_version":
        None,
        "pkey": {
            "name": "TRANS_ID",
            "stype": None,
            "dtype": "int",
            "na_values": None,
            "metadata": {},
            "drop_duplicates": False,
        },
        "fkeys": [
            {
                "name": "USER_ID",
                "stype": None,
                "dtype": "int",
                "na_values": None,
                "metadata": {},
                "dst_table": "USERS",
                "dst_key": None,
                "dst_data_source": None,
                "stats": {},
            },
            {
                "name": "PRODUCT_ID",
                "stype": None,
                "dtype": "int",
                "na_values": None,
                "metadata": {},
                "dst_table": "PRODUCTS",
                "dst_key": None,
                "dst_data_source": None,
                "stats": {},
            },
        ],
        "cols": [col("AMOUNT", "float")],
        "time_col":
        col("DATE", "time"),
        "source_name":
        None,
        "stats": {},
    }
    return {
        "name": "ENTITY_DATES",
        "data_source_dict": {},
        "tables": [users, products, trans],
    }


def _build_entity_dates_graph(
    tmp_path_factory,
    set_user_time: bool = True,
    set_product_time: bool = False,
    add_year_ago_data: bool = False,
) -> Tuple[Graph, str]:
    """Build + DG-index the entity-dates graph; return ``(graph, index_path)``
    in the SAME shape as the other DG fixtures.

    Mirrors the node-pred / link-pred fixtures' mechanics: write parquet per
    table, build a DG ingest config from a metadata.json-shaped dict, call the
    wheel's ``rust_ingest`` directly, then build the matching in-memory
    kumo.Graph via ``add_from_config`` against the same metadata so the schemas
    line up 1:1.
    """
    from diskgraph import ingest as rust_ingest

    parquet_dir = tmp_path_factory.mktemp("dg_entity_dates_parquet")
    tables = _entity_dates_tables(
        set_user_time=set_user_time,
        set_product_time=set_product_time,
        add_year_ago_data=add_year_ago_data,
    )
    for name, df in tables.items():
        df.to_parquet(osp.join(str(parquet_dir), f"{name}.parquet"),
                      index=False)

    metadata = _entity_dates_metadata(set_user_time=set_user_time,
                                      set_product_time=set_product_time)
    config = _build_diskgraph_ingest_config(str(parquet_dir), metadata)

    index_dir = tmp_path_factory.mktemp("dg_indexed_entity_dates_idx")
    rust_ingest(
        json.dumps(config),
        str(index_dir),
        col_stats=True,
        cat_count_strategy="hybrid",
        success_marker_content=json.dumps(
            {"source": "dg_indexed_entity_dates_graph fixture"}),
    )

    meta_path = osp.join(str(index_dir), "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f)

    graph = Graph(name="ENTITY_DATES")
    graph.connect(ParquetConnector("parquet", str(parquet_dir)))
    graph.add_from_config(meta_path, data_source="parquet")
    return graph, str(index_dir)


@pytest.fixture(scope="session")
def dg_indexed_entity_dates_graph(tmp_path_factory):
    """FACTORY fixture: returns a callable that builds + DG-indexes the
    entity-dates kumo.Graph and yields ``(graph, index_path)`` in the SAME
    shape as the other ``dg_indexed_*`` fixtures.

    Entity-dates graph used by the custom-prediction-time node-pred +
    forecasting batch-prediction rows. A FACTORY (not a built tuple) so callers
    can toggle the start-time slots and the year-ago duplication those rows
    need::

        def test_x(dg_indexed_entity_dates_graph, tmp_path):
            graph, index_path = dg_indexed_entity_dates_graph(
                set_user_time=True)

    USERS entity (own JOINED start-time) + PRODUCTS RHS (CREATED start-time)
    + TRANS two-fkey temporal fact table ($1/day x 30 days x 3 users, DATE
    time_col). 2023 origin (post-2001 ns landmine). NO end-time columns -- the
    unit DG ingest path has no end-time slot, so end-time-dependent rows
    (the custom-pred-time LINK-pred variants) stay parked; see the module note.
    """
    def _factory(
        set_user_time: bool = True,
        set_product_time: bool = False,
        add_year_ago_data: bool = False,
    ) -> Tuple[Graph, str]:
        return _build_entity_dates_graph(
            tmp_path_factory,
            set_user_time=set_user_time,
            set_product_time=set_product_time,
            add_year_ago_data=add_year_ago_data,
        )

    return _factory


# ---------------------------------------------------------------------------
# TEMPORAL FINANCIAL graph (ACCOUNT entity + dated TRANS fact) + DiskGraph index
# ---------------------------------------------------------------------------
#
# Why an ADDITIVE temporal sibling of dg_indexed_financial_graph: the session
# ``dg_indexed_financial_graph`` above is the canonical STATIC FINANCIAL fixture
# (FINANCIAL base tables + the static TRAINING_CLASSIFICATION node-pred target).
# It deliberately stays as-is; this block adds NOTHING to it (the existing
# fixture's tables / metadata / build path / return contract are untouched and
# byte-identical). A handful of parked FINANCIAL TEMPORAL tests
# (``test_train_generator_basic.py`` / ``test_rfm_parity.py``) drive a
# ``SUM(TRANS.AMOUNT, w0, w1)`` window query + a ``DateOffsetSplit`` over
# ACCOUNT/TRANS with ``TRANS.set_time_col('DATE')``. The shipped FINANCIAL data
# CANNOT back those: the parquet ``TRANS.DATE`` is a single degenerate value
# (``19930000`` reinterpreted as ns -> 1970-01-01T00:00:00.019930000,
# ``nunique == 1``), so a temporal pquery over it yields an empty train table
# (see the ``[fixture]`` triage note: "The FINANCIAL fixture data has NO usable
# temporal anchor").
#
# This fixture re-derives a FINANCIAL-shaped ACCOUNT + TRANS graph WITH a real
# datetime ``TRANS.DATE`` anchor by reading the source ``TRANS.csv`` /
# ``ACCOUNT.csv`` (whose ``DATE`` is the int ``YYYYMMDD`` column with 181
# distinct 1993 days), parsing it to a timestamp and shifting the origin
# 1993 -> 2023. The 1993 -> 2023 shift dodges the pre-2001 ns-DATE producer
# landmine the other DG fixtures pin (1970-era ns timestamps clamp to the
# Timestamp.max 2262 sentinel through the DG ingest path; a 2023 origin flows
# correctly). Day-granular spacing is preserved (+30 calendar years), so the
# temporal STRUCTURE is unchanged; only the absolute origin moves.
# Consequently, absolute-date / pinned-count assertions must be RECOMPUTED from
# real DG output, not copied from any pre-shift literals.
#
# Logical graph (FINANCIAL ACCOUNT/TRANS shape):
#   ACCOUNT  pkey=ACCOUNT_ID, cols DISTRICT_ID / FREQUENCY, NO time_col (static
#            entity, like the real FINANCIAL ACCOUNT table)
#   TRANS    pkey=TRANS_ID, fkey ACCOUNT_ID -> ACCOUNT, cols TYPE / OPERATION /
#            AMOUNT, time_col=DATE (the temporal anchor this fixture adds)
# Representative query (NOT run in the smoke test):
#   PREDICT SUM(TRANS.AMOUNT, 15, 30) FOR EACH ACCOUNT.ACCOUNT_ID

# ACCOUNT_ID ceiling kept small so the fixture is unit-scale (mirrors the
# legacy ``ACCOUNT.ACCOUNT_ID < 100`` entity filter); the real FINANCIAL CSV
# carries ~4500 accounts / 5000 transactions.
_TEMPORAL_FINANCIAL_MAX_ACCOUNT_ID = 100

# Year delta applied to the source 1993 dates so the anchor lands post-2001.
_TEMPORAL_FINANCIAL_YEAR_SHIFT = 30


def _temporal_financial_tables() -> Dict[str, "object"]:
    """Return ``{table_name: DataFrame}`` for the TEMPORAL FINANCIAL graph.

    Reads the source ``ACCOUNT.csv`` / ``TRANS.csv`` (semicolon-delimited),
    restricts to ``ACCOUNT_ID < _TEMPORAL_FINANCIAL_MAX_ACCOUNT_ID`` (unit
    scale), parses the int ``YYYYMMDD`` ``TRANS.DATE`` into a real
    ``datetime64[ns]`` column and shifts the origin
    1993 -> ``1993 + _TEMPORAL_FINANCIAL_YEAR_SHIFT`` so it clears the pre-2001
    ns-DATE producer landmine. ACCOUNT is a STATIC entity table (no time_col);
    TRANS carries the ``DATE`` time_col. Only the columns the parked temporal
    tests reference are kept (ACCOUNT: DISTRICT_ID / FREQUENCY; TRANS: TYPE /
    OPERATION / AMOUNT).
    """
    import pandas as pd

    csv_dir = financial_csv_dir()
    account = pd.read_csv(osp.join(csv_dir, "ACCOUNT.csv"), sep=";")
    trans = pd.read_csv(osp.join(csv_dir, "TRANS.csv"), sep=";")

    account = account[account["ACCOUNT_ID"] <
                      _TEMPORAL_FINANCIAL_MAX_ACCOUNT_ID].copy()
    trans = trans[trans["ACCOUNT_ID"].isin(account["ACCOUNT_ID"])].copy()

    # int YYYYMMDD -> datetime, shifted to a post-2001 origin (day spacing
    # preserved). The source DATE spans 1993-01-01 .. 1993-06-30 (181 days).
    parsed = pd.to_datetime(trans["DATE"].astype(str), format="%Y%m%d")
    trans["DATE"] = (parsed + pd.DateOffset(
        years=_TEMPORAL_FINANCIAL_YEAR_SHIFT)).astype("datetime64[ns]")

    account_df = account[["ACCOUNT_ID", "DISTRICT_ID", "FREQUENCY"]]
    account_df = account_df.reset_index(drop=True)
    trans_df = trans[[
        "TRANS_ID", "ACCOUNT_ID", "TYPE", "OPERATION", "AMOUNT", "DATE"
    ]].reset_index(drop=True)
    return {"ACCOUNT": account_df, "TRANS": trans_df}


def _temporal_financial_metadata() -> dict:
    """metadata.json-shaped dict for the TEMPORAL FINANCIAL graph.

    Same shape every fixture in this file uses so it drives BOTH the DG ingest
    config (:func:`_build_diskgraph_ingest_config`) and the in-memory kumo.Graph
    (``graph.add_from_config``). ACCOUNT is static (no time_col); TRANS declares
    ``time_col=DATE`` (a column-config dict, the GraphConfig-native shape) so DG
    ingest sets a ``time_column`` on the fact table block and the temporal
    sampler can resolve anchor dates.
    """
    def col(name: str, dtype: str) -> dict:
        return {
            "name": name,
            "stype": None,
            "dtype": dtype,
            "na_values": None,
            "metadata": {},
            "encoder": None,
            "stats": {},
        }

    account = {
        "table_name": "ACCOUNT",
        "data_source": None,
        "data_version": None,
        "pkey": {
            "name": "ACCOUNT_ID",
            "stype": None,
            "dtype": "int",
            "na_values": None,
            "metadata": {},
            "drop_duplicates": False,
        },
        "fkeys": [],
        "cols": [col("DISTRICT_ID", "int"),
                 col("FREQUENCY", "string")],
        "time_col": None,
        "source_name": None,
        "stats": {},
    }
    trans = {
        "table_name":
        "TRANS",
        "data_source":
        None,
        "data_version":
        None,
        "pkey": {
            "name": "TRANS_ID",
            "stype": None,
            "dtype": "int",
            "na_values": None,
            "metadata": {},
            "drop_duplicates": False,
        },
        "fkeys": [{
            "name": "ACCOUNT_ID",
            "stype": None,
            "dtype": "int",
            "na_values": None,
            "metadata": {},
            "dst_table": "ACCOUNT",
            "dst_key": None,
            "dst_data_source": None,
            "stats": {},
        }],
        "cols": [
            col("TYPE", "string"),
            col("OPERATION", "string"),
            col("AMOUNT", "float"),
        ],
        "time_col":
        col("DATE", "time"),
        "source_name":
        None,
        "stats": {},
    }
    return {
        "name": "FINANCIAL_TEMPORAL",
        "data_source_dict": {},
        "tables": [account, trans],
    }


def _build_temporal_financial_graph(tmp_path_factory, ) -> Tuple[Graph, str]:
    """Build + DG-index the TEMPORAL FINANCIAL graph; return
    ``(graph, index_path)`` in the SAME shape as every other fixture here.

    Mirrors the node-pred / entity-dates fixtures' mechanics: write parquet per
    table, build a DG ingest config from a metadata.json-shaped dict, call the
    wheel's ``rust_ingest`` directly, then build the matching in-memory
    kumo.Graph via ``add_from_config`` against the same metadata so the schemas
    line up 1:1.
    """
    from diskgraph import ingest as rust_ingest

    parquet_dir = tmp_path_factory.mktemp("dg_temporal_financial_parquet")
    tables = _temporal_financial_tables()
    for name, df in tables.items():
        df.to_parquet(osp.join(str(parquet_dir), f"{name}.parquet"),
                      index=False)

    metadata = _temporal_financial_metadata()
    config = _build_diskgraph_ingest_config(str(parquet_dir), metadata)

    index_dir = tmp_path_factory.mktemp("dg_indexed_temporal_financial_idx")
    rust_ingest(
        json.dumps(config),
        str(index_dir),
        col_stats=True,
        cat_count_strategy="hybrid",
        success_marker_content=json.dumps(
            {"source": "dg_indexed_temporal_financial_graph fixture"}),
    )

    meta_path = osp.join(str(index_dir), "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f)

    graph = Graph(name="FINANCIAL_TEMPORAL")
    graph.connect(ParquetConnector("parquet", str(parquet_dir)))
    graph.add_from_config(meta_path, data_source="parquet")
    return graph, str(index_dir)


@pytest.fixture(scope="session")
def dg_indexed_temporal_financial_graph(tmp_path_factory) -> Tuple[Graph, str]:
    """TEMPORAL FINANCIAL kumo.Graph + DiskGraph index, built once per session.
    Returns ``(graph, index_path)`` in the SAME shape as
    :func:`dg_indexed_financial_graph`.

    ADDITIVE sibling of :func:`dg_indexed_financial_graph`: that fixture is the
    canonical STATIC FINANCIAL graph and is left untouched. This one reproduces
    the FINANCIAL ACCOUNT/TRANS shape WITH a real ``TRANS.DATE`` timestamp anchor
    (2023 origin; see the module-level note for why the shipped degenerate
    parquet DATE cannot be used) so the parked FINANCIAL-temporal train-table
    tests (``SUM(TRANS.AMOUNT, ...)`` windows + ``DateOffsetSplit``) can
    materialize DG-native via :func:`materialize_task_dg`.
    """
    return _build_temporal_financial_graph(tmp_path_factory)


# ---------------------------------------------------------------------------
# TEMPORAL node-pred BANK graph (USERS entity + TRANSACTIONS temporal fact +
# BANKS RHS) + DiskGraph index.
# ---------------------------------------------------------------------------
#
# Why a separate fixture: a cluster of temporal node-pred generator tests build
# a USERS entity table + a TRANSACTIONS temporal fact table carrying an AMOUNT
# numeric and a BANK_ID column, with BANKS as the RHS table BANK_ID links to.
# They exercise query families the other DG fixtures do not cover:
#   * COUNT/MIN over a fkey value within a window + an entity WHERE-filter
#     (the "conversion" shape),
#   * a nested boolean target ``SUM(..)>30 OR COUNT(..)>1`` and its NOT(..)
#     inversion (BINARY_CLASSIFICATION),
#   * ``LIST_DISTINCT(TRANSACTIONS.BANK_ID, 0, 2) CLASSIFY`` with BANK_ID
#     re-typed as an ID *column* (MULTILABEL_CLASSIFICATION).
# None of the three existing node-pred / link-pred fixtures has a TRANSACTIONS
# table with a dual-role BANK_ID, so they are blocked purely on a DG-indexed
# graph of this shape.
#
# This fixture provides a 10-row TRANSACTIONS block (USER_ID / AMOUNT / BANK_ID
# / DATE) at a 2023 origin (a pre-2001 origin hits the ns-DATE landmine
# documented on the temporal node-pred fixture above; day-granular spacing is
# preserved, only the absolute origin shifts so window math / per-entity TARGET
# multisets are unchanged). It is a FACTORY so the BANK_ID role can be toggled:
#   * ``bank_id_is_col=False`` (default): BANK_ID is a fkey TRANSACTIONS ->
#     BANKS (the 'test_graph' shape used by the conversion / nested-invert
#     rows).
#   * ``bank_id_is_col=True``: BANK_ID is a plain column on TRANSACTIONS and
#     the BANKS table is dropped (the 'simple_graph' + ``remove_fkey`` /
#     ``add_col`` re-typing the LIST_DISTINCT multilabel rows do).
#
# Graph shape:
#   USERS(USER_ID pkey, AGE)
#   BANKS(BANK_ID pkey, BANK_TYPE)            # only when bank_id_is_col=False
#   TRANSACTIONS(TRANS_ID pkey,
#                USER_ID fkey -> USERS,
#                BANK_ID fkey -> BANKS | column,
#                AMOUNT float,
#                DATE time_col TIMESTAMP)


def _node_pred_bank_tables(
    bank_id_is_col: bool = False,
    add_na_fkeys: bool = False,
    add_na_timestamps: bool = False,
) -> Dict[str, "object"]:
    """Return ``{table_name: DataFrame}`` for the TEMPORAL node-pred BANK graph.

    The TRANSACTIONS values (USER_ID / AMOUNT / BANK_ID and the day-offset
    DATE sequence) form a fixed 10-row block with a stable ENTITY/TARGET
    sequence. When ``bank_id_is_col`` the BANKS table is dropped (BANK_ID
    becomes a plain TRANSACTIONS column).

    NA toggles (restore the legacy ``na_values`` axis):

    * ``add_na_fkeys``: append a TRANSACTIONS row whose ``USER_ID`` fkey is
      ``pd.NA`` -> DG drops the row (no entity) so ENTITY/TARGET are unchanged.
    * ``add_na_timestamps``: append a TRANSACTIONS row with ``NaT`` in the
      declared ``DATE`` time_col -> expected dropped (no anchor). UNVERIFIED on
      DG ingest (kept as a SEPARATE param id; see
      :func:`_temporal_synthetic_node_pred_tables`).
    """
    import pandas as pd

    users = pd.DataFrame({
        "USER_ID": [0, 1, 2, 3],
        "AGE": [19, 20, 21, 22],
    })
    # DATE day-offsets (4x +0, 2x +2, 3x +3, 1x +4) from a 2023 origin; spacing
    # chosen so window math is preserved.
    legacy_days = [0, 0, 0, 0, 2, 2, 3, 3, 3, 4]
    trans_id = list(range(10))
    user_id = [0, 1, 2, 3, 1, 3, 3, 0, 1, 0]
    amount = [0.0, 0.0, 0.0, 0.0, 150.0, 20.0, 30.0, 30.0, 150.0, 0.0]
    bank_id = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    dates = [(pd.Timestamp("2023-01-01") + pd.Timedelta(days=d))
             for d in legacy_days]

    next_id = 10
    if add_na_fkeys:
        # NA fkey: dropped, no new ENTITY appears.
        trans_id.append(next_id)
        user_id.append(pd.NA)
        amount.append(99.0)
        bank_id.append(0)
        dates.append(pd.Timestamp("2023-01-02"))
        next_id += 1
    if add_na_timestamps:
        # NaT anchor on an existing entity: expected dropped (no timeframe).
        # UNVERIFIED on DG ingest.
        trans_id.append(next_id)
        user_id.append(0)
        amount.append(99.0)
        bank_id.append(0)
        dates.append(pd.NaT)
        next_id += 1

    # Default (no-NA) path keeps USER_ID as a plain int column, byte-identical
    # to the pre-branch fixture, so non-NA BANK consumers are unaffected. Only
    # the NA variants promote to nullable Int64 so pd.NA survives the parquet
    # round-trip as a null key (a plain int64 column cannot hold pd.NA) -- mirrors
    # _temporal_synthetic_node_pred_tables / _temporal_node_pred_orders_tables.
    user_id_col = (pd.array(user_id, dtype="Int64") if
                   (add_na_fkeys or add_na_timestamps) else user_id)
    transactions = pd.DataFrame({
        "TRANS_ID": list(trans_id),
        "USER_ID": user_id_col,
        "AMOUNT": amount,
        "BANK_ID": bank_id,
        "DATE": pd.to_datetime(dates),
    })
    transactions["DATE"] = transactions["DATE"].astype("datetime64[ns]")
    tables: Dict[str, "object"] = {"USERS": users}
    if not bank_id_is_col:
        tables["BANKS"] = pd.DataFrame({
            "BANK_ID": [0, 1],
            "BANK_TYPE": ["A", "B"],
        })
    tables["TRANSACTIONS"] = transactions
    return tables


def _node_pred_bank_metadata(bank_id_is_col: bool = False) -> dict:
    """metadata.json-shaped dict for the TEMPORAL node-pred BANK graph.

    Drives BOTH the DG ingest config (:func:`_build_diskgraph_ingest_config`)
    and the in-memory kumo.Graph (``graph.add_from_config``). TRANSACTIONS
    declares ``time_col=DATE`` so DG ingest sets a ``time_column`` on the fact
    table block and the temporal sampler can resolve anchor dates. When
    ``bank_id_is_col`` the BANKS table + BANK_ID fkey are dropped and BANK_ID
    is emitted as a TRANSACTIONS column instead.
    """
    def col(name: str, dtype: str) -> dict:
        return {
            "name": name,
            "stype": None,
            "dtype": dtype,
            "na_values": None,
            "metadata": {},
            "encoder": None,
            "stats": {},
        }

    users = {
        "table_name": "USERS",
        "data_source": None,
        "data_version": None,
        "pkey": {
            "name": "USER_ID",
            "stype": None,
            "dtype": "int",
            "na_values": None,
            "metadata": {},
            "drop_duplicates": False,
        },
        "fkeys": [],
        "cols": [col("AGE", "int")],
        "time_col": None,
        "source_name": None,
        "stats": {},
    }

    trans_fkeys = [{
        "name": "USER_ID",
        "stype": None,
        "dtype": "int",
        "na_values": None,
        "metadata": {},
        "dst_table": "USERS",
        "dst_key": None,
        "dst_data_source": None,
        "stats": {},
    }]
    trans_cols = [col("AMOUNT", "float")]
    tables = [users]
    if bank_id_is_col:
        # BANK_ID is a plain TRANSACTIONS column (the LIST_DISTINCT multilabel
        # rows re-type it as an ID column post-build). No BANKS table.
        trans_cols.append(col("BANK_ID", "int"))
    else:
        banks = {
            "table_name": "BANKS",
            "data_source": None,
            "data_version": None,
            "pkey": {
                "name": "BANK_ID",
                "stype": None,
                "dtype": "int",
                "na_values": None,
                "metadata": {},
                "drop_duplicates": False,
            },
            "fkeys": [],
            "cols": [col("BANK_TYPE", "string")],
            "time_col": None,
            "source_name": None,
            "stats": {},
        }
        tables.append(banks)
        trans_fkeys.append({
            "name": "BANK_ID",
            "stype": None,
            "dtype": "int",
            "na_values": None,
            "metadata": {},
            "dst_table": "BANKS",
            "dst_key": None,
            "dst_data_source": None,
            "stats": {},
        })

    trans = {
        "table_name": "TRANSACTIONS",
        "data_source": None,
        "data_version": None,
        "pkey": {
            "name": "TRANS_ID",
            "stype": None,
            "dtype": "int",
            "na_values": None,
            "metadata": {},
            "drop_duplicates": False,
        },
        "fkeys": trans_fkeys,
        "cols": trans_cols,
        # GraphConfig requires ``time_col`` as a column-config dict;
        # ``_build_diskgraph_ingest_config`` / ``_time_col_name`` accept either
        # form and emit the DG ``time_column`` off the dict.
        "time_col": col("DATE", "time"),
        "source_name": None,
        "stats": {},
    }
    tables.append(trans)
    return {
        "name": "NODE_PRED_BANK",
        "data_source_dict": {},
        "tables": tables,
    }


def _build_node_pred_bank_graph(
    tmp_path_factory,
    bank_id_is_col: bool = False,
    add_na_fkeys: bool = False,
    add_na_timestamps: bool = False,
) -> Tuple[Graph, str]:
    """Build + DG-index the TEMPORAL node-pred BANK graph; return
    ``(graph, index_path)`` in the SAME shape as the other DG fixtures.

    Mirrors the node-pred / link-pred fixtures' mechanics: write parquet per
    table, build a DG ingest config from a metadata.json-shaped dict, call the
    wheel's ``rust_ingest`` directly, then build the matching in-memory
    kumo.Graph via ``add_from_config`` against the same metadata.

    ``add_na_fkeys`` / ``add_na_timestamps`` thread through to
    :func:`_node_pred_bank_tables` (the legacy ``na_values`` axis).
    """
    from diskgraph import ingest as rust_ingest

    parquet_dir = tmp_path_factory.mktemp("dg_node_pred_bank_parquet")
    tables = _node_pred_bank_tables(
        bank_id_is_col=bank_id_is_col,
        add_na_fkeys=add_na_fkeys,
        add_na_timestamps=add_na_timestamps,
    )
    for name, df in tables.items():
        df.to_parquet(osp.join(str(parquet_dir), f"{name}.parquet"),
                      index=False)

    metadata = _node_pred_bank_metadata(bank_id_is_col=bank_id_is_col)
    config = _build_diskgraph_ingest_config(str(parquet_dir), metadata)

    index_dir = tmp_path_factory.mktemp("dg_indexed_node_pred_bank_idx")
    rust_ingest(
        json.dumps(config),
        str(index_dir),
        col_stats=True,
        cat_count_strategy="hybrid",
        success_marker_content=json.dumps(
            {"source": "dg_indexed_node_pred_bank_graph fixture"}),
    )

    meta_path = osp.join(str(index_dir), "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f)

    graph = Graph(name="NODE_PRED_BANK")
    graph.connect(ParquetConnector("parquet", str(parquet_dir)))
    graph.add_from_config(meta_path, data_source="parquet")
    return graph, str(index_dir)


@pytest.fixture(scope="session")
def dg_indexed_node_pred_bank_graph(tmp_path_factory):
    """FACTORY fixture: returns a callable that builds + DG-indexes the
    TEMPORAL node-pred BANK kumo.Graph and yields ``(graph, index_path)`` in
    the SAME shape as the other ``dg_indexed_*`` fixtures.

    Node-pred BANK graph used by the temporal node-pred conversion /
    nested-invert / LIST_DISTINCT-multilabel generator rows. A FACTORY (not a
    built tuple) so callers can toggle the BANK_ID role those rows need::

        def test_x(dg_indexed_node_pred_bank_graph, tmp_path):
            # BANK_ID as a fkey TRANSACTIONS -> BANKS:
            graph, index_path = dg_indexed_node_pred_bank_graph()
            # BANK_ID as a plain TRANSACTIONS column (no BANKS table):
            graph, index_path = dg_indexed_node_pred_bank_graph(
                bank_id_is_col=True)

    USERS entity table + TRANSACTIONS temporal fact table (real ``DATE``
    time_col, AMOUNT numeric + BANK_ID) + BANKS RHS. 2023 origin (post-2001 ns
    landmine). Each ``_factory(...)`` call builds a FRESH index under its own
    tmp dir, so the two BANK_ID-role variants never share state.
    """
    def _factory(
        bank_id_is_col: bool = False,
        add_na_fkeys: bool = False,
        add_na_timestamps: bool = False,
    ) -> Tuple[Graph, str]:
        return _build_node_pred_bank_graph(tmp_path_factory,
                                           bank_id_is_col=bank_id_is_col,
                                           add_na_fkeys=add_na_fkeys,
                                           add_na_timestamps=add_na_timestamps)

    return _factory


# ---------------------------------------------------------------------------
# TEMPORAL node-pred TRANSACTIONS/SESSIONS graph
# ---------------------------------------------------------------------------
# Node-pred graph that, unlike the BANK fixture above, also carries a SESSIONS
# fact table (USER_ID fkey -> USERS, DURATION, DATE time_col) so the
# SESSIONS-driven WHERE/ASSUMING aggregation filters the forecasting rows use
# (e.g. ``WHERE SUM(SESSIONS.DURATION, -1, 0) > 0``) resolve against a real
# DG-indexed session edge.


def _node_pred_txn_sessions_tables(
    small_target_values: bool = False,
    with_category: bool = False,
    wide_span: bool = False,
) -> Dict[str, "object"]:
    """Return ``{table_name: DataFrame}`` for the TEMPORAL node-pred
    TRANSACTIONS/SESSIONS graph.

    The TRANSACTIONS values (USER_ID / AMOUNT / BANK_ID and the day-offset
    DATE sequence) and the SESSIONS values (USER_ID / DURATION and its
    day-offset DATE sequence) form a fixed 10-row TRANSACTIONS block and 4-row
    SESSIONS block with a stable ENTITY/TARGET sequence and SESSIONS-filter
    include set. Both DATE sequences use a 2023 origin (spacing: TRANSACTIONS
    4x day0, 2x day2, 3x day3, 1x day4; SESSIONS 2x day1, 2x day3). When
    ``small_target_values`` the AMOUNT column is divided by 10000 (used by the
    bin-width stats row).

    When ``with_category`` the TRANSACTIONS table additionally carries CATEGORY
    + IS_ONLINE columns so the conversion-parity row that predicates on
    ``TRANSACTIONS.CATEGORY`` can use this fixture. The columns are structural
    additions (10-row, few-value) and DEFAULT OFF so existing
    ``test_dg_node_pred_txn_sessions_*`` consumers keep their pinned TARGET
    multisets unchanged.

    When ``wide_span`` BOTH fact tables' DATE day-offsets are multiplied by 3
    (TRANSACTIONS 4x day0 / 2x day6 / 3x day9 / 1x day12; SESSIONS 2x day3 /
    2x day9), stretching the data span from 4 days to 12 days WITHOUT touching
    USER_ID / AMOUNT / CATEGORY / DURATION (only the absolute DATE moves). The
    narrow 4-day span resolves to <3 timeframes under ``DateOffsetSplit([-2,0],
    DAYS, timeframe=2)`` (train_table_validator "needs at least 3 timeframes" /
    DG empty ``train_table_stats``); the 3x-wide span resolves to 5 timeframes
    whose forward ``(0, 2)`` aggregation windows still capture the AMOUNT=30 /
    CATEGORY=1 transactions, so the array-target CONTAINS row materializes a
    non-empty train table with the same per-entity target set. DEFAULT OFF so
    existing consumers are unaffected.
    """
    import pandas as pd

    users = pd.DataFrame({
        "USER_ID": [0, 1, 2, 3],
        "AGE": [19, 20, 21, 22],
    })
    # TRANSACTIONS DATE day-offsets from a 2023 origin.
    # ``wide_span`` multiplies the day-offsets by 3 (4d span -> 12d span) so a
    # timeframe-2 DateOffsetSplit resolves >=3 timeframes while the forward
    # aggregation windows still cover the AMOUNT=30 / CATEGORY=1 rows.
    span_mult = 3 if wide_span else 1
    trans_days = [d * span_mult for d in [0, 0, 0, 0, 2, 2, 3, 3, 3, 4]]
    amount = [0.0, 0.0, 0.0, 0.0, 150.0, 20.0, 30.0, 30.0, 150.0, 0.0]
    if small_target_values:
        amount = [a / 10000 for a in amount]
    trans_cols = {
        "TRANS_ID": list(range(10)),
        "USER_ID": [0, 1, 2, 3, 1, 3, 3, 0, 1, 0],
        "AMOUNT": amount,
        "BANK_ID": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    }
    if with_category:
        # Fixed CATEGORY + IS_ONLINE values for the conversion-parity row.
        trans_cols["CATEGORY"] = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        trans_cols["IS_ONLINE"] = [
            True,
            False,
            False,
            False,
            False,
            True,
            True,
            False,
            False,
            True,
        ]
    trans_cols["DATE"] = [(pd.Timestamp("2023-01-01") + pd.Timedelta(days=d))
                          for d in trans_days]
    transactions = pd.DataFrame(trans_cols)
    transactions["DATE"] = pd.to_datetime(
        transactions["DATE"]).astype("datetime64[ns]")
    # Legacy SESSIONS: USER_ID [1,3,1,3] DURATION 1.0 dates 1990-01-02 (x2) and
    # 1990-01-04 (x2) -> day-offsets 1 and 3 from the 2023 origin. The same
    # ``span_mult`` stretches SESSIONS in lockstep so the anchor grid extends
    # to cover the stretched TRANSACTIONS.
    sess_days = [d * span_mult for d in [1, 1, 3, 3]]
    sessions = pd.DataFrame({
        "SESSION_ID":
        list(range(4)),
        "USER_ID": [1, 3, 1, 3],
        "DURATION": [1.0, 1.0, 1.0, 1.0],
        "DATE": [(pd.Timestamp("2023-01-01") + pd.Timedelta(days=d))
                 for d in sess_days],
    })
    sessions["DATE"] = pd.to_datetime(sessions["DATE"]).astype("datetime64[ns]")
    return {
        "USERS": users,
        "TRANSACTIONS": transactions,
        "SESSIONS": sessions,
    }


def _node_pred_txn_sessions_metadata(with_category: bool = False) -> dict:
    """metadata.json-shaped dict for the TEMPORAL node-pred
    TRANSACTIONS/SESSIONS graph.

    Drives BOTH the DG ingest config (:func:`_build_diskgraph_ingest_config`)
    and the in-memory kumo.Graph (``graph.add_from_config``). Both TRANSACTIONS
    and SESSIONS declare ``time_col=DATE`` so DG ingest sets a ``time_column``
    on each fact-table block (and a reverse-edge sort time on the USERS<-fact
    edges), and the temporal sampler can resolve anchor dates + apply the
    SESSIONS aggregation-filter time windows.

    When ``with_category`` the TRANSACTIONS block additionally declares the
    legacy CATEGORY (int) + IS_ONLINE (bool) columns; must stay in sync with the
    ``with_category`` branch of :func:`_node_pred_txn_sessions_tables`.
    """
    def col(name: str, dtype: str) -> dict:
        return {
            "name": name,
            "stype": None,
            "dtype": dtype,
            "na_values": None,
            "metadata": {},
            "encoder": None,
            "stats": {},
        }

    users = {
        "table_name": "USERS",
        "data_source": None,
        "data_version": None,
        "pkey": {
            "name": "USER_ID",
            "stype": None,
            "dtype": "int",
            "na_values": None,
            "metadata": {},
            "drop_duplicates": False,
        },
        "fkeys": [],
        "cols": [col("AGE", "int")],
        "time_col": None,
        "source_name": None,
        "stats": {},
    }

    transactions = {
        "table_name":
        "TRANSACTIONS",
        "data_source":
        None,
        "data_version":
        None,
        "pkey": {
            "name": "TRANS_ID",
            "stype": None,
            "dtype": "int",
            "na_values": None,
            "metadata": {},
            "drop_duplicates": False,
        },
        "fkeys": [{
            "name": "USER_ID",
            "stype": None,
            "dtype": "int",
            "na_values": None,
            "metadata": {},
            "dst_table": "USERS",
            "dst_key": None,
            "dst_data_source": None,
            "stats": {},
        }],
        "cols":
        ([col("AMOUNT", "float"), col("BANK_ID", "int")] +
         ([col("CATEGORY", "int"),
           col("IS_ONLINE", "bool")] if with_category else [])),
        "time_col":
        col("DATE", "time"),
        "source_name":
        None,
        "stats": {},
    }

    sessions = {
        "table_name":
        "SESSIONS",
        "data_source":
        None,
        "data_version":
        None,
        "pkey": {
            "name": "SESSION_ID",
            "stype": None,
            "dtype": "int",
            "na_values": None,
            "metadata": {},
            "drop_duplicates": False,
        },
        "fkeys": [{
            "name": "USER_ID",
            "stype": None,
            "dtype": "int",
            "na_values": None,
            "metadata": {},
            "dst_table": "USERS",
            "dst_key": None,
            "dst_data_source": None,
            "stats": {},
        }],
        "cols": [col("DURATION", "float")],
        "time_col":
        col("DATE", "time"),
        "source_name":
        None,
        "stats": {},
    }

    return {
        "name": "NODE_PRED_TXN_SESSIONS",
        "data_source_dict": {},
        "tables": [users, transactions, sessions],
    }


def _build_node_pred_txn_sessions_graph(
    tmp_path_factory,
    small_target_values: bool = False,
    with_category: bool = False,
    wide_span: bool = False,
) -> Tuple[Graph, str]:
    """Build + DG-index the TEMPORAL node-pred TRANSACTIONS/SESSIONS graph;
    return ``(graph, index_path)`` in the SAME shape as the other DG fixtures.

    Mirrors the BANK node-pred fixture's mechanics: write parquet per table,
    build a DG ingest config from a metadata.json-shaped dict, call the wheel's
    ``rust_ingest`` directly, then build the matching in-memory kumo.Graph via
    ``add_from_config`` against the same metadata. ``with_category`` adds the
    legacy CATEGORY + IS_ONLINE TRANSACTIONS columns (kept in sync across the
    tables + metadata builders). ``wide_span`` 3x-stretches both fact tables'
    DATE day-offsets (values unchanged) so a timeframe-2 DateOffsetSplit
    resolves >=3 timeframes (see :func:`_node_pred_txn_sessions_tables`).
    """
    from diskgraph import ingest as rust_ingest

    parquet_dir = tmp_path_factory.mktemp("dg_node_pred_txn_sessions_parquet")
    tables = _node_pred_txn_sessions_tables(
        small_target_values=small_target_values, with_category=with_category,
        wide_span=wide_span)
    for name, df in tables.items():
        df.to_parquet(osp.join(str(parquet_dir), f"{name}.parquet"),
                      index=False)

    metadata = _node_pred_txn_sessions_metadata(with_category=with_category)
    config = _build_diskgraph_ingest_config(str(parquet_dir), metadata)

    index_dir = tmp_path_factory.mktemp("dg_indexed_node_pred_txn_sessions_idx")
    rust_ingest(
        json.dumps(config),
        str(index_dir),
        col_stats=True,
        cat_count_strategy="hybrid",
        success_marker_content=json.dumps(
            {"source": "dg_indexed_node_pred_txn_sessions_graph fixture"}),
    )

    meta_path = osp.join(str(index_dir), "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f)

    graph = Graph(name="NODE_PRED_TXN_SESSIONS")
    graph.connect(ParquetConnector("parquet", str(parquet_dir)))
    graph.add_from_config(meta_path, data_source="parquet")
    return graph, str(index_dir)


@pytest.fixture(scope="session")
def dg_indexed_node_pred_txn_sessions_graph(tmp_path_factory):
    """FACTORY fixture: returns a callable that builds + DG-indexes the
    TEMPORAL node-pred TRANSACTIONS/SESSIONS kumo.Graph and yields
    ``(graph, index_path)`` in the SAME shape as the other ``dg_indexed_*``
    fixtures.

    TRANSACTIONS/SESSIONS node-pred graph used by the temporal node-pred
    forecasting / SESSIONS-filtered generator + stats rows. A FACTORY (not a
    built tuple) so callers can toggle the ``small_target_values`` AMOUNT
    scaling the bin-width stats row needs::

        def test_x(dg_indexed_node_pred_txn_sessions_graph, tmp_path):
            graph, index_path = dg_indexed_node_pred_txn_sessions_graph()
            # AMOUNT / 10000 (small_target_values toggle):
            graph, index_path = dg_indexed_node_pred_txn_sessions_graph(
                small_target_values=True)
            # +CATEGORY/IS_ONLINE TRANSACTIONS columns:
            graph, index_path = dg_indexed_node_pred_txn_sessions_graph(
                with_category=True)
            # 3x-wide DATE span (>=3 timeframes under a timeframe-2 split):
            graph, index_path = dg_indexed_node_pred_txn_sessions_graph(
                with_category=True, wide_span=True)

    USERS entity table + TRANSACTIONS temporal fact table (real ``DATE``
    time_col, AMOUNT numeric + BANK_ID) + SESSIONS temporal fact table (real
    ``DATE`` time_col, DURATION) -- both fact tables fkey -> USERS so
    SESSIONS-driven aggregation filters resolve. 2023 origin (post-2001 ns
    landmine). Each ``_factory(...)`` call builds a FRESH index under its own
    tmp dir, so the AMOUNT-scaling / wide-span variants never share state.
    """
    def _factory(small_target_values: bool = False, with_category: bool = False,
                 wide_span: bool = False) -> Tuple[Graph, str]:
        return _build_node_pred_txn_sessions_graph(
            tmp_path_factory, small_target_values=small_target_values,
            with_category=with_category, wide_span=wide_span)

    return _factory


# ---------------------------------------------------------------------------
# DG-native materialize helper
# ---------------------------------------------------------------------------


def _ttng_stage_masks_from_bundle(staging_dir: str, row_count: int):
    """Reconstruct per-row train/val/test boolean masks from the TT-NG durable
    staging bundle.

    Temporal splits fill stages by timestamp up to the ``split_stats.json``
    row counts; static splits match rows to the stage seed parts. The
    production native path consumes the stage tables directly.
    """
    import torch
    from kumoapi.train import TrainingStage

    stages = [TrainingStage.TRAIN, TrainingStage.VAL, TrainingStage.TEST]
    split_stats = json.loads(
        (Path(staging_dir) / 'split_stats.json').read_text())

    masks = {s: torch.zeros(row_count, dtype=torch.bool) for s in stages}
    if row_count == 0:
        return masks

    tt_path = Path(staging_dir) / 'tt.parquet'
    tt_df = pq.read_table(tt_path).to_pandas()

    mask_counts = {
        stage.value:
        int((split_stats.get(f'{stage.value}_table_stats')
             or {}).get('num_train_rows', 0))
        for stage in stages
    }
    ordered = [(s, int(mask_counts.get(s.value, 0))) for s in stages]

    # Temporal splits partition by anchor TIMESTAMP (ascending); the durable
    # ``tt.parquet`` is entity-ordered, so a contiguous row fill would misassign
    # whole entities to a single stage. Assign whole timestamps to stages in
    # ascending time order, filling each stage up to its ``mask_counts``.
    if 'TIMESTAMP' in tt_df.columns and tt_df['TIMESTAMP'].nunique() > 1:
        row_ts = tt_df['TIMESTAMP'].astype('int64').to_numpy()
        unique_ts = sorted(set(row_ts.tolist()))
        stage_iter = iter(ordered)
        cur_stage, cur_cap = next(stage_iter, (None, 0))
        assigned = 0
        for ts in unique_ts:
            while cur_stage is not None and cur_cap <= 0:
                cur_stage, cur_cap = next(stage_iter, (None, 0))
            if cur_stage is None:
                break
            sel = (row_ts == ts)
            masks[cur_stage][sel.nonzero()[0].tolist()] = True
            n = int(sel.sum())
            cur_cap -= n
            assigned += n
        if assigned:
            return masks

    # Static: tt.parquet row order is not stable across runs, so match each
    # row to the stage seed part that holds its (entity, timestamp).
    entity_col = 'ENTITY1' if 'ENTITY1' in tt_df.columns else 'ENTITY'

    def _keys(df):
        return zip(df[entity_col].tolist(),
                   df['TIMESTAMP'].astype('int64').tolist())

    remaining = {}
    for stage in stages:
        seeds = Path(staging_dir) / f'{stage.value}_seeds'
        remaining[stage] = Counter(
            _keys(pq.read_table(seeds).to_pandas()) if seeds.exists() else [])
    for row, key in enumerate(_keys(tt_df)):
        for stage in stages:
            if remaining[stage][key] > 0:
                remaining[stage][key] -= 1
                masks[stage][row] = True
                break
    return masks


def _ttng_stage_split_table_stats(tt_df, mask, task_type):
    """Build a per-stage ``SplitTableStats`` directly from the stage's slice of
    ``tt.parquet``.

    Reproduces the columns the split-stats consumers read: ``num_train_rows``,
    ``num_entities`` (distinct ENTITY1), ``num_targets`` (distinct TARGET1),
    ``target_distribution`` (categorical ``{count, percentage}`` for
    classification; numeric summary for regression), and the min/max/anchor
    dates. Returns ``None`` for an empty stage.
    """
    import numpy as np
    import pandas as pd

    from kumo.stats.train_table_stats import SplitTableStats

    idx = mask.nonzero(as_tuple=False).flatten().tolist()
    if not idx:
        return None
    stage_df = tt_df.iloc[idx]

    task_name = str(task_type) if task_type is not None else ''
    is_regression = task_name.endswith('REGRESSION')
    is_link_pred = 'LINK_PREDICTION' in task_name or task_name.endswith(
        'MULTILABEL_RANKING')

    num_rows = len(stage_df)
    entity_col = 'ENTITY1' if 'ENTITY1' in stage_df.columns else 'ENTITY'
    target_col = 'TARGET1' if 'TARGET1' in stage_df.columns else 'TARGET'
    num_entities = int(stage_df[entity_col].nunique())
    num_targets = None
    target_distribution = None
    if target_col in stage_df.columns:
        targets = stage_df[target_col].dropna()
    else:
        targets = None

    if targets is not None and len(targets) > 0 and not is_link_pred:
        # Scalar (node-pred) targets: distinct-count + distribution. LP targets
        # are ragged id lists (unhashable) whose per-stage distribution the DG
        # pre-split path does not surface, so they are left unset.
        num_targets = int(targets.nunique())
        if is_regression:
            arr = targets.to_numpy(dtype='float')
            arr = arr[np.isfinite(arr)]
            if len(arr):
                # 50-bin float histogram matches DG's NUM_BINS_HISTOGRAM=50
                # regression split-stats histogram.
                num_bins = 50
                lo, hi = float(np.min(arr)), float(np.max(arr))
                counts, edges = np.histogram(arr, bins=num_bins)
                bin_width = float(edges[1] -
                                  edges[0]) if len(edges) > 1 else 0.0
                target_distribution = {
                    'target_min': lo,
                    'target_max': hi,
                    'target_mean': float(np.mean(arr)),
                    'target_median': float(np.median(arr)),
                    'target_percentile_25': float(np.percentile(arr, 25)),
                    'target_percentile_75': float(np.percentile(arr, 75)),
                    'histogram': {
                        'bins': edges[:-1].tolist(),
                        'counts': counts.tolist(),
                        'bin_width': bin_width,
                        'num_bins': num_bins,
                    },
                }
        else:
            counts = targets.value_counts()
            total = int(counts.sum())
            target_distribution = {
                'count': {
                    str(k): int(v)
                    for k, v in counts.items()
                },
                'percentage': {
                    str(k): float(v) / total
                    for k, v in counts.items()
                },
            }

    min_date = max_date = anchor_date = None
    if 'TIMESTAMP' in stage_df.columns:
        ts = stage_df['TIMESTAMP'].dropna()
        if len(ts):

            def _utc(v):
                v = pd.Timestamp(v)
                return v.tz_localize('UTC') if v.tzinfo is None else v

            min_date = _utc(ts.min())
            max_date = _utc(ts.max())
            # Per-stage ``anchor_date`` is the EARLIEST anchor in the stage
            # (== ``min_date``), matching Kumo's split-stats convention that
            # every consumer/test asserts (node-pred + LP alike). This mirrors
            # the legacy DG ``split_stats.json`` per-stage payload; do not use
            # ``ts.max()`` here (the wheel's top-level aggregate anchor is a
            # separate field with its own convention).
            anchor_date = _utc(ts.min())

    return SplitTableStats(
        num_train_rows=num_rows,
        num_entities=num_entities,
        num_targets=num_targets,
        target_distribution=target_distribution,
        min_date=min_date,
        max_date=max_date,
        anchor_date=anchor_date,
    )


def _apply_ttng_native_split_from_bundle(train_table, staging_dir, task_type):
    """Apply TT-NG split masks + after-split stats to ``train_table`` from the
    durable ``.ttng_staging`` bundle.

    Replaces the legacy ``_apply_native_diskgraph_split`` /
    ``_apply_native_split_stats`` (which require ``parsed_split`` /
    ``parsed_split_stats`` manifest keys the TT-NG producer no longer emits).
    Split masks + per-stage stats are reconstructed from the artifacts the
    TT-NG bundle carries (``stats.json`` + ``tt.parquet`` + seed parts).
    """
    import json as _json

    from kumoapi.train import TrainingStage

    from kumo.data.generic.diskgraph_train_table import (
        resolve_durable_ttng_bundle,
    )
    from kumo.stats.train_table_stats import SplitTrainTableStats

    staging_dir = resolve_durable_ttng_bundle(str(staging_dir))
    stats = _json.loads((Path(staging_dir) / 'stats.json').read_text())
    row_count = int(stats.get('num_total_rows') or 0)

    masks = _ttng_stage_masks_from_bundle(staging_dir, row_count)
    for stage in TrainingStage.list_stages():
        train_table.data[stage.add_stage_prefix('mask')] = masks[stage]

    tt_df = pq.read_table(Path(staging_dir) / 'tt.parquet').to_pandas()
    split_stats = SplitTrainTableStats(
        train_table_stats=_ttng_stage_split_table_stats(
            tt_df, masks[TrainingStage.TRAIN], task_type),
        val_table_stats=_ttng_stage_split_table_stats(tt_df,
                                                      masks[TrainingStage.VAL],
                                                      task_type),
        test_table_stats=_ttng_stage_split_table_stats(
            tt_df, masks[TrainingStage.TEST], task_type),
    )
    train_table._split_train_table_stats = split_stats
    train_table._save_split_train_table_stats_to_job_dir(split_stats)
    train_table._update_and_save_stats_cache(split_stats)


def materialize_task_dg(
    pquery,
    index_path: str,
    tmp_path,
):
    """Generate train-table + materialize task using the DG-native
    manifest-based split pattern that production uses.

    Returns the materialized task with populated ``train_mask`` /
    ``val_mask`` / ``test_mask`` (via
    :func:`_apply_ttng_native_split_from_bundle`) and split-stats wired
    from the durable ``.ttng_staging`` bundle.

    Mirrors the prod TT-NG materialize path, condensed for unit-test use.
    """
    # Thread an explicit destination into generate_train_table (the sole
    # get_home_dir() fallback on this path) instead of setting a process-global
    # home dir via temp_home_dir. TrainingTable.materialize reads through the
    # table's explicit ParquetConnector, not the home dir.
    selected_train_table_path, job_id, _ = generate_train_table_dg(
        pquery, diskgraph_index_path=index_path, destination=str(tmp_path))
    return materialize_selected_task_dg(pquery, selected_train_table_path,
                                        job_id)


def materialize_selected_task_dg(pquery, selected_train_table_path, job_id):
    """Materialize an already-selected immutable train-table publication."""
    task = pquery.create_task(
        train_table_path=selected_train_table_path,
        job_id=job_id,
        ignore_encoders=True,
    )
    task.train_table.materialize(
        execute_split=False,
        calculate_split_stats=False,
        skip_compute_stats=True,
    )
    # TT-NG persists the split artifacts inside the durable
    # ``.ttng_staging`` bundle (stats.json + tt.parquet + seed parts)
    # rather than in a ``.manifest.json`` ``parsed_split`` blob; apply the
    # native split masks + after-split stats from that bundle.
    staging_dir = str(Path(selected_train_table_path).resolve().parent)
    _apply_ttng_native_split_from_bundle(task.train_table, staging_dir,
                                         task.task_type)
    return task


def generate_train_table_dg(pquery, *args, **kwargs):
    """Return the physical immutable TT parquet selected by production."""
    return pquery.generate_train_table(*args, **kwargs)


def materialize_bp_dg(
    graph,
    validated_query,
    index_path: str,
    tmp_path,
    pred_time=None,
    list_distinct_target_is_array: bool = False,
    forecast_length: int = 1,
    lag_timesteps=None,
    use_yoy_labels: bool = False,
):
    """Generate a BATCH-PREDICTION table on DG and read it back as a
    ``pandas.DataFrame`` (the bp sibling of :func:`materialize_task_dg`).

    The production batch-pred producer
    (:meth:`kumo.pquery.batch_pred_table_generator.
    PredictionTableGeneratorV2.generate_table`) WRITES parquet to
    ``output_path`` and returns ``None`` on DG (the legacy in-process
    producer that returned a DataFrame was retired in the DG cutover;
    calling it without ``diskgraph_index_path`` raises
    ``UnsupportedInDiskgraphError``). This helper threads a real
    ``diskgraph_index_path`` + an explicit ``output_path``, then reads the
    written parquet back so a migrated bp test can assert on a real
    DG-produced DataFrame -- exactly the read-back gap called out in the
    triage tracker (HELPER-INFRA 2.H).

    Output layout (TT-NG one-gen predict, from
    ``kumo/data/generic/diskgraph_batch_pred.py``): the bp producer requires a
    ``job_id`` (it names the KEPT predict staging bundle that
    ``materialize_prediction_task`` later ingests) and gens the predict SEEDS
    into ``<index>/train_tables/.<pred_<job_id>>.staging/tt.parquet`` — it does
    NOT write a serving ``<output_path>/part-00000.parquet``. This helper
    threads a synthetic ``job_id``, then reads the staged ``tt.parquet`` back so
    a migrated bp test can assert on a real DG-produced DataFrame.

    The staged seed schema carries the raw producer ``ENTITY1`` (and, for
    forecasting, ``autoregression_labels`` / ``year_over_year_labels``) columns;
    this helper renames slot-0 ``ENTITY1``/``TARGET1`` -> ``ENTITY``/``TARGET``
    so the migrated bp assertions (written against the Kumo-owned column names)
    keep working. There is NO ``TARGET`` column at predict time and NO separate
    RHS parquet on TT-NG: node-pred BP drops the target, and LP BP gens only the
    LHS seeds (the RHS candidate pool is seeded at INGEST, never materialized as
    a gen-time cross-product). Callers must not assert on a ``TARGET`` / RHS
    df / ``arange`` column — none is produced by the TT-NG one-gen path.

    The bp path is driven with an EXPLICIT ``output_path`` so it never falls
    through to ``get_home_dir()`` and never touches the process-global home dir.
    No ``temp_home_dir`` is opened here (and none must be nested by callers
    around this helper).

    Args:
        graph: the DG-indexed ``kumo.Graph`` (from a ``dg_indexed_*`` fixture).
        validated_query: the ``ValidatedPredictiveQuery`` (e.g. from
            ``kumo.testing.pquery.get_validated_query``).
        index_path: the real DG index path returned by the fixture.
        tmp_path: pytest ``tmp_path`` (used only as a scratch output root).
        pred_time: optional prediction anchor (``pd.Timestamp``). When
            ``None`` and the query is temporal, the producer anchors to the
            max target timestamp.
        list_distinct_target_is_array: array-target (separate-lhs-rhs) layout.
            On TT-NG only the LHS predict seeds are staged either way; the flag
            still selects the FK-free LP resolution the producer requires.
        forecast_length / lag_timesteps / use_yoy_labels: forecasting params,
            forwarded verbatim to ``generate_table``.

    Returns:
        ``pandas.DataFrame`` of the staged LHS predict seeds (columns renamed to
        the Kumo-owned names). No RHS df is returned — the RHS pool is an ingest
        concern on TT-NG.
    """
    from kumo.data.generic.diskgraph_staging import (
        predict_staging_tt_id,
        ttng_staging_dir,
    )
    from kumo.pquery.batch_pred_table_generator import (
        PredictionTableGeneratorV2,
    )

    # TT-NG bp REQUIRES a job_id (it names the kept predict staging bundle for
    # the one-gen handoff to materialize). Mirror the live prod path
    # (generate_current_pred_table auto-assigns one). Each invocation is a new
    # publication and therefore needs a distinct immutable output identity.
    call_id = uuid.uuid4().hex
    job_id = f"bp_{osp.basename(str(tmp_path).rstrip('/'))}_{call_id}"
    output_path = osp.join(str(tmp_path), f"dg_bp_pred_table_{call_id}")
    target_output_root = osp.join(str(tmp_path),
                                  f"dg_bp_pred_table_rhs_{call_id}")

    generator = PredictionTableGeneratorV2(graph, validated_query,
                                           forecast_length=forecast_length)
    generator.generate_table(
        pred_time=pred_time,
        lag_timesteps=lag_timesteps,
        use_yoy_labels=use_yoy_labels,
        diskgraph_index_path=index_path,
        list_distinct_target_is_array=list_distinct_target_is_array,
        output_path=output_path,
        target_file_path=target_output_root,
        job_id=job_id,
    )

    # Read back the KEPT predict-seed staging bundle (materialize ingests this
    # dir directly; no serving part-00000.parquet is written by gen on TT-NG).
    staging_dir = ttng_staging_dir(index_path, predict_staging_tt_id(job_id))
    df = pq.read_table(osp.join(staging_dir, "tt.parquet")).to_pandas()
    # Slot-0 ENTITY1/TARGET1 -> ENTITY/TARGET (a temporary alias that
    # disappears once kumo adopts DG's multi-entity schema); mirrors the
    # rename in _stream_to_parquet_ttng so migrated assertions keep working.
    return df.rename(columns={'ENTITY1': 'ENTITY', 'TARGET1': 'TARGET'})


# ---------------------------------------------------------------------------
# Fixture smoke test
# ---------------------------------------------------------------------------


def test_dg_indexed_financial_graph_fixture(dg_indexed_financial_graph):
    graph, index_path = dg_indexed_financial_graph
    assert isinstance(graph, Graph)
    assert os.path.exists(index_path)
    assert os.path.exists(os.path.join(index_path, "_SUCCESS"))


def test_dg_indexed_temporal_financial_graph_fixture(
        dg_indexed_temporal_financial_graph, tmp_path):
    """End-to-end smoke for the TEMPORAL FINANCIAL fixture.

    Parallels :func:`test_dg_indexed_financial_graph_fixture` but additionally
    proves the temporal anchor this sibling adds is real: TRANS carries a
    ``DATE`` time_col with a post-2001 (2023) spread, ACCOUNT stays static, and
    a ``SUM(TRANS.AMOUNT, ...)`` window query + ``DateOffsetSplit`` materializes
    a non-empty train table DG-native through :func:`materialize_task_dg`.
    """
    import pandas as pd
    from kumoapi.typing import TimeUnit

    from kumo.task.split import DateOffsetSplit
    from kumo.testing import get_validated_query

    graph, index_path = dg_indexed_temporal_financial_graph
    assert isinstance(graph, Graph)
    assert os.path.exists(index_path)
    assert os.path.exists(os.path.join(index_path, "_SUCCESS"))

    short_names = {name.split(".")[-1] for name in graph.table_names}
    assert {"ACCOUNT", "TRANS"} <= short_names

    # ACCOUNT is a STATIC entity; TRANS carries the temporal anchor.
    assert graph.table("ACCOUNT").time_col is None
    trans = graph.table("TRANS")
    assert trans.time_col is not None
    assert trans.time_col.name == "DATE"
    assert {fk.name for fk in trans.fkeys} >= {"ACCOUNT_ID"}

    # The DATE anchor must be a real 2023-era timestamp spread (NOT the single
    # degenerate value the shipped FINANCIAL parquet carries, and NOT the 2262
    # pre-2001 sentinel).
    trans_dates = pd.to_datetime(trans.pandas()["DATE"])
    assert trans_dates.dt.year.eq(2023).all()
    assert trans_dates.nunique() > 1

    # stype inferral needs the per-test autouse persist_dir.
    graph.infer_metadata()
    query = "PREDICT SUM(TRANS.AMOUNT, 15, 30) FOR EACH ACCOUNT.ACCOUNT_ID"
    validated_query = get_validated_query(graph, query)
    pquery = PredictiveQueryV2(
        name="temporal_financial_smoke",
        graph=graph,
        validated_predictive_query=validated_query,
        split=DateOffsetSplit([-30, 0], unit=TimeUnit.DAYS, timeframe=30),
    )
    selected_path, job_id, _ = generate_train_table_dg(
        pquery, diskgraph_index_path=index_path, destination=str(tmp_path))
    materialize_selected_task_dg(pquery, selected_path, job_id)
    train_df = pd.read_parquet(selected_path)
    assert len(train_df) > 0
    assert {"ENTITY1", "TARGET1", "TIMESTAMP"} <= set(train_df.columns)


def test_dg_indexed_synthetic_node_pred_graph_fixture(
        dg_indexed_synthetic_node_pred_graph):
    graph, index_path = dg_indexed_synthetic_node_pred_graph
    assert isinstance(graph, Graph)
    assert {"USERS", "USERS_TARGET"
            } <= {name.split(".")[-1]
                  for name in graph.table_names}
    assert os.path.exists(index_path)
    assert os.path.exists(os.path.join(index_path, "_SUCCESS"))


def test_dg_indexed_temporal_synthetic_node_pred_graph_fixture(
        dg_indexed_temporal_synthetic_node_pred_graph):
    graph, index_path = dg_indexed_temporal_synthetic_node_pred_graph
    assert isinstance(graph, Graph)
    assert {"USERS",
            "ORDERS"} <= {name.split(".")[-1]
                          for name in graph.table_names}
    # ORDERS must carry a real temporal anchor for the temporal node-pred
    # query family to produce non-empty train tables.
    orders = graph.table("ORDERS")
    assert orders.time_col is not None
    assert orders.time_col.name == "TIMESTAMP"
    assert os.path.exists(index_path)
    assert os.path.exists(os.path.join(index_path, "_SUCCESS"))


def test_dg_indexed_temporal_link_pred_graph_fixture(
        dg_indexed_temporal_link_pred_graph, tmp_path):
    """End-to-end smoke for the TEMPORAL link-pred FACTORY fixture.

    Builds the graph (verifying the two-fkey ORDERS edge table indexes
    cleanly), runs the default RANK-TOP-3 LP query + ``DateOffsetSplit`` through
    ``materialize_task_dg``, and asserts:
      * USERS / STORES / ORDERS are present and ORDERS carries the temporal
        anchor + two fkeys,
      * train / val / test masks are all non-empty,
      * per-split min/max dates are 2023-era (NOT the 2262 producer-bug
        sentinel) when surfaced.
    """
    import pandas as pd
    from kumoapi.typing import TimeUnit

    from kumo.task.split import DateOffsetSplit
    from kumo.testing import get_validated_query

    graph, index_path = dg_indexed_temporal_link_pred_graph()
    assert isinstance(graph, Graph)
    short_names = {name.split(".")[-1] for name in graph.table_names}
    assert {"USERS", "STORES", "ORDERS"} <= short_names
    assert os.path.exists(index_path)
    assert os.path.exists(os.path.join(index_path, "_SUCCESS"))

    # ORDERS is the two-fkey temporal edge/fact table.
    orders = graph.table("ORDERS")
    assert orders.time_col is not None
    assert orders.time_col.name == "TIMESTAMP"
    assert {fk.name for fk in orders.fkeys} >= {"USER_ID", "STORE_ID"}

    # stype inferral needs the per-test autouse persist_dir.
    graph.infer_metadata()
    query = ("PREDICT LIST_DISTINCT(ORDERS.STORE_ID, 0, 2) RANK TOP 3 "
             "FOR EACH USERS.USER_ID")
    validated_query = get_validated_query(graph, query)
    # The default LP split ``DateOffsetSplit([-30, 0], DAYS, timeframe=30)``
    # over-windows this small fixture: the 15 day-spaced edges span only 14 days
    # (2023-01-01 .. 2023-01-15), so a 30-day timeframe yields ZERO anchor
    # windows and the native-artifact path raises "split stats artifact must
    # include non-empty train_table_stats". A 2-day cadence
    # (``DateOffsetSplit([-2, 0], DAYS, timeframe=2)``) produces the canonical
    # 12-row / 6-timeframe link-pred train table, proving the fixture
    # end-to-end on non-empty splits.
    split = DateOffsetSplit([-2, 0], unit=TimeUnit.DAYS, timeframe=2)
    pquery = PredictiveQueryV2(
        name="TEMPORAL_LINK_PRED_SMOKE_v1",
        graph=graph,
        validated_predictive_query=validated_query,
        split=split,
    )

    task = materialize_task_dg(pquery, index_path, tmp_path)

    # All three split masks must be non-empty (proves the LP train table
    # generated rows end-to-end on the DG-native path).
    train_mask = task.train_table.data.train_mask
    val_mask = task.train_table.data.val_mask
    test_mask = task.train_table.data.test_mask
    assert int(train_mask.sum()) > 0
    assert int(val_mask.sum()) > 0
    assert int(test_mask.sum()) > 0

    # Per-split dates (when surfaced) must be 2023-era, NOT the 2262 sentinel
    # that the read-only manifest split-stats producer emits for pre-2001 ns
    # dates. This is the proven date-corruption tell.
    split_stats = task.train_table.get_train_table_stats(after_split=True)

    def _naive(ts) -> pd.Timestamp:
        ts = pd.Timestamp(ts)
        return ts.tz_localize(None) if ts.tzinfo is not None else ts

    for stage in (split_stats.train_table_stats, split_stats.val_table_stats,
                  split_stats.test_table_stats):
        if stage is None:
            continue
        for date in (stage.min_date, stage.max_date):
            if date is None:
                continue
            ts = _naive(date)
            assert ts.year == 2023, (
                f"expected 2023-era split date, got {ts!r} "
                f"(2262 sentinel => pre-2001 date-corruption bug)")


def test_dg_indexed_entity_dates_graph_fixture(dg_indexed_entity_dates_graph,
                                               tmp_path):
    """End-to-end smoke for the entity-dates FACTORY fixture.

    Builds the graph (verifying USERS carries its own start-time + the TRANS
    two-fkey temporal fact table indexes cleanly), then drives a
    custom-prediction-time node-pred batch table through ``materialize_bp_dg``
    and asserts:
      * USERS / PRODUCTS / TRANS are present; USERS.time_col == JOINED and
        TRANS.time_col == DATE,
      * the index + _SUCCESS marker exist on disk,
      * a custom pred_time + WHERE AGE<30 filter yields the start-time-eligible
        entities {0, 1} at the requested anchor (NOT the 2262 sentinel),
        proving the custom-pred-time codepath runs DG-native.
    """
    import pandas as pd

    from kumo.testing import get_validated_query

    graph, index_path = dg_indexed_entity_dates_graph(set_user_time=True)
    assert isinstance(graph, Graph)
    short_names = {name.split(".")[-1] for name in graph.table_names}
    assert {"USERS", "PRODUCTS", "TRANS"} <= short_names
    assert os.path.exists(index_path)
    assert os.path.exists(os.path.join(index_path, "_SUCCESS"))

    users = graph.table("USERS")
    assert users.time_col is not None
    assert users.time_col.name == "JOINED"
    trans = graph.table("TRANS")
    assert trans.time_col is not None
    assert trans.time_col.name == "DATE"
    assert {fk.name for fk in trans.fkeys} >= {"USER_ID", "PRODUCT_ID"}

    # stype inferral needs the per-test autouse persist_dir.
    graph.infer_metadata()
    query = ("PREDICT SUM(TRANS.AMOUNT, 0, 5) FOR EACH USERS.USER_ID "
             "WHERE USERS.AGE < 30")
    validated_query = get_validated_query(graph, query)
    pred_time = pd.Timestamp("2023-01-22")
    pred_df = materialize_bp_dg(graph, validated_query, index_path, tmp_path,
                                pred_time=pred_time)

    assert {"ENTITY", "TIMESTAMP"} <= set(pred_df.columns)
    # AGE < 30 keeps USERS {0, 1}; both JOINED before the 2023-01-22 anchor, so
    # both are eligible at this start-time-only fixture.
    assert set(pred_df["ENTITY"].tolist()) == {0, 1}
    assert pred_df["TIMESTAMP"].tolist() == [pred_time, pred_time]


def test_dg_indexed_node_pred_bank_graph_fixture(
        dg_indexed_node_pred_bank_graph, tmp_path):
    """End-to-end smoke for the TEMPORAL node-pred BANK FACTORY fixture.

    Exercises BOTH BANK_ID-role variants and proves each produces a non-empty
    DG-native train table:
      * fkey mode (``bank_id_is_col=False``): BANK_ID links TRANSACTIONS ->
        BANKS; run the nested boolean target ``SUM(..)>30 OR COUNT(..)>1``
        and assert the per-entity TARGET multiset,
      * column mode (``bank_id_is_col=True``): BANK_ID is a plain TRANSACTIONS
        column re-typed as an ID column; run the LIST_DISTINCT multilabel
        query and assert the per-entity TARGET.
    """
    import pandas as pd
    from kumo_core.types import TaskType
    from kumoapi.typing import Stype

    from kumo.task.split import TemporalSplit
    from kumo.testing import get_validated_query

    # --- fkey mode: BANK_ID -> BANKS, nested boolean target. ---
    graph, index_path = dg_indexed_node_pred_bank_graph()
    assert isinstance(graph, Graph)
    short_names = {name.split(".")[-1] for name in graph.table_names}
    assert {"USERS", "BANKS", "TRANSACTIONS"} <= short_names
    assert os.path.exists(index_path)
    assert os.path.exists(os.path.join(index_path, "_SUCCESS"))

    trans = graph.table("TRANSACTIONS")
    assert trans.time_col is not None
    assert trans.time_col.name == "DATE"
    assert {fk.name for fk in trans.fkeys} >= {"USER_ID", "BANK_ID"}

    graph.infer_metadata()
    query = ("PREDICT SUM(TRANSACTIONS.AMOUNT, 0, 2) > 30 OR "
             "COUNT(TRANSACTIONS.AMOUNT, 0, 2) > 1 FOR EACH USERS.USER_ID")
    validated_query = get_validated_query(graph, query)
    pquery = PredictiveQueryV2(
        name="NODE_PRED_BANK_SMOKE_FKEY_v1",
        graph=graph,
        validated_predictive_query=validated_query,
    )
    train_table_path, _, _ = generate_train_table_dg(
        pquery, diskgraph_index_path=index_path,
        destination=str(tmp_path / "fkey_home"))
    train_df = pd.read_parquet(train_table_path)
    train_df = train_df.sort_values("ENTITY1").reset_index(drop=True)
    assert train_df["ENTITY1"].tolist() == [0, 1, 2, 3]
    # Real DG output (probed): users 0 and 1 cross the SUM>30/COUNT>1 boolean.
    assert train_df["TARGET1"].tolist() == [True, True, False, False]

    # --- column mode: BANK_ID as an ID column, LIST_DISTINCT multilabel. ---
    graph2, index_path2 = dg_indexed_node_pred_bank_graph(bank_id_is_col=True)
    short_names2 = {name.split(".")[-1] for name in graph2.table_names}
    assert {"USERS", "TRANSACTIONS"} <= short_names2
    assert "BANKS" not in short_names2
    trans2 = graph2.table("TRANSACTIONS")
    assert "BANK_ID" not in {fk.name for fk in trans2.fkeys}
    assert os.path.exists(os.path.join(index_path2, "_SUCCESS"))

    graph2.infer_metadata()
    graph2["TRANSACTIONS", "BANK_ID"].stype = Stype.ID
    graph2["TRANSACTIONS"].infer_encoders()
    assert graph2["TRANSACTIONS", "BANK_ID"].stype == Stype.ID
    query2 = ("PREDICT LIST_DISTINCT(TRANSACTIONS.BANK_ID, 0, 2) CLASSIFY "
              "FOR EACH USERS.USER_ID")
    validated_query2 = get_validated_query(graph2, query2)
    pquery2 = PredictiveQueryV2(
        name="NODE_PRED_BANK_SMOKE_IDCOL_v1",
        graph=graph2,
        validated_predictive_query=validated_query2,
        split=TemporalSplit([0.5, 0.5]),
    )
    assert pquery2.get_task_type() == TaskType.MULTILABEL_CLASSIFICATION
    train_table_path2, _, _ = generate_train_table_dg(
        pquery2, diskgraph_index_path=index_path2,
        destination=str(tmp_path / "idcol_home"))
    train_df2 = pd.read_parquet(train_table_path2)
    train_df2 = train_df2.sort_values("ENTITY1").reset_index(drop=True)
    assert train_df2["ENTITY1"].tolist() == [0, 1, 2, 3]
    # Real DG output (probed): user 2 (no in-window BANK_ID) gets the empty
    # label; the rest distinct-list to '1'.
    assert train_df2["TARGET1"].tolist() == ["1", "1", "", "1"]


# ---------------------------------------------------------------------------
# DG-indexed STATIC link-pred fixture (B9a)
# ---------------------------------------------------------------------------
#
# Why a fifth fixture: the STATIC link-prediction test family (e.g.
# ``PREDICT LIST_DISTINCT(TRANS.PRODUCT_ID_FK) RANK TOP 2 FOR EACH
# USERS.USER_ID``) builds a BESPOKE bipartite purchase graph -- USERS entity x
# PRODUCTS RHS joined by a TRANS fact/edge table -- via the legacy conftest
# helper ``lp_test_graph`` (test/conftest.py). That helper has no DG index, so
# the parked static-LP tests are blocked. This fixture reproduces that graph
# shape DG-natively, following the universal build pattern of every fixture in
# this file (parquet per table -> metadata.json-shaped dict ->
# ``_build_diskgraph_ingest_config`` -> ``diskgraph.ingest`` ->
# ``add_from_config``).
#
# Logical graph (bipartite purchase graph, two-fkey edge table):
#   USERS          pkey=USER_ID, col AGE; optional DATE time_col
#                  (gated by ``set_entity_time_col``)
#   PRODUCTS       pkey=PRODUCT_ID (RHS / ranked target table),
#                  col BRAND (categorical), DATE date col (2023 origin)
#   PRODUCT_BRANDS fkey PRODUCT_ID -> PRODUCTS, cols BRAND / BRAND_VALUE,
#                  DATE date col (feature/dimension table off PRODUCTS)
#   TRANS          pkey=TRANS_ID, fkey USER_ID_FK -> USERS,
#                  fkey PRODUCT_ID_FK -> PRODUCTS, col AMOUNT (numerical),
#                  DATE; time_col=DATE gated by ``set_trans_time_col``
#                  (default True so the static-LP PQL works).
# Representative query (NOT run in the smoke test):
#   PREDICT LIST_DISTINCT(TRANS.PRODUCT_ID_FK) RANK TOP 2
#   FOR EACH USERS.USER_ID
#
# fkeys are named USER_ID_FK / PRODUCT_ID_FK (NOT matching the dst pkey names)
# to exercise explicit fkey mapping -- the ``key_name_match=False`` semantics of
# the legacy ``lp_test_graph``. ``_build_diskgraph_ingest_config`` keys the DG
# edge off ``fk["name"]`` (the src column) + ``fk["dst_table"]`` independently of
# the dst pkey name, so the mismatched names ingest cleanly.
#
# Sizing default: num_entities=3 users x 2 products, each user buying each
# product 15 times (30 trans/user, 90 trans total) so the realized link
# structure is dense and non-empty.


def _static_link_pred_tables(
    num_entities: int = 3,
    missing_link: bool = False,
    dangling_entity: bool = False,
    set_entity_time_col: bool = True,
    set_trans_time_col: bool = True,
    invalid_targets: bool = False,
) -> Dict[str, "object"]:
    """Return ``{table_name: DataFrame}`` for the STATIC link-pred graph.

    Reproduces the legacy ``lp_test_graph`` (test/conftest.py) bipartite
    USERS x PRODUCTS purchase graph at its canonical dense sizing: each of
    ``num_entities`` users buys each of the 2 products 15 times (30 trans per
    user). PRODUCTS is the RHS / ranked target table; TRANS is the two-fkey
    fact/edge table joining USERS and PRODUCTS.

    All date columns use a 2023 origin (NOT the legacy 1974/1970/2018 dates):
    the read-only DG manifest split-stats producer corrupts pre-2001
    nanosecond dates (clamps to the 2262 sentinel). A realistic 2023 origin
    flows correctly. Static-LP rows downstream assert COUNTS, not dates, but
    any date column included must still index cleanly.

    ``invalid_targets`` (ADDITIVE, default False): reproduces the legacy
    ``lp_test_graph_invalid_targets`` conftest helper -- inject TRANS rows
    whose PRODUCT_ID_FK points at a non-existent product id (100) or is NULL
    (NaN), i.e. links with NO matching PRODUCTS pkey that the train-table
    generator must DROP. Per user the 30 TRANS rows become ``[0]*10 + [1]*10
    + [NaN]*5 + [100]*5`` (matching the legacy pattern), so only the valid
    {0, 1} targets survive. Mutually exclusive with ``missing_link`` (both
    rewrite the same PRODUCT_ID_FK column); ``invalid_targets`` wins if both
    are set.
    """
    import numpy as np
    import pandas as pd

    # --- USERS entity table ---
    ages = list(range(17, 17 + num_entities * 8, 8))
    user_ids = list(range(num_entities))
    if dangling_entity:
        # Users with NO transactions (entities that connect to nothing).
        user_ids += [max(user_ids) + 1 + x for x in range(5)]
        ages += [100] * 5
    user_dates = (pd.Timestamp("2023-01-01") + pd.to_timedelta(
        np.arange(len(user_ids)), unit="D")).astype("datetime64[ns]")
    user_cols = {"USER_ID": user_ids, "AGE": ages}
    if set_entity_time_col:
        user_cols["DATE"] = user_dates
    user_df = pd.DataFrame(user_cols)

    # --- PRODUCTS RHS / target table (2 products, matching lp_test_graph) ---
    rhs_df = pd.DataFrame({
        "PRODUCT_ID":
        range(2),
        "BRAND": ["PATAGONIA", "KUMO"],
        "DATE":
        (pd.Timestamp("2023-01-01") +
         pd.to_timedelta(np.arange(2), unit="D")).astype("datetime64[ns]"),
    })

    # --- PRODUCT_BRANDS feature/dimension table hanging off PRODUCTS ---
    # Legacy lp_test_graph adds this table with NO pkey (only the PRODUCT_ID
    # fkey + cols). DG requires every table to declare a unique primary key, so
    # a synthetic per-row PB_ID is added here; PRODUCT_ID stays purely the fkey
    # into PRODUCTS. (Without a distinct pkey, the config builder would emit
    # PRODUCT_ID twice -- once as pkey, once as fkey -- which DG rejects as a
    # duplicate column.)
    product_brands_df = pd.DataFrame({
        "PB_ID":
        list(range(5)),
        "PRODUCT_ID": [0, 1, 0, 0, 1],
        "BRAND": ["PATAGONIA", "KUMO", "PATAGONIA", "PATAGONIA", "KUMO"],
        "BRAND_VALUE": [5, 5, 10, 20, 5],
        "DATE":
        (pd.Timestamp("2023-01-01") +
         pd.to_timedelta(np.arange(5), unit="D")).astype("datetime64[ns]"),
    })

    # --- TRANS fact/edge table: each user buys each product 15 times ---
    # 30 day-spaced timestamps repeated per user (2023 origin), mirroring the
    # legacy 30-row-per-user purchase cadence.
    base_dates = pd.Series(
        pd.Timestamp("2023-01-01") + pd.to_timedelta(np.arange(30), unit="D"))
    dates = pd.concat([base_dates] * num_entities, axis=0, ignore_index=True)
    user_ids_t: List[int] = []
    product_ids_t: List[object] = []
    amounts: List[float] = []
    if invalid_targets:
        # Reproduce the legacy lp_test_graph_invalid_targets pattern: per user
        # 10 trans -> product 0, 10 -> product 1, 5 -> NULL (no pkey), 5 ->
        # product 100 (no matching PRODUCTS pkey). The 100 / NULL links are
        # dangling fkeys the train-table generator must DROP, leaving only the
        # valid {0, 1} targets. (Takes precedence over missing_link: both
        # rewrite this same column.)
        for i in range(num_entities):
            user_ids_t += [i] * 30
            product_ids_t += [0] * 10 + [1] * 10 + [None] * 5 + [100] * 5
            amounts += [1.0, 2.0, 3.0] * 10
    elif missing_link:
        # Drop some user->product links so some entities have no targets.
        for i in range(num_entities):
            user_ids_t += [i] * 30
            if i % 3 == 0:
                product_ids_t += [0] * 15 + [1] * 15
            else:
                product_ids_t += [i % 2] * 30
            amounts += [1.0, 2.0, 3.0] * 10
    else:
        for i in range(num_entities):
            user_ids_t += [i] * 30
            product_ids_t += [0] * 15 + [1] * 15
            amounts += [1.0, 2.0, 3.0] * 10

    product_ids_col: object = product_ids_t
    if invalid_targets:
        # NULL targets force a nullable integer dtype (a plain int list with
        # None would coerce to float64 / object). Int64 keeps 100 as an int
        # (so it stays a clean "no matching pkey" id, not 100.0) while letting
        # the 5 NULL rows ingest as <NA>.
        product_ids_col = pd.array(product_ids_t, dtype="Int64")

    trans_cols = {
        "TRANS_ID": list(range(len(dates))),
        "USER_ID_FK": user_ids_t,
        "PRODUCT_ID_FK": product_ids_col,
        "AMOUNT": amounts,
    }
    # DATE is always written (so the column physically exists for ingest);
    # whether TRANS *uses* it as a time_col is gated by set_trans_time_col in
    # the metadata builder below.
    trans_cols["DATE"] = dates.astype("datetime64[ns]")
    trans_df = pd.DataFrame(trans_cols)

    return {
        "USERS": user_df,
        "PRODUCTS": rhs_df,
        "PRODUCT_BRANDS": product_brands_df,
        "TRANS": trans_df,
    }


def _static_link_pred_metadata(
    set_entity_time_col: bool = True,
    set_trans_time_col: bool = True,
) -> dict:
    """metadata.json-shaped dict for the STATIC link-pred graph.

    Same shape every fixture in this file uses, so it drives BOTH the DG
    ingest config (:func:`_build_diskgraph_ingest_config`) and the in-memory
    kumo.Graph (``graph.add_from_config``).

    TRANS declares TWO fkeys (USER_ID_FK -> USERS, PRODUCT_ID_FK -> PRODUCTS)
    -- the same two-fkey-on-one-table shape the temporal-LP ORDERS table
    already proved works with ``_build_diskgraph_ingest_config`` (it iterates
    ``tbl['fkeys']``). PRODUCTS (RHS) and PRODUCT_BRANDS (feature table) carry
    fkeys/cols but NO time_col, exercising the config builder's
    fkey-without-time_col branch.
    """
    def col(name: str, dtype: str) -> dict:
        return {
            "name": name,
            "stype": None,
            "dtype": dtype,
            "na_values": None,
            "metadata": {},
            "encoder": None,
            "stats": {},
        }

    user_cols = [col("AGE", "int")]
    users = {
        "table_name": "USERS",
        "data_source": None,
        "data_version": None,
        "pkey": {
            "name": "USER_ID",
            "stype": None,
            "dtype": "int",
            "na_values": None,
            "metadata": {},
            "drop_duplicates": False,
        },
        "fkeys": [],
        "cols": user_cols,
        # USERS.DATE time_col gated by set_entity_time_col (column-config dict
        # form; _time_col_name accepts either form and add_from_config requires
        # the dict).
        "time_col": col("DATE", "time") if set_entity_time_col else None,
        "source_name": None,
        "stats": {},
    }
    products = {
        "table_name": "PRODUCTS",
        "data_source": None,
        "data_version": None,
        "pkey": {
            "name": "PRODUCT_ID",
            "stype": None,
            "dtype": "int",
            "na_values": None,
            "metadata": {},
            "drop_duplicates": False,
        },
        "fkeys": [],
        "cols": [col("BRAND", "string"),
                 col("DATE", "date")],
        # RHS / target table: NO time_col (it is the ranked target, not a
        # temporal anchor).
        "time_col": None,
        "source_name": None,
        "stats": {},
    }
    product_brands = {
        "table_name":
        "PRODUCT_BRANDS",
        "data_source":
        None,
        "data_version":
        None,
        # Legacy lp_test_graph declares this dimension table with only
        # fkeys+cols (no pkey). DG requires a unique primary key per table, so a
        # synthetic per-row PB_ID is used here; PRODUCT_ID stays purely the fkey
        # into PRODUCTS (avoids the duplicate-column error a PRODUCT_ID pkey
        # would trigger when it is also a fkey).
        "pkey": {
            "name": "PB_ID",
            "stype": None,
            "dtype": "int",
            "na_values": None,
            "metadata": {},
            "drop_duplicates": False,
        },
        "fkeys": [
            {
                "name": "PRODUCT_ID",
                "stype": None,
                "dtype": "int",
                "na_values": None,
                "metadata": {},
                "dst_table": "PRODUCTS",
                "dst_key": None,
                "dst_data_source": None,
                "stats": {},
            },
        ],
        "cols": [
            col("BRAND", "string"),
            col("BRAND_VALUE", "int"),
            col("DATE", "date")
        ],
        "time_col":
        None,
        "source_name":
        None,
        "stats": {},
    }
    trans = {
        "table_name":
        "TRANS",
        "data_source":
        None,
        "data_version":
        None,
        "pkey": {
            "name": "TRANS_ID",
            "stype": None,
            "dtype": "int",
            "na_values": None,
            "metadata": {},
            "drop_duplicates": False,
        },
        "fkeys": [
            {
                "name": "USER_ID_FK",
                "stype": None,
                "dtype": "int",
                "na_values": None,
                "metadata": {},
                "dst_table": "USERS",
                "dst_key": None,
                "dst_data_source": None,
                "stats": {},
            },
            {
                "name": "PRODUCT_ID_FK",
                "stype": None,
                "dtype": "int",
                "na_values": None,
                "metadata": {},
                "dst_table": "PRODUCTS",
                "dst_key": None,
                "dst_data_source": None,
                "stats": {},
            },
        ],
        "cols": [col("AMOUNT", "float")],
        # TRANS.DATE time_col gated by set_trans_time_col (default True so the
        # static-LP PQL works). When False, DATE is added to the DG column list
        # by _build_diskgraph_ingest_config only if referenced; declare it as a
        # regular col in that case so it is still ingested.
        "time_col":
        col("DATE", "time") if set_trans_time_col else None,
        "source_name":
        None,
        "stats": {},
    }
    if not set_trans_time_col:
        trans["cols"].append(col("DATE", "date"))
    return {
        "name": "STATIC_LINK_PRED",
        "data_source_dict": {},
        "tables": [users, products, product_brands, trans],
    }


def _build_static_link_pred_graph(
    tmp_path_factory,
    num_entities: int = 3,
    missing_link: bool = False,
    dangling_entity: bool = False,
    set_entity_time_col: bool = True,
    set_trans_time_col: bool = True,
    invalid_targets: bool = False,
) -> Tuple[Graph, str]:
    """Build + DG-index the STATIC link-pred graph; return ``(graph,
    index_path)`` in the SAME shape as every other fixture in this file.

    Mirrors the universal build pattern: write parquet per table, build a DG
    ingest config from a metadata.json-shaped dict, call the wheel's
    ``rust_ingest`` directly, then build the matching in-memory kumo.Graph via
    ``add_from_config`` against the same metadata so the schemas line up 1:1.
    """
    from diskgraph import ingest as rust_ingest

    parquet_dir = tmp_path_factory.mktemp("dg_static_link_pred_parquet")
    tables = _static_link_pred_tables(
        num_entities=num_entities,
        missing_link=missing_link,
        dangling_entity=dangling_entity,
        set_entity_time_col=set_entity_time_col,
        set_trans_time_col=set_trans_time_col,
        invalid_targets=invalid_targets,
    )
    for name, df in tables.items():
        df.to_parquet(osp.join(str(parquet_dir), f"{name}.parquet"),
                      index=False)

    metadata = _static_link_pred_metadata(
        set_entity_time_col=set_entity_time_col,
        set_trans_time_col=set_trans_time_col,
    )
    config = _build_diskgraph_ingest_config(str(parquet_dir), metadata)

    index_dir = tmp_path_factory.mktemp("dg_indexed_static_link_pred_idx")
    rust_ingest(
        json.dumps(config),
        str(index_dir),
        col_stats=True,
        cat_count_strategy="hybrid",
        success_marker_content=json.dumps(
            {"source": "dg_indexed_static_link_pred_graph fixture"}),
    )

    meta_path = osp.join(str(index_dir), "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f)

    graph = Graph(name="STATIC_LINK_PRED")
    graph.connect(ParquetConnector("parquet", str(parquet_dir)))
    graph.add_from_config(meta_path, data_source="parquet")
    return graph, str(index_dir)


@pytest.fixture(scope="session")
def dg_indexed_static_link_pred_graph(tmp_path_factory):
    """FACTORY fixture: returns a callable that builds + DG-indexes a STATIC
    link-pred kumo.Graph and yields ``(graph, index_path)`` in the SAME shape
    as :func:`dg_indexed_financial_graph` / the other fixtures.

    Like :func:`dg_indexed_temporal_link_pred_graph`, this is a FACTORY so
    callers can pass sizing / variant flags::

        def test_x(dg_indexed_static_link_pred_graph, tmp_path):
            graph, index_path = dg_indexed_static_link_pred_graph()
            # or, for variants:
            graph, index_path = dg_indexed_static_link_pred_graph(
                missing_link=True, dangling_entity=True)

    Models the legacy conftest ``lp_test_graph`` bipartite USERS x PRODUCTS
    purchase graph: USERS entity + PRODUCTS RHS + PRODUCT_BRANDS feature table
    + TRANS two-fkey fact/edge table (USER_ID_FK -> USERS,
    PRODUCT_ID_FK -> PRODUCTS, key_name_match=False). With num_entities=3 and
    each user buying each of 2 products 15 times the realized link structure is
    dense and non-empty for ``PREDICT LIST_DISTINCT(TRANS.PRODUCT_ID_FK)
    RANK TOP 2 FOR EACH USERS.USER_ID``.

    Flags (all defaulted so a no-arg call yields the canonical dense graph):
      * ``num_entities`` -- number of USERS (default 3).
      * ``missing_link`` -- drop some user->product links so some entities have
        no targets.
      * ``dangling_entity`` -- include users with zero transactions.
      * ``set_entity_time_col`` -- add USERS.DATE as a time col (default True).
      * ``set_trans_time_col`` -- set TRANS.DATE as the time_col
        (default True so the static-LP PQL works).
      * ``invalid_targets`` -- inject TRANS rows whose PRODUCT_ID_FK is a
        non-existent product id (100) or NULL (dangling fkeys the train-table
        generator must DROP), reproducing the legacy
        ``lp_test_graph_invalid_targets`` conftest helper (default False).
    """
    def _factory(
        num_entities: int = 3,
        missing_link: bool = False,
        dangling_entity: bool = False,
        set_entity_time_col: bool = True,
        set_trans_time_col: bool = True,
        invalid_targets: bool = False,
    ) -> Tuple[Graph, str]:
        return _build_static_link_pred_graph(
            tmp_path_factory,
            num_entities=num_entities,
            missing_link=missing_link,
            dangling_entity=dangling_entity,
            set_entity_time_col=set_entity_time_col,
            set_trans_time_col=set_trans_time_col,
            invalid_targets=invalid_targets,
        )

    return _factory


def test_dg_indexed_static_link_pred_graph_fixture(
        dg_indexed_static_link_pred_graph):
    """Smoke test for the STATIC link-pred FACTORY fixture.

    Calls the factory with defaults and asserts the bipartite purchase graph
    indexes cleanly:
      * USERS / PRODUCTS / PRODUCT_BRANDS / TRANS are all present,
      * TRANS carries the temporal anchor (DATE) + its two fkeys
        (USER_ID_FK -> USERS, PRODUCT_ID_FK -> PRODUCTS),
      * a non-empty DG index path with a ``_SUCCESS`` marker exists.

    Does NOT validate PQL or stypes (columns are ``stype=None`` by design --
    see the module docstring); tests that need PQL call ``infer_metadata()``.
    """
    graph, index_path = dg_indexed_static_link_pred_graph()
    assert isinstance(graph, Graph)
    short_names = {name.split(".")[-1] for name in graph.table_names}
    assert {"USERS", "PRODUCTS", "PRODUCT_BRANDS", "TRANS"} <= short_names
    assert os.path.exists(index_path)
    assert os.path.exists(os.path.join(index_path, "_SUCCESS"))

    # TRANS is the two-fkey fact/edge table with key_name_match=False fkeys.
    trans = graph.table("TRANS")
    assert trans.time_col is not None
    assert trans.time_col.name == "DATE"
    assert set(trans.fkey_names) >= {"USER_ID_FK", "PRODUCT_ID_FK"}

    # PRODUCTS is the RHS / ranked target table; PRODUCT_BRANDS hangs off it.
    assert "BRAND" in graph.table("PRODUCTS").col_names
    product_brands = graph.table("PRODUCT_BRANDS")
    assert set(product_brands.fkey_names) >= {"PRODUCT_ID"}


# ---------------------------------------------------------------------------
# JOINT static node-pred graph (single entity+target table) + DiskGraph index.
# ---------------------------------------------------------------------------
#
# Why a dedicated fixture: the SEPARATE-entity/target synthetic fixture above
# (``dg_indexed_synthetic_node_pred_graph``) only supports queries of the shape
# ``PREDICT USERS_TARGET.<col> FOR EACH USERS.USER_ID`` (the target lives on a
# distinct table). A whole cluster of legacy node-pred tests query a column on
# the ENTITY table itself -- ``PREDICT USERS.STATUS FOR EACH USERS.USER_ID``,
# ``PREDICT USERS.PAYMENTS ...``, ``PREDICT USERS.STATUS_BOOL ...``,
# ``PREDICT USERS.MULTI_STATUS ...`` and nested-boolean / MemberOp variants.
# Those need the JOINT shape: a single USERS table carrying both the entity
# pkey and the target columns, with NO time_col (static). This is the DG-native
# rebuild of ``static_node_pred_test_graph(mode='static')`` in
# ``test/conftest.py`` -- identical USER_ID 0..3 rows / column values, only the
# producer changes (DG-native instead of the retired legacy generic path).
#
# Logical graph (single joint entity/target table):
#   USERS  pkey=USER_ID, cols PAYMENTS / STATUS / STATUS_BOOL / target /
#          MULTI_STATUS, NO time_col (static)
# Representative query: PREDICT USERS.STATUS FOR EACH USERS.USER_ID


def _static_node_pred_joint_tables() -> Dict[str, "object"]:
    """Return ``{table_name: DataFrame}`` for the JOINT static node-pred graph.

    Reproduces ``static_node_pred_test_graph(mode='static')`` (test/conftest.py)
    verbatim: a single USERS table, USER_ID 0..3, STATUS ``['A','B',NA,'B']``,
    target mirroring STATUS, PAYMENTS ``[10,20,40,80]``, STATUS_BOOL, and a
    comma-joined MULTI_STATUS multicategorical column. No DATE / time_col (the
    ``mode='static'`` variant drops it).
    """
    import pandas as pd

    users_df = pd.DataFrame({
        "USER_ID": [0, 1, 2, 3],
        "STATUS": ["A", "B", pd.NA, "B"],
        "target": ["A", "B", pd.NA, "B"],
        "PAYMENTS": [10.0, 20.0, 40.0, 80.0],
        "STATUS_BOOL": [True, False, False, True],
        "MULTI_STATUS": ["A,B", "C,D", pd.NA, "A,B,C"],
    })
    return {"USERS": users_df}


def _static_node_pred_joint_metadata() -> dict:
    """metadata.json-shaped dict for the JOINT static node-pred graph.

    Same shape the FINANCIAL / synthetic fixtures use so it drives BOTH the DG
    ingest config (:func:`_build_diskgraph_ingest_config`) and the in-memory
    kumo.Graph (``graph.add_from_config``). USERS is a static node table (pkey +
    feature/target cols, no time_col), so a static node-pred query
    (``PREDICT USERS.STATUS FOR EACH USERS.USER_ID``) under a ``RandomSplit``
    materializes one training row per entity.
    """
    def col(name: str, dtype: str) -> dict:
        return {
            "name": name,
            "stype": None,
            "dtype": dtype,
            "na_values": None,
            "metadata": {},
            "encoder": None,
            "stats": {},
        }

    users = {
        "table_name":
        "USERS",
        "data_source":
        None,
        "data_version":
        None,
        "pkey": {
            "name": "USER_ID",
            "stype": None,
            "dtype": "int",
            "na_values": None,
            "metadata": {},
            "drop_duplicates": False,
        },
        "fkeys": [],
        "cols": [
            col("STATUS", "string"),
            col("target", "string"),
            col("PAYMENTS", "float"),
            col("STATUS_BOOL", "bool"),
            col("MULTI_STATUS", "string"),
        ],
        "time_col":
        None,
        "source_name":
        None,
        "stats": {},
    }
    return {
        "name": "STATIC_NODE_PRED_JOINT",
        "data_source_dict": {},
        "tables": [users],
    }


def _build_static_node_pred_joint_graph(tmp_path_factory) -> Tuple[Graph, str]:
    """Build + DG-index the JOINT static node-pred graph; return
    ``(graph, index_path)`` in the SAME shape as every other fixture here.

    Mirrors the universal build pattern: write parquet per table, build a DG
    ingest config from a metadata.json-shaped dict, call the wheel's
    ``rust_ingest`` directly, then build the matching in-memory kumo.Graph via
    ``add_from_config`` against the same metadata so the schemas line up 1:1.
    The MULTI_STATUS column is marked multicategorical on the in-memory graph
    after ``add_from_config`` (mirrors the conftest fixture).
    """
    from diskgraph import ingest as rust_ingest
    from kumoapi.typing import Stype

    parquet_dir = tmp_path_factory.mktemp("dg_static_node_pred_joint_parquet")
    tables = _static_node_pred_joint_tables()
    for name, df in tables.items():
        df.to_parquet(osp.join(str(parquet_dir), f"{name}.parquet"),
                      index=False)

    metadata = _static_node_pred_joint_metadata()
    config = _build_diskgraph_ingest_config(str(parquet_dir), metadata)

    index_dir = tmp_path_factory.mktemp("dg_indexed_static_node_pred_joint_idx")
    rust_ingest(
        json.dumps(config),
        str(index_dir),
        col_stats=True,
        cat_count_strategy="hybrid",
        success_marker_content=json.dumps(
            {"source": "dg_indexed_static_node_pred_graph fixture"}),
    )

    meta_path = osp.join(str(index_dir), "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f)

    graph = Graph(name="STATIC_NODE_PRED_JOINT")
    graph.connect(ParquetConnector("parquet", str(parquet_dir)))
    graph.add_from_config(meta_path, data_source="parquet")
    graph["USERS", "MULTI_STATUS"].stype = Stype.multicategorical
    graph["USERS", "MULTI_STATUS"].multicategorical_delimiter = ","
    return graph, str(index_dir)


@pytest.fixture(scope="session")
def dg_indexed_static_node_pred_graph(tmp_path_factory) -> Tuple[Graph, str]:
    """JOINT static node-pred kumo.Graph + DiskGraph index, built once per
    session. Returns ``(graph, index_path)`` in the SAME shape as
    :func:`dg_indexed_synthetic_node_pred_graph`.

    DG-native rebuild of ``static_node_pred_test_graph(mode='static')``: one
    USERS table (USER_ID 0..3) carrying STATUS / target / PAYMENTS / STATUS_BOOL
    / MULTI_STATUS and NO time_col. Supports entity-table-target queries such as
    ``PREDICT USERS.STATUS FOR EACH USERS.USER_ID`` under a ``RandomSplit``
    (the DG static node-pred producer path). Tests that need PQL-validatable
    stypes must call ``graph.infer_metadata()`` themselves (stype inferral needs
    the per-test autouse persist_dir).
    """
    return _build_static_node_pred_joint_graph(tmp_path_factory)


def test_dg_indexed_static_node_pred_graph_fixture(
        dg_indexed_static_node_pred_graph):
    """Smoke test for the JOINT static node-pred fixture."""
    graph, index_path = dg_indexed_static_node_pred_graph
    assert isinstance(graph, Graph)
    short_names = {name.split(".")[-1] for name in graph.table_names}
    assert "USERS" in short_names
    users = graph.table("USERS")
    assert users.time_col is None
    assert {"STATUS", "PAYMENTS", "STATUS_BOOL",
            "MULTI_STATUS"} <= set(users.col_names)
    assert os.path.exists(index_path)
    assert os.path.exists(os.path.join(index_path, "_SUCCESS"))


# ---------------------------------------------------------------------------
# INVALID static node-pred graph (separate entity + DUPLICATE-pkey target
# table) + DiskGraph index.
# ---------------------------------------------------------------------------
#
# DG-native rebuild of ``static_node_pred_invalid_test_graph(mode='duplicate',
# user_has_pkey=False, ...)`` (test/pquery/conftest.py). The TARGET table
# (USERS_TARGET) carries DUPLICATE primary keys (each USER_ID appears twice),
# optionally with NA pkeys or empty-string targets. These graphs exercise the
# producer's de-duplication + NA/empty entity-key dropping for
# ``PREDICT USERS_TARGET.STATUS FOR EACH USERS_TARGET.USER_ID``.
#
# DG ingest tolerates the duplicate pkeys via ``drop_duplicates=True`` on the
# pkey spec (probed: the producer de-dups to the unique surviving entities,
# matching the legacy assertions: dup -> 3 unique rows, nan -> 2 rows w/ no
# null entity, empty -> 3 rows w/ no '' target).
#
# Logical graph:
#   USERS         fkey=USER_ID -> USERS_TARGET, col PAYMENTS, no time_col
#   USERS_TARGET  pkey=USER_ID (DUPLICATED rows), col STATUS, no time_col


def _invalid_static_node_pred_tables(
    target_key_has_nones: bool,
    target_key_has_empty_strings: bool,
) -> Dict[str, "object"]:
    """Return ``{table_name: DataFrame}`` for the INVALID static node-pred
    graph (separate entity + duplicate-pkey target), mirroring
    ``static_node_pred_invalid_test_graph(mode='duplicate')``.
    """
    import pandas as pd

    user_ids = [0, 1, 2, pd.NA] if target_key_has_nones else [0, 1, 2, 3]
    status_values = ["A", "B", pd.NA, "B"]
    if target_key_has_empty_strings:
        status_values = ["A", "B", "", "B"]

    entity_df = pd.DataFrame({
        "USER_ID": [0, 1, 2, 3],
        "PAYMENTS": [10.0, 20.0, 40.0, 80.0],
    })
    # Target table has DUPLICATE rows (each entity twice).
    target_df = pd.DataFrame({
        "USER_ID": user_ids * 2,
        "STATUS": status_values * 2,
    })
    return {"USERS": entity_df, "USERS_TARGET": target_df}


def _invalid_static_node_pred_metadata() -> dict:
    """metadata.json-shaped dict for the INVALID static node-pred graph.

    USERS is a pure fact/entity table with an ``USER_ID`` fkey -> USERS_TARGET
    (``user_has_pkey=False``); USERS_TARGET is the pkey/target table whose
    pkey spec sets ``drop_duplicates=True`` so DG ingest accepts the
    duplicated rows and the producer de-dups to unique entities.
    """
    def col(name: str, dtype: str) -> dict:
        return {
            "name": name,
            "stype": None,
            "dtype": dtype,
            "na_values": None,
            "metadata": {},
            "encoder": None,
            "stats": {},
        }

    users = {
        "table_name":
        "USERS",
        "data_source":
        None,
        "data_version":
        None,
        "pkey":
        None,
        "fkeys": [{
            "name": "USER_ID",
            "stype": None,
            "dtype": "int",
            "na_values": None,
            "metadata": {},
            "dst_table": "USERS_TARGET",
            "dst_key": None,
            "dst_data_source": None,
            "stats": {},
        }],
        "cols": [col("PAYMENTS", "float")],
        "time_col":
        None,
        "source_name":
        None,
        "stats": {},
    }
    users_target = {
        "table_name": "USERS_TARGET",
        "data_source": None,
        "data_version": None,
        "pkey": {
            "name": "USER_ID",
            "stype": None,
            "dtype": "int",
            "na_values": None,
            "metadata": {},
            "drop_duplicates": True,
        },
        "fkeys": [],
        "cols": [col("STATUS", "string")],
        "time_col": None,
        "source_name": None,
        "stats": {},
    }
    return {
        "name": "INVALID_STATIC_NODE_PRED",
        "data_source_dict": {},
        "tables": [users, users_target],
    }


def _build_invalid_static_node_pred_graph(
    tmp_path_factory,
    target_key_has_nones: bool,
    target_key_has_empty_strings: bool,
) -> Tuple[Graph, str]:
    """Build + DG-index the INVALID static node-pred graph; return
    ``(graph, index_path)`` in the SAME shape as every other fixture here.
    """
    from diskgraph import ingest as rust_ingest

    parquet_dir = tmp_path_factory.mktemp("dg_invalid_static_node_pred_parquet")
    tables = _invalid_static_node_pred_tables(target_key_has_nones,
                                              target_key_has_empty_strings)
    for name, df in tables.items():
        df.to_parquet(osp.join(str(parquet_dir), f"{name}.parquet"),
                      index=False)

    metadata = _invalid_static_node_pred_metadata()
    config = _build_diskgraph_ingest_config(str(parquet_dir), metadata)

    index_dir = tmp_path_factory.mktemp("dg_indexed_invalid_static_idx")
    rust_ingest(
        json.dumps(config),
        str(index_dir),
        col_stats=True,
        cat_count_strategy="hybrid",
        success_marker_content=json.dumps(
            {"source": "dg_indexed_invalid_static_node_pred_graph fixture"}),
    )

    meta_path = osp.join(str(index_dir), "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f)

    graph = Graph(name="INVALID_STATIC_NODE_PRED")
    graph.connect(ParquetConnector("parquet", str(parquet_dir)))
    graph.add_from_config(meta_path, data_source="parquet")
    graph.infer_links()
    return graph, str(index_dir)


@pytest.fixture(scope="session")
def dg_indexed_invalid_static_node_pred_graph(tmp_path_factory):
    """FACTORY fixture: returns a callable that builds + DG-indexes the INVALID
    static node-pred graph and yields ``(graph, index_path)``.

    DG-native rebuild of ``static_node_pred_invalid_test_graph(mode='duplicate',
    user_has_pkey=False)``. Flags mirror the legacy helper::

        graph, index_path = dg_indexed_invalid_static_node_pred_graph()
        graph, index_path = dg_indexed_invalid_static_node_pred_graph(
            target_key_has_nones=True)
        graph, index_path = dg_indexed_invalid_static_node_pred_graph(
            target_key_has_empty_strings=True)
    """
    def _factory(
        target_key_has_nones: bool = False,
        target_key_has_empty_strings: bool = False,
    ) -> Tuple[Graph, str]:
        return _build_invalid_static_node_pred_graph(
            tmp_path_factory,
            target_key_has_nones=target_key_has_nones,
            target_key_has_empty_strings=target_key_has_empty_strings,
        )

    return _factory


def test_dg_indexed_invalid_static_node_pred_graph_fixture(
        dg_indexed_invalid_static_node_pred_graph):
    """Smoke test for the INVALID static node-pred FACTORY fixture."""
    graph, index_path = dg_indexed_invalid_static_node_pred_graph()
    assert isinstance(graph, Graph)
    short_names = {name.split(".")[-1] for name in graph.table_names}
    assert {"USERS", "USERS_TARGET"} <= short_names
    assert os.path.exists(index_path)
    assert os.path.exists(os.path.join(index_path, "_SUCCESS"))
