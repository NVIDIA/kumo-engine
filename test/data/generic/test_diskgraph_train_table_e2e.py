# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end harness for diskgraph TT-gen against multicategorical sources.

Drives the kumo PQL → ``PredictiveQueryValidator`` → ``DiskGraphPlanCompiler``
→ ``execute_tt_plan_with_artifacts`` chain through a real diskgraph wheel
against small synthetic parquet, parametrized over the source-column on-disk
shape, with per-entity TARGET assertions. Runs in seconds, not minutes. No
Temporal dev server required.

Why this layer is new:

- Existing ``test_diskgraph_*`` files in ``test/data/`` and ``test/engine/``
  mock the wheel — they do not exercise per-token observation or the
  array-output writer arm against real parquet.
- ``test/rest/integ/test_diskgraph_e2e.py`` runs the real wheel, but only
  through the full Temporal-backed REST harness against a single hardcoded
  PQL (regression task, no multicategorical column). 5–15 min per run;
  assertions are coarse (training success + RMSE bounds), not per-entity
  TARGET tokens.
- ``test/pquery/train_table_generator/test_nodepred_train_generator.py``
  covers generator behavior without exercising the DiskGraph wheel.

This file establishes the missing unit-scale e2e layer: real wheel, small
synthetic parquet, parametrize over the source-shape axis, assertions on
per-entity TARGET tokens.

Canonical fixture:

    user 0:  ts=100 tags=[a,b], ts=200 tags=[b,c]
    user 1:  ts=300 tags=[a],   ts=400 tags=[d,e]
    user 2:  no orders

Anchor at ts=0, window [0, 5000] seconds → all 4 orders included.
Per-user distinct token sets after LIST_DISTINCT:

    user 0 → {a, b, c}  (b appears twice across two rows)
    user 1 → {a, d, e}
    user 2 → {}         (no entry under explode mode)

Source-shape axis covers the three on-disk shapes a multicategorical column
can take:

- ``utf8_delim``   — comma-joined ``string``; ingest's
  ``GraphConfig::normalize_multicat_pseudo_list`` flips it to StringArray at
  convert time. Exercises the convert-time normalization end-to-end.
- ``string_array`` — native ``list<utf8>``.
- ``int64_array``  — native ``list<int64>`` under the bijection
  a=1, b=2, c=3, d=4, e=5; tokens stringify to "1".."5" at ingest.

Run with::

    source ~/work/kumo/venv/bin/activate
    cd ~/work/kumo
    pytest test/data/generic/test_diskgraph_train_table_e2e.py -v
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from kumo_core.types import TaskType
from kumoapi.typing import Stype, TimeUnit

from kumo.connector.parquet_connector import ParquetConnector
from kumo.data.generic.diskgraph_staging import ensure_stage_ingested
from kumo.data.generic.diskgraph_train_table import (
    generate_train_table_via_diskgraph_ttng,
    persist_durable_ttng_bundle,
)
from kumo.graph.graph import Graph
from kumo.testing import get_validated_query

# NOTE on conftest's autouse ``run_before_and_after_tests`` fixture:
# we let it run rather than opting out via ``auto_setup_kumo_backend``. The
# autouse fixture wires up the data backend ``persist_dir`` that
# ``Graph.add`` needs at construction time (HybridBackend's __init_subclass__
# calls ``osp.join(get_persist_dir(), CACHE_DIR)``); skipping it leaves
# ``get_persist_dir()`` returning ``None`` and the graph build crashes with
# ``TypeError: expected str, bytes or os.PathLike object, not NoneType``.

# ---------------------------------------------------------------------------
# Canonical multicat fixture (token bijection + per-row layout)
# ---------------------------------------------------------------------------

# String → int bijection used to encode the int64_array shape. Decoded back
# via the inverse map for cross-shape parity assertions.
MULTICAT_TOKEN_INT_MAP: Dict[str, int] = {
    "a": 1,
    "b": 2,
    "c": 3,
    "d": 4,
    "e": 5
}
MULTICAT_INT_TO_TOKEN: Dict[int, str] = {
    v: k
    for k, v in MULTICAT_TOKEN_INT_MAP.items()
}

# 4 orders, 3 users (user 2 has zero orders → empty target set).
TAGS_PER_ROW: List[List[str]] = [
    ["a", "b"],  # row 0: user 0, ts=100
    ["b", "c"],  # row 1: user 0, ts=200
    ["a"],  # row 2: user 1, ts=300
    ["d", "e"],  # row 3: user 1, ts=400
]
ORDER_USER_IDS: List[int] = [0, 0, 1, 1]
ORDER_TIMESTAMPS: List[int] = [100, 200, 300, 400]
USER_IDS: List[int] = [0, 1, 2]

# Per-user expected DISTINCT token set under LIST_DISTINCT semantics.
# Node-pred multilabel-classification dedups tokens per entity at the writer
# (the per-row-frequency information is preserved separately in col_stats's
# MULTI_CATEGORY_COUNTS, asserted at the diskgraph layer).
# Keys omit user 2 because exploded mode emits no row for an entity with empty
# target; array mode includes the empty row but the per-token assertion only
# checks non-empty entities.
EXPECTED_DISTINCT_TOKENS_PER_ENTITY: Dict[int, List[str]] = {
    0: ["a", "b", "c"],
    1: ["a", "d", "e"],
}

# Cross-entity per-token row count (a appears in both users' source rows once
# each → 2; b appears once for user 0 → 1; etc.). Used by the col_stats
# MULTI_CATEGORY_COUNTS assertion.
EXPECTED_CROSS_ENTITY_TOKEN_COUNTS: Dict[str, int] = {
    "a": 2,
    "b": 1,
    "c": 1,
    "d": 1,
    "e": 1,
}

# ---------------------------------------------------------------------------
# Fixture builder — parquet + Graph
# ---------------------------------------------------------------------------


def _write_users_parquet(path: Path) -> None:
    pq.write_table(
        pa.table({"user_id": pa.array(USER_IDS, type=pa.int64())}),
        str(path),
    )


def _write_orders_parquet(path: Path, shape: str) -> None:
    """Write the orders parquet for the requested source shape.

    ``shape``:
      - ``"utf8_delim"``   — tags as comma-joined ``string``
      - ``"string_array"`` — tags as ``list<utf8>``
      - ``"int64_array"``  — tags as ``list<int64>`` under the bijection
    """
    if shape == "utf8_delim":
        tags_arr = pa.array(
            [",".join(toks) for toks in TAGS_PER_ROW],
            type=pa.string(),
        )
    elif shape == "string_array":
        tags_arr = pa.array(TAGS_PER_ROW, type=pa.list_(pa.string()))
    elif shape == "int64_array":
        tags_arr = pa.array(
            [[MULTICAT_TOKEN_INT_MAP[t] for t in toks]
             for toks in TAGS_PER_ROW],
            type=pa.list_(pa.int64()),
        )
    else:
        raise ValueError(f"unknown shape: {shape!r}")

    table = pa.table({
        "order_id":
        pa.array(list(range(len(ORDER_USER_IDS))), type=pa.int64()),
        "user_id":
        pa.array(ORDER_USER_IDS, type=pa.int64()),
        "ts":
        pa.array(ORDER_TIMESTAMPS, type=pa.int64()),
        "tags":
        tags_arr,
    })
    pq.write_table(table, str(path))


def _build_kumo_graph(data_dir: Path, shape: str) -> Graph:
    """Build a programmatic Graph over the multicat fixture parquets."""
    _write_users_parquet(data_dir / "users.parquet")
    _write_orders_parquet(data_dir / "orders.parquet", shape)

    graph = Graph(name=f"multicat_e2e_{shape}")
    graph.connect(ParquetConnector("local", root_dir=str(data_dir)))
    graph.add("users", pkey="user_id")
    graph.add("orders", cols=["tags"], time_col="ts", pkey="order_id")
    graph.link(src_table="orders", src_key="user_id", dst_table="users")

    graph = graph.infer_metadata()

    # Override the inferred stype: ingest must persist
    # `semantic_type=multicategorical` into `<index>/diskgraph_config.json`
    # so TT-gen can dispatch per-token observation at the read sites.
    graph["orders", "tags"].stype = Stype.multicategorical
    if shape == "utf8_delim":
        graph["orders", "tags"].multicategorical_delimiter = ","
    return graph


# ---------------------------------------------------------------------------
# Link-prediction fixture (FK-target, exercises the array-output writer arm
# via the kumo-PQL-reachable path)
# ---------------------------------------------------------------------------
#
# Multicat-source per-token observe and the array-output writer
# (`PredictionType::Link { explode: false }`) are separate diskgraph code
# paths. The multicat arm is not reachable via kumo PQL: get_rhs_lp_table
# rejects a non-FK link-pred target. What customers can reach is
# `LIST_DISTINCT(orders.product_id)` over an FK column, which exercises the
# `List<Utf8>` array-output writer arm (not multicat per-token observe). This
# fixture covers that path so the harness validates both writer arms reachable
# from kumo PQL.


def _build_linkpred_graph(data_dir: Path) -> Graph:
    """Build a 3-table graph for link-pred with FK target.

    users → orders → products (FK link)

    Same canonical 3-user / 4-order layout as the multicat fixture, but with
    `tags` replaced by `product_id` referencing a 5-product table. Distinct
    products per user under the canonical fixture:

        user 0 (orders 0, 1) → {p_a, p_b, p_c}  (4 product refs)
        user 1 (orders 2, 3) → {p_a, p_d, p_e}
        user 2: no orders → {}
    """
    products = pa.table({
        "product_id":
        pa.array(["p_a", "p_b", "p_c", "p_d", "p_e"], type=pa.string())
    })
    pq.write_table(products, str(data_dir / "products.parquet"))

    _write_users_parquet(data_dir / "users.parquet")

    # Mirror the canonical token layout but as FK refs. Each `tags` row
    # becomes one order's product. Two orders per user 0 / 1; user 2 unmoved.
    # We compress the 4-row × 2-tag layout into 7 single-FK orders so the
    # distinct-product set per user matches the canonical token set:
    #   user 0: 3 orders → p_a, p_b, p_c (b twice across 4-tag layout
    #     becomes p_b once in distinct set; that maps to a 3-set here)
    #   user 1: 3 orders → p_a, p_d, p_e
    order_user_id = [0, 0, 0, 1, 1, 1]
    order_ts = [100, 150, 200, 300, 350, 400]
    order_product = ["p_a", "p_b", "p_c", "p_a", "p_d", "p_e"]

    orders = pa.table({
        "order_id":
        pa.array(list(range(len(order_user_id))), type=pa.int64()),
        "user_id":
        pa.array(order_user_id, type=pa.int64()),
        "ts":
        pa.array(order_ts, type=pa.int64()),
        "product_id":
        pa.array(order_product, type=pa.string()),
    })
    pq.write_table(orders, str(data_dir / "orders.parquet"))

    graph = Graph(name="linkpred_fk_e2e")
    graph.connect(ParquetConnector("local", root_dir=str(data_dir)))
    graph.add("users", pkey="user_id")
    graph.add("products", pkey="product_id")
    graph.add("orders", cols=[], time_col="ts", pkey="order_id")
    graph.link(src_table="orders", src_key="user_id", dst_table="users")
    graph.link(src_table="orders", src_key="product_id", dst_table="products")
    graph = graph.infer_metadata()
    return graph


def _ingest_linkpred_index(data_dir: Path, index_dir: Path) -> str:
    """Build the diskgraph index for the link-pred FK fixture."""
    from diskgraph import ingest as rust_ingest

    config = {
        "dataset_name":
        "linkpred_fk_e2e",
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
                "name": "products",
                "columns": [{
                    "name": "product_id",
                    "ingested_dtype": "string"
                }],
                "primary_key": "product_id",
                "source": {
                    "type": "parquet",
                    "paths": [str(data_dir / "products.parquet")],
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
                    {
                        "name": "product_id",
                        "ingested_dtype": "string"
                    },
                ],
                "primary_key":
                "order_id",
                "fkeys": [
                    {
                        "src_column": "user_id",
                        "dst_table": "users",
                        "reverse": {
                            "time": {
                                "table": "orders",
                                "column": "ts",
                                "unit": "s",
                            }
                        },
                    },
                    {
                        "src_column": "product_id",
                        "dst_table": "products",
                    },
                ],
                "source": {
                    "type": "parquet",
                    "paths": [str(data_dir / "orders.parquet")],
                },
            },
        ],
    }
    index_dir.mkdir(parents=True, exist_ok=True)
    rust_ingest(
        json.dumps(config),
        str(index_dir),
        col_stats=True,
        cat_count_strategy="hybrid",
    )
    return str(index_dir)


# ---------------------------------------------------------------------------
# Real ingest invocation
# ---------------------------------------------------------------------------
#
# We bypass kumo's ``run_diskgraph_ingest`` and the kumo→diskgraph
# ``DiskgraphConfigConverter`` for ingest. The converter has its own
# kumo-config-shape integration tests; this file is here to exercise the
# *runtime* path (PQL → validator → DiskGraphPlanCompiler →
# execute_tt_plan_with_artifacts). Hand-rolling the diskgraph ingest config
# decouples this test from any kumo-config inference quirks (e.g. ``infer_metadata``
# emitting ``dtype: "string"`` for native-list columns) and matches the
# pattern used in the diskgraph repo's ``test_materialize_e2e.py``.
#
# The Graph object is still built — it's required by ``get_validated_query``
# for PQL validation and by ``generate_train_table_via_diskgraph`` to resolve
# the LP RHS entity table for link-pred filtering — but its ``config()`` is
# not consumed by ingest.


def _ingest_index_handrolled(data_dir: Path, index_dir: Path,
                             shape: str) -> str:
    """Build the diskgraph index by calling the wheel's ``ingest`` function
    directly with a hand-rolled diskgraph config JSON.

    Mirrors the approach in the diskgraph repo's ``test_materialize_e2e.py``.
    Returns the index path.
    """
    from diskgraph import ingest as rust_ingest

    if shape == "utf8_delim":
        tags_col = {
            "name": "tags",
            "ingested_dtype": "string",
            "semantic_type": "multicategorical",
            "multicategorical_delimiter": ",",
        }
    elif shape == "string_array":
        tags_col = {
            "name": "tags",
            "ingested_dtype": "string_array",
            "semantic_type": "multicategorical",
        }
    elif shape == "int64_array":
        tags_col = {
            "name": "tags",
            "ingested_dtype": "int64_array",
            "semantic_type": "multicategorical",
        }
    else:
        raise ValueError(f"unknown shape: {shape!r}")

    config = {
        "dataset_name":
        f"multicat_e2e_{shape}",
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
                    tags_col,
                ],
                "primary_key":
                "order_id",
                "fkeys": [
                    {
                        "src_column": "user_id",
                        "dst_table": "users",
                        "reverse": {
                            "time": {
                                "table": "orders",
                                "column": "ts",
                                "unit": "s",
                            }
                        },
                    },
                ],
                "source": {
                    "type": "parquet",
                    "paths": [str(data_dir / "orders.parquet")],
                },
            },
        ],
    }
    index_dir.mkdir(parents=True, exist_ok=True)
    rust_ingest(
        json.dumps(config),
        str(index_dir),
        col_stats=True,
        cat_count_strategy="hybrid",
    )
    return str(index_dir)


# ---------------------------------------------------------------------------
# Output parquet readers + token assertions
# ---------------------------------------------------------------------------


def _read_nodepred_distinct_tokens(
    output_path: str,
    decode_int: bool = False,
) -> Dict[int, List[str]]:
    """Map entity → sorted distinct tokens for non-array node-pred output.

    Node-pred multilabel-classification (non-array) renders the LIST_DISTINCT
    target as a `|`-joined multicategorical string per entity row. We split
    that back into tokens. Empty target strings (entities with no in-window
    rows) are dropped from the returned mapping.

    For ``int64_array`` source shape, tokens stringify to "1".."5"; under
    ``decode_int=True`` they are mapped back through the bijection so the
    cross-shape parity assertion can compare against canonical letters.
    """
    table = pq.read_table(output_path)
    df = table.to_pandas()
    out: Dict[int, List[str]] = {}
    # TT-NG materializes the train table with the raw producer column names
    # ``ENTITY1``/``TARGET1`` (the ``ENTITY``/``TARGET`` rename was a legacy
    # ``execute_tt_plan_with_artifacts`` behavior). The multicat target is still
    # rendered as a ``|``-joined string per entity row.
    for entity, target in zip(df["ENTITY1"], df["TARGET1"]):
        # Entity rows with no in-window source produce an empty TARGET string;
        # drop them so cross-mode comparisons line up with array-mode (which
        # also drops empty target lists in the helper above).
        if target is None or target == "":
            continue
        tokens = str(target).split("|")
        if decode_int:
            tokens = [MULTICAT_INT_TO_TOKEN[int(t)] for t in tokens]
        tokens.sort()
        # Multiple anchor timestamps would emit one row per (entity, anchor);
        # this fixture uses a single timestamp so we expect exactly one row
        # per non-empty entity.
        out[int(entity)] = tokens
    return out


# ---------------------------------------------------------------------------
# Cube parametrization + tests
#
# Node-pred multilabel-classification is parametrized over the three
# multicategorical source-column on-disk shapes. The array-output writer arm
# (``list_distinct_target_is_array=True``, ``TARGET=List<Utf8>``) is
# link-prediction-only in the current compiler: ``DiskGraphPlanCompiler.compile``
# honors the flag inside the ``link_prediction`` branch but ignores it for
# ``node_prediction``. The link-pred FK-target test below covers that arm.
# ---------------------------------------------------------------------------

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
def test_nodepred_multilabel_classification_exploded(
    tmp_path: Path,
    source_shape: str,
):
    """Node-pred multilabel-classification with multicat target, exploded.

    Drives the kumo PQL → validator → ``DiskGraphPlanCompiler`` →
    ``execute_tt_plan_with_artifacts`` chain end-to-end. Exploded writer arm
    emits one row per (entity, distinct token), one ``__kumo_internal_frequency_column__``
    column carrying per-token row count. Asserts per-entity (token, freq) set
    matches the canonical fixture.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    index_dir = tmp_path / "index"

    graph = _build_kumo_graph(data_dir, source_shape)
    index_path = _ingest_index_handrolled(data_dir, index_dir, source_shape)

    pql = "PREDICT LIST_DISTINCT(orders.tags, 0, 5000) CLASSIFY FOR EACH users.user_id"
    validated_query = get_validated_query(graph, pql)

    output_path = str(tmp_path / "tt.parquet")
    out_path, _manifest_path = generate_train_table_via_diskgraph_ttng(
        query=validated_query,
        graph=graph,
        timestamps=[0],
        task_type=TaskType.MULTILABEL_CLASSIFICATION.value.lower(),
        forecast_length=1,
        lag_timesteps=None,
        list_distinct_target_is_array=False,
        include_frequency=False,
        entity_time_col=None,
        index_path=index_path,
        output_path=output_path,
        tt_id="tt-exploded",
        split_config={
            "_target_": "kumo.task.split.RandomSplit",
            "ratio": [1.0, 0.0, 0.0]
        },
        split_family="RandomSplit",
        downsampling=None,
        input_timestamp_unit=TimeUnit.SECONDS,
        target_stype="multicategorical",
    )
    assert os.path.exists(out_path)

    # TT-NG is a two-phase pipeline: ``generate_train_table_via_diskgraph_ttng``
    # runs GEN-only (stages tt.parquet + col-stats + tt_ingest_spec.json), and
    # ``ensure_stage_ingested`` runs INGEST, publishing the materialized train
    # table extension (the artifact training/stats actually read).
    producer_staging_dir = os.path.dirname(out_path)
    staging_dir = str(tmp_path / "tt-exploded.ttng_staging")
    persist_durable_ttng_bundle(producer_staging_dir, staging_dir)
    ext = ensure_stage_ingested(index_path=index_path,
                                target_tt_id="tt-exploded",
                                staging_dir=staging_dir,
                                predict_stage_name=None)
    materialized_tt = os.path.join(ext, "tt.parquet")

    decode_int = (source_shape == "int64_array")
    observed = _read_nodepred_distinct_tokens(materialized_tt,
                                              decode_int=decode_int)
    assert observed == EXPECTED_DISTINCT_TOKENS_PER_ENTITY, (
        f"per-entity distinct tokens mismatch on shape={source_shape}; "
        f"got {observed}")

    # Temporal node-pred train tables (LIST_DISTINCT with explicit offsets)
    # must always retain the TIMESTAMP column — it carries the real anchor
    # time, not a sentinel.
    pf_schema = pq.read_table(materialized_tt).schema
    assert "TIMESTAMP" in pf_schema.names, (
        "TIMESTAMP column must be retained on temporal node-pred train "
        f"tables; got schema={pf_schema.names}")


def test_nodepred_multilabel_classification_split_stats(tmp_path: Path):
    """Native split-stats ``target_distribution`` for multilabel classification.

    Regression test for the kumo-diskgraph ``build_stage_stats_from_columns``
    gap: multilabel tasks fell through the task-type match and left
    ``target_distribution = null``. Drives the real diskgraph wheel end-to-end
    on the canonical multicat fixture through the TT-NG two-phase flow (GEN via
    ``generate_train_table_via_diskgraph_ttng`` then INGEST via
    ``ensure_stage_ingested``) and asserts the ingest-published native
    ``stats.json`` carries a populated ``{count, percentage}`` target
    distribution.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    index_dir = tmp_path / "index"

    graph = _build_kumo_graph(data_dir, "utf8_delim")
    index_path = _ingest_index_handrolled(data_dir, index_dir, "utf8_delim")

    pql = "PREDICT LIST_DISTINCT(orders.tags, 0, 5000) CLASSIFY FOR EACH users.user_id"
    validated_query = get_validated_query(graph, pql)

    output_path = str(tmp_path / "tt.parquet")
    out_path, _manifest_path = generate_train_table_via_diskgraph_ttng(
        query=validated_query,
        graph=graph,
        timestamps=[0],
        task_type=TaskType.MULTILABEL_CLASSIFICATION.value.lower(),
        forecast_length=1,
        lag_timesteps=None,
        list_distinct_target_is_array=False,
        include_frequency=False,
        entity_time_col=None,
        index_path=index_path,
        output_path=output_path,
        tt_id="tt-split-stats",
        split_config={
            "_target_": "kumo.task.split.RandomSplit",
            "ratio": [1.0, 0.0, 0.0]
        },
        split_family="RandomSplit",
        downsampling=None,
        input_timestamp_unit=TimeUnit.SECONDS,
        target_stype="multicategorical",
    )

    # INGEST publishes the extension; training-side stats come from the
    # published native ``stats.json`` sidecar (never the gen staging manifest).
    producer_staging_dir = os.path.dirname(out_path)
    staging_dir = str(tmp_path / "tt-split-stats.ttng_staging")
    persist_durable_ttng_bundle(producer_staging_dir, staging_dir)
    ext = ensure_stage_ingested(index_path=index_path,
                                target_tt_id="tt-split-stats",
                                staging_dir=staging_dir,
                                predict_stage_name=None)
    with open(os.path.join(ext, "stats.json")) as f:
        stats = json.load(f)

    dist = stats.get("target_distribution")
    assert dist is not None, (
        "multilabel target_distribution is null — the diskgraph producer did "
        "not emit it (the regression this test guards)")
    assert set(dist.keys()) >= {"count", "percentage"}, dist
    assert dist["count"] and dist["percentage"], "distribution maps are empty"
    assert set(dist["count"].keys()) == set(dist["percentage"].keys())

    # Labels must be a subset of the per-entity distinct tokens unioned across
    # the (single, train-only) split; utf8_delim tokens are the canonical
    # letters directly.
    expected_labels = {
        tok
        for toks in EXPECTED_DISTINCT_TOKENS_PER_ENTITY.values()
        for tok in toks
    }
    assert set(dist["count"].keys()) <= expected_labels, (
        f"unexpected labels: {set(dist['count'].keys()) - expected_labels}")
    assert all(v > 0 for v in dist["count"].values())
    # Per-row prevalence: native stats emit fractions in (0, 1]; multilabel
    # distinct-token prevalence per label never exceeds the full population.
    assert all(0.0 < v <= 1.0 for v in dist["percentage"].values())


# ---------------------------------------------------------------------------
# Array-output writer arm (`list_distinct_target_is_array=True`)
#
# Exercises the link-prediction-style writer that emits ``TARGET=List<Utf8>``
# (one row per entity, list contains distinct tokens). The kumo PQL validator
# only accepts this on `string`-dtype source columns when ALLOW_ARRAY_TARGETS
# is enabled — native `stringlist`/`int64list` columns raise
# ``ArrayDtype not supported for Dtype stringlist`` at validation time. The
# flag also flips the compiler's task routing: when set, the
# ``link_prediction(explode=False)`` branch fires instead of
# ``node_prediction()``.
# ---------------------------------------------------------------------------


def _read_arrayout_distinct_tokens(output_path: str) -> Dict[int, List[str]]:
    """Map entity → sorted distinct tokens for array-output writer.

    Array mode emits one row per entity; TARGET = ``List<Utf8>`` containing
    the distinct tokens. Empty target lists are dropped from the returned
    mapping so cross-mode comparisons line up.
    """
    table = pq.read_table(output_path)
    out: Dict[int, List[str]] = {}
    entities = table.column("ENTITY").to_pylist()
    targets = table.column("TARGET").to_pylist()
    for entity, target_list in zip(entities, targets):
        if not target_list:
            continue
        toks = sorted(str(t) for t in target_list)
        out[int(entity)] = toks
    return out


# Per-user expected distinct products under the link-pred FK fixture
# (`_build_linkpred_graph`). Three orders per user mapping to three distinct
# products; user 2 has no orders so emits an empty target list under array
# mode (or no row under exploded mode).
EXPECTED_DISTINCT_PRODUCTS_PER_ENTITY: Dict[int, List[str]] = {
    0: ["p_a", "p_b", "p_c"],
    1: ["p_a", "p_d", "p_e"],
}


# Parked: kumo-diskgraph's Rust `build_split_stats_artifact` rejects the
# fk_target_arrayout column dtype on the train-table path. Upstream wheel gap,
# not a kumo-side bug — the BP sibling exercises the same FK-target +
# array-output arm and passes (BP doesn't route through split-stats).
@pytest.mark.skip(
    reason=("kumo-diskgraph build_split_stats_artifact rejects the "
            "fk_target_arrayout column dtype on the train-table path"))
def test_linkpred_fk_target_arrayout(tmp_path):
    """Link-prediction with FK target, array-output writer arm.

    Kumo customers reach the ``PredictionType::Link { explode: false }``
    writer arm via link-pred PQL like
    ``PREDICT LIST_DISTINCT(orders.product_id) RANK TOP 3 FOR EACH users.user_id``
    (or CLASSIFY TOP k for multilabel-classification link-pred).

    This test covers that path end-to-end: real wheel ingest, real PQL →
    validator → ``DiskGraphPlanCompiler.link_prediction(explode=False)`` →
    ``execute_tt_plan_with_artifacts``. Asserts:
    - TARGET column dtype is ``List<Utf8>`` (the link-pred array-out writer
      arm fired, not the node-pred ``|``-joined string writer)
    - One row per entity in the array writer arm (including empty rows)
    - Per-entity distinct product sets match the canonical link-pred fixture

    Note: this exercises the array writer arm but does NOT cover the
    multicategorical per-token observation path — the FK target is
    categorical-by-stype, not multicategorical. That intersection is
    unreachable via standard kumo PQL because ``get_rhs_lp_table`` requires
    the link-pred target column to be a ``ForeignKey`` and rejects
    multicategorical columns; per-token observation is covered in the
    diskgraph repo's own materialize tests.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    index_dir = tmp_path / "index"

    graph = _build_linkpred_graph(data_dir)
    index_path = _ingest_linkpred_index(data_dir, index_dir)

    pql = ("PREDICT LIST_DISTINCT(orders.product_id, 0, 5000) RANK TOP 3 "
           "FOR EACH users.user_id")
    validated_query = get_validated_query(graph, pql)

    output_path = str(tmp_path / "tt.parquet")
    out_path, _manifest_path = generate_train_table_via_diskgraph_ttng(
        query=validated_query,
        graph=graph,
        timestamps=[0],
        task_type=TaskType.TEMPORAL_LINK_PREDICTION.value.lower(),
        forecast_length=1,
        lag_timesteps=None,
        list_distinct_target_is_array=True,
        include_frequency=False,
        entity_time_col=None,
        index_path=index_path,
        output_path=output_path,
        tt_id="tt-lp-arrayout",
        split_config={
            "_target_": "kumo.task.split.RandomSplit",
            "ratio": [1.0, 0.0, 0.0]
        },
        split_family="RandomSplit",
        downsampling=None,
        input_timestamp_unit=TimeUnit.SECONDS,
        target_stype="multicategorical",
    )
    assert os.path.exists(out_path)

    pf = pq.ParquetFile(out_path)
    cols = {f.name: f.type for f in pf.schema_arrow}
    target_type = cols["TARGET"]
    assert pa.types.is_list(target_type), (
        f"TARGET must be List<Utf8> when list_distinct_target_is_array=True "
        f"on link-pred; got {target_type}")
    assert pa.types.is_string(target_type.value_type), (
        f"TARGET value_type must be string; got {target_type.value_type}")

    observed = _read_arrayout_distinct_tokens(out_path)
    assert observed == EXPECTED_DISTINCT_PRODUCTS_PER_ENTITY, (
        f"per-entity distinct products mismatch on link-pred array-out arm; "
        f"got {observed}")
