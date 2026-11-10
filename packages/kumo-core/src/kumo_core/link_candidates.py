# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The target-filter conditions that select link-prediction candidates."""

from __future__ import annotations

import shutil
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import Any

import numpy as np
from kumoapi.pquery.AST import Filter
from kumopql.rewriting import BPFilterTransformer

from .pquery import NativeGraph, ValidatedNativeQuery, to_api_graph
from .sources.cpu import cpu_budget
from .types import TaskType


def candidate_node_ids(
    query: ValidatedNativeQuery,
    graph: NativeGraph,
    *,
    index_path: str,
    tt_id: str,
    runtime: Any,
) -> np.ndarray | None:
    r"""Sorted RHS node IDs that pass the query's candidate-table
    conditions, or ``None`` when it has none."""
    if query.task_family != "link_prediction":
        return None
    candidate_query = BPFilterTransformer().generate_bp_query(
        deepcopy(query.query), TaskType(query.task_type), to_api_graph(graph))
    target = candidate_query.get_final_target_aggregation().target
    if not isinstance(target, Filter):
        return None
    table = graph.table(query.rhs_node_type)
    plan = runtime.DiskGraphPlanCompiler(index_path=index_path).compile(
        query=candidate_query,
        timestamps=[],
        task_type=query.task_type,
        input_timestamp_unit="nanoseconds",
        is_prediction=True,
        target_stype=query.target_stype,
        lp_prediction_mode="RHS",
        lp_rhs_entity_table=table.name,
        lp_rhs_entity_column=table.primary_key,
        list_distinct_target_is_array=True,
    )
    generated = runtime.execute_tt_gen(plan, index_path, tt_id)
    try:
        engine = runtime.SamplerEngine(index_path)

        def lookup(batch: Any) -> np.ndarray:
            return np.asarray(
                engine.lookup_node_ids(table.name, batch.column(0)),
                dtype=np.int64)

        # Each lookup releases the GIL.
        with ThreadPoolExecutor(cpu_budget()) as pool:
            parts = list(
                pool.map(
                    lookup,
                    runtime.read_tt_parquet_batches(
                        generated["tt_parquet_path"], ["ENTITY1"])))
    finally:
        shutil.rmtree(generated["staging_dir"], ignore_errors=True)
    node_ids = np.sort(np.concatenate([np.empty(0, dtype=np.int64), *parts]))
    if not node_ids.size:
        raise ValueError(
            f"No {table.name} rows pass the prediction query's target "
            f"filter {target.condition}, so there is nothing to rank.")
    return node_ids[np.r_[True, node_ids[1:] != node_ids[:-1]]]
