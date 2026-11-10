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
from kumoapi.pquery.AST import Filter, Join
from kumopql.rewriting import BPFilterTransformer

from .pquery import NativeGraph, ValidatedNativeQuery, to_api_graph
from .sources.cpu import cpu_budget
from .types import TaskType


def target_contract(query: ValidatedNativeQuery) -> tuple[str, str | None]:
    r"""The target without its filter, and the filter conditions a
    prediction query must keep.

    A link-prediction query may change the conditions on its candidate
    table; they choose what is ranked, not what the model learned.
    """
    if query.task_family != "link_prediction":
        return str(query.query.target_ast), None
    target = deepcopy(query.query.target_ast)
    aggregation = target.rhs_target if isinstance(target, Join) else target
    if not isinstance(aggregation.target, Filter):
        return str(target), None
    split = BPFilterTransformer().split_target_filter(query.query.target_ast,
                                                      query.entity_table,
                                                      query.rhs_node_type)
    kept = (split.non_applicable_filter
            if split.non_applicable_is_valid else aggregation.target.condition)
    aggregation.target = aggregation.target.target
    return str(target), None if kept is None else str(kept)


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
