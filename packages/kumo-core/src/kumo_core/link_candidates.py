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
from kumoapi.graph import GraphDefinition
from kumoapi.pquery import ValidatedPredictiveQuery
from kumoapi.pquery.AST import Filter, Join
from kumopql.rewriting import BPFilterTransformer
from kumopql.validator.utils import get_rhs_lp_table

from .pquery import ValidatedNativeQuery
from .sources.cpu import cpu_budget
from .types import TaskType


def target_contract(query: ValidatedNativeQuery) -> tuple[str, str | None]:
    r"""The target without its filter, and the conditions a prediction
    must keep."""
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
    query: ValidatedPredictiveQuery,
    graph: GraphDefinition,
    *,
    task_type: TaskType,
    index_path: str,
    tt_id: str,
    runtime: Any,
) -> np.ndarray | None:
    r"""Sorted RHS node IDs passing the candidate-table conditions."""
    if not task_type.is_link_pred or not isinstance(
            query.get_final_target_aggregation().target, Filter):
        return None
    candidate_query = BPFilterTransformer().generate_bp_query(
        deepcopy(query), task_type, graph)
    target = candidate_query.get_final_target_aggregation().target
    if not isinstance(target, Filter):
        return None
    table = get_rhs_lp_table(candidate_query, graph)
    primary_key = graph.tables[table].pkey
    plan = runtime.DiskGraphPlanCompiler(index_path=index_path).compile(
        query=candidate_query,
        timestamps=[],
        task_type=task_type.value,
        input_timestamp_unit="nanoseconds",
        is_prediction=True,
        target_stype=candidate_query.target_stype.value,
        lp_prediction_mode="RHS",
        lp_rhs_entity_table=table,
        lp_rhs_entity_column=primary_key,
        list_distinct_target_is_array=True,
    )
    generated = runtime.execute_tt_gen(plan, index_path, tt_id)
    try:
        engine = runtime.SamplerEngine(index_path)

        def lookup(batch: Any) -> np.ndarray:
            return np.asarray(engine.lookup_node_ids(table, batch.column(0)),
                              dtype=np.int64)

        batches = runtime.read_tt_parquet_batches(generated["tt_parquet_path"],
                                                  ["ENTITY1"])
        with ThreadPoolExecutor(cpu_budget()) as pool:
            parts = [np.empty(0, dtype=np.int64), *pool.map(lookup, batches)]
    finally:
        shutil.rmtree(generated["staging_dir"], ignore_errors=True)
    node_ids = np.sort(np.concatenate(parts))
    if not node_ids.size:
        raise ValueError(f"No row of table {table!r} passes "
                         f"{target.condition}; loosen the target filter.")
    return node_ids[np.r_[True, node_ids[1:] != node_ids[:-1]]]
