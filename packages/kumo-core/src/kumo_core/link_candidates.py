# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The target-filter conditions that select link-prediction candidates."""

from __future__ import annotations

import os
import shutil
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
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

PAIRWISE_CANDIDATES = (
    "A pairwise model cannot apply target-filter conditions on the "
    "candidate table; remove them from the query.")


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


def candidate_query(query: ValidatedPredictiveQuery, graph: GraphDefinition,
                    task_type: TaskType) -> ValidatedPredictiveQuery | None:
    r"""The query keeping only its candidate-table conditions, or ``None``
    when it has none."""
    if not task_type.is_link_pred or not isinstance(
            query.get_final_target_aggregation().target, Filter):
        return None
    candidates = BPFilterTransformer().generate_bp_query(
        deepcopy(query), task_type, graph)
    if not isinstance(candidates.get_final_target_aggregation().target,
                      Filter):
        return None
    return candidates


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
    candidates = candidate_query(query, graph, task_type)
    if candidates is None:
        return None
    target = candidates.get_final_target_aggregation().target
    table = get_rhs_lp_table(candidates, graph)
    primary_key = graph.tables[table].pkey
    plan = runtime.DiskGraphPlanCompiler(index_path=index_path).compile(
        query=candidates,
        timestamps=[],
        task_type=task_type.value,
        input_timestamp_unit="nanoseconds",
        is_prediction=True,
        target_stype=candidates.target_stype.value,
        lp_prediction_mode="RHS",
        lp_rhs_entity_table=table,
        lp_rhs_entity_column=primary_key,
        list_distinct_target_is_array=True,
    )
    staging_dir = os.path.join(index_path, "train_tables", f".{tt_id}.staging")
    try:
        generated = runtime.execute_tt_gen(plan, index_path, tt_id)
        engine = runtime.SamplerEngine(index_path)
        node_ids = np.empty(generated["row_count"], dtype=np.int64)

        def lookup(batch: Any, start: int) -> None:
            node_ids[start:start + batch.num_rows] = engine.lookup_node_ids(
                table, batch.column(0))

        workers = cpu_budget()
        pending: deque[Future[None]] = deque()
        start = 0
        with ThreadPoolExecutor(workers) as pool:
            for batch in runtime.read_tt_parquet_batches(
                    generated["tt_parquet_path"], ["ENTITY1"]):
                if len(pending) == 2 * workers:
                    pending.popleft().result()
                pending.append(pool.submit(lookup, batch, start))
                start += batch.num_rows
            for future in pending:
                future.result()
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)
    if not node_ids.size:
        raise ValueError(f"No row of table {table!r} passes "
                         f"{target.condition}; loosen the target filter.")
    node_ids.sort()
    return node_ids[np.r_[True, node_ids[1:] != node_ids[:-1]]]
