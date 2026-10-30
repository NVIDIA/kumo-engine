# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any, Dict, List, Optional, Tuple

from kumoapi.pquery.AST import (Aggregation, ASTNode, Column, Condition,
                                Constant, Filter, Join, LogicalOperation)
from kumoapi.pquery.validated_predictive_query import ValidatedPredictiveQuery
from kumoapi.typing import BoolOp

from ._c import (DiskgraphConfigError, DiskgraphDataError,
                 DiskgraphSystemError, DiskGraphTTBuilder, EdgeRef, RowFilter,
                 StageFilter, TimeUnit, read_index_metadata)
from .split_contract import SplitSpec

VALID_TASK_TYPES = {
    "binary_classification",
    "multiclass_classification",
    "multilabel_classification",
    "multilabel_ranking",
    "regression",
    "temporal_link_prediction",
    "static_link_prediction",
    "link_prediction",
    "forecasting",
}

LINK_PRED_TASKS = {
    "static_link_prediction",
    "temporal_link_prediction",
    "link_prediction",
}

# Multilabel classification can learn from an all-negative target set, while
# multilabel ranking needs at least one relevant item to provide supervision.
TASK_TYPES_REQUIRING_NON_EMPTY_TARGET_LIST = {"multilabel_ranking"}

LP_BP_LAYOUTS = {"hstack", "separate_lhs_rhs"}

# Canonical output column names for the autoregression ops emitted by
# forecasting plans. These are the contract between DG and kumo-pred —
# kumo-pred expects to find these exact column names in the BP/TT parquet
# output. The canonical strings are defined kumo-side in
# ``kumo/utils/autoregression.py``; if either side renames, the
# integration tests on both sides fail together.
AUTOREGRESSION_OUTPUT_ALIAS = "autoregression_labels"
FUTURE_LABELS_OUTPUT_ALIAS = "future_labels"
YOY_OUTPUT_ALIAS = "year_over_year_labels"

PRIMARY_ENTITY_COL = "ENTITY1"
PRIMARY_TARGET_COL = "TARGET1"

def _map_op(op: str) -> str:
    if op == '=':
        return '=='
    # Multicategorical membership (StrOp.CONTAINS / NOT_CONTAINS, e.g. from a
    # PQL `LIKE '%Comedy%'` rewrite). The Rust filter eval decodes the
    # native-list column per-token; scalar ops are untouched.
    if op == 'CONTAINS':
        return 'contains'
    if op == 'NOT CONTAINS':
        return 'not_contains'
    if op in ('IN', 'IS IN'):
        return 'in'
    return op

def _validate_lp_layout(name: str, value: Optional[str], allowed: set[str]) -> None:
    if value is None or value in allowed:
        return
    choices = ", ".join(sorted(allowed))
    raise DiskgraphConfigError(f"{name} must be one of: {choices}")

def _requires_non_empty_target_list(task_type: str) -> bool:
    return task_type in TASK_TYPES_REQUIRING_NON_EMPTY_TARGET_LIST

def index_timestamp_unit(index_path: Optional[str]) -> str:
    """The clock the index at ``index_path`` stores timestamps in (seconds when omitted)."""
    if index_path is None:
        return "seconds"
    return json.loads(read_index_metadata(index_path)).get("timestamp_unit", "seconds")


class DiskGraphPlanCompiler:
    """
    Translates a ValidatedPredictiveQuery (AST) into a DiskGraphTTPlan (JSON).
    Plans are written in the clock of the index at ``index_path`` (seconds when omitted).
    """

    def __init__(self, index_path: Optional[str] = None):
        self._timestamp_unit = index_timestamp_unit(index_path)
        self.builder = DiskGraphTTBuilder(self._timestamp_unit)
        self._alias_counter = 0
        self._is_link_pred: bool = False
        self._lp_rhs_entity_table: Optional[str] = None

    def _next_alias(self, prefix: str) -> str:
        self._alias_counter += 1
        return f"{prefix}_{self._alias_counter}"

    def compile(
        self,
        query: ValidatedPredictiveQuery,
        timestamps: List[int],
        task_type: str,
        input_timestamp_unit: 'TimeUnit',
        target_stype: str,
        forecast_length: int = 1,
        lag_timesteps: Optional[int] = None,
        list_distinct_target_is_array: bool = False,
        include_frequency: bool = False,
        entity_time_col: Optional[str] = None,
        is_prediction: bool = False,
        lp_prediction_mode: Optional[str] = None,
        lp_rhs_entity_table: Optional[str] = None,
        lp_rhs_entity_column: Optional[str] = None,
        lp_bp_layout: Optional[str] = None,
        lag_offsets_secs: Optional[List[int]] = None,
        future_offsets_secs: Optional[List[int]] = None,
        yoy_offsets_secs: Optional[List[int]] = None,
        split: Optional[SplitSpec] = None,
    ) -> str:
        """
        Build the DiskGraph execution plan.

        Args:
            query: The ValidatedPredictiveQuery AST from Kumo.
            timestamps: Raw observation anchor times in the unit declared by
                ``input_timestamp_unit``. Rust converts to the index unit.
            task_type: The ML task type (e.g., 'binary_classification', 'temporal_link_prediction').
            forecast_length: Number of future timesteps to ignore before the target window.
            lag_timesteps: Lookback window for autoregressive features.
            list_distinct_target_is_array: If True, output Link Prediction targets as a single LIST column (one row per anchor).
            include_frequency: When True and target is LIST_DISTINCT, include a
                __kumo_internal_frequency_column__ column counting occurrences of each distinct value.
                Set for RANK problems; unset for CLASSIFY.
            entity_time_col: Optional column name to use as static observation timestamps.
            input_timestamp_unit: The unit of the raw ``timestamps`` values
                (e.g. 'seconds', 'microseconds', 'nanoseconds'). Rust converts
                to the index unit via the UnitSystem trait.
            is_prediction: Whether this plan is for generating a batch prediction table.
            lp_prediction_mode: 'LHS' or 'RHS'. When set, emits a pass-through plan for Link Prediction candidates.
            lp_rhs_entity_table: The LP RHS entity table name. Two uses:
                - When lp_prediction_mode == 'RHS': identifies the entity table
                  for RHS candidate extraction (BP flow).
                - When is_link_pred and not is_prediction (TT-gen LP): drives the
                  orphan + temporal filter on ListDistinct targets.
            lp_rhs_entity_column: The LP RHS entity PK column (used when lp_prediction_mode == 'RHS').
            lp_bp_layout: Optional LP-only batch-prediction artifact layout
                override. "hstack" uses the legacy single-parquet shape.
                "separate_lhs_rhs" uses separate LHS and RHS candidate outputs.
                If omitted, derived from list_distinct_target_is_array for
                compatibility.
            lag_offsets_secs: Optional lag offsets to be used in bp forecasting task.
            yoy_offsets_secs: Optional yoy offsets to be used in bp forecasting task.
            split: Optional canonical SplitSpec. Legacy dict payloads are not supported.
        """
        task_type = task_type.lower()
        if task_type not in VALID_TASK_TYPES:
            raise DiskgraphConfigError(f"Invalid task_type '{task_type}'. Must be one of {VALID_TASK_TYPES}")

        # 0. Target semantic type
        self.builder.set_target_stype(target_stype)
        self.builder.set_is_prediction(is_prediction)

        # 1. Temporal Spec — pass raw timestamps with declared source unit to
        # Rust, which converts to the index unit system via UnitSystem trait.
        if query.query_type.name == "TEMPORAL" and timestamps:
            unit_str = str(input_timestamp_unit).rsplit('.', 1)[-1].lower()
            self.builder.temporal(
                timestamps, forecast_length, lag_timesteps,
                unit=unit_str,
            )

        # 2. Prediction Type detection
        is_link_pred = task_type in LINK_PRED_TASKS
        self._is_link_pred = is_link_pred
        self._lp_rhs_entity_table = lp_rhs_entity_table

        _validate_lp_layout("lp_bp_layout", lp_bp_layout, LP_BP_LAYOUTS)
        if not is_link_pred and lp_bp_layout is not None:
            raise DiskgraphConfigError("lp_bp_layout is only valid for link prediction tasks")

        resolved_lp_bp_layout = lp_bp_layout
        if resolved_lp_bp_layout is None:
            resolved_lp_bp_layout = "separate_lhs_rhs" if list_distinct_target_is_array else "hstack"

        # 3. Entity Spec
        target_for_type = query.get_final_target_aggregation()

        if lp_prediction_mode == 'RHS' and is_link_pred and is_prediction:
            if not lp_rhs_entity_table or not lp_rhs_entity_column:
                raise DiskgraphConfigError("lp_rhs_entity_table and lp_rhs_entity_column are required for RHS extraction")

            entity_table = lp_rhs_entity_table
            entity_col = lp_rhs_entity_column

            # Evaluate target filter natively
            target_inner = target_for_type.target
            filter_node = None
            if isinstance(target_inner, Filter):
                filter_node = target_inner.condition

            # Kumo guarantees that any filter conditions remaining on the target
            # after BPFilterTransformer are strictly local to the target node table.
            # We swap the target column of the filter to the physical Node PK.
            if filter_node:
                dummy_col = Column(fqn=f"{entity_table}.{entity_col}")
                query_entity_ast = Filter(target=dummy_col, condition=filter_node)
            else:
                query_entity_ast = None
        else:
            entity_table, entity_col = query.entity_column.split('.')
            query_entity_ast = query.entity_ast

        # By default, the output time column is TIMESTAMP and populated with 0 for static queries.
        # For static link prediction queries with legacy format, Kumo expects
        # the time column to be named 'PSEUDO_TIMESTAMP' and populated with MAX_PSEUDO_TIME.
        output_time_col_name = 'TIMESTAMP'
        default_time_value = 0
        is_static_query = query.query_type.name == "STATIC"
        if is_static_query and is_link_pred and (not list_distinct_target_is_array or resolved_lp_bp_layout == "hstack"):
            output_time_col_name = 'PSEUDO_TIMESTAMP'
            default_time_value = 100 * (10**12)  # 100 trillion

        self.builder.for_each(entity_table, entity_col, output_time_col_name, default_time_value)

        # Always pass entity_time_col when available. For temporal queries this
        # enables create_time filtering (skip entity/timestamp pairs where
        # timestamp < create_time). For static queries it lets the executor
        # emit actual timestamps instead of 0.
        if entity_time_col:
            self.builder.entity_time_col(entity_time_col)

        # Handle simple entity filters (WHERE col = constant)
        if isinstance(query_entity_ast, Filter):
            self._compile_entity_filter(query_entity_ast.condition, query, root_table=entity_table)
        # 4. Set Prediction Type in Builder
        if is_link_pred:
            self.builder.link_prediction(
                include_frequency=include_frequency,
            )
        else:
            self.builder.node_prediction()

        split_spec = split
        if split_spec is not None and not isinstance(split_spec, SplitSpec):
            raise DiskgraphConfigError(
                'diskgraph compiler split contract requires split=SplitSpec; '
                'legacy dict split payloads are no longer supported'
            )
        if split_spec is not None:
            split_spec.validate_for_compiler()
            self.builder.split_ratio(split_spec)

        # 4. What-If (Assuming) — only for training, not prediction.
        # GE's batch_pred_table_generator skips ASSUMING entirely;
        # match that behaviour here.
        if query.whatif_ast and not is_prediction:
            sf = self._compile_stage_filter(query.whatif_ast, query)
            self.builder.filter(sf)

        # 6. Target AST Traversal
        is_forecasting = task_type == 'forecasting'

        if is_forecasting:
            self.builder.forecasting_prediction()
            # Forecasting: Build historical lags.
            # In Training mode, the Rust executor dynamically appends a '0' offset
            # to the sequence to capture the label target without duplicating the AST traversal.
            self._traverse_target(query.target_ast, query)
            target_op_json = self.builder.pop_last_op_json()

            # We expect the `lag_offsets_secs` to be passed for both training and BP
            if not lag_offsets_secs:
                raise DiskgraphConfigError("lag_offsets_secs is required for forecasting tasks")

            if lag_offsets_secs:
                self.builder.autoregression(AUTOREGRESSION_OUTPUT_ALIAS, target_op_json, lag_offsets_secs)
            if future_offsets_secs and not is_prediction:
                self.builder.future_labels(FUTURE_LABELS_OUTPUT_ALIAS, target_op_json, future_offsets_secs)
            if yoy_offsets_secs:
                self.builder.autoregression(YOY_OUTPUT_ALIAS, target_op_json, yoy_offsets_secs)

            if not is_prediction:
                # Training Table Generation (Forecasting)
                # Preserve the entity-column anchor expected by downstream plan
                # consumers. Rust resolves TARGET1 from the canonical
                # autoregression op's inner aggregate, not from this column.
                self.builder.column("entity_id_target", entity_table, entity_col)

        elif is_prediction:
            if lp_prediction_mode in ('LHS', 'RHS'):
                self.builder.pass_through_prediction()
                # For Link Prediction separate dataset mode (or LHS pass), we stop here.
                # No target operations are evaluated or injected.
                return self.builder.build()

            if not is_link_pred:
                # Node prediction: emit entities only (PassThrough), no target.
                # Static tasks still restrict to unlabeled entities (TARGET IS
                # NULL): a HasEdges topology check keeps entities that traversed
                # no edges, falling back to is_null for direct-column targets.
                if is_static_query:
                    try:
                        has_edges_alias = self._has_target_edges(query.target_ast, query)
                        self.builder.filter(StageFilter.compare(has_edges_alias, "==", False))
                    except DiskgraphConfigError:
                        target_alias = self._traverse_target(query.target_ast, query)
                        self.builder.filter(StageFilter.compare(target_alias, "is_null", 0))
                self.builder.pass_through_prediction()
            else:
                # Temporal / hstack Link Prediction: emit the entity ID as a dummy
                # target to satisfy the engine's requirement for at least one Op.
                self.builder.column("entity_id_target", entity_table, entity_col)
        else:
            # Training Table Generation (Node/Link Prediction): evaluate target label natively.
            target_alias = self._traverse_target(query.target_ast, query)
            if _requires_non_empty_target_list(task_type):
                if target_stype.lower() != "multicategorical":
                    raise DiskgraphConfigError(
                        f"{task_type} requires target_stype='multicategorical' "
                        "for non-empty target-list admission"
                    )
                self.builder.filter(StageFilter.non_empty_list(target_alias))

        fkey_mapping = [(PRIMARY_ENTITY_COL, entity_table)]
        if is_link_pred and lp_rhs_entity_table is not None:
            if not is_prediction and not list_distinct_target_is_array:
                fkey_mapping.append((PRIMARY_TARGET_COL, lp_rhs_entity_table))

        self.builder.set_fkey_mapping(fkey_mapping)
        return self.builder.build()

    def _extract_path(self, node: ASTNode, query: ValidatedPredictiveQuery, root_table: Optional[str] = None) -> Tuple[List[EdgeRef], ASTNode]:
        path = []
        current = node
        entity_table = root_table or query.entity_table

        # Unpack wrappers
        while isinstance(current, (Filter, Aggregation)):
            current = current.target

        while isinstance(current, Join):
            src_table, src_col = current.lhs_key.split('.')
            dst_table, dst_col = current.rhs_key.split('.')
            path.append(EdgeRef(src_table, src_col, dst_table, dst_col))
            current = current.rhs_target

        # If path is still empty, check if we are on a different table than entity
        if not path:
            if isinstance(current, Column):
                target_table = current.fqn.split('.')[0]
                if target_table != entity_table:
                    # Target is on a different table. Try to find the link.
                    fkey = query.get_single_target_fkey(lhs=False)
                    if fkey:
                        src_table = query.entity_table
                        src_col = query.entity_column.split('.')[1]
                        dst_table = fkey.split('.')[0]
                        dst_col = fkey.split('.')[1]
                        path.append(EdgeRef(src_table, src_col, dst_table, dst_col))
                    else:
                        # Final fallback: Assume the target table has a column named exactly after entity table or 'user_id'
                        # This handles some PQLParser edge cases.
                        src_table = query.entity_table
                        src_col = query.entity_column.split('.')[1]
                        dst_table = target_table
                        dst_col = src_col # Default to same column name (e.g. user_id -> user_id)
                        path.append(EdgeRef(src_table, src_col, dst_table, dst_col))

        return path, current

    def _compile_entity_filter(self, node: ASTNode, query: ValidatedPredictiveQuery, root_table: Optional[str] = None):
        """Routes filters in the entity FOR EACH clause."""
        # Try as RowFilter first (more efficient, happens during scan)
        try:
            # We check if it involves any Aggregations. If so, it CANNOT be a RowFilter.
            if self._has_aggregation(node):
                raise DiskgraphConfigError("Involves aggregation")
            rf = self._compile_row_filter(node, query, root_table=root_table)
            self.builder.entity_filter(rf)
        except (DiskgraphConfigError, AttributeError):
            # If it involves aggregations, it must be a StageFilter
            sf = self._compile_stage_filter(node, query)
            self.builder.filter(sf)

    def _has_aggregation(self, node: ASTNode) -> bool:
        if isinstance(node, Aggregation):
            return True
        for child in getattr(node, 'children', []):
            if isinstance(child, ASTNode) and self._has_aggregation(child):
                return True
        return False

    def _compile_row_filter(self, node: ASTNode, query: ValidatedPredictiveQuery, context_table: str = None, root_table: Optional[str] = None) -> RowFilter:
        if isinstance(node, Condition):
            # When context_table is set (row filter inside an aggregation),
            # columns on the context table should use Column LHS (not Remote).
            # The row filter is evaluated on each destination node during traversal,
            # so columns on that table are directly accessible.
            if context_table and isinstance(node.target, Column):
                target_table = node.target.fqn.split('.')[0]
                if target_table == context_table:
                    col_name = node.target.fqn.split('.')[1]
                    op = _map_op(node.op.value if hasattr(node.op, 'value') else str(node.op))
                    if isinstance(node.value, Column):
                        ent_col = node.value.fqn.split('.')[1]
                        return RowFilter.compare_to_entity(col_name, op, ent_col)
                    else:
                        val = node.get_typed_value()
                        return RowFilter.compare(col_name, op, val, self._timestamp_unit)

            path, leaf = self._extract_path(node.target, query, root_table=root_table)
            col_name = leaf.fqn.split('.')[1]
            op = _map_op(node.op.value if hasattr(node.op, 'value') else str(node.op))

            # Resolve RHS
            if isinstance(node.value, Column):
                ent_col = node.value.fqn.split('.')[1]
                if not path:
                    return RowFilter.compare_to_entity(col_name, op, ent_col)
                else:
                    return RowFilter.compare_remote_to_entity(path, col_name, op, ent_col)
            else:
                val = node.get_typed_value()
                if val is None:
                    op = "is_null" if op in ("==", "eq") else "is_not_null"
                    val = 0
                if not path:
                    return RowFilter.compare(col_name, op, val, self._timestamp_unit)
                else:
                    return RowFilter.compare_remote_lhs(path, col_name, op, val,
                                                        self._timestamp_unit)

        elif isinstance(node, LogicalOperation):
            left = self._compile_row_filter(node.left, query, context_table=context_table)
            if node.bool_op == BoolOp.NOT:
                return RowFilter.not_(left)
            right = self._compile_row_filter(node.right, query, context_table=context_table)
            if node.bool_op == BoolOp.AND:
                return RowFilter.and_(left, right)
            elif node.bool_op == BoolOp.OR:
                return RowFilter.or_(left, right)

        raise DiskgraphConfigError(f"Node {type(node)} cannot be a RowFilter")

    def _compile_stage_filter(self, node: ASTNode, query: ValidatedPredictiveQuery) -> StageFilter:
        if isinstance(node, Condition):
            op = _map_op(node.op.value if hasattr(node.op, 'value') else str(node.op))

            # LHS is often an aggregation in StageFilters
            if isinstance(node.target, Aggregation):
                alias = self._traverse_target(node.target, query)
                if isinstance(node.value, Column):
                    return StageFilter.compare_to_entity(alias, op, node.value.fqn.split('.')[1])
                elif isinstance(node.value, Aggregation):
                    rhs_alias = self._traverse_target(node.value, query)
                    return StageFilter.compare_to_alias(alias, op, rhs_alias)
                else:
                    val = node.get_typed_value()
                    if val is None:
                        op = "is_null" if op in ("==", "eq") else "is_not_null"
                        val = 0
                    return StageFilter.compare(alias, op, val, self._timestamp_unit)

            # LHS is a Join wrapping an aggregation (edge traversal in stage filter)
            elif isinstance(node.target, Join):
                alias = self._traverse_target(node.target, query)
                if isinstance(node.value, Column):
                    return StageFilter.compare_to_entity(alias, op, node.value.fqn.split('.')[1])
                elif isinstance(node.value, Aggregation):
                    rhs_alias = self._traverse_target(node.value, query)
                    return StageFilter.compare_to_alias(alias, op, rhs_alias)
                else:
                    val = node.get_typed_value()
                    if val is None:
                        op = "is_null" if op in ("==", "eq") else "is_not_null"
                        val = 0
                    return StageFilter.compare(alias, op, val, self._timestamp_unit)

            # Simple Column LHS in StageFilter — must emit a column-fetch
            # op so the Rust engine can resolve the alias at execution time.
            elif isinstance(node.target, Column):
                alias = self._traverse_target(node.target, query)
                if isinstance(node.value, Column):
                    return StageFilter.compare_to_entity(alias, op, node.value.fqn.split('.')[1])
                else:
                    val = node.get_typed_value()
                    if val is None:
                        op = "is_null" if op in ("==", "eq") else "is_not_null"
                        val = 0
                    return StageFilter.compare(alias, op, val, self._timestamp_unit)

        elif isinstance(node, LogicalOperation):
            left = self._compile_stage_filter(node.left, query)
            if node.bool_op == BoolOp.NOT:
                return StageFilter.not_(left)
            right = self._compile_stage_filter(node.right, query)
            if node.bool_op == BoolOp.AND:
                return StageFilter.and_(left, right)
            elif node.bool_op == BoolOp.OR:
                return StageFilter.or_(left, right)

        raise DiskgraphConfigError(f"Unsupported stage filter node: {type(node)}")

    def _has_target_edges(self, node: ASTNode, query: ValidatedPredictiveQuery) -> str:
        """
        Evaluates the target purely to see if any edges were traversed.
        Returns the alias of the boolean `has_edges` result.
        """
        while True:
            if isinstance(node, Join):
                node = node.rhs_target
            elif isinstance(node, Condition):
                # If target is SUM(...) > 0, the topology is defined by the LHS aggregation
                node = node.target
            elif isinstance(node, Filter):
                node = node.target
            else:
                break

        if not isinstance(node, Aggregation):
            raise DiskgraphConfigError("Target must contain an Aggregation to check for edges")

        alias = self._next_alias("has_edges")

        # 1. Path and Row Filter
        path, leaf = self._extract_path(node.target, query)
        row_filter = None
        if isinstance(leaf, Filter):
            context_table = path[-1].dst_table if path else query.entity_table
            row_filter = self._compile_row_filter(leaf.condition, query, context_table=context_table)

        # 2. Temporal Window
        window = None
        window_unit = None
        time_col = None
        if node.aggr_time_range:
            start = node.aggr_time_range.start
            end = node.aggr_time_range.end
            if start == "-INF" or start is None:
                start = -2**60
            window = (int(start), int(end))
            window_unit = node.aggr_time_range.unit.value

        if node.time_col:
            time_col = node.time_col.split('.')[1]

        self.builder.has_edges(
            alias=alias,
            path=path,
            time_col=time_col,
            window=window,
            window_unit=window_unit,
            row_filter=row_filter,
        )
        return alias

    def _traverse_target(self, node: ASTNode, query: ValidatedPredictiveQuery,
                         outer_path: Optional[List[EdgeRef]] = None) -> str:
        if isinstance(node, Column):
            alias = self._next_alias("col")
            table, col = node.fqn.split('.')
            if outer_path is None and table != query.entity_table:
                raise DiskgraphConfigError(
                    f"Direct column '{node.fqn}' is outside the entity table "
                    f"'{query.entity_table}' but has no validated relationship path"
                )
            self.builder.column(alias, table, col, path=outer_path)
            return alias

        elif isinstance(node, Aggregation):
            alias = self._next_alias("agg")

            # Extract path FIRST so we know the destination table for row filter context
            filter_node = None
            target_inner = node.target
            if isinstance(target_inner, Filter):
                filter_node = target_inner.condition
                target_inner = target_inner.target

            path, leaf = self._extract_path(target_inner, query)

            # If a parent Join node already resolved the edge path, use it
            # instead of the (possibly incorrect) inferred path.
            if outer_path is not None:
                path = outer_path

            # Determine the destination table: row filter columns resolve against it.
            # The leaf from _extract_path is always a Column on the destination table.
            if isinstance(leaf, Column):
                dest_table = leaf.fqn.split('.')[0]
            else:
                dest_table = None

            row_filter = None
            if filter_node is not None:
                row_filter = self._compile_row_filter(filter_node, query, context_table=dest_table)

            target_col = None
            if isinstance(leaf, Column):
                col_name = leaf.fqn.split('.')[1]
                if col_name != '*':
                    target_col = col_name

            window = None
            window_unit = None
            if node.aggr_time_range:
                start = node.aggr_time_range.start
                end = node.aggr_time_range.end
                if start == "-INF" or start is None:
                    start = -2**60 # Use large negative for -INF
                window = (int(start), int(end))
                window_unit = node.aggr_time_range.unit.value

            time_col = None
            if node.time_col:
                time_col = node.time_col.split('.')[1]

            # NOTE: limit stays None for training table generation.
            # top_k from PQL is only used at ML prediction time, not here.

            # Orphan + temporal filter: lp_rhs_entity_table is resolved by
            # the caller (kumo) from the graph schema and passed through
            # compile(). Used during LP TT-gen to filter LIST_DISTINCT
            # targets against the destination table's PK map + temporal cols.
            # NOTE: AggregationType.LIST_DISTINCT.value is uppercase
            # ("LIST_DISTINCT"); compare case-insensitively since the Rust
            # parser lowercases func anyway.
            target_entity_table = None
            if (
                node.aggr.value.lower() == "list_distinct"
                and self._is_link_pred
                and isinstance(leaf, Column)
            ):
                target_entity_table = self._lp_rhs_entity_table

            self.builder.aggregate(
                alias=alias,
                path=path,
                func=node.aggr.value,
                target_col=target_col,
                time_col=time_col,
                window=window,
                window_unit=window_unit,
                row_filter=row_filter,
                target_entity_table=target_entity_table,
            )
            return alias

        elif isinstance(node, Condition):
            alias = self._next_alias("cond")
            lhs_alias = self._traverse_target(node.target, query)
            op = _map_op(node.op.value if hasattr(node.op, 'value') else str(node.op))
            self.builder.compare(alias, lhs_alias, op, node.get_typed_value())
            return alias

        elif isinstance(node, Join):
            # Join nodes encode validated edge paths from the PQL validator.
            # Extract the edge(s) and pass them to the inner node so the
            # Aggregation handler uses the correct path instead of
            # re-inferring it via get_single_target_fkey (which is
            # query-scoped and may point to a different table).
            join_path: List[EdgeRef] = []
            current = node
            while isinstance(current, Join):
                src_table, src_col = current.lhs_key.split('.')
                dst_table, dst_col = current.rhs_key.split('.')
                join_path.append(EdgeRef(src_table, src_col, dst_table, dst_col))
                current = current.rhs_target
            return self._traverse_target(current, query, outer_path=join_path)

        elif isinstance(node, LogicalOperation):
            left_alias = self._traverse_target(node.left, query)
            if node.bool_op == BoolOp.NOT:
                alias = self._next_alias("logic")
                self.builder.logical_op(alias, "not", [left_alias])
                return alias
            right_alias = self._traverse_target(node.right, query)
            alias = self._next_alias("logic")
            op_str = "and" if node.bool_op == BoolOp.AND else "or"
            self.builder.logical_op(alias, op_str, [left_alias, right_alias])
            return alias

        raise DiskgraphConfigError(f"Unsupported target node: {type(node)}")
