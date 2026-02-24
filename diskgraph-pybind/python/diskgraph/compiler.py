from typing import List, Optional, Tuple, Union
from kumoapi.pquery.validated_predictive_query import ValidatedPredictiveQuery
from kumoapi.pquery.AST import (
    Aggregation,
    ASTNode,
    Column,
    Condition,
    Constant,
    Filter,
    Join,
    LogicalOperation,
)
from kumoapi.typing import BoolOp

from ._c import DiskGraphTTBuilder, EdgeRef, RowFilter, StageFilter

def _map_op(op: str) -> str:
    if op == '=':
        return '=='
    return op

class DiskGraphPlanCompiler:
    """
    Translates a ValidatedPredictiveQuery (AST) into a DiskGraphTTPlan (JSON).
    """

    def __init__(self, builder: Optional[DiskGraphTTBuilder] = None):
        self.builder = builder or DiskGraphTTBuilder()
        self._alias_counter = 0

    def _next_alias(self, prefix: str) -> str:
        self._alias_counter += 1
        return f"{prefix}_{self._alias_counter}"

    def compile(
        self,
        query: ValidatedPredictiveQuery,
        timestamps: List[int],
        forecast_length: int = 1,
        lag_timesteps: Optional[int] = None,
        list_distinct_target_is_array: bool = False,
        include_frequency: bool = False,
        entity_time_col: Optional[str] = None,
    ) -> str:
        """
        Build the DiskGraph execution plan.

        Args:
            include_frequency: When True and target is LIST_DISTINCT, include a
                FREQUENCY column counting occurrences of each distinct value.
                Set for RANK problems; unset for CLASSIFY.
        """
        # 1. Temporal Spec
        if query.query_type.name == "TEMPORAL" and timestamps:
            self.builder.temporal(timestamps, forecast_length, lag_timesteps)

        # 2. Entity Spec
        entity_table, entity_col = query.entity_column.split('.')
        self.builder.for_each(entity_table, entity_col)

        # Always pass entity_time_col when available. For temporal queries this
        # enables create_time filtering (skip entity/timestamp pairs where
        # timestamp < create_time). For static queries it lets the executor
        # emit actual timestamps instead of 0.
        if entity_time_col:
            self.builder.entity_time_col(entity_time_col)

        # Handle simple entity filters (WHERE col = constant)
        if isinstance(query.entity_ast, Filter):
            self._compile_entity_filter(query.entity_ast.condition, query)

        # 3. Prediction Type
        # Unwrap Join wrappers to find the actual aggregation for type detection.
        target_for_type = query.target_ast
        while isinstance(target_for_type, Join):
            target_for_type = target_for_type.rhs_target
        if (isinstance(target_for_type, Aggregation) and
            target_for_type.aggr.value.lower() in ('list_distinct', 'listdistinct')):
            self.builder.link_prediction(
                explode=not list_distinct_target_is_array,
                include_frequency=include_frequency,
            )
        else:
            self.builder.node_prediction()

        # 4. What-If (Assuming)
        if query.whatif_ast:
            sf = self._compile_stage_filter(query.whatif_ast, query)
            self.builder.filter(sf)

        # 5. Target AST Traversal
        # NOTE: top_k is intentionally NOT passed through. In Kumo, top_k is
        # only used at ML prediction time, not during training table generation.
        # The TT executor collects ALL distinct values.
        self._traverse_target(query.target_ast, query)

        return self.builder.build()

    def _extract_path(self, node: ASTNode, query: ValidatedPredictiveQuery) -> Tuple[List[EdgeRef], ASTNode]:
        path = []
        current = node

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
                if target_table != query.entity_table:
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

    def _compile_entity_filter(self, node: ASTNode, query: ValidatedPredictiveQuery):
        """Routes filters in the entity FOR EACH clause."""
        # Try as RowFilter first (more efficient, happens during scan)
        try:
            # We check if it involves any Aggregations. If so, it CANNOT be a RowFilter.
            if self._has_aggregation(node):
                raise ValueError("Involves aggregation")
            rf = self._compile_row_filter(node, query)
            self.builder.entity_filter(rf)
        except (ValueError, AttributeError):
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

    def _compile_row_filter(self, node: ASTNode, query: ValidatedPredictiveQuery, context_table: str = None) -> RowFilter:
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
                        return RowFilter.compare(col_name, op, val)

            path, leaf = self._extract_path(node.target, query)
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
                if not path:
                    return RowFilter.compare(col_name, op, val)
                else:
                    return RowFilter.compare_remote_lhs(path, col_name, op, val)

        elif isinstance(node, LogicalOperation):
            left = self._compile_row_filter(node.left, query, context_table=context_table)
            if node.bool_op == BoolOp.NOT:
                return RowFilter.not_(left)
            right = self._compile_row_filter(node.right, query, context_table=context_table)
            if node.bool_op == BoolOp.AND:
                return RowFilter.and_(left, right)
            elif node.bool_op == BoolOp.OR:
                return RowFilter.or_(left, right)

        raise ValueError(f"Node {type(node)} cannot be a RowFilter")

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
                    return StageFilter.compare(alias, op, node.get_typed_value())

            # Simple Column LHS in StageFilter
            elif isinstance(node.target, Column):
                col_name = node.target.fqn.split('.')[1]
                if isinstance(node.value, Column):
                    return StageFilter.compare_to_entity(col_name, op, node.value.fqn.split('.')[1])
                else:
                    return StageFilter.compare(col_name, op, node.get_typed_value())

        elif isinstance(node, LogicalOperation):
            left = self._compile_stage_filter(node.left, query)
            if node.bool_op == BoolOp.NOT:
                return StageFilter.not_(left)
            right = self._compile_stage_filter(node.right, query)
            if node.bool_op == BoolOp.AND:
                return StageFilter.and_(left, right)
            elif node.bool_op == BoolOp.OR:
                return StageFilter.or_(left, right)

        raise ValueError(f"Unsupported stage filter node: {type(node)}")

    def _traverse_target(self, node: ASTNode, query: ValidatedPredictiveQuery) -> str:
        if isinstance(node, Column):
            alias = self._next_alias("col")
            table, col = node.fqn.split('.')
            self.builder.column(alias, table, col)
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

            self.builder.aggregate(
                alias=alias,
                path=path,
                func=node.aggr.value,
                target_col=target_col,
                time_col=time_col,
                window=window,
                window_unit=window_unit,
                row_filter=row_filter,
            )
            return alias

        elif isinstance(node, Condition):
            alias = self._next_alias("cond")
            lhs_alias = self._traverse_target(node.target, query)
            op = _map_op(node.op.value if hasattr(node.op, 'value') else str(node.op))
            self.builder.compare(alias, lhs_alias, op, node.get_typed_value())
            return alias

        elif isinstance(node, Join):
            # Join wraps the target AST for link prediction queries.
            # The Join's lhs_key/rhs_key encode the edge path, which
            # _extract_path already knows how to extract when it finds
            # Join nodes inside an Aggregation's target. So we just
            # recurse into the rhs_target (the actual Aggregation).
            return self._traverse_target(node.rhs_target, query)

        raise ValueError(f"Unsupported target node: {type(node)}")
