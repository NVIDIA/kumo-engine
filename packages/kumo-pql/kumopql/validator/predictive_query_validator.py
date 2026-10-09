# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from kumoapi.common import (
    ValidationError,
    ValidationResponse,
    ValidationWarning,
)
from kumoapi.graph import ColumnKey, GraphDefinition
from kumoapi.pquery import ParsedPredictiveQuery, ValidatedPredictiveQuery
from kumoapi.pquery.AST import (
    Aggregation,
    ASTNode,
    Column,
    Condition,
    Filter,
    Join,
)
from kumoapi.task import TaskType
from kumoapi.typing import AggregationType, ProblemType
from kumopql.validator.join_validator import JoinValidator
from kumopql.validator.problem_type_validator import ProblemTypeValidator
from kumopql.validator.time_range_validator import TimeRangeValidator
from kumopql.validator.type_validator import TypeValidator
from kumopql.validator.utils import (
    col_name,
    get_rhs_lp_table,
    merge,
    table_name,
)


class PredictiveQueryValidator:
    r"""This class contains all the validation and metadata inferral business
    logic, for turning a :class:`ParsedPredictiveQuery` into a
    :class:`ValidatedPredictiveQuery`.

    Args:
        graph: Graph on which the query is defined.
        allow_array_targets: Have LIST_DISTINCT generate an array instead of
            a "|"-separated string.
        allow_timestamp_arrays: If :obj:`False`, block LIST_DISTINCT on time
            columns. Used for SPCS where timestamps are not handled well.
    """
    def __init__(
        self,
        graph: GraphDefinition,
        allow_array_targets: bool = False,
        allow_timestamp_arrays: bool = True,
    ) -> None:
        self.graph = graph
        self.allow_array_targets = allow_array_targets
        self.allow_timestamp_arrays = allow_timestamp_arrays

    def validate_predictive_query(
        self,
        parsed_query: ParsedPredictiveQuery,
    ) -> tuple[ValidatedPredictiveQuery | None, ValidationResponse]:
        r"""Turns a :class:`ParsedPredictiveQuery` into a
        :class:`ValidatedPredictiveQuery`. It infers and validates query dtypes
        and stypes, validates the existence of all columns, and infers and
        validates all the join keys.

        Args:
            parsed_query: the input parsed query.

        Returns:
            Validated `parsed_query` with metadata and validation response
            if no errors were encountered. If errors were encountered,
            a tuple of :obj:`None` and a validation response are returned.

        Raises:
            ValueError: if the query is not valid.
        """
        response = ValidationResponse()
        response = merge(response, self.validate_columns(parsed_query))
        response = merge(response, self.validate_wildcard(parsed_query))
        if not response.ok:
            return None, response

        # infer and validate types
        self.type_validator = TypeValidator(
            self.graph,
            allow_array_targets=self.allow_array_targets,
            allow_timestamp_arrays=self.allow_timestamp_arrays,
        )
        self.type_validator.infer_dtypes(parsed_query)
        self.type_validator.infer_stypes(parsed_query)
        response = merge(response,
                         self.type_validator.validate_dtypes(parsed_query))
        if not response.ok:
            return None, response
        response = merge(response,
                         self.type_validator.validate_stypes(parsed_query))
        if not response.ok:
            return None, response

        # validate problem type
        self.problem_type_validator = ProblemTypeValidator()
        response = merge(response,
                         self.problem_type_validator.validate(parsed_query))
        if not response.ok:
            return None, response

        # validate time ranges
        self.time_range_validator = TimeRangeValidator(self.graph)
        response = merge(
            response,
            self.time_range_validator.validate_time_ranges(parsed_query))
        if not response.ok:
            return None, response

        # target filter validation
        response = merge(response, self.validate_target_filters(parsed_query))
        if not response.ok:
            return None, response

        # infer and validate joins
        self.join_validator = JoinValidator(self.graph)
        response = merge(
            response,
            self.join_validator.infer_and_validate_joins(parsed_query))
        if not response.ok:
            return None, response

        # list distinct special rules validation
        response = merge(response, self.validate_list_distinct(parsed_query))
        if not response.ok:
            return None, response

        response = merge(response, self.target_not_entity_pkey(parsed_query))
        if not response.ok:
            return None, response

        return ValidatedPredictiveQuery(
            entity_ast=parsed_query.entity_ast,
            target_ast=parsed_query.target_ast,
            validation_response=response,
            whatif_ast=parsed_query.whatif_ast,
            top_k=parsed_query.top_k,
            num_forecasts=parsed_query.num_forecasts,
            problem_type=parsed_query.problem_type,
            for_each=parsed_query.for_each,
        ), response

    def validate_columns(
            self, parsed_query: ParsedPredictiveQuery) -> ValidationResponse:
        r"""Validates that all tables and columns that appear in the query are
        part of the graph.

        Args:
            parsed_query (ParsedPredictiveQuery): the input parsed query.

        Returns:
                ValidationResponse: Errors and warnings encountered.
        """
        response = ValidationResponse()
        unknown_tables = []
        unknown_columns = []

        for (fully_qualified_name,
             location) in parsed_query.all_query_columns_with_locations:
            assert len(fully_qualified_name.split('.')) == 2
            column = col_name(fully_qualified_name)
            table = table_name(fully_qualified_name)

            if table not in self.graph.tables:
                unknown_tables.append((table, location))
                continue

            if column != '*' and column not in [
                    c.name for c in self.graph.tables[table].cols
            ]:
                unknown_columns.append((table, column, location))

        if len(unknown_tables) > 0:
            for table, location in unknown_tables:
                response.errors.append(
                    ValidationError(
                        title='Unknown table name',
                        message=f"{location.message_start}: Table '{table}' "
                        f"does not exist in the "
                        f"graph. Please be mindful of case sensitivity and "
                        f"ensure that the name is spelled correctly."))

        if len(unknown_columns) > 0:
            for table, column, location in unknown_columns:
                response.errors.append(
                    ValidationError(
                        title='Unknown column name',
                        message=f"{location.message_start}: Column "
                        f"'{column}' does not exist in table '{table}'. "
                        f"Please be mindful of case sensitivity and ensure "
                        f"that the name is spelled correctly."))

        # Only do this check if no errors were encountered to avoid crashes
        # due to unknown columns
        entity_table = table_name(parsed_query.entity_column)
        entity_pkey = col_name(parsed_query.entity_column)
        if response.ok and self.graph.tables[entity_table].pkey != entity_pkey:
            location = parsed_query.entity_column_obj.get_location()
            response.errors.append(
                ValidationError(
                    title='Invalid entity',
                    message=f"{location.message_start}: Entity "
                    f"'{parsed_query.entity_column}' "
                    f"is not a primary key. The column used in 'FOR EACH'"
                    f" has to be a primary key."))

        return response

    def validate_wildcard(
            self, parsed_query: ParsedPredictiveQuery) -> ValidationResponse:
        r"""Validates that all wildcard columns that appear in the query are
        correctly nested in a `COUNT` aggregation.

        Args:
            parsed_query (ParsedPredictiveQuery): the input parsed query.

        Returns:
                ValidationResponse: Errors and warnings encountered.
        """
        response = ValidationResponse()
        response = merge(response,
                         self._validate_wildcard_col(parsed_query.entity_ast))
        response = merge(response,
                         self._validate_wildcard_col(parsed_query.target_ast))
        if parsed_query.whatif_ast is not None:
            response = merge(
                response, self._validate_wildcard_col(parsed_query.whatif_ast))
        return response

    def _validate_wildcard_col(
        self,
        ast_node: ASTNode,
        count_target: bool = False,
    ) -> ValidationResponse:
        response = ValidationResponse()
        for child in ast_node.children:
            child_is_count_target = (isinstance(ast_node, Aggregation)
                                     and ast_node.aggr == AggregationType.COUNT
                                     and child is ast_node.target) or (
                                         isinstance(ast_node, Filter)
                                         and count_target
                                         and child is ast_node.target)
            response = merge(
                response,
                self._validate_wildcard_col(child, child_is_count_target))
        if (isinstance(ast_node, Column) and col_name(ast_node.fqn) == '*'
                and not count_target):
            response.errors.append(
                ValidationError(
                    title='Invalid wildcard column',
                    message=f'{ast_node.get_location().message_start}: '
                    f'Wildcard column {ast_node.fqn} appeared in an '
                    f'invalid position. It can only be used in a `COUNT` '
                    f'aggregation.'))
        return response

    def validate_target_filters(
            self, parsed_query: ParsedPredictiveQuery) -> ValidationResponse:
        r"""Validations of filters within the target AST. At the moment,
        this just checks if the filters should be moved to the entity table.

        Args:
            parsed_query: the input parsed query.

        Returns:
            List of encountered errors/warnings.
        """
        entity_table = table_name(parsed_query.entity_column)
        filters = self._find_filters(parsed_query.target_ast)
        response = ValidationResponse()
        entity_filters = []
        for filter in filters:
            all_tables = set(
                table_name(col) for col in filter.condition.all_query_columns)
            if all_tables == {entity_table}:
                entity_filters.append(repr(filter.condition))
        if len(entity_filters) > 0:
            and_combined = " AND ".join(list(set(entity_filters)))
            msg_location = parsed_query.target_ast.get_location()
            response.warnings.append(
                ValidationWarning(
                    title='Entity filter in the target definition',
                    message=f'{msg_location.message_start}'
                    f': WHERE conditions {entity_filters} in '
                    f'the target aggregation only '
                    f'contains references to entity table {entity_table}. '
                    f'It can likely be moved into the entity '
                    f'definition as follows: '
                    f'"FOR EACH {parsed_query.entity_column} WHERE '
                    f'{and_combined}". Leaving it in the target '
                    f'definition may result in unnecessary slowdowns '
                    f'or even incorrect labels.'))
        return response

    def _find_filters(self, node: ASTNode) -> list[Filter]:
        if isinstance(node, Filter):
            return [node]
        filters = []
        for child in node.children:
            filters.extend(self._find_filters(child))
        return filters

    def validate_list_distinct(
            self, parsed_query: ParsedPredictiveQuery) -> ValidationResponse:
        r"""Validates checks related to link prediction and multilabel
        classification/ranking specifically. This contains any validations that
        should only be performed for the root list_distinct specifically
        and are not already captured by other validators. More specifically,
        we validate that the aggregation isn't FIRST/LAST, that `CLASSIFY`
        isn't used with static queries and that entity table does not appear
        in the target filter.

        Args:
            parsed_query: the input parsed query.

        Returns:
            Errors and warnings encountered.
        """
        response = ValidationResponse()
        # TODO (vid) turn below checks into a util method `is_link_pred`
        # after LAST/FIRST errors are dropped.
        target_ast = parsed_query.target_ast
        if isinstance(target_ast, Join):
            target_ast = target_ast.rhs_target
        if not isinstance(target_ast, Aggregation):
            return response
        target_fqn = target_ast.get_target_column_name()
        if col_name(target_fqn) == '*':
            return response

        if (target_ast.aggr == AggregationType.LIST_DISTINCT
                and parsed_query.problem_type == ProblemType.CLASSIFY
                and target_ast.aggr_time_range is None):
            # There is some overlap between this check and
            # TimeRangeValidator._validate_aggr where
            # most invalid static aggregations should already be caught
            response.errors.append(
                ValidationError(
                    title='Unsupported syntax', message=(
                        f'{target_ast.get_location().message_start}: '
                        'The combination of LIST_DISTINCT and CLASSIFY is not '
                        'supported right now when there is no time range '
                        'in LIST_DISTINCT.')))

        if (self.graph.tables[table_name(target_fqn)].pkey
                == col_name(target_fqn)
                and target_ast.aggr == AggregationType.LIST_DISTINCT):
            response.errors.append(
                ValidationError(
                    title='Aggregation not supported on primary key columns',
                    message=f'{target_ast.get_location().message_start}: '
                    f'LIST_DISTINCT target aggregations are not supported on'
                    f'primary key columns like {target_fqn}.'))
            return response
        all_keys = []
        for col_group in self.graph.col_groups:
            all_keys.extend(list(col_group.columns))
        if (ColumnKey(table_name(target_fqn), col_name(target_fqn))
                not in all_keys):
            return response
        if target_ast.aggr in [AggregationType.FIRST, AggregationType.LAST]:
            response.errors.append(
                ValidationError(
                    title='Aggregation not supported on foreign key columns',
                    message=f'{target_ast.get_location().message_start}: '
                    f'LAST and FIRST aggregations are not supported on'
                    f'foreign key columns like {target_fqn} yet. '
                    f'Currently, only LIST_DISTINCT is supported.'))
        # All aggregations are nested in a join
        assert isinstance(parsed_query.target_ast, Join)
        if target_ast.aggr != AggregationType.LIST_DISTINCT:
            return response
        if parsed_query.target_ast.rhs_key == target_fqn:
            # The query is predicting the primary key
            # Find alternate entity tables that have a primary key
            # and are connected to the target table
            valid_entities = []
            target_table = table_name(target_fqn)
            target_col = col_name(target_fqn)
            for col_group in self.graph.col_groups:
                target_col_key = ColumnKey(target_table, target_col)
                # col_group must contain a different fkey from target table
                if target_col_key in col_group.columns:
                    continue
                if target_table not in [
                        c.table_name for c in col_group.columns
                ]:
                    continue
                for col in col_group.columns:
                    if self.graph.tables[col.table_name].pkey != col.col_name:
                        continue
                    valid_entities.append(
                        f"FOR EACH {col.table_name}.{col.col_name}")
            # Show the error message to the user
            error_msg = (
                f'{target_ast.get_location().message_start}: '
                f'Target {target_fqn} creates a loop by pointing to the '
                f'entity table. A foreign key used in the target expression '
                f'must not point to the entity table to define a valid '
                f'predictive problem.')
            if len(valid_entities) > 0:
                entity_string = ' '.join(valid_entities)
                error_msg += (f' To fix this, you can change the entity '
                              f'to one of the following: {entity_string}.')
            response.errors.append(
                ValidationError(title='Nothing to predict', message=error_msg))

        if parsed_query.problem_type == ProblemType.CLASSIFY:
            return response
        if isinstance(target_ast.target, Filter):
            entity_table = table_name(parsed_query.entity_column)
            filter_condition = target_ast.target.condition
            for col, loc in filter_condition.all_query_columns_with_locations:
                if table_name(col) == entity_table:
                    response.errors.append(
                        ValidationError(
                            title='Invalid target filter',
                            message=f'{loc.message_start}: '
                            f'Entity table {entity_table} appears '
                            f'inside of the target filter {filter_condition}. '
                            f'References to the entity table are only '
                            f'supported in the entity definition.'))
                    break

        # Validate target table timestamp for static link pred
        target_col_name = target_ast.get_target_column_name()
        target_table = table_name(target_col_name)
        if parsed_query.target_ast.date_offset_range is None:
            if self.graph.tables[target_table].time_col is not None:
                time_col_name = self.graph.tables[target_table].time_col
                location_msg = (
                    parsed_query.target_ast.get_location().message_start)
                response.errors.append(
                    ValidationError(
                        title="Time range required for LIST_DISTINCT with "
                        "time column", message=(
                            f"{location_msg}: "
                            f"When using LIST_DISTINCT on a table with a time "
                            f"column ({time_col_name}), you must specify a "
                            f"time range. For example, use LIST_DISTINCT("
                            f"{target_col_name}, 0, 30, days) instead of "
                            f"LIST_DISTINCT({target_col_name}). "
                            f"Alternatively, you can remove {time_col_name} "
                            f"from {target_table} if time is not relevant.")))

            # Validate timestamps in non-target tables for static link pred
            fully_qualified_time_col_names = []
            for t_name, table in self.graph.tables.items():
                if t_name != target_table and table.time_col is not None:
                    time_col_name = table.time_col
                    assert time_col_name is not None
                    fully_qualified_time_col_names.append(
                        f"{t_name}.{time_col_name}")
            if len(fully_qualified_time_col_names) > 0:
                fully_qualified_time_cols_str = ' '.join(
                    fully_qualified_time_col_names)
                location_msg = (
                    parsed_query.target_ast.get_location().message_start)
                response.errors.append(
                    ValidationError(
                        title="Unexpected time columns in tables.", message=(
                            f"{location_msg}: "
                            f"Cannot perform LIST_DISTINCT query without "
                            f"temporal aggregation when the following time "
                            f"columns {fully_qualified_time_cols_str} are "
                            f"observed in graph. Since LIST_DISTINCT and the "
                            f"columns do not contain any time information, "
                            f"this is likely to result in an incorrect data "
                            f"split and information leakage. To fix this, "
                            f"you can add the time interval in LIST_DISTINCT "
                            f"command or drop the columns "
                            f"{fully_qualified_time_cols_str} from the graph."
                        )))
        return response

    def target_not_entity_pkey(
        self,
        parsed_query: ParsedPredictiveQuery,
    ) -> ValidationResponse:
        r"""Validates that the target is not predicting the entity
        primary key column.

        Args:
            parsed_query (ParsedPredictiveQuery): the input parsed query.

        Returns:
            ValidationResponse: Errors and warnings encountered.
        """
        response = ValidationResponse()
        # Get the entity primary key column ast
        entity_name: str = parsed_query.entity_column

        # Check if the target is predicting the entity primary key column
        target_ast: ASTNode = parsed_query.target_ast
        # Recursively loop through the syntax tree to get all the columns
        # that equal to entity primary key column
        ast_cols = self._target_not_entity_pkey(
            target_ast,
            entity_name,
        )
        for ast_col in ast_cols:
            response.errors.append(
                ValidationError(
                    title='Predicting entity primary key column',
                    message=f"{ast_col.get_location().message_start}: "
                    f"Predicting the entity primary key column "
                    f"is not allowed."))
        return response

    def _target_not_entity_pkey(
        self,
        ast_node: ASTNode,
        entity_name: str,
    ) -> list[Column]:
        r"""Helper function to recursively traverse the syntax tree to get
        all the columns that equal to the entity primary key column.

        Args:
            ast_node: The current node in the syntax tree.
            entity_name: The fully qualified name of the entity primary
                key column.

        Returns: List of columns that store the entity
                primary key column found in the syntax tree of the target.
        """
        ast_cols = []
        if isinstance(ast_node, Column) and ast_node.fqn == entity_name:
            ast_cols.append(ast_node)
        elif isinstance(ast_node, Condition):
            # Only check target, condition may contain entity pkey
            ast_cols.extend(
                self._target_not_entity_pkey(ast_node.target, entity_name))
        elif isinstance(ast_node, Filter | Aggregation):
            # * Aggregation will be blocked by other validation errors
            # such as "Aggregation xxx is not intended to operate on foreign "
            # keys and ID-type columns"
            # * Filter: Filter on target column will be blocked by
            # validate_target_filters
            return ast_cols
        else:
            # Handle other AST types such as LogicalOperation, Join, etc.
            for child in ast_node.children:
                ast_cols.extend(self._target_not_entity_pkey(
                    child, entity_name))
        return ast_cols

    def validate_bp_filter_overrides(
        self,
        validated_query: ValidatedPredictiveQuery,
        task_type: TaskType,
    ) -> ValidationResponse:
        r"""Validates checks related to batch prediction filters specifically.
        More specifically, we validate that only the RHS entity table appears
        in the target filter of a link prediction query.

        Args:
            validated_query: the input validated query.
            task_type: Task type of the query.

        Returns:
                ValidationResponse: Errors and warnings encountered.
        """
        response = ValidationResponse()
        if not task_type.is_link_pred:
            return response
        rhs_entity_table = get_rhs_lp_table(validated_query, self.graph)
        aggr_node = validated_query.target_ast
        if isinstance(aggr_node, Join):
            aggr_node = aggr_node.rhs_target
        assert isinstance(aggr_node, Aggregation)
        if isinstance(aggr_node.target, Filter):
            filter_condition = aggr_node.target.condition
            for col, loc in filter_condition.all_query_columns_with_locations:
                if table_name(col) != rhs_entity_table:
                    response.errors.append(
                        ValidationError(
                            title='Invalid target filter',
                            message=f'{loc.message_start}: '
                            f'Table {table_name(col)} appears '
                            f'in the target filter edit {filter_condition}. '
                            f'During batch prediction, only conditions that '
                            f'refer to {rhs_entity_table} table can be added.'))
                    break
        return response
