# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from kumoapi.common import (
    ValidationError,
    ValidationResponse,
    ValidationWarning,
)
from kumoapi.graph import ColumnKey, GraphDefinition
from kumoapi.pquery import ParsedPredictiveQuery
from kumoapi.pquery.AST import Aggregation, ASTNode, Condition, Filter, Join
from kumoapi.pquery.utils import validate_int
from kumoapi.typing import AggregationType, ProblemType, Stype
from kumopql.validator.utils import col_name, merge, table_name


class TimeRangeValidator:
    r"""Class with all the logic around :class:`ASTNode` time range validation.

    Args:
        graph: Graph on which the query is defined.
    """

    ENTITY = 'ENTITY'  # Validate entity definition
    TARGET = 'TARGET'  # Validate target definition
    ASSUMPTION = 'ASSUMPTION'  # Validate whatif definition

    def __init__(self, graph: GraphDefinition) -> None:
        self.graph = graph

    def validate_tree(
        self,
        node: ASTNode,
        query_name: str,
        problem_type: ProblemType,
        allow_future_looking: bool = True,
        allow_past_looking: bool = True,
        allow_static: bool = True,
        allow_temporal: bool = True,
        is_label: bool = False,
    ) -> ValidationResponse:
        r"""Validate time ranges in the filter.

        Args:
            node: Node to be validated
            query_name: The name of the part of the query that is
                displayed in the error message.
            problem_type: Problem type of the query.
            allow_future_looking: If :obj:`False`, an error
                will be raised for any future-looking time ranges.
            allow_past_looking: If :obj:`False`, an error
                will be raised for any past-looking time ranges. This does not
                apply to nested time-ranges that appear as part of the filter
                conditions because one should be able to, e.g., say "count the
                number of next-week purchases from companies that showed ads
                last week."
            allow_static: If :obj:`False`, an error will be
                raised for any static condition in the query.
            allow_temporal: If :obj:`False`, an error will be
                raised for any temporal condition/aggregation in the query.
            is_label: If :obj:`True`, the output of this
                node is to be directly used as a label. Used for evaluations
                where the label needs to conform to different standards than
                the rest of the query, e.g. static node prediction. For
                example, the aggregation node in `PREDICT SUM(table.col, 0, 1)`
                is the label, but `PREDICT SUM(table.col, 0, 1) > 10` isn't.
                In the latter case, the condition is the label.

        Returns:
            Validation response to the checks.
        """
        response = ValidationResponse()
        if isinstance(node, Filter):
            # Special handling of tree traversal
            return self._validate_filter(
                node,
                query_name,
                problem_type,
                allow_future_looking=allow_future_looking,
                allow_past_looking=allow_past_looking,
                allow_static=allow_static,
                allow_temporal=allow_temporal,
                is_label=False,
            )
        for child in node.children:
            response = merge(
                response,
                self.validate_tree(
                    child,
                    query_name,
                    problem_type,
                    allow_future_looking=allow_future_looking,
                    allow_past_looking=allow_past_looking,
                    allow_static=allow_static,
                    allow_temporal=allow_temporal,
                    is_label=is_label and isinstance(node, Join),
                ))
        if isinstance(node, Aggregation):
            response = merge(
                response,
                self._validate_aggr(
                    node,
                    query_name,
                    problem_type,
                    allow_future_looking=allow_future_looking,
                    allow_past_looking=allow_past_looking,
                    allow_static=allow_static,
                    allow_temporal=allow_temporal,
                    is_label=is_label,
                ))
        if isinstance(node, Condition):
            response = merge(
                response,
                self._validate_condition(
                    node,
                    query_name,
                    problem_type,
                    allow_future_looking=allow_future_looking,
                    allow_past_looking=allow_past_looking,
                    allow_static=allow_static,
                    allow_temporal=allow_temporal,
                ))

        return response

    def _validate_filter(
        self,
        node: Filter,
        query_name: str,
        problem_type: ProblemType,
        allow_future_looking: bool = True,
        allow_past_looking: bool = True,
        allow_static: bool = True,
        allow_temporal: bool = True,
        is_label: bool = False,
    ) -> ValidationResponse:
        response = ValidationResponse()
        response = merge(
            response,
            self.validate_tree(
                node.target,
                query_name,
                problem_type,
                allow_future_looking=allow_future_looking,
                allow_past_looking=allow_past_looking,
                allow_static=allow_static,
                allow_temporal=allow_temporal,
            ))
        # past-looking conditions are acceptable even with
        # allow_past_looking=False, see docstring of `validate_tree
        # for the explanation
        # Similarly, static conditions are always allowed.
        # Even in ASSUMING, static filter can only appear nested in an
        # aggregation, making it acceptable.
        response = merge(
            response,
            self.validate_tree(
                node.condition,
                query_name,
                problem_type,
                allow_future_looking=allow_future_looking,
                allow_past_looking=True,
                allow_static=True,
                allow_temporal=allow_temporal,
            ))
        return response

    def _validate_aggr(
        self,
        node: Aggregation,
        query_name: str,
        problem_type: ProblemType,
        allow_future_looking: bool = True,
        allow_past_looking: bool = True,
        allow_static: bool = True,
        allow_temporal: bool = True,
        is_label: bool = False,
    ) -> ValidationResponse:
        response = ValidationResponse()
        target_col_fqn = node.get_target_column_name()
        target_table_name = table_name(target_col_fqn)
        if not allow_temporal and node.aggr_time_range is not None:
            response.errors.append(
                ValidationError(
                    title='Temporal aggregation not allowed',
                    message=f'{node.get_location().message_start}: '
                    f'{query_name} aggregation {node} has a time '
                    f'range but temporal aggregation is not supported here. '
                    f'This error usually happens when one tries to use a '
                    f'temporal aggregation in a filter in a query with a '
                    f'static target that does not contain any temporal '
                    f'aggregations.'))
        if self.graph.tables[target_table_name].time_col is None:
            if node.aggr_time_range is not None:
                response.errors.append(
                    ValidationError(
                        title='Table missing a time column',
                        message=f"{node.get_location().message_start}: "
                        f"Missing a \"Create date\" column in the table "
                        f"'{target_table_name}'. A \"Create date\" column is "
                        f"required to use a time range since it defines "
                        f"a temporal prediction. To fix this, you can "
                        f"update your graph to include a \"Create date\" "
                        f"column in {target_table_name} or remove the "
                        f"aggregation and turn this into a static prediction "
                        f"problem with \"PREDICT {target_col_fqn}\"."))
                return response
        if node.aggr_time_range is None:
            # `list_distinct` operator without time ranges is only allowed for
            # target definition, with other additional restrictions below
            is_target_list_distinct = (query_name == self.TARGET and node.aggr
                                       == AggregationType.LIST_DISTINCT)
            if col_name(target_col_fqn) == '*':
                target_is_fkey = False
                target_is_categorical_col = False
                is_ranking = False
            else:
                all_keys = []
                for col_group in self.graph.col_groups:
                    all_keys.extend(list(col_group.columns))
                target_is_fkey = (ColumnKey(target_table_name,
                                            col_name(target_col_fqn))
                                  in all_keys)
                if (self.graph.tables[target_table_name].pkey == col_name(
                        target_col_fqn)):
                    target_is_fkey = False
                target_col = [
                    col for col in self.graph.tables[target_table_name].cols
                    if col.name == col_name(target_col_fqn)
                ][0]
                target_is_categorical_col = (
                    target_col.stype == Stype.categorical)
                is_ranking = problem_type != ProblemType.CLASSIFY
            valid_static_list_distinct = (is_target_list_distinct and is_label
                                          and target_is_fkey and is_ranking)

            # For static list_distinct, there are 4 possible combinations when
            # used in the target
            # 1. rank + fkey (allowed)
            # 2. rank + non-fkey (not allowed)
            # 3. classify + fkey (not allowed)
            # 4. classify + non-fkey (not allowed)
            if not valid_static_list_distinct:
                if not is_target_list_distinct:
                    response.errors.append(
                        ValidationError(
                            title='Aggregation missing a time range',
                            message=f"{node.get_location().message_start}: "
                            f"{query_name} aggregation {node} "
                            "is missing a time range. At the moment, static "
                            "usage aggregations is only supported for "
                            "LIST_DISTINCT combined with RANK."))
                # Make sure 3 and 4 are not allowed
                elif not is_ranking:
                    response.errors.append(
                        ValidationError(
                            title='Aggregation missing a time range',
                            message=f"{node.get_location().message_start}: "
                            f"{query_name} aggregation {node} "
                            "is missing a time range. At the moment, static "
                            "usage of LIST_DISTINCT is only possible for "
                            "RANK problems."))
                # Make sure 2 is not allowed
                elif target_is_categorical_col:
                    response.errors.append(
                        ValidationError(
                            title='Aggregation missing a time range',
                            message=f"{node.get_location().message_start}: "
                            f"{query_name} aggregation {node} "
                            f"is missing a time range. At the moment, static "
                            f"usage of LIST_DISTINCT is only possible for "
                            f"the target clause with a foreign key. "
                            f"(Currently using LIST_DISTINCT in "
                            f"{query_name}). If you want to use LIST_DISTINCT "
                            f"in a filter please make sure to add "
                            f"a time range."))
                else:
                    # Make sure static LP is not used in non-targets
                    response.errors.append(
                        ValidationError(
                            title='Aggregation missing a time range',
                            message=f"{node.get_location().message_start}: "
                            f"{query_name} aggregation {node} "
                            f"is missing a time range. At the moment, static "
                            f"usage of LIST_DISTINCT is only possible for "
                            f"the target clause with a foreign key or "
                            f"categorical target. (Currently using "
                            f"LIST_DISTINCT in {query_name}). If you want to "
                            f"use LIST_DISTINCT in a filter please make "
                            f"sure to add a time range. If you want to use "
                            f"LIST_DISTINCT with a column, change its "
                            f"semantic type to cateogrical (currently "
                            f"{target_col.stype} (foreign key: "
                            f"{target_is_fkey})"))
            return response
        response = merge(response, validate_int(node.aggr_time_range.end))
        if node.aggr_time_range.start is not None:
            response = merge(response, validate_int(node.aggr_time_range.start))
        if node.aggr_time_range.end > 0 and not allow_future_looking:
            response.errors.append(
                ValidationError(
                    title="Future-looking time range not allowed",
                    message=(f"{node.get_location().message_start}: "
                             f"{node} within the {query_name} contains a "
                             f"future-looking time range. This is not allowed "
                             f"as it would fail during batch prediction due to "
                             f"unavailable future data. Consider using the "
                             f"ASSUMING clause for future scenarios instead.")))
        if ((node.aggr_time_range.is_open or node.aggr_time_range.start < 0)
                and not allow_past_looking):
            response.errors.append(
                ValidationError(
                    title="Invalid time range",
                    message=f"{node.get_location().message_start}: "
                    f"{query_name} cannot have an aggregation in "
                    f"the past. Both start and end need to be non negative."))
        time_col_name = self.graph.tables[target_table_name].time_col
        node.time_col = target_table_name + '.' + time_col_name
        return response

    def _validate_condition(
        self,
        node: Condition,
        query_name: str,
        problem_type: ProblemType,
        allow_future_looking: bool = True,
        allow_past_looking: bool = True,
        allow_static: bool = True,
        allow_temporal: bool = True,
    ) -> ValidationResponse:
        response = ValidationResponse()
        if node.date_offset_range is None and not allow_static:
            if query_name == self.TARGET:
                response.errors.append(
                    ValidationError(
                        title='Mixing static and temporal conditions '
                        'is not allowed',
                        message=f'{node.get_location().message_start}: '
                        f'Mixing static and temporal targets is '
                        f'not supported yet. Contact Kumo for help.'))
            else:
                response.errors.append(
                    ValidationError(
                        title='Static condition not allowed',
                        message=f'{node.get_location().message_start}: '
                        f'{query_name} condition {node} has no time '
                        f'range but a temporal aggregation is required.'))
        return response

    def static_query_validations(
            self, parsed_query: ParsedPredictiveQuery) -> ValidationResponse:
        response = ValidationResponse()
        entity_table_name = table_name(parsed_query.entity_column)
        if self.graph.tables[entity_table_name].time_col is None:
            cols_with_timestamp = []
            for col in parsed_query.all_query_columns:
                if self.graph.tables[table_name(col)].time_col is not None:
                    cols_with_timestamp.append(col)
            if len(cols_with_timestamp) > 0:
                message = parsed_query.entity_ast.get_location().message_start
                response.warnings.append(
                    ValidationWarning(
                        title='Potential data leakage', message=f'{message}: '
                        f'Entity table {entity_table_name} does '
                        f'not have a "Create date", but several other '
                        f'columns in the query do: {cols_with_timestamp}. '
                        f'Training/validation/test data split will be '
                        f'done at random, potentially resulting in '
                        f'data leakage. To prevent this and remove this '
                        f'message, add a "Create date" column to your '
                        f'entity table, corresponding to the time '
                        f'when the prediction would have been done.'))
        if parsed_query.whatif_ast is not None:
            message = parsed_query.whatif_ast.get_location().message_start
            response.errors.append(
                ValidationError(
                    title='ASSUMING not allowed for static queries',
                    message=f'{message}: '
                    f'Target {parsed_query.target_ast} contains no '
                    f'temporal aggregations. ASSUMING is not allowed for such '
                    f'queries.',
                ))
        return response

    def validate_time_ranges(
        self,
        parsed_query: ParsedPredictiveQuery,
    ) -> ValidationResponse:
        r"""Validate time range of every node in the AST, returning the list
        of errors/warnings. We perform three types of validation:
            1. All nodes that require a time interval have one
            2. All nodes with a time interval operate on a table with a time
                col
            3. Entity does not look into future, target and whatif do not look
                into the past.

        Args:
            parsed_query: the input parsed query.

        Returns:
            List of encountered errors/warnings.
        """
        response = ValidationResponse()
        allow_temporal = True
        allow_static_target = True
        if parsed_query.target_ast.date_offset_range is None:
            response = merge(response,
                             self.static_query_validations(parsed_query))
            allow_temporal = False
        else:
            allow_static_target = False
        response = merge(
            response,
            self.validate_tree(
                parsed_query.entity_ast,
                self.ENTITY,
                parsed_query.problem_type,
                allow_future_looking=False,
                allow_temporal=allow_temporal,
                is_label=False,
            ))
        # TODO (vid) future-looking requirement for target is expected to be
        # further softened in the future once we implement operations between
        # aggregations and want to predict queries such as "will a user spend
        # more next week than last week"
        response = merge(
            response,
            self.validate_tree(
                parsed_query.target_ast,
                self.TARGET,
                parsed_query.problem_type,
                allow_past_looking=False,
                allow_static=allow_static_target,
                is_label=True,
            ))
        if parsed_query.whatif_ast is not None:
            response = merge(
                response,
                self.validate_tree(
                    parsed_query.whatif_ast,
                    self.ASSUMPTION,
                    parsed_query.problem_type,
                    allow_past_looking=False,
                    allow_static=False,
                    is_label=False,
                ))
        return response
