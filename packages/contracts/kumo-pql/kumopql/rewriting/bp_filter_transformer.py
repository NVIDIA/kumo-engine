# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from dataclasses import dataclass, field

from kumoapi.common import (
    ValidationError,
    ValidationResponse,
    ValidationWarning,
)
from kumoapi.graph import GraphDefinition
from kumoapi.pquery import ValidatedPredictiveQuery
from kumoapi.pquery.AST import (
    Aggregation,
    ASTNode,
    Column,
    Condition,
    Filter,
    Join,
    LogicalOperation,
)
from kumoapi.task import TaskType
from kumoapi.typing import AggregationType, BoolOp, Dtype, ProblemType, Stype
from kumopql.parser import PQLParser
from kumopql.rewriting.filter_splitter import FilterSplitter, SplitFilter
from kumopql.validator import PredictiveQueryValidator
from kumopql.validator.utils import get_rhs_lp_table, merge, table_name


@dataclass
class EditableField:
    r"""A single part of the query that may be edited. Corresponds to a single
    substring of the entire query.

    Args:
        field_name: Name of the field that can be used in user-facing
            messages, e.g. 'entity filter'.
        text: The substring of the query that corresponds to this field.
        position:  Position in the full query, given as a
            (start, end) tuple. When inserting user edits,
            `query.text[position[0]:position[1]]` will be dropped and replaced
            with text, defined by `full_filter_format_string` (if user input is
            non-empty) and `empty_filter_default` otherwise.
        full_filter_format_string: Format string that defines how the
            user-input is inserted into the entire filter.
        empty_filter_default: Default value to insert when the user
            input is empty (no filter).
        bold_text_position: Position in `query.text` to be
            displayed in bold text.
    """
    field_name: str
    text: str
    position: tuple[int, int]
    full_filter_format_string: str = ' WHERE {0}'
    empty_filter_default: str = ''
    bold_text_position: tuple[int, int] = (0, 0)


@dataclass
class EditableQuery:
    r"""Dataclass with the batch prediction query as string and a list of its
    fields that may be edited before the BP. Note that `text` might not be the
    same as the original query when dealing with, e.g. link prediction target
    filters. `validation_response` contains warnings and errors related to the
    rewriting - a warning if rewriting took place, an error if MAX_N_CONDITIONS
    were exceeded.
    """
    text: str
    editable_fields: list[EditableField]
    validation_response: ValidationResponse = field(
        default_factory=ValidationResponse())


@dataclass
class EditableQueryResponse:
    r"""Response to the editable query that contains user changes, if any.

    Args:
        editable_query: The `EditableQuery` object that the
            response is for.
        user_changes: A dict of user overwrites, given as
            a mapping from a field_name to a user_change.
    """
    editable_query: EditableQuery
    user_changes: dict[str, str] = field(default_factory=dict)


class BPFilterTransformer(FilterSplitter):
    r"""This class contains all transformations and methods related to
    transforming a predictive query for batch prediction.
    This involves overriding entity and target filters and validating the
    replacements.
    """
    def generate_bp_query(
        self,
        query: ValidatedPredictiveQuery,
        task_type: TaskType,
        graph: GraphDefinition,
    ) -> ValidatedPredictiveQuery:
        r"""Given a `ValidatedPredictiveQuery` object, returns a new
        `ValidatedPredictiveQuery`, with fields correctly replaced for batch
        prediction.

        Args:
            query: Query to be transformed.
            task_type: Task type.
            graph: Graph on which the query is performed.

        Returns:
            A new query object, if transformations
            were necessary for batch prediction or the same object if no
            transformations were necessary. The rewritten query is not
            guaranteed to contain correct location information in all nodes.
            If location is required, use `ASTNode.to_string()` and parse again.

        """
        query.whatif_ast = None
        if not task_type.is_link_pred:
            return query
        lhs_table = table_name(query.entity_column)
        rhs_table = get_rhs_lp_table(query, graph)

        split_target_filter = self.split_target_filter(query.target_ast,
                                                       lhs_table, rhs_table)
        new_filter = split_target_filter.applicable_filter

        assert isinstance(query.target_ast, Join)
        assert isinstance(query.target_ast.rhs_target, Aggregation)
        aggr_node = query.target_ast.rhs_target
        assert aggr_node.aggr == AggregationType.LIST_DISTINCT

        # No filter
        if not isinstance(aggr_node.target, Filter):
            assert new_filter is None
            return query

        if new_filter is None:
            # No applicable filter, we drop the filter
            aggr_node.target = aggr_node.target.target
        else:
            # There is an applicable filter, we replace the condition
            aggr_node.target.condition = new_filter

        return query

    def _get_editable_entity(self, entity: ASTNode) -> EditableField:
        r"""Genedates `EditableField` for the entity AST. Entity filter is
        labeled as fine to edit.
        """
        if isinstance(entity, Column):
            query_repr = ''
            strlen = len(str(entity))
            return EditableField(field_name='Entity filter', text=query_repr,
                                 position=(strlen, strlen),
                                 bold_text_position=(strlen, strlen))
        assert isinstance(entity, Filter)
        query_repr = str(entity.condition)
        filter_start = len(str(entity.target))
        filter_end = filter_start + len(' WHERE ') + len(query_repr)
        return EditableField(
            field_name='Entity filter', text=query_repr,
            position=(filter_start, filter_end),
            bold_text_position=(filter_start + len(' WHERE '), filter_end))

    def _get_editable_split_filter(
        self, target: ASTNode, split_filter: SplitFilter
    ) -> tuple[str, EditableField, ValidationResponse]:
        r"""Given a target AST, it returns a transformed string representation
        of the target, an editable field, and a validation response with info
        about rewriting, if it took place.
        """
        if isinstance(target, Join):
            target = target.rhs_target
        assert isinstance(target, Aggregation)
        # To get the filter of the original target, we can reuse handling of
        # entity. We then have to fix the field name and position
        editable_field = self._get_editable_entity(target.target)
        editable_field.field_name = 'Target filter'
        prefix_len = len(target.aggr.value) + 1
        editable_field.position = _tuple_inc(editable_field.position,
                                             prefix_len)
        editable_field.bold_text_position = _tuple_inc(
            editable_field.bold_text_position, prefix_len)
        assert split_filter.non_applicable_is_valid
        response = ValidationResponse()
        full_target_str = str(target)
        if (split_filter.applicable_filter is None
                and split_filter.non_applicable_filter is None):
            # other fields are handled by default values
            editable_field.text = ''
        elif split_filter.applicable_filter is None:
            # If there's no bp_applicable_filter, we just use the original
            # untouched text instead of the possibly-rewritten version in
            # split_fitler.bp_non_applicable_filter
            editable_field.full_filter_format_string = (
                f' WHERE ({editable_field.text}) AND ({{0}})')
            editable_field.empty_filter_default = (
                f' WHERE {editable_field.text}')
            editable_field.text = ''
            editable_field.bold_text_position = (0, 0)  # No bold text
        elif split_filter.non_applicable_filter is None:
            # If there's no bp_non_applicable_filter, we just use the original
            # untouched text instead of the possibly-rewritten version in
            # split_fitler.bp_applicable_filter
            # TODO (vid) Make this info-level if possible.
            response.warnings.append(
                ValidationWarning(
                    title='Target filter applied during batch prediction.',
                    message='The target filter did not contain references to '
                    'any columns with unavailable data at prediction '
                    'time. Kumo can therefore guarantee that bolded '
                    'conditions are met by all predicted links.'))
        else:
            editable_field.full_filter_format_string = (
                f' WHERE ({split_filter.non_applicable_filter}) AND ({{0}})')
            editable_field.empty_filter_default = (
                f' WHERE {split_filter.non_applicable_filter}')
            editable_field.text = str(split_filter.applicable_filter)
            new_filter = (f' WHERE ({split_filter.non_applicable_filter}) '
                          f'AND ({split_filter.applicable_filter})')
            full_target_str = (full_target_str[:editable_field.position[0]] +
                               new_filter +
                               full_target_str[editable_field.position[1]:])
            editable_field.position = (editable_field.position[0],
                                       editable_field.position[0] +
                                       len(new_filter))
            non_bold_prefix_len = len(
                f' WHERE ({split_filter.non_applicable_filter}) AND (')
            editable_field.bold_text_position = (
                editable_field.position[0] + non_bold_prefix_len,
                editable_field.position[0] + non_bold_prefix_len +
                len(str(split_filter.applicable_filter)))
            # TODO (vid) Make this info-level if possible.
            response.warnings.append(
                ValidationWarning(
                    title='Target filter applied during batch prediction.',
                    message='Kumo has identified the part of the target '
                    'filter that only contains references to the data that is '
                    'available at prediction time. '
                    'Kumo can therefore guarantee that bolded '
                    'conditions are all met by all '
                    'predicted links.'))
            if split_filter.rewritten:
                response.warnings.append(
                    ValidationWarning(
                        title='Target filter was rewritten',
                        message='In order to identify the subset of the '
                        'target filter that could be explicitly guaranteed '
                        'during prediction, the logical expression has '
                        'been rewritten. The new expression is equivalent '
                        'to the one in your input query.'))
        return full_target_str, editable_field, response

    def get_bp_editable_query(
        self,
        query: ValidatedPredictiveQuery,
        task_type: TaskType,
        graph: GraphDefinition,
        original_query_text: str,
    ) -> EditableQuery:
        r"""Given a `ValidatedPredictiveQuery`, this method returns the
        transformed query for BP as well as metadata on all the editable
        fields to display in the UI.

        Args:
            query: Query that the user may modify.
            task_type: Task type.
            graph: Graph on which the query is performed.
            original_query_text: Retained for caller compatibility. The
                updated query is reconstructed from the parsed tree.

        Returns:
            EditableQuery: Information on the raw updated query and parts that
                may be modified.

        """
        del original_query_text
        # TODO (vid): This method should be improved once we start storing
        # locations in the original query as part of better error reporting.
        # At the moment, this code recovers the query from the parsed tree
        # instead of using `original_query_text`. This is because it's hard
        # to determine which parts of the original query were dropped because
        # they don't belong into batch prediction, which parts were dropped
        # because they were comments/whitespace, and which parts were added
        # automatically.
        # This is not ideal because it doesn't contain user-added line breaks
        # and comments.

        lhs_table = table_name(query.entity_column)
        editable_fields = []
        response = ValidationResponse()
        if task_type.is_link_pred:
            rhs_table = get_rhs_lp_table(query, graph)
            split_target_filter = self.split_target_filter(
                query.target_ast, lhs_table, rhs_table)
            if not split_target_filter.non_applicable_is_valid:
                number_of_terms = (
                    split_target_filter.n_applicable_conditions +
                    split_target_filter.n_non_applicable_conditions)
                response.warnings.append(
                    ValidationWarning(
                        title='Rewritten target filter too large.',
                        message=f'To enable target filter override, Kumo '
                        f'tried to rewrite the target filter and identify the '
                        f'subset that could be applied/overridden at batch '
                        f'prediction time. Unfortunately, the size of the '
                        f'target filter grew too large in the process '
                        f'({number_of_terms} conditions). To this end, filter '
                        f'overrides will not be supported for this query. '
                        f'Contact Kumo for further assistance with this if '
                        f'required.'))
                target_repr = 'PREDICT ' + str(query.target_ast)
            else:
                (target_repr, target_field,
                 edit_response) = self._get_editable_split_filter(
                     query.target_ast, split_target_filter)
                # edit_response should only contain warnings, and even these
                # are info-level, rather than actual warnings
                assert edit_response.ok
                response = merge(response, edit_response)
                predict_prefix = 'PREDICT '
                target_field.position = _tuple_inc(target_field.position,
                                                   len(predict_prefix))
                target_field.bold_text_position = _tuple_inc(
                    target_field.bold_text_position, len(predict_prefix))
                target_repr = predict_prefix + target_repr
                editable_fields.append(target_field)
        else:
            target_repr = 'PREDICT ' + str(query.target_ast)
        if query.problem_type == ProblemType.FORECAST:
            target_repr += f'\nFORECAST {query.num_forecasts} TIMEFRAMES'
            target_repr += '\nFOR EACH '
        else:
            if query.problem_type is not None:
                target_repr += f' {query.problem_type.value}'
            if query.top_k is not None:
                target_repr += f' TOP {query.top_k}'
            target_repr += ' FOR EACH '
        prefix_len = len(target_repr)
        entity_field = self._get_editable_entity(query.entity_ast)
        entity_field.position = _tuple_inc(entity_field.position, prefix_len)
        entity_field.bold_text_position = _tuple_inc(
            entity_field.bold_text_position, prefix_len)
        editable_fields.append(entity_field)
        query_repr = target_repr + str(query.entity_ast)
        return EditableQuery(query_repr, editable_fields, response)

    def insert_filter_overrides(
        self,
        user_edits: EditableQueryResponse,
        graph: GraphDefinition,
        task_type: TaskType,
    ) -> tuple[EditableQuery, ValidatedPredictiveQuery | None]:
        r"""This query takes user edits of the filter and produces a new
        editable query to show to the user. If errors/warnings were
        encountered, they're given as EditableQuery.validation_response.
        If no errors were encountered, the method also returns a validated
        predictive query AST (otherwise :obj:`None`). The text in
        `EditableQuery` may not match the validated predictive query!
        The former is user-displayed query and may (in the case of LP)
        contain references to tables not available at BP time. The latter
        is ready for execution.

        Args:
            user_edits: EditableQuery and user edits.
            graph: The graph to use for validation.
            task_type: Task type of the query.

        Returns:
            EditableQuery for user display (+ validation) and validated
            predictive query for execution, if no errors were encountered.
            Otherwise, :obj:`None`.
        """
        editable_query = deepcopy(user_edits.editable_query)
        editable_query.validation_response = ValidationResponse()

        # a counter of position changes due to user edits
        prefix_len_delta = 0
        # Text without the ignored non-applicable_parts
        applicable_text = editable_query.text
        appl_delta = 0

        # We process fields from left to right to keep track of the position
        # changes
        for edit_field in sorted(editable_query.editable_fields,
                                 key=lambda x: x.position):
            edit_field.position = _tuple_inc(edit_field.position,
                                             prefix_len_delta)
            edit_field.bold_text_position = _tuple_inc(
                edit_field.bold_text_position, prefix_len_delta)
            if edit_field.field_name not in user_edits.user_changes:
                if len(edit_field.text) != 0:
                    # edit_field.text is the applicable/user editable text
                    # insert that instead of the original query
                    new_applicable_text = f' WHERE {edit_field.text}'
                    applicable_text = (
                        applicable_text[:edit_field.position[0] + appl_delta] +
                        new_applicable_text +
                        applicable_text[edit_field.position[1] + appl_delta:])
                    appl_delta += len(new_applicable_text) - (
                        edit_field.position[1] - edit_field.position[0])
                continue
            user_change = user_edits.user_changes[edit_field.field_name]
            if user_change == '':
                bold_text_offset = 0
                new_text = edit_field.empty_filter_default
            else:
                bold_text_offset = edit_field.full_filter_format_string.find(
                    '{0}')
                new_text = edit_field.full_filter_format_string.format(
                    user_change)
            editable_query.text = (
                editable_query.text[:edit_field.position[0]] + new_text +
                editable_query.text[edit_field.position[1]:])
            prefix_len_delta += len(new_text) - (edit_field.position[1] -
                                                 edit_field.position[0])
            edit_field.text = user_change
            # Applicable query is only updated with the user change, the
            # non-applicable part is not inserted since it's ... non-applicable
            new_applicable_text = ''
            if len(user_change) > 0:
                new_applicable_text = f' WHERE {user_change}'
            applicable_text = (
                applicable_text[:edit_field.position[0] + appl_delta] +
                new_applicable_text +
                applicable_text[edit_field.position[1] + appl_delta:])
            appl_delta += len(new_applicable_text) - len(new_text)

            edit_field.position = (edit_field.position[0],
                                   edit_field.position[0] + len(new_text))
            edit_field.bold_text_position = (edit_field.position[0] +
                                             bold_text_offset,
                                             edit_field.position[0] +
                                             bold_text_offset +
                                             len(user_change))
        # We've got two queries, the user-displayed one and the actually
        # applicable one from which we've removed the non-applicable bits.
        # We parse the first one for validation, since that's what user sees
        # The second one is parsed for execution.
        parser = PQLParser()
        try:
            parsed_query = parser.to_parsed_predictive_query(
                editable_query.text)
        except ValueError as e:
            editable_query.validation_response.errors.append(
                ValidationError(title='Invalid query syntax.', message=str(e)))
            return editable_query, None
        parsed_applicable_query = parser.to_parsed_predictive_query(
            applicable_text)
        validator = PredictiveQueryValidator(graph)
        _, response = validator.validate_predictive_query(parsed_query)
        editable_query.validation_response = merge(
            editable_query.validation_response, response)
        (validated_applicable_query, appl_response
         ) = validator.validate_predictive_query(parsed_applicable_query)
        if response.ok:
            assert appl_response.ok
            assert validated_applicable_query is not None
            editable_query.validation_response = merge(
                editable_query.validation_response,
                validator.validate_bp_filter_overrides(
                    validated_applicable_query, task_type))
            if not editable_query.validation_response.ok:
                validated_applicable_query = None
        return editable_query, validated_applicable_query

    def split_target_filter(self, node: ASTNode, lhs_table: str,
                            rhs_table: str) -> SplitFilter:
        r"""Splits the target ast defined by `node` in the following way:
        If there are any references to the `lhs_table`, the method raises
        an error. Such references are not allowed in the target filter because
        we do not support applying them at BP time yet. Otherwise it extracts a
        the target filter and splits it into two objects:
        1. a filter that only contains references to `rhs_table`.
        2. a filter that contains the rest of the conditions.
        3. an indicator whether rewriting was required for the above split.
        The first filter is designed to be conservative in the sense that it
        should only filter RHS targets that the original filter would surely
        have filtered.
        E.g. `edge.X>1 AND rhs_table.Y=1` should get split into
        `rhs_table.Y=1` and `edge.X>1`. The first of the two filters can be
        applied at BP time because any filter that doesn't meet that simpler
        condition will surely fail the full condition. On the other hand,
        `edge.X > 1 OR rhs_table.Y = 1` cannot be transformed into another
        filter with only `rhs_table` reference because a row that does not
        meet the right side of the condition might still meet the full
        condition, if `edge.X > 1`, which is unknown at BP time.
        In that case, the method will retun `None` as the
        `bp_applicable_filter` and the entire original filter as
        `bp_non_applicable_filter`. This can be interpreted as "no filter
        should be applied at BP time".
        Args:
            node (ASTNode): Target filter to be transformed.
            lhs_table (str): Name of the table of LHS nodes (entity table).
            rhs_table (str): Name of the table of RHS nodes.

        Returns:
            SplitTargetFilter: The split filters. If one of the filters is
            empty, :obj:`None` is returned instead of a new node.
        """
        # Note: Without hops, the transformed target filter can only be
        # static because RHS table is a dim table and one-hop filter can
        # only be static.
        # Revisit this code when hops are introduced or when fkeys can be
        # specified as we'll need to support two additional scenarios:
        # 1. Temporal aggregation one hop away from the RHS entity
        # (two hops away from the target), e.g. only RHS targets with at
        # least N sessions.
        # 2. The filter is using RHS table, but it's connected through a
        # fkey that's not the target column - it should be dropped.
        if isinstance(node, Filter):
            return self._recursively_split_filter(
                node=node.condition, original_filter=node.condition,
                lhs_table=lhs_table, rhs_table=rhs_table)
        if isinstance(node, Join):
            return self.split_target_filter(node.rhs_target, lhs_table,
                                            rhs_table)
        if isinstance(node, Aggregation):
            # must be the target aggregation
            assert node.aggr == AggregationType.LIST_DISTINCT
            return self.split_target_filter(node.target, lhs_table, rhs_table)
        if isinstance(node, Column):
            return SplitFilter(
                applicable_filter=None,
                non_applicable_filter=None,
                rewritten=False,
                n_applicable_conditions=0,
                n_non_applicable_conditions=0,
                non_applicable_is_valid=True,
                original_filter=None,
            )
        raise AssertionError(f"Unknown node type {type(node)}.")

    def _condition_base_case(  # type: ignore
        self,
        node: Condition,
        original_filter: Condition | LogicalOperation | None,
        lhs_table: str,
        rhs_table: str,
    ) -> SplitFilter:
        """Base case for the recursive filter splitting. A condition is not
        applicable if its target contains references to the edge table.
        Otherwise, it should be applicable.

        Args:
            node: The condition to split.
            original_filter: The original full filter being split.
            lhs_table: The name of the LHS table.
            rhs_table: The name of the RHS table.

        Raises:
            ValueError: If the condition contains a reference to the LHS table.

        Returns:
            A split filter describing whether the input condition
            is applicable or not.
        """
        target = node.target
        if isinstance(target, Join):
            target = target.rhs_target
        if isinstance(target, Aggregation):
            # Cannot be in RHS until multi-hops
            return SplitFilter(
                applicable_filter=None,
                non_applicable_filter=node,
                rewritten=False,
                n_applicable_conditions=0,
                n_non_applicable_conditions=1,
                non_applicable_is_valid=True,
                original_filter=original_filter,
            )
        assert isinstance(target, Column)
        if table_name(target.fqn) == lhs_table:
            # Should be handled by the PredictiveQueryValidator already.
            raise ValueError(f'Target filter condition {node} contains a '
                             f'reference to the entity table. Entity table '
                             f'references in target filters are not '
                             f'supported, please use entity filter instead.')
        if table_name(target.fqn) != rhs_table:
            return SplitFilter(
                applicable_filter=None,
                non_applicable_filter=node,
                rewritten=False,
                n_applicable_conditions=0,
                n_non_applicable_conditions=1,
                non_applicable_is_valid=True,
                original_filter=original_filter,
            )
        return SplitFilter(
            applicable_filter=node,
            non_applicable_filter=None,
            rewritten=False,
            n_applicable_conditions=1,
            n_non_applicable_conditions=0,
            non_applicable_is_valid=True,
            original_filter=original_filter,
        )

    def generate_backtesting_query(
        self,
        training_query: ValidatedPredictiveQuery,
        prediction_query: ValidatedPredictiveQuery,
        task_type: TaskType,
        graph: GraphDefinition,
    ) -> ValidatedPredictiveQuery:
        r"""Combines the training query and the prediction query that
        maybe contains overrides into a single backtesting query.
        Backtesting means testing prediciton on ground-truth labels, once
        available. This requires combining filter overrides with the training
        query, if any were used.
        This query contains entity filter from the `prediction_query` and
        whatif filter from the `training_query`.
        If node prediction, target is taken from `training_query`. If
        link prediction, target is obtained by combining `training_query`
        and possible filter overrides in `prediction_query`. The code should
        work regardless of whether `prediction_query` contains the
        non-BP-applicable part of the filter or not.

        Args:
            training_query: Query used during
                training.
            prediction_query: Query used during
                batch prediction.
            task_type: Task type.
            graph: Graph on which the query is performed.

        Returns:
            The combined query. If no filter overrides were applied, this is
            equivalent to `training_query`.

        Raises:
            ValueError: If the `training_query` contains more than
                `MAX_N_CONDITIONS` conditions after rewriting.
        """
        if not task_type.is_link_pred:
            # If node prediction, there are no target filter overrides
            # As such, we take the entity from the prediction query
            # to get the entity filter, any target (they are the same), and
            # training whatif ast.
            return ValidatedPredictiveQuery(
                entity_ast=prediction_query.entity_ast,
                target_ast=training_query.target_ast,
                validation_response=ValidationResponse(),
                whatif_ast=training_query.whatif_ast,
                top_k=prediction_query.top_k,
                num_forecasts=prediction_query.num_forecasts,
                problem_type=training_query.problem_type,
            )
        entity_table = table_name(training_query.entity_column)
        # given in "connector.table_name" format
        rhs_table = get_rhs_lp_table(training_query, graph)
        split_training_filter = self.split_target_filter(
            training_query.target_ast, entity_table, rhs_table)
        split_prediction_filter = self.split_target_filter(
            prediction_query.target_ast, entity_table, rhs_table)
        combined_target = None
        if (split_prediction_filter.applicable_filter ==
                split_training_filter.applicable_filter):
            # No overrides happened, we can use the original target
            # This thus works even if `non_applicable_is_valid` is False.
            combined_target = training_query.target_ast
        elif not split_training_filter.non_applicable_is_valid:
            raise ValueError(
                f"Backtesting not supported for this query because of the "
                f"complicated filter in the target. Target contains "
                f"{split_training_filter.n_non_applicable_conditions} "
                f"conditions after splitting.")
        elif (split_prediction_filter.applicable_filter is None
              and split_training_filter.non_applicable_filter is None):
            # This can only happen if the training filter only contained a
            # bp_applicable bit and it got deleted for target filter, resulting
            # in an empty filter.
            assert split_prediction_filter.non_applicable_filter is None
            combined_target = prediction_query.target_ast
        else:
            # We need to combine the two filters.
            combined_filter = None
            if split_training_filter.non_applicable_filter is None:
                combined_filter = split_prediction_filter.applicable_filter
            elif split_prediction_filter.applicable_filter is None:
                combined_filter = split_training_filter.non_applicable_filter
            else:
                combined_filter = LogicalOperation(
                    left=split_training_filter.non_applicable_filter,
                    bool_op=BoolOp.AND,
                    right=split_prediction_filter.applicable_filter,
                    dtype_maybe=Dtype.bool,
                    stype_maybe=Stype.categorical,
                )
            # To avoid re-inferring metadata, we take one of the existing
            # target ASTs with a filter and copy it, then modify it in place
            # One of them is guaranteed to have it.
            if split_prediction_filter.applicable_filter is None:
                combined_target = deepcopy(training_query.target_ast)
            else:
                combined_target = deepcopy(prediction_query.target_ast)
            # locate the original filter and overwrite it
            assert isinstance(combined_target, Join)
            assert isinstance(combined_target.rhs_target, Aggregation)
            assert (combined_target.rhs_target.aggr ==
                    AggregationType.LIST_DISTINCT)
            assert isinstance(combined_target.rhs_target.target, Filter)
            assert combined_filter is not None
            combined_target.rhs_target.target.condition = combined_filter
            print(f'KOMBINACIJA -- {combined_target}')
        return ValidatedPredictiveQuery(
            entity_ast=prediction_query.entity_ast,
            target_ast=combined_target,
            validation_response=ValidationResponse(),
            whatif_ast=training_query.whatif_ast,
            top_k=prediction_query.top_k,
            num_forecasts=prediction_query.num_forecasts,
            problem_type=training_query.problem_type,
        )


def _tuple_inc(tup: tuple[int, int], inc: int) -> tuple[int, int]:
    return (tup[0] + inc, tup[1] + inc)
