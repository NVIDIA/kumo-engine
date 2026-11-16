# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from kumoapi.common import ValidationError, ValidationResponse
from kumoapi.graph import ColumnKey, GraphDefinition
from kumoapi.pquery import ParsedPredictiveQuery
from kumoapi.pquery.AST import (
    Aggregation,
    ASTNode,
    Column,
    Condition,
    Filter,
    Join,
    LogicalOperation,
)
from kumopql.validator.utils import col_name, merge, table_name


class JoinValidator:
    r"""Class with all the logic around :class:`ASTNode` join inferral and
    validation.

    Args:
        graph: Graph on which the query is defined.
    """
    def __init__(self, graph: GraphDefinition) -> None:
        self.graph = graph
        self.target_filter_example: str | None = None
        self.entity_filter_tables: set[str] = set()
        self.entity_filter_is_row_level = False
        self.allow_temporal_entity_filter = False

    def infer_and_validate_joins(
        self,
        parsed_query: ParsedPredictiveQuery,
    ) -> ValidationResponse:
        r"""Modifies the AST in place, inserting :class:`Join` nodes where
        appropriate. Returns errors/warnings if there are issues resolving
        fkey to pkey relations, like ambiguous references, or missing keys.

        Args:
            parsed_query: the input parsed query.

        Returns:
            ValidationResponse: List of encountered errors/warnings.
        """
        entity_column = parsed_query.entity_column
        entity_table = table_name(entity_column)
        entity_filter_condition = None
        self.entity_filter_tables = set()
        self.entity_filter_is_row_level = False
        if isinstance(parsed_query.entity_ast, Filter):
            condition = parsed_query.entity_ast.condition
            assert condition is not None
            entity_filter_condition = repr(condition)
            self.entity_filter_tables = {
                table_name(column)
                for column in condition.all_query_columns
            }
            self.entity_filter_is_row_level = not self._contains_aggregation(
                condition)
        self.allow_temporal_entity_filter = (
            parsed_query.target_ast.date_offset_range is not None)
        self.target_filter_example = self._get_target_filter_example(
            parsed_query)
        response = ValidationResponse()
        entity_ast, entity_response = self._infer_joins(
            parsed_query.entity_ast,
            entity_table,
            in_entity_filter=isinstance(parsed_query.entity_ast, Filter),
            entity_filter_condition=entity_filter_condition,
        )

        assert isinstance(entity_ast, Column | Filter)
        parsed_query.entity_ast = entity_ast
        parsed_query.target_ast, target_response = self._infer_joins(
            parsed_query.target_ast, entity_table)
        response = merge(response, entity_response)
        response = merge(response, target_response)
        if parsed_query.whatif_ast is not None:
            whatif_ast, whatif_response = self._infer_joins(
                parsed_query.whatif_ast, entity_table)
            assert isinstance(whatif_ast, Condition) or isinstance(
                whatif_ast, LogicalOperation)
            parsed_query.whatif_ast = whatif_ast
            response = merge(response, whatif_response)
        return response

    def _get_target_filter_example(
            self, parsed_query: ParsedPredictiveQuery) -> str | None:
        entity_ast = parsed_query.entity_ast
        target_ast = parsed_query.target_ast
        if (not isinstance(entity_ast, Filter)
                or not isinstance(target_ast, Aggregation)
                or not isinstance(target_ast.target, Column)):
            return None
        condition = entity_ast.condition
        assert condition is not None
        if self._contains_aggregation(condition):
            return None

        condition_tables = {
            table_name(column)
            for column in condition.all_query_columns
        }
        target_column = target_ast.target.fqn
        if condition_tables != {table_name(target_column)}:
            return None

        filtered_column = f'{target_column} WHERE {entity_ast.condition}'
        return repr(target_ast).replace(target_column, filtered_column, 1)

    def _contains_aggregation(self, node: ASTNode) -> bool:
        r"""Returns whether the AST subtree contains an aggregation."""
        if isinstance(node, Aggregation):
            return True
        return any(self._contains_aggregation(child) for child in node.children)

    def _find_keys(
        self,
        node: ASTNode,
        fkey_table: str,
        pkey_table: str,
        is_target_entity: bool = False,
        in_entity_filter: bool = False,
        entity_filter_condition: str | None = None,
    ) -> tuple[list[str], ValidationResponse]:
        r"""Tries to find the unique foreign key that points from `fkey_table`
        to `pkey_table`. If it cannot be determined uniquely, an error is
        returned instead.

        Args:
            node: Subquery, output of which needs to be joined.
                Currently only used for a more informative error message when
                the key can't be determined.
            fkey_table: Name of the table that contains the foreign key.
            pkey_table: Name of the table that contains the primary key.
            is_target_entity: Whether it's finding keys between target
                and entity tables. If :obj:`True`, it will add an additional
                error message to the response.
            in_entity_filter: Whether the node is used directly in an entity
                filter.
            entity_filter_condition: The entity filter condition that contains
                the node.

        Returns:
            The list of all valid foreign
            keys and a validation response. Validation response contains an
            error if the length of the list is not 1.
        """
        response = ValidationResponse()
        keys = []
        reverse_keys = []
        for col_group in self.graph.col_groups:
            is_keys_group = False
            is_rev_keys_group = False
            table_of_interest = None
            if (ColumnKey(pkey_table, self.graph.tables[pkey_table].pkey)
                    in col_group.columns):
                is_keys_group = True
                table_of_interest = fkey_table
            elif (self.graph.tables[fkey_table].pkey is not None
                  and ColumnKey(fkey_table, self.graph.tables[fkey_table].pkey)
                  in col_group.columns):
                is_rev_keys_group = True
                table_of_interest = pkey_table
            if not is_keys_group and not is_rev_keys_group:
                continue
            assert isinstance(table_of_interest, str)
            # When fkey_table == pkey_table (self-referential), exclude the
            # pkey itself from the results - we only want actual foreign keys.
            pkey_col = self.graph.tables[pkey_table].pkey
            curr_keys = [
                f'{col.table_name}.{col.col_name}' for col in col_group.columns
                if col.table_name == table_of_interest
                and not (fkey_table == pkey_table and col.col_name == pkey_col)
            ]
            if is_keys_group:
                keys = curr_keys
            else:
                assert is_rev_keys_group
                reverse_keys = curr_keys

        if isinstance(node, Aggregation):
            pkey_name = self.graph.tables[pkey_table].pkey
            additional_explanation = (
                f"To unambiguously link each row in '{fkey_table}' to a "
                f"'{pkey_table}.{pkey_name}', exactly one foreign key should "
                f"point from '{fkey_table}' to '{pkey_table}.{pkey_name}'.")
        else:
            additional_explanation = (
                f"To unambiguously link each row in '{fkey_table}' to a "
                f"'{node}', exactly one foreign key should "
                f"point from '{fkey_table}' to '{pkey_table}'.")

            # Add an additional error message to the response
            # which captures the case that the target column is not
            # reachable from the source table through a foreign key.
            # Here is an example where target column
            # TRANSACTIONS.AMOUNT is not connected to the source table
            # USERS through a foreign key in the USERS table:
            # PREDICT TRANSACTIONS.AMOUNT FOR EACH USERS.USER_ID
            # Here source table is the fkey_table
            if is_target_entity:
                additional_explanation += (
                    f" This may also mean there is an incorrect "
                    f"target column specification. Column {node.fqn} is "
                    f"not reachable through a foreign key in table "
                    f"{fkey_table}.")
        if len(keys) == 0:
            # This is likely the case where the target column is not reachable
            # unambiguously from the entity table. For example, for the query:
            # PREDICT transactions.age FOR EACH customers.customer_id
            # where there is a foreign key from transactions to customers (
            # containing the primary key). We do not support this if there is
            # no aggregation.
            is_aggregation = False
            if len(reverse_keys) > 0:
                if not isinstance(node, Aggregation):
                    assert isinstance(node, Column)
                    column_name = col_name(node.fqn)
                else:
                    is_aggregation = True
                if not is_aggregation and in_entity_filter:
                    condition = entity_filter_condition or repr(node)
                    fixes = []
                    if self.target_filter_example is not None:
                        fixes.append(
                            'To filter events used by the prediction target, '
                            f"use '{self.target_filter_example}'.")
                    if (self.allow_temporal_entity_filter
                            and entity_filter_condition is not None
                            and self.entity_filter_is_row_level
                            and self.entity_filter_tables == {pkey_table} and
                            self.graph.tables[pkey_table].time_col is not None):
                        fixes.append(
                            'To select entities that have matching events, '
                            f"use 'COUNT({pkey_table}.* WHERE {condition}, "
                            "-INF, 0) > 0'.")
                    message = (
                        f'{node.get_location().message_start}: '
                        f"The entity filter condition '{condition}' "
                        f"references event column '{node.fqn}' directly. "
                        f"Table '{pkey_table}' can contain multiple rows for "
                        f"each '{fkey_table}', so there is no single value to "
                        f'compare for each entity. Move this condition inside '
                        f'an aggregation.')
                    if fixes:
                        message += f" {' '.join(fixes)}"
                    response.errors.append(
                        ValidationError(
                            title='Invalid entity filter',
                            message=message,
                        ))
                    return keys, response
                if not is_aggregation:
                    column_name = col_name(node.fqn)
                    response.errors.append(
                        ValidationError(
                            title=f"Couldn't pick a unique {node.fqn} for "
                            f"each {fkey_table}",
                            message=f"{node.get_location().message_start}: "
                            "Encountered an error, we have found multiple "
                            f"'{column_name}' values in the '{pkey_table}' "
                            f"table for a single '{fkey_table}'. To fix "
                            f"this, you have two options:\n"
                            "1. Aggregate the values:"
                            f" If your '{pkey_table}' table has a "
                            f"'created at' column, use an aggregation "
                            f"function to reduce them to one value. For "
                            f"example: PREDICT LAST({node.fqn}, 0, 30, "
                            f"days).\n"
                            "2. Rewrite your query:"
                            " If you are writing a static node "
                            "prediction predictive query, make sure that "
                            "you either use the same table to define your "
                            "target and entity, or you have a foreign key "
                            "from the entity pointing to the target table, "
                            "not the other way around.",
                        ))
            if is_aggregation or len(reverse_keys) == 0:
                response.errors.append(
                    ValidationError(
                        title="Missing foreign key",
                        message=f"{node.get_location().message_start}: "
                        f"Encountered an error when processing '{node}'. "
                        f"There is no foreign key from table "
                        f"{fkey_table} to table {pkey_table}. "
                        f"{additional_explanation}"))
        elif len(keys) > 1:
            response.errors.append(
                ValidationError(
                    title="Ambiguous link between tables",
                    message=f"{node.get_location().message_start}: "
                    f"Encountered an error when processing '{node}'. "
                    f"The foreign key from table {fkey_table} "
                    f"to table {pkey_table} is not unique (got {keys}). "
                    f"{additional_explanation}"))
        return keys, response

    def _infer_joins(
        self,
        node: ASTNode,
        source_table: str,
        in_entity_filter: bool = False,
        entity_filter_condition: str | None = None,
    ) -> tuple[ASTNode, ValidationResponse]:
        r"""Ensures that :obj:`node` is reachable from `source_table` in a
        single hop, adding joins if necessary. All nodes are modified in-place
        except where a new :class:`Join` needs to
        be added.

        Args:
            node (ASTNode): Node with missing joins. This is typically an
                aggregation, a column or a condition, which is performed
                in the context of an entity or a target.
            source_table (str): name of the table that we are trying to
                reach :obj:`node` from.
            in_entity_filter: Whether the node is used directly in an entity
                filter.
            entity_filter_condition: The entity filter condition that contains
                the node.

        Returns:
            tuple[ASTNode, ValidationResponse]: a modified node with added
                joins and validation errors/warnings encountered.
        """
        # We handle five cases:
        # 1. Column might get replaced with Join->Column if
        #    it's one-hop away
        # 2. Aggregation always gets replaced with Join->Aggregation
        # 3. Condition is the only node where children might get replaced and
        #    not just modified in-place and return nodes need to be handled
        # 4. Filter has special handling of source table - source table for
        #    condition is the target table.
        # 5. LogicalOperation does not need any extra action
        response = ValidationResponse()
        if isinstance(node, Column):
            # fkey -> pkey hop (if necessary)
            col_table = table_name(node.fqn)
            if source_table != col_table:
                # There is a link to a separate table, ensure that it's unique
                keys, response = self._find_keys(
                    node, source_table, col_table, is_target_entity=True,
                    in_entity_filter=(in_entity_filter),
                    entity_filter_condition=(entity_filter_condition))
                if not response.ok:
                    return node, response
                rhs_pkey = self.graph.tables[col_table].pkey
                return Join(
                    rhs_target=node,
                    lhs_key=keys[0],
                    rhs_key=col_table + '.' + rhs_pkey,
                ), response
        elif isinstance(node, Aggregation):
            # pkey -> fkey hop
            target_table = table_name(node.get_target_column_name())
            node.target, target_response = self._infer_joins(
                node.target, target_table)
            response = merge(response, target_response)
            keys, join_response = self._find_keys(node, target_table,
                                                  source_table)
            response = merge(response, join_response)
            if not response.ok:
                return node, response
            node.group_by = keys[0]
            lhs_pkey = self.graph.tables[source_table].pkey
            return Join(rhs_target=node, lhs_key=source_table + '.' + lhs_pkey,
                        rhs_key=keys[0]), response
        elif isinstance(node, Condition):
            # replace the target
            condition_text = entity_filter_condition
            if in_entity_filter and condition_text is None:
                condition_text = repr(node)
            node.target, target_response = self._infer_joins(
                node.target,
                source_table,
                in_entity_filter=in_entity_filter,
                entity_filter_condition=condition_text,
            )
            response = merge(response, target_response)
        elif isinstance(node, Filter):
            # special handling of condition source table
            target_table = table_name(node.target.fqn)
            target, target_response = self._infer_joins(node.target,
                                                        source_table)
            assert isinstance(target, Column)
            node.target = target
            response = merge(response, target_response)
            condition, condition_response = self._infer_joins(
                node.condition,
                target_table,
                in_entity_filter=in_entity_filter,
                entity_filter_condition=entity_filter_condition,
            )
            assert isinstance(condition, Condition) or isinstance(
                condition, LogicalOperation)
            node.condition = condition
            response = merge(response, condition_response)
        else:
            for child in node.children:
                modified_child, child_response = self._infer_joins(
                    child,
                    source_table,
                    in_entity_filter=in_entity_filter,
                    entity_filter_condition=entity_filter_condition,
                )
                # assert that no re-initialisation is necessary
                assert child is modified_child
                response = merge(response, child_response)
        return node, response
