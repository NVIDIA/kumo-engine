# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
from typing import Any

import pandas as pd
from kumoapi.common import (
    ValidationError,
    ValidationResponse,
    ValidationWarning,
)
from kumoapi.graph import ColumnKey, GraphDefinition
from kumoapi.pquery import ParsedPredictiveQuery
from kumoapi.pquery.AST import (
    Aggregation,
    ASTNode,
    Column,
    Condition,
    Constant,
    Filter,
    LogicalOperation,
)
from kumoapi.pquery.AST.ast_node import ArrayDtype
from kumoapi.pquery.utils import validate_int
from kumoapi.typing import (
    AggregationType,
    Dtype,
    MemberOp,
    RelOp,
    StrOp,
    Stype,
)
from kumopql.validator.utils import col_name, merge, table_name


class NodeTypeInferrer:
    r"""Class with all the logic around :class:`ASTNode` specific
    :class:`Dtype` and :class:`Stype` inferral.

    Args:
        graph: Graph on which the inferrals are performed.
        allow_array_targets: Have LIST_DISTINCT generate an array instead of
            a "|"-separated string.
    """
    def __init__(
        self,
        graph: GraphDefinition,
        allow_array_targets: bool = False,
    ) -> None:
        self.graph = graph
        self.allow_array_targets = allow_array_targets

    @abstractmethod
    def infer_dtype(self, node: ASTNode) -> None:
        r"""Infers data type of the node, writing it to ASTNode.dtype_maybe.

        Args:
            node: the input AST node.

        Raises:
            ValueError: If `self.graph` comes without data types .
        """
        raise NotImplementedError

    @abstractmethod
    def infer_stype(self, node: ASTNode) -> None:
        r"""Infers semantic type of the node, writing it to
        ASTNode.stype_maybe.

        Args:
            node: the input AST node.

        Raises:
            ValueError: If `self.graph` comes without semantic types .
        """
        raise NotImplementedError


class AggrNodeTypeInferrer(NodeTypeInferrer):
    def _infer_dtype_aggr(self, node: Aggregation) -> None:
        if node.aggr in [
                AggregationType.COUNT,
                AggregationType.COUNT_DISTINCT,
        ]:
            node.dtype_maybe = Dtype.int
            return
        if node.aggr in [
                AggregationType.AVG,
        ]:
            node.dtype_maybe = Dtype.float
            return
        if node.aggr in [
                AggregationType.LIST_DISTINCT,
        ]:
            if self.allow_array_targets:
                assert isinstance(node.target.dtype_maybe, Dtype | ArrayDtype)
                if isinstance(node.target.dtype_maybe, ArrayDtype):
                    node.dtype_maybe = node.target.dtype_maybe
                else:
                    node.dtype_maybe = ArrayDtype(node.target.dtype_maybe)
            else:
                assert isinstance(node.target.dtype_maybe, Dtype)
                node.dtype_maybe = Dtype.string
            return
        # SUM, MIN, MAX, LAST, FIRST
        if node.aggr in [
                AggregationType.SUM,
        ] and node.target.dtype_maybe == Dtype.bool:
            node.dtype_maybe = Dtype.int
            return
        node.dtype_maybe = node.target.dtype_maybe

    def _infer_stype_aggr(self, node: Aggregation) -> None:
        if node.aggr in [
                AggregationType.AVG,
                AggregationType.COUNT,
                AggregationType.COUNT_DISTINCT,
                AggregationType.SUM,
        ]:
            node.stype_maybe = Stype.numerical
            return
        if node.aggr in [
                AggregationType.LIST_DISTINCT,
        ]:
            node.stype_maybe = Stype.multicategorical
            return
        # MIN, MAX, LAST, FIRST
        node.stype_maybe = node.target.stype_maybe

    def infer_dtype(self, node: ASTNode) -> None:
        assert isinstance(node, Aggregation)
        self._infer_dtype_aggr(node)

    def infer_stype(self, node: ASTNode) -> None:
        assert isinstance(node, Aggregation)
        self._infer_stype_aggr(node)


class FilterNodeTypeInferrer(NodeTypeInferrer):
    def _infer_dtype_copy_target(self, node: Filter) -> None:
        node.dtype_maybe = node.target.dtype

    def _infer_stype_copy_target(self, node: Filter) -> None:
        node.stype_maybe = node.target.stype

    def infer_dtype(self, node: ASTNode) -> None:
        assert isinstance(node, Filter)
        self._infer_dtype_copy_target(node)

    def infer_stype(self, node: ASTNode) -> None:
        assert isinstance(node, Filter)
        self._infer_stype_copy_target(node)


class ColumnNodeTypeInferrer(NodeTypeInferrer):
    def _infer_stype_from_fqn(self, node: Column) -> None:
        assert isinstance(node, Column)
        if col_name(node.fqn) == '*':
            node.stype_maybe = Stype.categorical
            return
        target_col = [
            col for col in self.graph.tables[table_name(node.fqn)].cols
            if col.name == col_name(node.fqn)
        ][0]
        node.stype_maybe = target_col.stype

    def _infer_dtype_from_fqn(self, node: Column) -> None:
        assert isinstance(node, Column)
        if col_name(node.fqn) == '*':
            node.dtype_maybe = Dtype.int
            return
        target_col = [
            col for col in self.graph.tables[table_name(node.fqn)].cols
            if col.name == col_name(node.fqn)
        ][0]
        target_col_dtype = target_col.dtype
        if self.allow_array_targets:
            if target_col_dtype == Dtype.intlist:
                node.dtype_maybe = ArrayDtype(Dtype.int)
            elif target_col_dtype == Dtype.floatlist:
                node.dtype_maybe = ArrayDtype(Dtype.float)
            elif (target_col_dtype == Dtype.string
                  and target_col.stype == Stype.multicategorical):
                node.dtype_maybe = ArrayDtype(Dtype.string)
            else:
                node.dtype_maybe = target_col_dtype
        else:
            node.dtype_maybe = target_col_dtype

    def infer_dtype(self, node: Column) -> None:
        self._infer_dtype_from_fqn(node)

    def infer_stype(self, node: Column) -> None:
        self._infer_stype_from_fqn(node)


class ConstantNodeTypeInferrer(NodeTypeInferrer):
    def infer_dtype(self, node: ASTNode) -> None:
        assert isinstance(node, Constant)
        if not isinstance(node.value, list):
            return
        nested_types = [
            c.dtype_maybe for c in node.value if c.dtype_maybe is not None
        ]
        if all(t.is_int() for t in nested_types):
            node.dtype_maybe = Dtype.intlist
        elif all(t.is_numerical() for t in nested_types):
            node.dtype_maybe = Dtype.floatlist
        elif all(t.is_string() for t in nested_types):
            node.dtype_maybe = Dtype.stringlist

    def infer_stype(self, node: ASTNode) -> None:
        pass


class LogicalOperationNodeTypeInferrer(NodeTypeInferrer):
    def _infer_dtype_bool(self, node: LogicalOperation) -> None:
        node.dtype_maybe = Dtype.bool

    def _infer_stype_categorical(self, node: LogicalOperation) -> None:
        node.stype_maybe = Stype.categorical

    def infer_dtype(self, node: ASTNode) -> None:
        assert isinstance(node, LogicalOperation)
        self._infer_dtype_bool(node)

    def infer_stype(self, node: ASTNode) -> None:
        assert isinstance(node, LogicalOperation)
        self._infer_stype_categorical(node)


class ConditionNodeTypeInferrer(NodeTypeInferrer):
    def _infer_dtype_bool(self, node: Condition) -> None:
        node.dtype_maybe = Dtype.bool

    def _infer_stype_categorical(self, node: Condition) -> None:
        node.stype_maybe = Stype.categorical

    def infer_dtype(self, node: ASTNode) -> None:
        assert isinstance(node, Condition)
        self._infer_dtype_bool(node)

    def infer_stype(self, node: ASTNode) -> None:
        assert isinstance(node, Condition)
        self._infer_stype_categorical(node)


class NodeTypeValidator:
    r"""Class with all the logic around :class:`ASTNode` specific
    :class:`Dtype` and :class:`Stype` validation.

    Args:
        graph: Graph on which the query is defined.
        allow_array_targets: Have LIST_DISTINCT generate an array instead of
            a "|"-separated string.
        allow_timestamp_ararys: If :obj:`False`, block LIST_DISTINCT on time
            columns. Used for SPCS where timestamps are not handled well.
            Only applicable when `allow_array_targets` is :obj:`True`.
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

    @abstractmethod
    def validate_dtype(self, node: ASTNode) -> ValidationResponse:
        r"""Validate data type of every node in the AST, returning the list
        of errors.

        Args:
            node: the input node.

        Returns:
            ValidationResponse: List of encountered errors.
        """
        raise NotImplementedError

    @abstractmethod
    def validate_stype(self, node: ASTNode) -> ValidationResponse:
        r"""Validate semantic type of every node in the AST, returning the list
        of errors.

        Args:
            node: the input parsed query.

        Returns:
            ValidationResponse: List of encountered errors.
        """
        raise NotImplementedError


class AggrNodeTypeValidator(NodeTypeValidator):
    def _validate_dtype_aggr(self, node: Aggregation) -> ValidationResponse:
        response = ValidationResponse()
        if node.aggr in [
                AggregationType.AVG,
                AggregationType.SUM,
        ]:
            assert node.target.dtype is not None
            if not node.target.dtype.is_numerical():
                response.errors.append(
                    ValidationError(
                        message=f'{node.get_location().message_start}: '
                        f'Aggregation {node.aggr.value} can only '
                        f'operate on integers and floats, but {node.target} '
                        f'has data type {node.target.dtype.value}.',
                        title='Type Mismatch'))
        if node.aggr in [
                AggregationType.MIN,
                AggregationType.MAX,
        ]:
            assert node.target.dtype is not None
            if node.target.dtype == Dtype.bool:
                response.warnings.append(
                    ValidationWarning(
                        message=f'{node.get_location().message_start}: '
                        f'Aggregation {node.aggr.value} is not '
                        f'intended to operate on bool columns. During this '
                        f'label computation, True will be considered larger '
                        f'than False. To avoid this warning, save column '
                        f'{node.target} as an integer column instead.',
                        title='Type Mismatch'))

        # Snowflake has a bug where list_distinct on timestamps ends up as
        # an array of strings.
        if (self.allow_array_targets and not self.allow_timestamp_arrays
                and node.target.dtype.is_timestamp()):
            response.errors.append(
                ValidationError(
                    message=f'{node.get_location().message_start}: '
                    f'Aggregation {node.aggr.value} on columns of type '
                    f'timestamp is not supported on Snowflake.',
                    title='Unsupported type'))

        return response

    def _validate_stype_aggr(self, node: Aggregation) -> ValidationResponse:
        response = ValidationResponse()
        target_is_key = False
        target_fqn_name = node.get_target_column_name()
        if self.graph.tables[table_name(target_fqn_name)].pkey == col_name(
                target_fqn_name):
            target_is_key = True
        all_keys = []
        for col_group in self.graph.col_groups:
            all_keys.extend(list(col_group.columns))
        if (ColumnKey(table_name(target_fqn_name), col_name(target_fqn_name))
                in all_keys):
            target_is_key = True
        if (node.aggr not in [
                AggregationType.LIST_DISTINCT,
                AggregationType.LAST,
                AggregationType.FIRST,
                AggregationType.COUNT,
                AggregationType.MIN,
                AggregationType.MAX,
        ] and (node.target.stype in [Stype.ID] or target_is_key)):
            # unsupported aggregations on fkey columns can cause crashes
            response.errors.append(
                ValidationError(
                    title='Semantic Type Mismatch',
                    message=f'{node.get_location().message_start}: '
                    f'Aggregation {node.aggr} is not intended to '
                    f'operate on foreign keys and ID-type columns.'))
        if (node.aggr in [AggregationType.MIN, AggregationType.MAX]
                and node.target.stype
                not in [Stype.numerical, Stype.text, Stype.timestamp]):
            response.warnings.append(
                ValidationWarning(
                    message=f'Aggregation {node.aggr} is intended to operate '
                    f'on numerical, text, and timestamp columns, but '
                    f'{node.target} has semantic type {node.target.stype}. '
                    f'Consider changing the underlying semantic type to fit '
                    f'the task better.', title='Semantic Type Mismatch'))
        if (node.aggr in [AggregationType.AVG, AggregationType.SUM]
                and node.target.stype != Stype.numerical):
            response.warnings.append(
                ValidationWarning(
                    message=f'{node.get_location().message_start}: '
                    f'Aggregation {node.aggr} is intended to operate '
                    f'on numerical columns, but {node.target} has semantic '
                    f'type {node.target.stype}. Consider changing the '
                    f'underlying semantic type to fit the task better.',
                    title='Semantic Type Mismatch'))
        if (node.aggr == AggregationType.LIST_DISTINCT and node.target.stype
                not in [Stype.categorical, Stype.multicategorical, Stype.ID]):
            response.warnings.append(
                ValidationWarning(
                    message=f'{node.get_location().message_start}: '
                    f'Aggregation {node.aggr} is intended to operate '
                    f'on categorical and multicategorical columns, but '
                    f'{node.target} has semantic type {node.target.stype}. '
                    f'Consider changing the underlying semantic type to fit '
                    f'the task better.', title='Semantic Type Mismatch'))
        return response

    def validate_dtype(self, node: ASTNode) -> ValidationResponse:
        assert isinstance(node, Aggregation)
        return self._validate_dtype_aggr(node)

    def validate_stype(self, node: ASTNode) -> ValidationResponse:
        assert isinstance(node, Aggregation)
        return self._validate_stype_aggr(node)


class ColumnNodeTypeValidator(NodeTypeValidator):
    def validate_dtype(self, node: ASTNode) -> ValidationResponse:
        return ValidationResponse()

    def validate_stype(self, node: ASTNode) -> ValidationResponse:
        return ValidationResponse()


class ConstantNodeTypeValidator(NodeTypeValidator):
    def validate_dtype(self, node: ASTNode) -> ValidationResponse:
        return ValidationResponse()

    def validate_stype(self, node: ASTNode) -> ValidationResponse:
        return ValidationResponse()


class ConditionNodeTypeValidator(NodeTypeValidator):
    def _validate_dtype_condition(self, node: Condition) -> ValidationResponse:
        response = ValidationResponse()
        if isinstance(node.op, MemberOp):
            response = merge(response, self._validate_array_dtype(node))
        if isinstance(node.op, StrOp):
            response = merge(response, self._validate_strop_dtype(node))
        if isinstance(node.op, RelOp):
            response = merge(response, self._validate_relop_dtype(node))
        if (isinstance(node.value, Constant)
                and node.value.dtype_maybe is not None
                and node.value.dtype.is_int()):
            response = merge(response, validate_int(node.value.typed_value()))
        return response

    def _value_dtype_mismatch(self, dtype: Dtype | ArrayDtype, value: Any,
                              allow_null: bool) -> bool:
        if isinstance(value, str):
            return not dtype.is_string()
        elif isinstance(value, bool):
            return not dtype.is_bool()
        elif isinstance(value, int) or isinstance(value, float):
            return not dtype.is_numerical()
        elif pd.api.types.is_datetime64_any_dtype(type(value)) or isinstance(
                value, pd.Timestamp):
            return not dtype.is_timestamp()
        elif value is None:
            # Validated elsewhere based on relop
            return not allow_null
        # fallback for nested arrays which we do not
        # support with array operations
        return True

    def _validate_relop_dtype(self, node: Condition) -> ValidationResponse:
        response = ValidationResponse()
        assert isinstance(node.value, Constant)
        typed_val = node.value.typed_value()
        if isinstance(typed_val, list):
            response.errors.append(
                ValidationError(
                    message=f'{node.get_location().message_start}: '
                    f'Operator {node.op.value} does not support '
                    f'operations on arrays.', title='Type Mismatch'))
            return response
        if self._value_dtype_mismatch(node.target.dtype, typed_val,
                                      allow_null=True):
            value_str = node.value.to_string()
            quote_helper = ""
            if isinstance(node.value, str):
                value_str = f'"{value_str}"'
            elif node.target.dtype.is_string():
                quote_helper = " Did you forget to add quotes?"
            response.errors.append(
                ValidationError(
                    message=f'{node.get_location().message_start}: '
                    f'Expression {node.target} has type '
                    f'{node.target.dtype} but constant {value_str} '
                    f'has type {node.value.dtype_maybe}.{quote_helper}',
                    title='Type Mismatch'))
        if typed_val is None and node.op not in {RelOp.NEQ, RelOp.EQ}:
            # Comparing NULL with <,>,<=, >= is not permitted
            response.errors.append(
                ValidationError(
                    message=f'{node.get_location().message_start}: '
                    f'Expression {node.target} has type '
                    f'{node.target.dtype} but operator {node.op.value} '
                    f'does not support comparison with NULL. '
                    f'Please use IS NULL or IS NOT NULL.',
                    title='Type Mismatch'))
        if isinstance(typed_val, bool) and node.op not in {RelOp.NEQ, RelOp.EQ}:
            # Comparing bools with <,>,<=, >= is not permitted
            response.errors.append(
                ValidationError(
                    message=f'{node.get_location().message_start}: '
                    f'Expression {node.target} has type '
                    f'{node.target.dtype} but operator {node.op.value} '
                    f'does not support comparisons of booleans.',
                    title='Type Mismatch'))
        return response

    def _validate_strop_dtype(self, node: Condition) -> ValidationResponse:
        response = ValidationResponse()
        assert isinstance(node.value, Constant)
        typed_val = node.value.typed_value()
        value_str = node.value.to_string()
        assert node.target.dtype is not None
        if (self.allow_array_targets
                and isinstance(node.target.dtype, ArrayDtype)
                and node.op in [StrOp.CONTAINS, StrOp.NOT_CONTAINS]):
            # CONTAINS and NOT CONTAINS are supported on ArrayDtype
            if self._value_dtype_mismatch(node.target.dtype.nested_dtype,
                                          typed_val, allow_null=False):
                response.errors.append(
                    ValidationError(
                        message=f'{node.get_location().message_start}: '
                        f'Expression {node.target} has type '
                        f'{node.target.dtype.value} but constant {value_str} '
                        f'has type {node.value.dtype_maybe}.',
                        title='Type Mismatch'))
            return response
        elif not node.target.dtype.is_string():
            response.errors.append(
                ValidationError(
                    message=f'{node.get_location().message_start}: '
                    f'Operator {node.op.value} operates on strings '
                    f'but {node.target} has data '
                    f'type {node.target.dtype.value}.', title='Type Mismatch'))
        if not isinstance(typed_val, str):
            response.errors.append(
                ValidationError(
                    message=f'{node.get_location().message_start}: '
                    f'Operator {node.op.value} operates on strings '
                    f'but {value_str} has data type {node.value.dtype_maybe}. '
                    f'Did you forget to add quotes?', title='Type Mismatch'))
        return response

    def _validate_array_dtype(self, node: Condition) -> ValidationResponse:
        response = ValidationResponse()
        assert isinstance(node.value, Constant)
        typed_val = node.value.typed_value()
        value_str = node.value.to_string()
        if node.op == MemberOp.IS_IN:
            response.warnings.append(
                ValidationWarning(
                    message=f'{node.get_location().message_start}: '
                    f'Operator {node.op.value} is deprecated '
                    f'and will be removed in future versions of Kumo. '
                    f'Please use operator `IN` instead.',
                    title='Operation Deprecated'))
        if not isinstance(typed_val, list):
            response.errors.append(
                ValidationError(
                    message=f'{node.get_location().message_start}: '
                    f'Operator {node.op.value} expects an array '
                    f'constant but {value_str} has data type '
                    f'{node.value.dtype_maybe}.', title='Type Mismatch'))
            return response
        assert node.target.dtype is not None
        assert isinstance(node.value.value, list)
        for element in node.value.value:
            if self._value_dtype_mismatch(node.target.dtype,
                                          element.typed_value(),
                                          allow_null=False):
                response.errors.append(
                    ValidationError(
                        message=f'{node.get_location().message_start}: '
                        f'Expression {node.target} has type '
                        f'{node.target.dtype} but array contains element '
                        f'{element.to_string()} of type '
                        f'{element.dtype_maybe}.', title='Type Mismatch'))
                # Break to avoid reporting a mismatch for each element if the
                # error is on the target side.
                break
        return response

    def _validate_stype_condition(self, node: Condition) -> ValidationResponse:
        response = ValidationResponse()
        assert node.target.stype not in [Stype.unsupported]
        if (node.target.stype
                in [Stype.categorical, Stype.multicategorical, Stype.ID]
                and node.op in [RelOp.LEQ, RelOp.GEQ, RelOp.LT, RelOp.GT]):
            response.warnings.append(
                ValidationWarning(
                    message=f'{node.get_location().message_start}: '
                    f'Expression {node.target} has semantic '
                    f'type {node.target.stype}. But you are comparing it by '
                    f'value with operator {node.op.value}. Consider changing '
                    f'the underlying semantic type to fit the task better.',
                    title='Semantic Type Mismatch'))
        if node.target.stype in [Stype.ID] and isinstance(node.op, StrOp):
            response.warnings.append(
                ValidationWarning(
                    message=f'{node.get_location().message_start}: '
                    f'Expression {node.target} has semantic '
                    f'type {node.target.stype}. But you are comparing it by '
                    f'value with operator {node.op.value}. Consider changing '
                    f'the underlying semantic type to fit the task better.',
                    title='Semantic Type Mismatch'))
        return response

    def validate_dtype(self, node: ASTNode) -> ValidationResponse:
        assert isinstance(node, Condition)
        return self._validate_dtype_condition(node)

    def validate_stype(self, node: ASTNode) -> ValidationResponse:
        assert isinstance(node, Condition)
        return self._validate_stype_condition(node)


class FilterNodeTypeValidator(NodeTypeValidator):
    def _validate_dtype(self, node: Filter) -> ValidationResponse:
        # Note: it is currently impossible to trigger this error as a user
        # since the grammar does not permit it.
        assert node.condition.dtype == Dtype.bool
        return ValidationResponse()

    def _validate_stype(self, node: Filter) -> ValidationResponse:
        # Note: it is currently impossible to trigger this error as a user
        # since the grammar does not permit it.
        assert node.condition.stype == Stype.categorical
        return ValidationResponse()

    def validate_dtype(self, node: ASTNode) -> ValidationResponse:
        assert isinstance(node, Filter)
        return self._validate_dtype(node)

    def validate_stype(self, node: ASTNode) -> ValidationResponse:
        assert isinstance(node, Filter)
        return self._validate_stype(node)


class LogicalOperationNodeTypeValidator(NodeTypeValidator):
    def _validate_dtype(self, node: LogicalOperation) -> ValidationResponse:
        # Note: it is currently impossible to trigger this error as a user
        # since the grammar does not permit it.
        for child in node.children:
            assert child.dtype == Dtype.bool
        return ValidationResponse()

    def _validate_stype(self, node: LogicalOperation) -> ValidationResponse:
        # Note: it is currently impossible to trigger this error as a user
        # since the grammar does not permit it.
        for child in node.children:
            assert child.stype == Stype.categorical
        return ValidationResponse()

    def validate_dtype(self, node: ASTNode) -> ValidationResponse:
        assert isinstance(node, LogicalOperation)
        return self._validate_dtype(node)

    def validate_stype(self, node: ASTNode) -> ValidationResponse:
        assert isinstance(node, LogicalOperation)
        return self._validate_stype(node)


class TypeValidator:
    r"""Class with all the logic around :class:`ASTNode` :class:`Dtype` and
    :class:`Stype` inferral and validation.

    Args:
        graph: Graph on which the query is defined.
        allow_array_targets: Have LIST_DISTINCT generate an array instead of
            a "|"-separated string.
        allow_timestamp_ararys: If :obj:`False`, block LIST_DISTINCT on time
            columns. Used for SPCS where timestamps are not handled well.
            Only applicable when `allow_array_targets` is :obj:`True`.
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
        self.node_type_inferrer: dict[str, NodeTypeInferrer] = {
            'LogicalOperation':
            LogicalOperationNodeTypeInferrer(
                self.graph,
                self.allow_array_targets,
            ),
            'Condition':
            ConditionNodeTypeInferrer(
                self.graph,
                self.allow_array_targets,
            ),
            'Filter':
            FilterNodeTypeInferrer(
                self.graph,
                self.allow_array_targets,
            ),
            'Aggregation':
            AggrNodeTypeInferrer(
                self.graph,
                self.allow_array_targets,
            ),
            'Column':
            ColumnNodeTypeInferrer(
                self.graph,
                self.allow_array_targets,
            ),
            'Constant':
            ConstantNodeTypeInferrer(
                self.graph,
                self.allow_array_targets,
            ),
        }
        self.node_type_validator: dict[str, NodeTypeValidator] = {
            'LogicalOperation':
            LogicalOperationNodeTypeValidator(
                self.graph,
                self.allow_array_targets,
                self.allow_timestamp_arrays,
            ),
            'Condition':
            ConditionNodeTypeValidator(
                self.graph,
                self.allow_array_targets,
                self.allow_timestamp_arrays,
            ),
            'Filter':
            FilterNodeTypeValidator(
                self.graph,
                self.allow_array_targets,
                self.allow_timestamp_arrays,
            ),
            'Aggregation':
            AggrNodeTypeValidator(
                self.graph,
                self.allow_array_targets,
                self.allow_timestamp_arrays,
            ),
            'Column':
            ColumnNodeTypeValidator(
                self.graph,
                self.allow_array_targets,
                self.allow_timestamp_arrays,
            ),
            'Constant':
            ColumnNodeTypeValidator(
                self.graph,
                self.allow_array_targets,
                self.allow_timestamp_arrays,
            ),
        }

    def _infer_dtypes(self, node: ASTNode) -> None:
        for child in node.children:
            self._infer_dtypes(child)
        self.node_type_inferrer[node.__class__.__name__].infer_dtype(node)

    def _infer_stypes(self, node: ASTNode) -> None:
        for child in node.children:
            self._infer_stypes(child)
        self.node_type_inferrer[node.__class__.__name__].infer_stype(node)

    def infer_dtypes(self, parsed_query: ParsedPredictiveQuery) -> None:
        r"""Infers data type of every node in the AST, writing it to
        ASTNode.dtype_maybe.

        Args:
            parsed_query: the input parsed query.

        Raises:
            ValueError: If `self.graph` comes without data types .
        """
        self._infer_dtypes(parsed_query.entity_ast)
        self._infer_dtypes(parsed_query.target_ast)
        if parsed_query.whatif_ast is not None:
            self._infer_dtypes(parsed_query.whatif_ast)

    def infer_stypes(self, parsed_query: ParsedPredictiveQuery) -> None:
        r"""Infers semantic type of every node in the AST, writing it to
        ASTNode.stype_maybe.

        Args:
            parsed_query: the input parsed query.

        Raises:
            ValueError: If `self.graph` comes without semantic types .
        """
        self._infer_stypes(parsed_query.entity_ast)
        self._infer_stypes(parsed_query.target_ast)
        if parsed_query.whatif_ast is not None:
            self._infer_stypes(parsed_query.whatif_ast)

    def _validate_dtypes(self, node: ASTNode) -> ValidationResponse:
        response = ValidationResponse()
        for child in node.children:
            response = merge(response, self._validate_dtypes(child))
        response = merge(
            response, self.node_type_validator[
                node.__class__.__name__].validate_dtype(node))
        return response

    def _validate_stypes(self, node: ASTNode) -> ValidationResponse:
        response = ValidationResponse()
        for child in node.children:
            response = merge(response, self._validate_stypes(child))
        response = merge(
            response, self.node_type_validator[
                node.__class__.__name__].validate_stype(node))
        return response

    def validate_dtypes(
            self, parsed_query: ParsedPredictiveQuery) -> ValidationResponse:
        r"""Validate data type of every node in the AST, returning the list
        of errors.

        Args:
            parsed_query: the input parsed query.

        Returns:
            ValidationResponse: List of encountered errors.
        """
        response = ValidationResponse()
        response = merge(response,
                         self._validate_dtypes(parsed_query.entity_ast))
        response = merge(response,
                         self._validate_dtypes(parsed_query.target_ast))
        if isinstance(parsed_query.target_ast.dtype, ArrayDtype):
            nested_type = parsed_query.target_ast.dtype.nested_dtype
            if not (nested_type == Dtype.string or nested_type.is_numerical()):
                message = parsed_query.target_ast.get_location().message_start
                response.errors.append(
                    ValidationError(
                        title='Invalid target type', message=f'{message}: '
                        f'Target label has data type '
                        f'{parsed_query.target_ast.dtype}, which Kumo '
                        f'currently cannot predict.'))

        if parsed_query.whatif_ast is not None:
            response = merge(response,
                             self._validate_dtypes(parsed_query.whatif_ast))
        return response

    def validate_stypes(
            self, parsed_query: ParsedPredictiveQuery) -> ValidationResponse:
        r"""Validate semantic type of every node in the AST, returning the list
        of errors.

        Args:
            parsed_query: the input parsed query.

        Returns:
            ValidationResponse: List of encountered errors.
        """
        response = ValidationResponse()
        response = merge(response,
                         self._validate_stypes(parsed_query.entity_ast))
        response = merge(response,
                         self._validate_stypes(parsed_query.target_ast))
        if parsed_query.target_ast.stype in [
                Stype.text,
                Stype.timestamp,
                Stype.sequence,
                Stype.unsupported,
        ]:
            message = parsed_query.target_ast.get_location().message_start
            response.errors.append(
                ValidationError(
                    title='Invalid target type', message=f'{message}: '
                    f'Target label has semantic type '
                    f'{parsed_query.target_ast.stype}, which Kumo currently '
                    f'cannot predict.'))
        if parsed_query.whatif_ast is not None:
            response = merge(response,
                             self._validate_stypes(parsed_query.whatif_ast))
        return response
