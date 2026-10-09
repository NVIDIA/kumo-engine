# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: MIT

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

from pydantic.dataclasses import dataclass

from kumoapi.json_serde import from_json
from kumoapi.pquery.AST.date_offset_range import DateOffsetRange
from kumoapi.pquery.AST.location_interval import ASTQueryLocationInterval
from kumoapi.typing import _FORBID_EXTRA_CONFIG, Dtype, Stype


@dataclass(config=_FORBID_EXTRA_CONFIG)
class ArrayDtype:
    r"""Class for array typing. Used for internal predictive query validation.
    """

    nested_dtype: Dtype

    def __post_init__(self) -> None:
        if isinstance(self.nested_dtype, str):
            self.nested_dtype = Dtype(self.nested_dtype)
        if self.nested_dtype in [
                Dtype.floatlist, Dtype.intlist, Dtype.stringlist
        ]:
            raise ValueError(
                f'ArrayDtype not supported for Dtype {self.nested_dtype}.')

    def to_dtype(self) -> Dtype:
        if self.nested_dtype.is_float():
            return Dtype.floatlist
        elif self.nested_dtype.is_int():
            return Dtype.intlist
        elif self.nested_dtype == Dtype.string:
            return Dtype.stringlist
        raise ValueError(f'ArrayDtype.to_dtype() only supported for numerical '
                         f'nested types, got {self.nested_dtype}.')

    def is_bool(self) -> bool:
        return False

    def is_int(self) -> bool:
        return False

    def is_float(self) -> bool:
        return False

    def is_string(self) -> bool:
        return False

    def is_temporal(self) -> bool:
        return False

    def is_timestamp(self) -> bool:
        return False

    def is_numerical(self) -> bool:
        return False

    def is_list(self) -> bool:
        return True

    def is_unsupported(self) -> bool:
        return self.nested_dtype.is_unsupported()

    @property
    def value(self) -> str:
        return f'List[{self.nested_dtype.value}]'


@dataclass(repr=False, config=_FORBID_EXTRA_CONFIG)
class ASTNode(ABC):
    r"""A base class for all abstract syntax tree nodes in
    :class:`~kumo.pquery.PredictiveQuery`.

    Args:
        children: Any children of this node. Default is an
            empty list.
        date_offset_range: Date offset range of the
            entire subtree. For internal dataclass loading only.
        dtype_maybe:  Data type of the tree.
            For internal dataclass loading only.
        stype_maybe:  Semantic type of the tree. For internal
            dataclass loading only.
        location: Interval in the input
            query that corresponds to this AST subtree.
    """
    date_offset_range: Optional[DateOffsetRange] = None
    dtype_maybe: Optional[Union[Dtype, ArrayDtype, str]] = None
    stype_maybe: Optional[Union[Stype, str]] = None
    location: Optional[ASTQueryLocationInterval] = None

    def __post_init__(self) -> None:
        if isinstance(self.dtype_maybe, str):
            self.dtype_maybe = Dtype(self.dtype_maybe)
        if isinstance(self.stype_maybe, str):
            self.stype_maybe = Stype(self.stype_maybe)
        self.date_offset_range = ASTNode.get_combined_time(self)

    @property
    def children(self) -> List['ASTNode']:
        return []

    def get_location(self) -> ASTQueryLocationInterval:
        r"""Returns the location of this subtree in the input query text."""
        location = self._get_location()
        if location is None:
            location = ASTQueryLocationInterval(0, 0, 0, 0, False)
        return location

    def _get_location(self) -> Optional[ASTQueryLocationInterval]:
        location = self.location
        for child in self.children:
            child_location = child._get_location()
            if location is None:
                location = child_location
            elif child_location is not None:
                location = ASTQueryLocationInterval.merge(
                    location, child_location)
        return location

    @property
    def end_date_offset(self) -> Optional[int]:
        r"""The end of the time range of this subtree.
        Returns :obj:`None` if there is no time range."""
        if self.date_offset_range is None:
            return None
        return self.date_offset_range.end

    @staticmethod
    def get_combined_time(
            node: Union['ASTNode', Dict]) -> Optional[DateOffsetRange]:
        r'''This is a mildly hacky method that combines date ranges of all
        children with the time range of the current node during the post_init.
        Since attributes of a pydantic dataclass sometimes aren't correctly
        loaded at the __post_init__ time, this method needs to be able to
        handle uninitialized children that are still config dicts.'''
        date_range = None
        if not isinstance(node, ASTNode):
            date_range = node.get('date_offset_range', None)
            children = node.get('children', [])
        else:
            date_range = node.date_offset_range
            children = node.children
        for child in children:
            child_date_range = ASTNode.get_combined_time(child)
            if child_date_range is None:
                continue
            if date_range is None:
                date_range = child_date_range
                continue
            # If Pydantic hasn't loaded attributes yet, they might still
            # be dicts
            if isinstance(date_range, dict):
                date_range = from_json(date_range, DateOffsetRange)
            if isinstance(child_date_range, dict):
                child_date_range = from_json(child_date_range, DateOffsetRange)
            date_range = date_range.merge_ranges(date_range, child_date_range)
        return date_range

    def non_inf_date_offset_range(self) -> Optional[DateOffsetRange]:
        r"""The full time range of this subtree, excluding any ranges with
        infinities, given as a `DateOffsetRange`.
        Returns :obj:`None` if there is no time range."""
        result = None
        for child in self.children:
            child_range = child.non_inf_date_offset_range()
            if child_range is None or child_range.start is None:
                continue
            if result is None:
                result = child_range
            else:
                result = result.merge_ranges(result, child_range)
        return result

    @abstractmethod
    def to_string(self, rich: bool = False) -> str:
        pass

    @property
    def dtype(self) -> Union[Dtype, ArrayDtype]:
        r"""Dtype of the output of this expression, if known."""
        if self.dtype_maybe is None:
            raise ValueError('`dtype` has not been inferred yet.')
        assert isinstance(self.dtype_maybe, (Dtype, ArrayDtype))
        return self.dtype_maybe

    @property
    def stype(self) -> Stype:
        r"""Stype of the output of this expression, if known."""
        if self.stype_maybe is None:
            raise ValueError('`stype` has not been inferred yet.')
        assert isinstance(self.stype_maybe, Stype)
        return self.stype_maybe

    def __repr__(self) -> str:
        return self.to_string()

    @property
    def all_query_columns_with_locations(
            self) -> List[Tuple[str, ASTQueryLocationInterval]]:
        r"""List of all columns that explicitly appear in the query, given
        with in a fully-qualified name format: `table.column` and their
        corresponding locations."""
        targets = []
        for child in self.children:
            targets.extend(child.all_query_columns_with_locations)
        return list(set(targets))

    @property
    def all_query_columns(self) -> List[str]:
        r"""List of all columns that explicitly appear in the query, given
        with in a fully-qualified name format: `table.column`."""
        targets = []
        for child in self.children:
            targets.extend(child.all_query_columns)
        return list(set(targets))

    @property
    def all_join_columns(self) -> List[str]:
        r"""List of all columns that are needed for joins in the query, given
        with in a fully-qualified name format: `table.column`."""
        targets = []
        for child in self.children:
            targets.extend(child.all_join_columns)
        return list(set(targets))

    @property
    def all_time_columns(self) -> List[str]:
        r"""List of all columns that are needed for temporal aggregations in
        the query, given with in a fully-qualified name format:
        `table.column`."""
        targets = []
        for child in self.children:
            targets.extend(child.all_time_columns)
        return list(set(targets))
