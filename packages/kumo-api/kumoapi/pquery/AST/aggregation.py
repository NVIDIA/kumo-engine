# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: MIT

from typing import List, Optional, Union

import pydantic
from pydantic.dataclasses import dataclass

from kumoapi.json_serde import from_json
from kumoapi.pquery.AST.ast_node import ASTNode
from kumoapi.pquery.AST.column import Column
from kumoapi.pquery.AST.date_offset_range import DateOffsetRange
from kumoapi.pquery.utils import maybe_bold
from kumoapi.typing import _FORBID_EXTRA_CONFIG, AggregationType, TimeUnit


@dataclass(repr=False, config=_FORBID_EXTRA_CONFIG)
class Aggregation(ASTNode):
    r"""An generic aggregation description within
    :class:`~kumo.pquery.PredictiveQuery`.

    Args:
        target: The ASTNode defining the target to be aggregated.
            (:obj:`table_name.col_name`) on which the aggregation is performed.
        aggr: The type of aggregation.
        aggr_time_range: The date offset
            range of the aggregation. (default: :obj:`None`)
        group_by: The column to group by during aggregation.
            For internal dataclass loading use only.
        time_col: The time column to use during aggregation.
            For internal dataclass loading use only.
    """
    target: Optional[Union['Filter', Column]] = None
    aggr: Union[AggregationType, str] = ''
    aggr_time_range: Optional[DateOffsetRange] = None
    group_by: Optional[str] = None
    time_col: Optional[str] = None

    def __post_init__(self) -> None:
        if self.target is None:
            raise ValueError(f"Class '{self.__class__.__name__}' is missing a "
                             f"target.")
        if self.aggr_time_range is not None:
            if isinstance(self.aggr_time_range, dict):
                # If Pydantic hasn't loaded attributes yet, they might still
                # be dicts
                self.aggr_time_range = from_json(self.aggr_time_range,
                                                 DateOffsetRange)
            if self.date_offset_range is not None:
                if isinstance(self.date_offset_range, dict):
                    self.date_offset_range = from_json(self.date_offset_range,
                                                       DateOffsetRange)
                self.date_offset_range = DateOffsetRange.merge_ranges(
                    self.date_offset_range, self.aggr_time_range)
            else:
                self.date_offset_range = self.aggr_time_range

        self.aggr = AggregationType(self.aggr)
        super().__post_init__()

    @property
    def children(self) -> List['ASTNode']:
        assert self.target is not None
        return [self.target]

    def non_inf_date_offset_range(self) -> Optional[DateOffsetRange]:
        r"""The full time range of this subtree, excluding any ranges with
        infinities, given as a `DateOffsetRange`.
        Returns :obj:`None` if there is no time range."""
        child_range = super().non_inf_date_offset_range()
        if self.aggr_time_range is None or self.aggr_time_range.is_open:
            return child_range
        elif child_range is None:
            return self.aggr_time_range
        else:
            return child_range.merge_ranges(child_range, self.aggr_time_range)

    @property
    def is_static(self) -> bool:
        return self.aggr_time_range is None

    def to_string(self, rich: bool = False) -> str:
        # {aggr}({target}, {date_offset_range})
        # Example: COUNT(TRANSACTIONS.PRICE, -30, 0, days)
        assert isinstance(self.aggr, AggregationType)
        aggr_name = maybe_bold(self.aggr.value, rich)
        assert self.target is not None
        target_repr = self.target.to_string(rich=rich)
        if self.aggr_time_range is None:
            return f'{aggr_name}({target_repr})'

        start_offset = self.aggr_time_range.start
        if start_offset is None:
            start_offset = "-INF"
        assert isinstance(self.aggr_time_range.unit, TimeUnit)
        return (f'{aggr_name}('
                f'{target_repr}, '
                f'{start_offset}, '
                f'{self.aggr_time_range.end}, '
                f'{self.aggr_time_range.unit.value}'
                f')')

    def _get_target_column_name(self) -> str:
        return self.get_target_column_name()

    def get_target_column_name(self) -> str:
        r"""Returns the fully-qualified name (``table.column``) of the leaf
        column being aggregated.
        """
        if isinstance(self.target, Column):
            return self.target.fqn
        else:
            assert isinstance(self.target, Filter)
            assert isinstance(self.target.target, Column)
            return self.target.target.fqn

    @property
    def all_time_columns(self) -> List[str]:
        r"""List of all columns that are needed for temporal aggregations in
        the query, given with in a fully-qualified name format:
        `table.column`."""
        assert self.target is not None
        targets = self.target.all_time_columns
        if self.time_col is not None:
            targets.append(self.time_col)
        return list(set(targets))


from kumoapi.pquery.AST.filter import Filter  # noqa: E402

if pydantic.__version__.startswith('1.'):
    Aggregation.__pydantic_model__.update_forward_refs()
