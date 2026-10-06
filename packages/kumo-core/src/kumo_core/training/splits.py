# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Canonical Kumo split constructors.

A split is stored in a model plan as its constructor repr, so the constructor
is the schema: argument order, defaults, the values a split derives from its
own arguments, and the repr that carries them all live here. Portable training
resolves a plan's split against these classes, and `kumo.task.split` builds its
registered, mask-generating splits on top of them, so both sides read one
definition.

Behaviour that needs the platform graph -- mask generation, boundary
adjustment against a `Table` -- stays in `kumo.task.split`.

Annotations here are deliberately not postponed: `kumo.task.split` inherits
these constructors, and `register` builds a pydantic dataclass from the
inherited signature in `torch_geometric.config_store`, where a string
annotation naming `TimeUnit` cannot be resolved.
"""

from datetime import datetime, timezone
from enum import Enum
from numbers import Integral
from typing import Any

import pandas as pd
from kumoapi.pquery import QueryType
from kumoapi.typing import TimeUnit

from ..types import TaskType


def validate_split_ratios(
    ratio: list[float],
    check_unused_data: bool = True,
) -> None:
    if len(ratio) > 3:
        raise ValueError("No more than three training splits is supported")
    if len(ratio) < 2:
        raise ValueError("At least two training splits are required")

    for val in ratio:
        if val < 0:
            raise ValueError("'ratio' can not contain negative values")

    if sum(ratio) - 1 > 1e-2:
        raise ValueError("'ratio' exceeds more than 100% of the data")

    if check_unused_data and 1 - sum(ratio) > 1e-2:
        raise ValueError("'ratio' does not use 100% of the data")


def _time_unit(value: TimeUnit | str) -> TimeUnit:
    return TimeUnit(value.lower() if isinstance(value, str) else value)


def epoch_seconds(value: Any) -> int:
    """Resolve one split boundary to exact epoch seconds."""

    if isinstance(value, bool):
        raise ValueError("split boundary must not be a bool")
    if isinstance(value, pd.Timestamp):
        return int(value.timestamp())
    if isinstance(value, Integral):
        return int(value)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(
            f"split boundary {value!r} must be an ISO-8601 date or an "
            "integer epoch second") from error
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp())


class SplitArguments:
    """Base for every split whose arguments a model plan can express."""
    def to_split_config(self) -> dict[str, Any]:
        """Return the exact split config `build_split_spec` accepts.

        Deliberately not named `config`: the platform subclasses also inherit
        `ConfigMixin.config`, and a same-named method here would shadow it.
        """
        raise NotImplementedError


class TemporalSplitArguments(SplitArguments):
    """Shared temporal arguments, and the pandas offsets derived from them."""
    def __init__(
        self,
        unit: TimeUnit | str = 'days',
        timeframe: int | None = None,
        anchor_start_offset: int | None = 0,
        timeframe_step: int | None = None,
        flexible_boundaries: bool = True,
    ) -> None:
        self.unit = _time_unit(unit)
        self.timeframe = timeframe
        self.anchor_start_offset = anchor_start_offset
        self.timeframe_step = timeframe_step
        self.flexible_boundaries = flexible_boundaries

        self.timeframe_offset = None
        if self.timeframe is not None:
            self.timeframe_offset = pd.DateOffset(
                **{self.unit.value: timeframe})

        self.anchor_start_date_offset = None
        if anchor_start_offset is not None:
            self.anchor_start_date_offset = pd.DateOffset(
                **{self.unit.value: anchor_start_offset})

        self.timeframe_step_offset = None
        if timeframe_step is not None:
            self.timeframe_step_offset = pd.DateOffset(
                **{self.unit.value: timeframe_step})
        self.boundaries = None

    def to_split_config(self) -> dict[str, Any]:
        return {
            "unit": self.unit.value,
            "timeframe": self.timeframe,
            "anchor_start_offset": self.anchor_start_offset,
            "timeframe_step": self.timeframe_step,
            "flexible_boundaries": self.flexible_boundaries,
        }


class RandomSplit(SplitArguments):
    def __init__(self, ratio: list[float], seed: int | None = None) -> None:
        validate_split_ratios(ratio=ratio, check_unused_data=False)

        self.ratio = ratio
        self.seed = seed

    @property
    def num_splits(self) -> int:
        return len(self.ratio)

    def to_split_config(self) -> dict[str, Any]:
        return {"ratio": self.ratio, "rnd_seed": self.seed}

    def __repr__(self) -> str:
        if self.seed is not None:
            return f'{self.__class__.__name__}({self.ratio}, seed={self.seed})'
        return f'{self.__class__.__name__}({self.ratio})'


class ManualSplit(SplitArguments):
    """Splits on a column the training table already carries.

    The values are compared as strings, matching `ColumnDeciderManual`, which
    routes a row by its column value and then drops the column from the seed
    output. A row whose value names none of the three splits is unused.

    `col` is the column's name, not a column object: nothing about the split
    depends on the column's type or statistics, and a plan stores this
    constructor's repr, so a name is the whole of what has to survive the
    round trip.

    `test_value` defaults to `None`, which produces no test stage --
    `num_splits` reads 2 rather than 3. Callers that hold out a test set say
    so; defaulting one into existence would silently turn every two-way
    split into a three-way one.
    """
    def __init__(
        self,
        col: str,
        train_value: Any = 0,
        val_value: Any = 1,
        test_value: Any = None,
    ) -> None:
        if not isinstance(col, str) or not col:
            raise ValueError("'col' must be a non-empty column name")

        self.col = col
        self.train_value = train_value
        self.val_value = val_value
        self.test_value = test_value

    @property
    def num_splits(self) -> int:
        return 3 if self.test_value is not None else 2

    def to_split_config(self) -> dict[str, Any]:
        def values(value: Any) -> list[str] | None:
            if value is None:
                return None
            listed = value if isinstance(value, (list, tuple)) else [value]
            return [str(item) for item in listed]

        return {
            "split_column": self.col,
            "manual_train_values": values(self.train_value),
            "manual_val_values": values(self.val_value),
            "manual_test_values": values(self.test_value),
        }

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(col={self.col!r}, '
                f'train_value={self.train_value!r}, '
                f'val_value={self.val_value!r}, '
                f'test_value={self.test_value!r})')


class RandomLinkSplit(SplitArguments):
    def __init__(self, ratio: list[float]) -> None:
        validate_split_ratios(ratio=ratio, check_unused_data=False)

        self.ratio = ratio

    @property
    def num_splits(self) -> int:
        return len(self.ratio)

    def to_split_config(self) -> dict[str, Any]:
        return {"ratio": self.ratio}

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.ratio})'


class ApproxDateOffsetSplit(SplitArguments):
    def __init__(self, ratio: list[float]) -> None:
        validate_split_ratios(ratio)
        self.ratio = ratio

    @property
    def num_splits(self) -> int:
        return len(self.ratio)

    def to_split_config(self) -> dict[str, Any]:
        return {"ratio": self.ratio}

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.ratio})'


class TemporalSplit(TemporalSplitArguments):
    def __init__(
        self,
        ratio: list[float],
        unit: TimeUnit | str = 'days',
        timeframe: int | None = None,
        anchor_start_offset: int | None = 0,
    ) -> None:
        validate_split_ratios(ratio)
        super().__init__(
            unit=unit,
            timeframe=timeframe,
            anchor_start_offset=anchor_start_offset,
        )
        self.ratio = ratio

    @property
    def num_splits(self) -> int:
        return len(self.ratio)

    def to_split_config(self) -> dict[str, Any]:
        return {**super().to_split_config(), "ratio": self.ratio}

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.ratio})'


def normalize_static_link_split(
    split: SplitArguments,
    *,
    default_is_random_link_split: bool,
) -> SplitArguments:
    """Use per-link splits for static array targets, as hosted queries do."""
    if (isinstance(split, (RandomSplit, TemporalSplit))
            and default_is_random_link_split):
        return RandomLinkSplit(split.ratio)
    return split


class DateOffsetSplit(TemporalSplitArguments):
    def __init__(
        self,
        offsets: list[int],
        offset_unit: TimeUnit | str = 'days',
        unit: TimeUnit | str = 'days',
        timeframe: int | None = None,
        anchor_start_offset: int = 0,
        timeframe_step: int | None = None,
        flexible_boundaries: bool = True,
    ) -> None:
        if len(offsets) < 1:
            raise ValueError("Provide at least one element in 'offsets'")
        if len(offsets) > 2:
            raise ValueError("No more than three training splits supported")
        for offset in offsets:
            if offset > 0:
                raise ValueError(f"Offsets in '{self.__class__.__name__}' "
                                 f"need to be negative (got {offset})")
        for left_offset, right_offset in zip(offsets[:-1], offsets[1:]):
            if left_offset > right_offset:
                raise ValueError(f"Offsets in '{self.__class__.__name__}' "
                                 f"need to be monotonically increasing (got "
                                 f"{left_offset} and {right_offset})")

        self.offset_unit = _time_unit(offset_unit)
        if timeframe is None:
            if len(offsets) == 2:
                timeframe = offsets[1] - offsets[0]
            else:
                timeframe = -offsets[0]
        super().__init__(unit=unit, timeframe=timeframe,
                         anchor_start_offset=anchor_start_offset,
                         timeframe_step=timeframe_step,
                         flexible_boundaries=flexible_boundaries)
        self.offsets = offsets

    @property
    def num_splits(self) -> int:
        return len(self.offsets) + 1

    def to_split_config(self) -> dict[str, Any]:
        return {
            **super().to_split_config(),
            "offsets": self.offsets,
            "offset_unit": self.offset_unit.value,
        }

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.offsets}, '
                f'offset_unit={self.offset_unit.value}, '
                f'unit={self.unit.value}, timeframe={self.timeframe}, '
                f'anchor_start_offset={self.anchor_start_offset}, '
                f'timeframe_step={self.timeframe_step})')


class TimeRangeSplit(TemporalSplitArguments):
    def __init__(
        self,
        ranges: list[tuple[Any, Any]],
        unit: TimeUnit | str = 'days',
        timeframe: int | None = None,
        anchor_start_offset: int | None = 0,
        timeframe_step: int | None = None,
        flexible_boundaries: bool = True,
    ) -> None:
        if len(ranges) > 3:
            raise ValueError("No more than three training splits is supported")
        if len(ranges) < 2:
            raise ValueError("At least two training splits are required")
        type_safe_ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []

        for i, pair in enumerate(ranges):
            if len(pair) != 2:
                raise ValueError(
                    "In time range split, each range needs an exact start and "
                    f"end. However we have {pair!r}. ")
            start, end = pair
            if not isinstance(start, pd.Timestamp):
                start = pd.Timestamp(start)
            if not isinstance(end, pd.Timestamp):
                end = pd.Timestamp(end)
            if end < start:
                raise ValueError(
                    f"In time range split, end date needs to be smaller "
                    f"than start date. However we have {end=} and {start=}. ")
            if i > 0:
                prev_end = ranges[i - 1][-1]
                if not isinstance(prev_end, pd.Timestamp):
                    prev_end = pd.Timestamp(prev_end)
                if start < prev_end:
                    raise ValueError(
                        f"In time range split, different time ranges can "
                        f"not overlap. However, we have both "
                        f"{ranges[i-1]} and {ranges[i]}. ")
            type_safe_ranges.append((start, end))

        super().__init__(unit=unit, timeframe=timeframe,
                         anchor_start_offset=anchor_start_offset,
                         timeframe_step=timeframe_step,
                         flexible_boundaries=flexible_boundaries)
        self.ranges = type_safe_ranges

    @property
    def num_splits(self) -> int:
        return len(self.ranges)

    def to_split_config(self) -> dict[str, Any]:
        return {
            **super().to_split_config(),
            "ranges": [[epoch_seconds(start),
                        epoch_seconds(end)] for start, end in self.ranges],
        }

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.ranges}, '
                f'unit={self.unit.value}, timeframe={self.timeframe}, '
                f'anchor_start_offset={self.anchor_start_offset}, '
                f'timeframe_step={self.timeframe_step})').replace("'", "")


class TrainTableTime(Enum):
    NONE = "none"
    PSEUDO = "pseudo"
    REAL = "real"


def train_table_time(
    task_type: TaskType,
    query_type: QueryType,
    *,
    list_distinct_target_is_array: bool,
    entity_table_has_time_col: bool,
) -> TrainTableTime:
    if (task_type == TaskType.STATIC_LINK_PREDICTION
            and not list_distinct_target_is_array):
        return TrainTableTime.PSEUDO
    if query_type == QueryType.TEMPORAL or entity_table_has_time_col:
        return TrainTableTime.REAL
    return TrainTableTime.NONE


STATIC_LINK_SEED_TIMES = {"train": 1, "val": 1, "test": 2}


def uses_pseudo_time(task_type: Any) -> bool:
    """Static link prediction hides later splits' links by pseudo time."""
    value = getattr(task_type, "value", task_type)
    return value == TaskType.STATIC_LINK_PREDICTION.value


def pseudo_seed_time(task_type: Any, stage: str) -> int | None:
    if not uses_pseudo_time(task_type):
        return None
    return STATIC_LINK_SEED_TIMES.get(stage)


def default_split_for_query(
    task_type: TaskType,
    *,
    query_is_static: bool,
    entity_has_time_column: bool,
    list_distinct_target_is_array: bool,
    offset_start: int | None,
    offset_end: int | None,
    unit: TimeUnit | str,
    forecast_length: int,
    timeframe_step: int | None,
) -> SplitArguments:
    """Choose the legacy default split from resolved query facts."""
    if query_is_static:
        if task_type == TaskType.STATIC_LINK_PREDICTION:
            split = (RandomLinkSplit
                     if list_distinct_target_is_array else RandomSplit)
            return split([0.8, 0.1, 0.1])
        split = TemporalSplit if entity_has_time_column else RandomSplit
        return split([0.8, 0.1, 0.1])

    assert offset_start is not None
    assert offset_end is not None
    timeframe = offset_end - offset_start
    if task_type == TaskType.FORECASTING:
        split_length = forecast_length * timeframe
        return DateOffsetSplit(
            [-2 * split_length + timeframe, -split_length + timeframe],
            offset_unit=unit,
            unit=unit,
            timeframe=timeframe,
            anchor_start_offset=offset_start,
            timeframe_step=timeframe_step,
            flexible_boundaries=True,
        )
    if (task_type != TaskType.TEMPORAL_LINK_PREDICTION and offset_start == 0):
        return ApproxDateOffsetSplit([0.8, 0.1, 0.1])
    return DateOffsetSplit(
        [-offset_end, 0],
        offset_unit=unit,
        unit=unit,
        timeframe=timeframe,
        anchor_start_offset=offset_start,
        timeframe_step=timeframe_step,
        flexible_boundaries=False,
    )


#: The split families a model plan can express, keyed by constructor name.
PLAN_SPLITS: dict[str, type[SplitArguments]] = {
    "ApproxDateOffsetSplit": ApproxDateOffsetSplit,
    "DateOffsetSplit": DateOffsetSplit,
    "ManualSplit": ManualSplit,
    "RandomLinkSplit": RandomLinkSplit,
    "RandomSplit": RandomSplit,
    "TemporalSplit": TemporalSplit,
    "TimeRangeSplit": TimeRangeSplit,
}

__all__ = [
    "ApproxDateOffsetSplit",
    "DateOffsetSplit",
    "ManualSplit",
    "PLAN_SPLITS",
    "RandomLinkSplit",
    "RandomSplit",
    "SplitArguments",
    "TemporalSplit",
    "TemporalSplitArguments",
    "TimeRangeSplit",
    "epoch_seconds",
    "validate_split_ratios",
]
