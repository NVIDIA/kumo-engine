# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json as pyjson
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import field, is_dataclass, replace
from functools import cached_property
from hashlib import md5
from typing import Any

import numpy as np
import pandas as pd
from kumo_core.training import splits as core_splits
from kumo_core.training.splits import validate_split_ratios
from kumo_core.types.conversions import DataFrame
from kumo_core.types.schema import FPID
from kumo_core.utils.registry import register
from kumoapi.typing import TimeUnit

from io_utils import logging
from io_utils.locate import locate
from kumo.config import ConfigMixin, SplitConfig
from kumo.config.task import Mask  # type: ignore
from kumo.graph import Table
from kumo.graph.column import BaseColumn
from kumo.stats.train_table_stats import (
    TemporalLinkTrainTableStats,
    TemporalNodeTrainTableStats,
)
from statedb.utils.errors import UnsupportedInDiskgraphError

logger = logging.getLogger(__name__)


class Split(ConfigMixin, ABC):
    # Determines which options to expose as part of the model plan.
    _external_args: set[str]

    @abstractmethod
    def get_masks(self, table: Table) -> list[Mask]:
        pass

    @property
    @abstractmethod
    def is_temporal_split(self) -> bool:
        """Returns :obj:`True` if split
        is based on :obj:`table.time_col`."""

    @property
    @abstractmethod
    def num_splits(self) -> int:
        pass

    @classmethod
    def from_config(cls, cfg: Any, *args, **kwargs) -> Any:
        r"""Rebuilds a split, reading any superseded serialization.

        `ConfigMixin.from_config` resolves `_target_` to the concrete class
        inside its own frame, so an override on that class is never reached
        for the ``Split.from_config(cfg)`` call every caller makes
        (`kumo/task/task.py`, `rest/.../activities/split.py`). This is the
        one entry point that is.
        """
        return super().from_config(_normalize_legacy_split(cfg), *args,
                                   **kwargs)

    @cached_property
    def fpid(self):
        return self.get_fpid_from_config(self.config())

    @staticmethod
    def get_fpid_from_config(config: SplitConfig) -> FPID:
        cfg_str = pyjson.dumps(config, sort_keys=True, default=str)
        fpid = FPID(md5(cfg_str.encode()).hexdigest())
        logger.debug(f'Split fpid: {fpid} for config: {config}')
        return fpid

    def post_table_generation_transform(self, table: DataFrame) -> DataFrame:
        r"""A split-specific transformation of the training table.
        This operation is called on the remote EMR (if used) directly at the
        end of the training table generation and just before saving the
        training table anywhere. It's thus performed before masks are
        generated.

        Args:
            table (DataFrame): The training table.

        Returns:
            DataFrame: The transformed training table.
        """
        return table

    def assert_has_time_col(self, table: Table) -> None:
        if not table.has_time_col():
            raise ValueError(f"Cannot apply '{self.__class__.__name__}' "
                             f"without a 'time_col' in table '{table.name}'")


@register(group='split', base_cls=SplitConfig)
class RandomSplit(core_splits.RandomSplit, Split):
    r"""Defines a random split of the training table according to a
    :obj:`ratio`.

    Args:
        ratio (List[float]): The ratio for each split.
        seed (int, optional): Random seed for reproducible splits.
    """
    _external_args: set[str] = {'ratio', 'seed'}

    @property
    def is_temporal_split(self) -> bool:
        return False

    def get_masks(self, table: Table) -> list[Mask]:
        raise UnsupportedInDiskgraphError(
            "whole-table split via Split.get_masks is dead on TT-NG; "
            "per-stage train/val/test masks come from the native "
            "DiskGraph split (split-stats sidecar / bundle)")


@register(group='split', base_cls=SplitConfig)
class RandomLinkSplit(core_splits.RandomLinkSplit, Split):
    r"""Defines a random split of the links in the training table according
    to the :obj:`ratio`, used when the target column contains arrays that
    can be split further. As part of the split, an array can be split into
    multiple smaller arrays and an entity may appear in the table multiple
    times (but only once per set).

    Args:
        ratio (List[float]): The ratio for each split.
    """
    _external_args: set[str] = {'ratio'}
    _PERMUTATION_COL = '__internal_kumo_permutation_col__'
    _SPLIT_ID_COL = '__internal_kumo_split_id_col__'

    @property
    def is_temporal_split(self) -> bool:
        return False

    def get_masks(self, table: Table) -> list[Mask]:
        raise UnsupportedInDiskgraphError(
            "whole-table split via Split.get_masks is dead on TT-NG; "
            "per-stage train/val/test masks come from the native "
            "DiskGraph split (split-stats sidecar / bundle)")

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.ratio})'


@register(group='split', base_cls=SplitConfig)
class FixedSplit(Split):
    r"""Defines a split of the training table according to a :obj:`ratio` that
    respects its ordering.

    Args:
        ratio (List[float]): The ratio for each split.
    """
    _external_args: set[str] = {'ratio'}

    def __init__(self, ratio: list[float]) -> None:
        validate_split_ratios(ratio)

        self.ratio = ratio

    @property
    def is_temporal_split(self) -> bool:
        return False

    @property
    def num_splits(self) -> int:
        return len(self.ratio)

    def get_masks(self, table: Table) -> list[Mask]:
        raise UnsupportedInDiskgraphError(
            "whole-table split via Split.get_masks is dead on TT-NG; "
            "per-stage train/val/test masks come from the native "
            "DiskGraph split (split-stats sidecar / bundle)")

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.ratio})'


class BaseTemporalSplit(core_splits.TemporalSplitArguments, Split):
    r"""Defines a temporal split of the training table according to the
    (relative) offsets from the max date.

    Args:
        unit (TimeUnit or str, optional): The time unit of the timeframe.
            (default: :obj:`"days"`)
        timeframe (int, optional): Timeframe of a query, used to
            detect and remove any examples that would overlap with multiple
            splits. If :obj:`None`, it is assumed to be the minimal
            value/difference between offsets.
        anchor_start_offset (int, optional): Time between the anchor time and
            the beginning of the timeframe. Used to detect and remove any
            examples where the anchor time overlaps with the previous bucket.
        timeframe_step (int, optional): If not :obj:`None`, object
             will adjust offsets to ensure that at least one sample interval
             appears in val/test set. No-op when `timeframe_step` is not used
             or when its value is a multiple of `timeframe`.
        flexible_boundaries (bool, optional): If :obj:`True`, the system will
            shift the provided boundaries to ensure the maximal use of the
            data. (default: :obj:`True`)
    """
    @property
    def is_temporal_split(self) -> bool:
        return True

    def get_boundaries(self) -> list[tuple[pd.Timestamp, pd.Timestamp]] | None:
        r"""Returns the final split boundaries after all adjustments.
        Returns:
            (List[Tuple[pd.Timestamp, pd.Timestamp]], optional): When split
                boundary is not finalized, returns `None`. Otherwise, returns
                the final adjusted split boundary.
        """
        return self.boundaries

    def adjust_boundaries(
        self,
        boundaries: list[tuple[pd.Timestamp, pd.Timestamp]],
    ) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        r"""Adjusts boundaries `boundaries` to ensure that each split set
        starts with a training sample and that each set is non-empty.

        Args:
            boundaries (List[Tuple[pd.Timestamp, pd.Timestamp]]): The list of
                time intervals, each corresponding to one split set, given in
                the increasing order (train, val, test). Each set is
                described as a pair of the start and end timestamp,
                respectively.

        Returns:
            List[Tuple[pd.Timestamp, pd.Timestamp]]: The adjusted boundaries.

        """
        if not self.flexible_boundaries or self.timeframe_offset is None:
            return boundaries
        assert self.anchor_start_date_offset is not None

        new_boundaries = []
        # cumulative offset - all data with dates larger than cum_offset is
        # part of some split already and should not be used in other splits
        cum_offset: pd.Timestamp = boundaries[-1][1]
        # step_offset - counter that moves in the `step` intervals and
        # corresponds to timestamps at which data is sampled.
        step_offset: pd.Timestamp = boundaries[-1][1] - self.timeframe_offset
        step = self.timeframe_step_offset or self.timeframe_offset

        for start, finish in boundaries[::-1]:
            # There should be at least one timeframe in a split window
            # The exception is the test set (cum_offset = 0) where there is
            # guaranteed to be at least one timeframe worth of data after it.
            if cum_offset < finish:
                finish = cum_offset
            if (start + self.anchor_start_date_offset + self.timeframe_offset
                    > finish):
                start = (finish - self.timeframe_offset -
                         self.anchor_start_date_offset)
            # if timeframe_step is not None, samples will be generated at
            # timestamps - timeframe - k * step. Make sure that
            # the left boundary/offset coincides with one generated sample
            # to ensure a non-empty set and/or make the most of generated data.
            while step_offset - self.anchor_start_date_offset > start:
                step_offset -= step
            start = step_offset - self.anchor_start_date_offset
            cum_offset = start
            new_boundaries.append((start, finish))
        new_boundaries = new_boundaries[::-1]
        # recover the min date
        new_boundaries[0] = (boundaries[0][0], new_boundaries[0][1])
        if new_boundaries != boundaries:
            logger.warning(
                f"Changed boundaries {boundaries} to {new_boundaries} "
                f"to ensure that all offsets contain at least "
                f"one full timeframe (value {self.timeframe}).")

        if (new_boundaries[0][0] + self.timeframe_offset +
                self.anchor_start_date_offset > new_boundaries[0][1]):
            raise ValueError(
                'Not enough data for the split - the training set will end up '
                'empty. Provide a longer history of data or reduce the size '
                'of the window.')

        self.boundaries = new_boundaries
        return new_boundaries


@register(group='split', base_cls=SplitConfig)
class TemporalSplit(core_splits.TemporalSplit, BaseTemporalSplit):
    r"""Defines a temporal split of the training table according to a
    :obj:`ratio`.
    Requires that the training table has a :obj:`time_col` defined.

    Args:
        ratio (List[float]): The ratio for each split.
    """
    _external_args: set[str] = {'ratio'}

    def get_masks(
        self,
        table: Table,
    ) -> list[Mask]:
        raise UnsupportedInDiskgraphError(
            "whole-table split via Split.get_masks is dead on TT-NG; "
            "per-stage train/val/test masks come from the native "
            "DiskGraph split (split-stats sidecar / bundle)")


@register(group='split', base_cls=SplitConfig,
          map_args={'unit': (TimeUnit, TimeUnit.DAYS)})
class DateOffsetSplit(core_splits.DateOffsetSplit, BaseTemporalSplit):
    r"""Defines a date offset split of the training table according to the
    (relative) offsets from the max date.

    Args:
        offsets (List[int]): The relative and negative offsets that define
            boundaries of training splits according to the max date.
        offset_unit (TimeUnit or None, optional): The time unit of the offsets.
            (default: :obj:`:"days"`)
        unit (TimeUnit or str, optional): The time unit of the
            timeframe, timeframe_step, and start offset.
            (default: :obj:`"days"`)
        timeframe (int, optional): Timeframe of a query, used to detect and
            remove any examples that would overlap with multiple splits.
            If :obj:`None`, it is assumed to be the minimal value/difference
            between offsets.
        anchor_start_offset (int, optional): Time between the anchor time and
            the beginning of the timeframe. Used to detect and remove any
            examples where the anchor time overlaps with the previous bucket.
            (default: 0)
        timeframe_step (int, optional): If not :obj:`None`, object will adjust
             offsets to ensure that at least one sample interval appears in
             val/test set. No-op when `timeframe_step` is not used or
             when its value is a multiple of `timeframe`.
        flexible_boundaries (bool, optional): If :obj:`True`, the system will
            shift the provided boundaries to ensure the maximal use of the
            data. (default: :obj:`True`)
    """
    _external_args: set[str] = {'offsets', 'offset_unit'}

    def get_masks(
        self,
        table: Table,
    ) -> list[Mask]:
        raise UnsupportedInDiskgraphError(
            "whole-table split via Split.get_masks is dead on TT-NG; "
            "per-stage train/val/test masks come from the native "
            "DiskGraph split (split-stats sidecar / bundle)")

    def get_adjusted_boundaries(
            self, max_date: pd.Timestamp,
            min_date: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        """A helper function that gets the boundaries of the split based on
        the max date and the offsets. It also adjusts the boundaries to ensure
        that each split set starts with a training sample and that each set is
        non-empty.
        Args:
            max_date (pd.Timestamp): The maximum date in the training table.
            min_date (pd.Timestamp): The minimum date in the training table.
        Returns:
            List[Tuple[pd.Timestamp, pd.Timestamp]]: The boundaries of the
                split. Each split is described as a pair of the start and end
                timestamp, respectively.
        """
        boundaries = [min_date]
        for offset in self.offsets:
            boundaries.append(max_date + pd.DateOffset(
                **{self.offset_unit.value: offset}))  # type: ignore
        assert self.timeframe_offset is not None
        assert self.anchor_start_date_offset is not None
        boundaries.append(max_date + self.timeframe_offset +
                          self.anchor_start_date_offset)
        return self.adjust_boundaries(
            [x for x in zip(boundaries[:-1], boundaries[1:])])


@register(group='split', base_cls=SplitConfig)
class ApproxDateOffsetSplit(core_splits.ApproxDateOffsetSplit, Split):
    r"""Defines an approximate date offset split of the training table
    according to a :obj:`ratio`.
    Requires that the training table has a :obj:`time_col` defined.
    In the more general case, where train/val/test sets are all required,
    we will select train set such that it contains an integer
    value of timeframes, the number of data point in train set is equal to or
    slightly larger than its ratio in the whole training table. train set
    must leave at least 2 timeframes out for val/test. Similarly val set
    will leave at least 1 timeframe for test set and the remaining timeframe
    will be assigned to test set. Similar logics apply when only train/val
    sets are required.

    Note that we can think of ApproxDateOffsetSplit as a wrapper of
    DateOffsetSplit where the split boundaries in DateOffsetSplit is inferred
    using the data size distribution of training table timeframes and ratio.

    ApproxDateOffsetSplit is different from TemporalSplit in that TemporalSplit
    could split one training table timeframe into 2 sets, e.g. train/val,
    val/test, while timeframes will be in only one train/val/test set under
    ApproxDateOffsetSplit.

    Args:
        ratio (List[float]): The ratio for each split.
    """
    _external_args: set[str] = {'ratio'}

    #: The `DateOffsetSplit` this split resolves to once boundaries are
    #: inferred from the training table. Declared here rather than in an
    #: `__init__` so the canonical constructor stays the only one.
    split: DateOffsetSplit | None = None

    @property
    def is_temporal_split(self) -> bool:
        return True

    def get_boundaries(self) -> list[tuple[pd.Timestamp, pd.Timestamp]] | None:
        boundaries = None
        if self.split is not None:
            boundaries = self.split.boundaries
        return boundaries

    def set_date_offset_boundary(
        self,
        train_table_stats: (TemporalNodeTrainTableStats
                            | TemporalLinkTrainTableStats),
        timeframe: int,
        unit: TimeUnit | str = 'days',
        timeframe_step: int | None = None,
        flexible_boundaries: bool = True,
    ) -> None:
        r"""Fully initialize ApproxDateOffsetSplit by defining a
        DateOffsetSplit object according to train_table_stats and split ratio.

        Args:
            train_table_stats (Union[TemporalNodeTrainTableStats,
                TemporalLinkTrainTableStats]): The temporal train table stats
                of the generated training table.
            timeframe (int, optional): Timeframe of a query, used to detect
                and remove any examples that would overlap with multiple
                splits. If :obj:`None`, it is assumed to be the minimal
                value/difference between offsets.
            unit (TimeUnit or str, optional): The time unit of the offsets.
                (default: :obj:`"days"`)
            timeframe_step (int, optional): If not :obj:`None`, object will
                adjust offsets to ensure that at least one sample interval
                appears in val/test set. No-op when `timeframe_step` is not
                used or when its value is a multiple of `timeframe`.
            flexible_boundaries (bool, optional): If :obj:`True`, the system
                will shift the provided boundaries to ensure the maximal use
                of the data. (default: :obj:`True`)
        """
        # 1. Obtain the data size ratios of each timeframe stats
        num_split = len(self.ratio)
        timeframe_stats = train_table_stats.stats_per_timeframe
        assert isinstance(timeframe_stats, list)
        num_timeframes = len(timeframe_stats)

        anchor_time_list = [
            timeframe_stat.anchor_date for timeframe_stat in timeframe_stats
        ]
        assert anchor_time_list == sorted(anchor_time_list, reverse=True)

        # timeframe stats is stored from the latest timeframe to earlier
        # timeframes in train table stats. We will reverse it in the process
        # of determining train/val split boundary and val/test split boundary
        from kumo_core.training.options import allow_array_targets
        use_num_links = (  # If LP, use link ratio, not row ratio
            allow_array_targets()
            and isinstance(train_table_stats, TemporalLinkTrainTableStats)
            and train_table_stats.num_links is not None
            and train_table_stats.num_links > 0)
        timeframe_ratios = np.array(
            list(
                reversed([
                    (
                        timeframe_stat.num_links  # type: ignore
                        if use_num_links else timeframe_stat.num_train_rows)
                    /  # type: ignore
                    train_table_stats.num_train_rows
                    for timeframe_stat in timeframe_stats
                ])))
        timeframe_cum_ratios = np.cumsum(timeframe_ratios)

        # 2. Obtain the boundary of train/val set and val/test set
        # obtain the start index of val set.
        train_val_index = int(
            np.searchsorted(timeframe_cum_ratios, self.ratio[0], side='left'))
        # Make sure there are enough timeframes left for val/test set
        train_val_index = min(train_val_index, num_timeframes - num_split)
        train_val_timeframe_index = timeframe_stats[num_timeframes -
                                                    train_val_index - 2].index
        assert train_val_timeframe_index is not None

        # obtain the start index of test set
        val_test_timeframe_index = None
        if num_split == 3:
            val_test_index = int(
                np.searchsorted(
                    timeframe_cum_ratios,
                    self.ratio[0] + self.ratio[1],
                    side='left',
                ))
            # Make sure there are enough timeframes left for test set
            val_test_index = min(val_test_index, num_timeframes - 2)
            # Make sure val set contains at least one timeframe
            val_test_index = max(val_test_index, train_val_index + 1)
            val_test_timeframe_index = timeframe_stats[num_timeframes -
                                                       val_test_index - 2].index
            assert val_test_timeframe_index is not None

        # 3. Adjusting the split boundaries of date offset split.
        diff = (timeframe if timeframe_step is None else timeframe_step)

        # obtain the index of the last non empty timeframe
        max_boundary_index = timeframe_stats[0].index
        assert max_boundary_index is not None
        offsets = None
        # get the time difference betwen max date and train set end date
        train_val_offset = (max_boundary_index -
                            train_val_timeframe_index) * diff
        offsets = [train_val_offset]
        if num_split == 3:
            assert val_test_timeframe_index is not None
            # get the time difference betwen max date and val set end date
            val_test_offset = (max_boundary_index -
                               val_test_timeframe_index) * diff
            offsets.append(val_test_offset)

        # 4. Initialize a DateOffsetSplit object.
        self.split = DateOffsetSplit(offsets, offset_unit=unit, unit=unit,
                                     timeframe=timeframe,
                                     timeframe_step=timeframe_step,
                                     flexible_boundaries=flexible_boundaries)

    @property
    def is_initialized(self) -> bool:
        return self.split is not None

    def get_masks(self, table: Table) -> list[Mask]:
        raise UnsupportedInDiskgraphError(
            "whole-table split via Split.get_masks is dead on TT-NG; "
            "per-stage train/val/test masks come from the native "
            "DiskGraph split (split-stats sidecar / bundle)")

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.ratio})'


@register(
    group='split',
    base_cls=SplitConfig,
    map_args={'ranges': (list[Any], field(default_factory=list))},
)
class TimeRangeSplit(core_splits.TimeRangeSplit, BaseTemporalSplit):
    r"""Defines a time range split of the training table according to a list of
    given start and end times.
    Requires that the training table has a :obj:`time_col` defined.

    Args:
        ranges (List[Tuple[Any, Any]]): The :obj:`(start_time, end_time)` for
            each split.
        unit (TimeUnit or str, optional): The time unit of the timeframe.
            (default: :obj:`"days"`)
        timeframe (int, optional): Timeframe of a query, used to
            detect and remove any examples that would overlap with multiple
            splits. If :obj:`None`, it is assumed to be the minimal
            value/difference between offsets.
        anchor_start_offset (int, optional): Time between the anchor time and
            the beginning of the timeframe. Used to detect and remove any
            examples where the anchor time overlaps with the previous bucket.
            (default: 0)
        timeframe_step (int, optional): If not :obj:`None`, object
            will adjust offsets to ensure that at least one sample interval
            appears in val/test set. No-op when `timeframe_step` is not used or
            when its value is a multiple of `timeframe`.
        flexible_boundaries (bool, optional): If :obj:`True`, the system will
            shift the provided boundaries to ensure the maximal use of the
            data. (default: :obj:`True`)
    """
    _external_args: set[str] = {'ranges'}

    @property
    def type_safe_ranges(self) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        return self.ranges

    def get_masks(self, table: Table) -> list[Mask]:
        raise UnsupportedInDiskgraphError(
            "whole-table split via Split.get_masks is dead on TT-NG; "
            "per-stage train/val/test masks come from the native "
            "DiskGraph split (split-stats sidecar / bundle)")


@register(group='split', base_cls=SplitConfig)
class BacktestingSplit(Split):
    r"""Defines a backtesting split of the training table that only contains
    the test set and generates the labels for a specific timestamp. All
    datapoints will be part of the test set.

    Args:
        timestamp (pd.Timestamp or str): The timestamp to generate the labels
            for.
    """
    _external_args: set[str] = {'timestamp'}

    def __init__(
        self,
        timestamp: Any,
    ) -> None:
        if not isinstance(timestamp, pd.Timestamp):
            timestamp = pd.Timestamp(timestamp)
        assert isinstance(timestamp, pd.Timestamp)
        self.timestamp = timestamp

    @property
    def is_temporal_split(self) -> bool:
        return True

    @property
    def num_splits(self) -> int:
        return 1

    def get_boundaries(self) -> None:
        return None

    def get_masks(self, table: Table) -> list[Mask]:
        raise UnsupportedInDiskgraphError(
            "whole-table split via Split.get_masks is dead on TT-NG; "
            "per-stage train/val/test masks come from the native "
            "DiskGraph split (split-stats sidecar / bundle)")

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.timestamp})')


@register(group='split', base_cls=SplitConfig)
class ColumnSplit(Split):
    r"""Defines a split of the training table according to a user specified
    column :obj:`col` in the training table.

    Args:
        col (BaseColumn): The column name of training table (feature column,
            pkey, fkeys) that will be sorted in ascending order, which will
            then be use to create splits based on the :obj:`ratio` argument
        ratio (List[float]): The ratio for each split.
    """
    _external_args: set[str] = {'col', 'ratio'}

    def __init__(self, col: str | BaseColumn, ratio: list[float]) -> None:
        validate_split_ratios(ratio)

        self.col = BaseColumn.cast(col)
        # todo: infer how to create splits based on column data
        self.ratio = ratio

    @property
    def is_temporal_split(self) -> bool:
        return False

    @property
    def num_splits(self) -> int:
        return len(self.ratio)

    def get_masks(self, table: Table) -> list[Mask]:
        raise UnsupportedInDiskgraphError(
            "whole-table split via Split.get_masks is dead on TT-NG; "
            "per-stage train/val/test masks come from the native "
            "DiskGraph split (split-stats sidecar / bundle)")

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(col={self.col.name}, '
                f'ratio={self.ratio})')


@register(group='split', base_cls=SplitConfig)
class ManualSplit(core_splits.ManualSplit, Split):
    r"""Defines a split of the training table according to a column in the
    training table with specified values for each split.

    Args:
        col (str): The name of the training-table column that will be used to
            define the training / validation / test split.
        train_value (optional, Any or List[Any]): The column value(s) that will
            be used to indicate the training split (default: :obj:`0`).
        val_value (optional, Any or List[Any]): The column value(s) that will
            be used to indicate the validation split (default: :obj:`1`).
        test_value (optional, Any or List[Any]): The column value(s) that will
            be used to indicate the test split (default: :obj:`None`, for
            which an all-zero test split is produced).
    """
    _external_args: set[str] = {'col', 'train_value', 'val_value', 'test_value'}

    @property
    def is_temporal_split(self) -> bool:
        return False

    def get_masks(self, table: Table) -> list[Mask]:
        raise UnsupportedInDiskgraphError(
            "whole-table split via Split.get_masks is dead on TT-NG; "
            "per-stage train/val/test masks come from the native "
            "DiskGraph split (split-stats sidecar / bundle)")


def _normalize_legacy_split(cfg: Any) -> Any:
    """Rewrites a split config whose serialized shape has since changed."""
    if isinstance(cfg, Mapping):
        target = cfg.get('_target_')
    else:
        target = getattr(cfg, '_target_', None)
    if not isinstance(target, str) or not target:
        return cfg
    try:
        target_cls = locate(target)
    except Exception:
        return cfg  # Let `from_config` report an unresolvable target.
    # Deliberately not every split carrying a `col`: `ColumnSplit` still
    # takes a whole column, and reading only its name would drop the rest.
    if isinstance(target_cls, type) and issubclass(target_cls, ManualSplit):
        return _with_column_name(cfg)
    return cfg


def _with_column_name(cfg: Any) -> Any:
    """Returns `cfg` with a serialized `col` replaced by its column name."""
    is_mapping = isinstance(cfg, Mapping)
    col = cfg.get('col') if is_mapping else getattr(cfg, 'col', None)
    if not isinstance(col, Mapping):
        return cfg
    name = col.get('name')
    if not isinstance(name, str) or not name:
        # Not the shape this reads; let the constructor refuse it by name.
        return cfg
    if is_mapping:
        return {**cfg, 'col': name}
    if is_dataclass(cfg) and not isinstance(cfg, type):
        return replace(cfg, col=name)
    return cfg


###############################################################################
