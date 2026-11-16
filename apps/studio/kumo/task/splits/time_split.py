# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List, Literal, Tuple, Union

import numpy as np
import pandas as pd
import torch
from kumo_core.train.mapping import NODE_INDEX_COLUMN
from kumo_core.utils.registry import register
from kumoapi.typing import TimeUnit

from io_utils import logging
from kumo.config import SplitConfig
from kumo.config.task import Mask, TimePeriod
from kumo.graph import Table
from kumo.task import Split
from kumo.task.splits.util import SMALLEST_OFFSET_UNIT, SPLIT_STAGES

logger = logging.getLogger(__name__)


@register(group='split', base_cls=SplitConfig)
class TimeSplit(Split):
    r"""A time-based split implementation that supports multiple ways of
    splitting data.

    This class provides four mutually exclusive ways to define splits:
    1. timestamp_boundaries: List of timestamp string pairs defining split
        ranges.
    2. offset_boundaries: List of integer offset pairs defining split ranges
        in time units (e.g. -180 means 180 days/hours/months ago from latest
        timestamp).
    3. fractions: List of fractional pairs defining relative split ranges.
    4. indices: List of integer pairs defining split ranges by numbering
        timeframes (-1 is latest timeframe, -2 is second latest, etc.).

    Args:
        timestamp_boundaries (List[Tuple[str, str]] | List[List[str]] | None):
            List of timestamp string pairs defining split ranges (e.g.
            '2022-01-01'). (default: :obj:`None`)
        offset_boundaries (List[Tuple[Union[int, Literal['-INF']], int]] |
            List[List[Union[int, Literal['-INF']]]] | None): List of integer
            List of integer offset pairs or -INF for unbounded start.
            (default: :obj:`None`)
        fractions (List[float] | None): List of float values defining
            relative split ranges. (default: :obj:`None`)
        indices (List[Tuple[Union[int, Literal['-INF']], int]] |
            List[List[Union[int, Literal['-INF']], int]] | None): List of
            integer pairs defining which of the generated timeframes are
            used in each split where -1 is the last timeframe, -2 is the
            second to last, etc. (default: :obj:`None`)
        strict_split_boundaries (bool): Whether to enforce strict boundaries
            between splits. When True, ensures no overlap between splits.
            (default: :obj:`True`)
        unit (TimeUnit | str | None): Time unit for interpreting intervals.
            One of: days, months, hours, minutes. (default: :obj:`None`)
        timeframe_size (int | None): Size of the timeframe window.
            (default: :obj:`None`)
        anchor_start_offset (int | None): Offset from anchor point to start of
            timeframe. (default: :obj:`None`)

    Raises:
        ValueError: If more than one split type is specified or if no split
            type is specified.
    """
    _external_args: set[str] = {
        'timestamp_boundaries',
        'offset_boundaries',
        'fractions',
        'indices',
        'strict_split_boundaries',
    }

    def __init__(
        self,
        # User-specified arguments
        timestamp_boundaries: List[Tuple[str, str]] | List[List[str]]
        | None = None,
        offset_boundaries: List[Tuple[Union[int, Literal['-INF']], int]]
        | List[List[Union[int, Literal['-INF']]]] | None = None,
        fractions: List[float] | None = None,
        indices: List[Tuple[Union[int, Literal['-INF']], int]]
        | List[List[Union[int, Literal['-INF']]]] | None = None,
        strict_split_boundaries: bool = True,
        # These arguments come from the Predictive Query and are not specified
        # by the user.
        unit: TimeUnit | str | None = None,
        timeframe_size: int | None = None,
        anchor_start_offset: int | None = None,
    ) -> None:
        self.timestamp_boundaries = timestamp_boundaries
        self.offset_boundaries = offset_boundaries
        self.fractions = fractions
        self.indices = indices
        self.strict_split_boundaries = strict_split_boundaries
        self.unit = TimeUnit(
            unit.lower() if isinstance(unit, str) else unit) if unit else None

        self.timeframe_size = timeframe_size
        self.timeframe_size_date_offset = self.int_to_date_offset(
            timeframe_size, self.unit)

        self.anchor_start_offset = anchor_start_offset
        self.anchor_start_offset_date_offset = self.int_to_date_offset(
            anchor_start_offset, self.unit)

        # Validate the arguments
        self.validator = TimeSplitValidator(self)
        self.validator.validate()

        # Initialize the timestamp boundaries to None
        self._boundaries = None

    @staticmethod
    def int_to_date_offset(
        value: int | None,
        unit: TimeUnit | None,
    ) -> pd.DateOffset | None:
        r"""Convert an integer value and time unit into a pandas DateOffset.

        This method converts a numeric interval and time unit into a pandas
        DateOffset object that can be used for time-based calculations.

        Args:
            value (int | None): The numeric value to convert. If None,
                returns None.
            unit (TimeUnit | None): The time unit to use for conversion.
                Must be one of the supported TimeUnit values
                (DAYS, MONTHS, HOURS, MINUTES). If None, returns None.

        Returns:
            pd.DateOffset if both value and unit are provided, None otherwise.

        Raises:
            ValueError: If unit is provided but not one of the supported
                TimeUnit values.
        """
        if value is None or unit is None:
            return None
        if unit == TimeUnit.DAYS:
            return pd.DateOffset(days=value)
        elif unit == TimeUnit.MONTHS:
            return pd.DateOffset(months=value)
        elif unit == TimeUnit.HOURS:
            return pd.DateOffset(hours=value)
        elif unit == TimeUnit.MINUTES:
            return pd.DateOffset(minutes=value)
        raise ValueError(f"Invalid unit: {unit}")

    def _create_time_range_masks(
        self,
        table: Table,
        boundaries: List[Tuple[pd.Timestamp, pd.Timestamp]],
    ) -> List[Mask]:
        r"""
        Given a Table (usually a TrainingTable), and a list of boundaries, take
        the table and use its time column to generate masks for rows that fall
        within the boundaries.
        Args:
            table (Table): The table to split.
            boundaries (List[Tuple[pd.Timestamp, pd.Timestamp]]): The
                boundaries to split the table into.
        Returns:
            List[Mask]: The masks corresponding to the boundaries.
        """
        # Get the time column from the table. For training tables, this is the
        # column containing the timestamps for each timeframe.
        time = table.time_col.pandas_indexed()
        assert isinstance(time, pd.DataFrame)
        time_col = time[table.time_col.name]
        assert isinstance(time_col, pd.Series)

        masks: List[Mask] = []

        # Process each boundary
        for start_ts, end_ts in boundaries:
            assert end_ts >= start_ts

            # Static Query
            # NOTE: for raw data, the split boundary is left exclusive and
            # right inclusive.
            if self.timeframe_size_date_offset is None:
                assert self.anchor_start_offset_date_offset is None
                rows_after_start = (time_col > start_ts).values
                rows_before_end = (time_col <= end_ts).values
            else:
                # Temporal Query
                assert self.anchor_start_offset_date_offset is not None

                # Apply timeframe offsets sequentially to check if each row
                # fits within boundaries.
                time_with_offsets = time_col
                if self.strict_split_boundaries:
                    time_with_offsets = (time_with_offsets +
                                         self.timeframe_size_date_offset +
                                         self.anchor_start_offset_date_offset)

                # NOTE: for train data, i.e. data generated with aggregation,
                # against the anchor time of the temporal aggregations, the
                # split boundary is left inclusive and right exclusive. This is
                # actually consistent with the raw data split boundary, i.e.
                # the split boundary is left exclusive and right inclusive
                # against the raw data timestamps. Because the temporal
                # aggregation is left exclusive. As a result, the raw data in
                # start_ts is not included in the split.
                rows_after_start = (time_col >= start_ts).values
                rows_before_end = (time_with_offsets < end_ts).values

                # Since for temporal query, aggregation boundaries are left
                # open and right closed, we should use <= to check whether the
                # max date of the previous split is smaller or equal to the
                # anchor date of the next split.
                if self.strict_split_boundaries:
                    rows_before_end = (time_with_offsets <= end_ts).values

            # The mask is comprised of rows that fit both criteria.
            mask = torch.from_numpy(rows_after_start) & torch.from_numpy(
                rows_before_end)

            # If the table has an entity include bit column, we need to make
            # sure the mask only includes rows where the entity include bit
            # is set.
            from kumo.task.training_table import TrainingTable
            if isinstance(table,
                          TrainingTable) and table.has_entity_include_bit_col():
                mask &= torch.from_numpy(
                    table.data.entity_include_bit.values.astype(np.bool_))

            # Convert the mask to a numpy array and construct the mask object.
            mask_arr: np.ndarray = mask.numpy()
            indexed_mask = time[mask_arr][NODE_INDEX_COLUMN]
            masks.append(
                Mask(
                    tensor=mask,
                    index_tensor=torch.from_numpy(indexed_mask.values),
                    time_period=TimePeriod.cast(start_ts, end_ts),
                ))

        return masks

    def set_boundaries(
            self, boundaries: List[Tuple[pd.Timestamp, pd.Timestamp]]) -> None:
        """Set the timestamp boundaries for this split.

        Args:
            boundaries (List[Tuple[pd.Timestamp, pd.Timestamp]]): List of
                timestamp pairs defining the start and end of each split.
        """
        self._boundaries = boundaries

    def create_timestamp_boundaries(
            self) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        r"""Get timestamp boundaries for each split based on the configured
        timestamp boundaries.

        This method takes the configured timestamp boundaries and converts
        them to pandas Timestamps before generating timestamp boundaries for
        each split (train/val/test). The boundaries must be specified as
        timestamp strings (e.g. "2022-01-01").

        Args:
            table (Table): The table to generate masks for. Must have a time
                column.

        Returns:
            List[Tuple[pd.Timestamp, pd.Timestamp]]: A list of timestamp pairs,
                one for each split. Each pair defines the start and end of a
                split.

        Raises:
            AssertionError: If timestamp_boundaries is None.
        """
        assert self.timestamp_boundaries is not None

        def to_timestamp(
                boundary: Union[str, int, Literal['-INF']]) -> pd.Timestamp:
            if boundary == '-INF':
                ts = pd.Timestamp.min
            else:
                ts = pd.Timestamp(boundary)
            assert isinstance(ts, pd.Timestamp)
            return ts

        timestamp_boundaries = [(to_timestamp(start), to_timestamp(end))
                                for start, end in self.timestamp_boundaries]

        # Store the computed boundaries
        self.set_boundaries(timestamp_boundaries)
        return timestamp_boundaries

    def _offset_boundary_start_end_ts_adjustment(
        self,
        idx: int,
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp,
        start_is_inf: bool,
    ) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """Adjusts the start and end timestamps for a given split index based
        on offset and indices boundaries.

        This method is used when calculating the time boundaries for a split
        in a TimeSplit configuration. It handles special cases like infinite
        start times and adjusts the timestamps based on the split's offset and
        indices configuration.

        Args:
            idx (int): The index of the split being processed. From the oldest
                data to the newest data.
            start_ts (pd.Timestamp): The initial start timestamp for the split.
            end_ts (pd.Timestamp): The initial end timestamp for the split.
            start_is_inf (bool): Whether the start timestamp represents
                negative infinity.

        Returns:
            Tuple[pd.Timestamp, pd.Timestamp]: A tuple containing the
                adjusted (start_ts, end_ts) timestamps that account for the
                split's offset and indices boundaries.

        Note:
            - If start_is_inf is True, the start timestamp will be adjusted
              based on the split's offset.
            - The end timestamp may be adjusted based on the split's indices
              configuration.
        """
        # Temporal Query
        if self.timeframe_size_date_offset is not None:
            # This adjustment is necessary because the split is
            # defined on raw data for training data, we need to
            # account for the anchor start offset and timeframe size.
            if not start_is_inf:
                start_ts += self.anchor_start_offset_date_offset
                start_ts += self.timeframe_size_date_offset
            end_ts += self.anchor_start_offset_date_offset
            end_ts += self.timeframe_size_date_offset
        else:
            # Static Query
            # This adjustment is necessary because for static query, the
            # split boundaries are left exclusive and right inclusive.
            # As a result, the raw data (target table) in start_ts is not
            # included even when start boundary is -INF in the split without
            # this adjustment.
            if start_is_inf:
                start_ts -= SMALLEST_OFFSET_UNIT
        return start_ts, end_ts

    def _create_offset_boundaries(
            self, table: Table) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        r"""Get timestamp boundaries for each split based on the configured
        offset boundaries.

        This method takes the configured offset boundaries and converts them
        to pandas Timestamps before generating timestamp boundaries for each
        split (train/val/test). For each tuple pair of offset boundaries
        (start_offset, end_offset), both offsets are inclusive, meaning the
        split will include timestamps at both start_offset and end_offset=
        positions.

        For example, given timestamps
        [2021-01-01, 2021-01-02, 2021-01-03, 2021-01-04]:
          - indices=[(-4,-3), (-2,-1)] would create:
          - First split: [2021-01-01, 2021-01-02]  # includes both -4 and -3
          - Second split: [2021-01-03, 2021-01-04]  # includes both -2 and -1

        Args:
            table (Table): The table to generate masks for. Must have a time
                column.

        Returns:
            List[Tuple[pd.Timestamp, pd.Timestamp]]: A list of timestamp pairs,
                one for each split. Each pair defines the start and end of a
                split.

        Raises:
            AssertionError: If indices is None.
            ValueError: If any index is out of bounds.
        """
        assert self.offset_boundaries is not None

        def to_timestamp(
                boundary: Union[str, int, Literal['-INF']]) -> pd.Timestamp:
            if boundary == '-INF':
                # Get the earliest timestamp from the table
                time_col = table.time_col.pandas()
                assert isinstance(time_col, pd.Series)
                return pd.Timestamp(time_col.min())  # type: ignore
            else:
                assert isinstance(boundary, int)
                # Convert offset to timestamp using the latest timestamp
                time_col = table.time_col.pandas()
                assert isinstance(time_col, pd.Series)
                latest_ts = pd.Timestamp(time_col.max())  # type: ignore
                offset = self.int_to_date_offset(abs(boundary), self.unit)
                assert offset is not None
                return latest_ts - offset

        timestamp_boundaries = []
        for idx, (start, end) in enumerate(self.offset_boundaries):
            start_ts = to_timestamp(start)
            end_ts = to_timestamp(end)
            start_is_inf = start == '-INF'
            start_ts, end_ts = (self._offset_boundary_start_end_ts_adjustment(
                idx=idx,
                start_ts=start_ts,
                end_ts=end_ts,
                start_is_inf=start_is_inf,
            ))
            timestamp_boundaries.append((start_ts, end_ts))

        # Store the computed boundaries
        self.set_boundaries(timestamp_boundaries)
        return timestamp_boundaries

    def _create_indices_boundaries(
            self, table: Table) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        r"""Get masks for each split based on the configured indices.

        This method takes the configured indices and converts them to timestamp
        boundaries for each split (train/val/test). For each tuple pair of
        indices (start_idx, end_idx), both indices are inclusive, meaning the
        split will include timestamps at both start_idx and end_idx positions.

        For example, given timestamps
        [2021-01-01, 2021-01-02, 2021-01-03, 2021-01-04]:
          - indices=[(-4,-3), (-2,-1)] would create:
          - First split: [2021-01-01, 2021-01-02]  # includes both -4 and -3
          - Second split: [2021-01-03, 2021-01-04]  # includes both -2 and -1

        Args:
            table (Table): The table to generate masks for. Must have a time
                column.

        Returns:
            List[Tuple[pd.Timestamp, pd.Timestamp]]: A list of timestamp pairs,
                one for each split. Each pair defines the start and end of a
                split.

        Raises:
            AssertionError: If indices is None.
            ValueError: If any index is out of bounds.
        """
        assert self.indices is not None

        # Get sorted unique timestamps from the table
        time_col = table.time_col.pandas()
        assert isinstance(time_col, pd.Series)
        sorted_timestamps = sorted(time_col.unique())
        sorted_timestamps_length = len(sorted_timestamps)

        # Validate indices before processing
        self.validator.validate_indices_against_num_timestamps(
            sorted_timestamps_length)

        # Convert indices to timestamp boundaries
        timestamp_boundaries = []
        for idx, (start_idx, end_idx) in enumerate(self.indices):
            # Handle '-INF' for first start index
            start_idx_is_inf = False
            if start_idx == '-INF':
                start_idx_is_inf = True
                start_idx = -sorted_timestamps_length + 1
            else:
                start_idx = int(start_idx)
            end_idx = int(end_idx)

            # Convert indices since sorted_timestamps is in ascending order.
            start_idx = sorted_timestamps_length + start_idx - 1
            end_idx = sorted_timestamps_length + end_idx - 1

            # For temporal queries, where the current start idx is not -INF,
            # we need to adjust the start idx by 1 to account for the fact that
            # the start idx is exclusive on raw data (target table) but
            # inclusive on the training data.
            if (self.timeframe_size_date_offset is not None
                    and not start_idx_is_inf):
                start_idx += 1

            # For temporal queries, we need to adjust the end idx by 1 to
            # account for the fact that the end idx is right inclusive on raw
            # data but right exclusive on the training data.
            if (self.timeframe_size_date_offset is not None
                    and idx < self.num_splits - 1):
                end_idx += 1

            start_ts = pd.Timestamp(sorted_timestamps[start_idx])
            end_ts = pd.Timestamp(sorted_timestamps[end_idx])

            # Temporal Query
            if self.timeframe_size_date_offset is not None:
                if idx == self.num_splits - 1:
                    end_ts += self.anchor_start_offset_date_offset
                    end_ts += self.timeframe_size_date_offset
            else:
                # Static Query
                if start_idx_is_inf:
                    start_ts -= SMALLEST_OFFSET_UNIT

            timestamp_boundaries.append((start_ts, end_ts))

        self.set_boundaries(timestamp_boundaries)
        return timestamp_boundaries

    def _create_fractions_boundaries(
            self, table: Table) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        r"""Get masks for each split based on the configured fractions.

        This method takes the configured fractions and generates boundaries
        that define which rows belong to each split (train/val/test). It
        ensures that data points with the same anchor date are assigned to the
        same split.

        Args:
            table (Table): The table to generate masks for. Must have a time
                column.

        Returns:
            List[Tuple[pd.Timestamp, pd.Timestamp]]: A list of timestamp pairs,
                one for each split. Each pair defines the start and end of a
                split.

        Raises:
            AssertionError: If fractions is None.
            ValueError: If there aren't enough anchor dates for the
                requested split.
        """
        from kumo_core.training.options import allow_array_targets

        from kumo.pquery.train_table_generator import TrainingTableGeneratorV2
        assert self.fractions is not None

        # Extract the table data and time column name
        table_data = table.pandas()
        time_col_name = table.time_col.name

        # Get unique timestamps and their counts, possibly weighted by TARGET
        # list length
        TARGET = TrainingTableGeneratorV2.TARGET
        # ``allow_array_targets()`` is blanket-True under TT-NG, so on its own it
        # can no longer distinguish array (link-prediction) targets from scalar
        # ones -- every live consumer additionally gates on ``is_link_pred``
        # (see ``diskgraph_train_table``). Here we discriminate straight from the
        # data: the array-length weighting only applies when the ``TARGET``
        # column actually holds per-row lists. Scalar targets (binary/regression)
        # keep the row-count behavior.
        has_array_target = (allow_array_targets()
                            and TARGET in table_data.columns
                            and len(table_data) > 0
                            and isinstance(table_data[TARGET].iloc[0],
                                           (list, np.ndarray)))
        if has_array_target:
            # Group by timestamp and sum the length of TARGET lists for each
            # timestamp
            unique_times = table_data.groupby(time_col_name).apply(
                lambda group: sum(len(target)
                                  for target in group[TARGET])).sort_index()
        else:
            # Standard behavior - count rows per timestamp
            time_col = table_data[time_col_name]
            assert isinstance(time_col, pd.Series)
            unique_times = time_col.value_counts().sort_index()

        total_rows = unique_times.sum()
        num_anchor_dates = len(unique_times)

        # Validate we have enough anchor dates
        self.validator.validate_num_splits_against_num_timestamps(
            num_anchor_dates)

        # Calculate cumulative ratios for each unique timestamp
        cumulative_counts = unique_times.cumsum()
        cumulative_ratios = cumulative_counts / total_rows

        # Find split boundaries using searchsorted, ensuring each split gets
        # at least one anchor date and leaving enough dates for remaining
        # splits
        split_boundaries = []
        cumsum = 0.0
        remaining_splits = self.num_splits
        for frac in self.fractions[:-1]:  # Process all but last fraction
            remaining_splits -= 1
            cumsum += frac

            # Find the index where cumulative ratio exceeds target
            idx = np.searchsorted(cumulative_ratios.values, cumsum,
                                  side='right')

            # Ensure we leave enough anchor dates for remaining splits
            max_allowed_idx = num_anchor_dates - remaining_splits
            if idx >= max_allowed_idx:
                idx = max_allowed_idx

            # Ensure we move forward at least one anchor date
            if len(split_boundaries) > 0:
                last_boundary_idx = np.searchsorted(unique_times.index,
                                                    split_boundaries[-1])
            else:
                last_boundary_idx = 0
            if idx <= last_boundary_idx:
                idx = last_boundary_idx + 1

            assert 0 <= idx < len(unique_times)
            split_boundaries.append(unique_times.index[idx])

        # Convert split boundaries to timestamp pairs
        timestamp_boundaries = []
        start_time = pd.Timestamp(unique_times.index[0])

        # For static queries, we need to adjust the start time by the smallest
        # offset unit to account for the fact that the start time is exclusive
        # on raw data but inclusive on the training data.
        if self.timeframe_size_date_offset is None:
            start_time -= SMALLEST_OFFSET_UNIT
        for end_time in split_boundaries:
            end_time = pd.Timestamp(end_time)
            timestamp_boundaries.append((start_time, end_time))
            start_time = end_time

        # Add final boundary
        max_time = pd.Timestamp(unique_times.index[-1])
        if self.timeframe_size_date_offset is not None:
            max_time += self.anchor_start_offset_date_offset
            max_time += self.timeframe_size_date_offset
        timestamp_boundaries.append((start_time, max_time))

        # Store the computed boundaries
        self.set_boundaries(timestamp_boundaries)
        return timestamp_boundaries

    def create_boundaries(
            self, table: Table) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        if self.timestamp_boundaries is not None:
            return self.create_timestamp_boundaries()
        elif self.offset_boundaries is not None:
            return self._create_offset_boundaries(table)
        elif self.indices is not None:
            return self._create_indices_boundaries(table)
        elif self.fractions is not None:
            return self._create_fractions_boundaries(table)
        raise NotImplementedError

    def get_masks(self, table: Table) -> List[Mask]:
        r"""Generate the masks for each split based on the configured
        split type.

        This method returns a list of masks that define which rows belong to
        each split (train/val/test). The masks are generated based on the
        split configuration - either using timestamp_boundaries,
        offset_boundaries, indices, or fractions.

        Args:
            table (Table): The table to generate masks for. Must have a time
                column.

        Returns:
            List[Mask]: A list of masks, one for each split. Each mask
                indicates which rows belong to that split.
        """
        boundaries = self.create_boundaries(table)
        logger.info(f"Generated TimeSplit Boundaries: {boundaries}")
        return self._create_time_range_masks(
            table=table,
            boundaries=boundaries,
        )

    @property
    def is_temporal_split(self) -> bool:
        """Returns :obj:`True` if split is based on :obj:`table.time_col`."""
        return True

    def get_boundaries(self) -> List[Tuple[pd.Timestamp, pd.Timestamp]] | None:
        r"""Returns the final split boundaries after all adjustments.

        Returns:
            (List[Tuple[pd.Timestamp, pd.Timestamp]] | None): When split
                boundaries are not finalized, returns `None`. Otherwise,
                returns the final calculated boundaries.
        """
        return self._boundaries

    @property
    def num_splits(self) -> int:
        if self.indices is not None:
            return len(self.indices)
        elif self.timestamp_boundaries is not None:
            return len(self.timestamp_boundaries)
        elif self.offset_boundaries is not None:
            return len(self.offset_boundaries)
        elif self.fractions is not None:
            return len(self.fractions)
        raise ValueError(
            "No valid split type found - one of indices, boundaries, or "
            "fractions must be set")

    def __repr__(self) -> str:
        if self.indices is not None:
            return (f'{self.__class__.__name__}('
                    f'indices={self.indices}, '
                    f'strict_split_boundaries={self.strict_split_boundaries})')
        elif self.timestamp_boundaries is not None:
            return (f'{self.__class__.__name__}('
                    f'timestamp_boundaries={self.timestamp_boundaries}, '
                    f'strict_split_boundaries={self.strict_split_boundaries})')
        elif self.offset_boundaries is not None:
            return (f'{self.__class__.__name__}('
                    f'offset_boundaries={self.offset_boundaries}, '
                    f'strict_split_boundaries={self.strict_split_boundaries})')
        elif self.fractions is not None:
            return (f'{self.__class__.__name__}('
                    f'fractions={self.fractions}, '
                    f'strict_split_boundaries={self.strict_split_boundaries})')
        raise ValueError(
            "No valid split type found - one of indices, boundaries, or "
            "fractions must be set")


class TimeSplitValidator():
    def __init__(self, split: TimeSplit) -> None:
        self.split = split

    def _validate_timeframe_params(self) -> None:
        r"""Validates that timeframe parameters are consistent.

        Raises:
            ValueError: If any of anchor_start_offset or timeframe_size
                is None while others are not None, or vice versa.
        """
        has_none = (self.split.anchor_start_offset is None
                    or self.split.timeframe_size is None)
        all_none = (self.split.anchor_start_offset is None
                    and self.split.timeframe_size is None)
        if has_none != all_none:
            raise ValueError("anchor_start_offset, timeframe_size must all be "
                             "None or all be not None")

    def _validate_num_splits(
        self, data: List[Tuple[Union[int, Literal['-INF']], int]]
        | List[List[Union[int, Literal['-INF']]]]
        | List[Tuple[pd.Timestamp, pd.Timestamp]]
        | List[List[pd.Timestamp]]
        | List[Tuple[float, float]] | List[List[float]]
        | List[float]
    ) -> None:
        r"""Validate that the number of splits is between 2 and 3.

        Args:
            data: A list of tuples or lists containing either integers,
                timestamps, or floats that define the split ranges.

        Raises:
            ValueError: If number of splits is less than 2 or more than 3
        """
        if len(data) < 2:
            raise ValueError("At least two splits are required")
        if len(data) > 3:
            raise ValueError("No more than three splits is supported")

    def _validate_tuple_structure(
        self, data: List[Tuple[Union[int, Literal['-INF']], int]]
        | List[List[Union[int, Literal['-INF']]]]
        | List[Tuple[str, str]] | List[List[str]]
        | List[Tuple[float, float]] | List[List[float]]
        | List[float]
    ) -> None:
        r"""Validate that each item in the data is a tuple/list with exactly
        2 elements.

        Args:
            data: List of items to validate.

        Raises:
            ValueError: If any item is not a tuple/list with exactly 2
                elements.
        """
        for item in data:
            if not (isinstance(item, tuple)
                    or isinstance(item, list)) or len(item) != 2:
                raise ValueError(
                    "Data must be a list of tuples with two elements")

    def _validate_arg_exclusivity(self) -> None:
        r"""Validates that exactly one split configuration argument is
        specified.

        The TimeSplit class accepts three mutually exclusive ways to
            define splits:
        - boundaries: List of timestamp pairs defining split ranges.
        - fractions: List of fractional pairs defining relative split ranges.
        - indices: List of integer pairs defining offset-based split ranges.

        Raises:
            ValueError: If zero or multiple split configuration arguments are
                specified. The error message will indicate which arguments
                conflict or that one is required.
        """
        # Track which split arguments are set
        split_args = {
            'timestamp_boundaries': self.split.timestamp_boundaries is not None,
            'offset_boundaries': self.split.offset_boundaries is not None,
            'fractions': self.split.fractions is not None,
            'indices': self.split.indices is not None
        }

        # Count how many split arguments are set
        num_args_set = sum(split_args.values())

        # Must specify exactly one argument
        if num_args_set == 0:
            raise ValueError(
                "Must specify exactly one of: boundaries, fractions, "
                "or indices")
        elif num_args_set > 1:
            # Find which arguments are in conflict
            set_args = [arg for arg, is_set in split_args.items() if is_set]
            raise ValueError(f"Cannot specify multiple split arguments. Found: "
                             f"{', '.join(set_args)}")

    def _validate_indices_or_boundaries(
        self,
        data: List[Tuple[int | Literal['-INF'], int]]
        | List[List[int | Literal['-INF']]]
        | List[Tuple[pd.Timestamp, pd.Timestamp]]
        | List[List[pd.Timestamp]],
    ) -> None:
        r"""Validate the generic structure and ordering of split data.

        This method performs common validation checks for different types of
        split configurations (boundaries, indices). It ensures:
        1. The correct number of splits (between 2 and 3).
        2. Each split has exactly two elements (start and end).
        3. Proper ordering within each split (start < end).
        4. No overlap between consecutive splits.

        Args:
            data: List of splits, where each split is a tuple/list of two
                elements representing start and end values. Supports various
                types:
                - Timestamp pairs for time boundaries.
                - Integer pairs for indices.

        Raises:
            ValueError: If any validation check fails:
                - Less than 2 or more than 3 splits.
                - Invalid split format (not a tuple/list of 2 elements).
                - Start value greater than end value within a split.
                - Overlapping splits (end of one split > start of next split).
            AssertionError: If data is None or start/end types don't match.
        """
        assert data is not None
        self._validate_num_splits(data)

        for idx, item in enumerate(data):
            # Check length
            if not (isinstance(item, tuple)
                    or isinstance(item, list)) or len(item) != 2:
                raise ValueError(
                    "Data must be a list of tuples with two elements")

            # Check ordering
            start = item[0]
            end = item[1]

            # Skip comparison if start is '-INF' (only valid for first index)
            if idx == 0 and start == '-INF':
                continue

            if start > end:  # type: ignore
                stage = SPLIT_STAGES[idx]
                raise ValueError(f"The start ({start}) of the "
                                 f"({stage.value}) split must be strictly "
                                 f"smaller than the end ({end}).")

            if idx < len(data) - 1:
                next_start = data[idx + 1][0]
                current_stage = SPLIT_STAGES[idx]
                next_stage = SPLIT_STAGES[idx + 1]
                if end > next_start:  # type: ignore
                    raise ValueError(
                        f"The end ({end}) of the "
                        f"({current_stage.value}) split must be "
                        f"strictly smaller than the start ({next_start}) of "
                        f"the next ({next_stage.value}) split.")

    def _validate_indices_are_non_positive(
        self, data: List[Tuple[int | Literal['-INF'], int]]
        | List[List[int | Literal['-INF']]]
    ) -> None:
        r"""Validate that all indices are non-positive (<=0) or '-INF' for
        first start.

        Args:
            data: List of index pairs to validate.

        Raises:
            ValueError: If any index is positive, if '-INF' is used
                incorrectly, or if any index is used more than once
        """
        for idx, (start, end) in enumerate(data):
            # Allow '-INF' only for the first start index
            if idx == 0 and start == '-INF':
                continue

            if not isinstance(start, int) or not isinstance(end, int):
                stage = SPLIT_STAGES[idx]
                raise ValueError(
                    f"Indices must be integers (except '-INF' for first "
                    f"start), but got {start, end} for the ({stage.value}) "
                    f"split")

            if start > 0 or end > 0:
                stage = SPLIT_STAGES[idx]
                raise ValueError(f"Indices must be non-positive (<=0), but got "
                                 f"{start, end} for the ({stage.value}) split")

    def _validate_indices(self) -> None:
        assert self.split.indices is not None
        indices = self.split.indices
        self._validate_tuple_structure(indices)
        self._validate_indices_are_non_positive(indices)
        self._validate_indices_or_boundaries(indices)

    def _boundary_string_to_date(self, boundary: str) -> pd.Timestamp:
        return pd.Timestamp(boundary)

    def _validate_boundary_values(
        self, boundaries: List[Tuple[Union[int, Literal['-INF']], int]]
        | List[List[Union[int, Literal['-INF']]]]
        | List[Tuple[str, str]]
        | List[List[str]]
    ) -> None:
        r"""Validate that boundary values are consistent in type and format.

        Args:
            boundaries: List of boundary pairs to validate. Can be either:
                - List of string timestamp tuples/lists
                - List of integer offset tuples/lists (with optional '-INF'
                    at first start)

        Raises:
            ValueError: If boundaries mix different types or have invalid
                formats.
        """
        # Check first value to determine expected type
        first_val = boundaries[0][0]

        # Case 1: All string timestamps
        if isinstance(first_val, str) and first_val != '-INF':
            for idx, (start, end) in enumerate(boundaries):
                if not (isinstance(start, str) and isinstance(end, str)):
                    raise ValueError(
                        f"When using timestamp strings, all boundaries must "
                        f"be strings. Found non-string values in boundary "
                        f"{idx}: ({start}, {end})")
                # Validate strings are valid timestamps
                try:
                    pd.Timestamp(start)
                    pd.Timestamp(end)
                except ValueError as e:
                    raise ValueError(
                        f"Invalid timestamp format in boundary {idx}: "
                        f"({start}, {end}). Error: {e}")

        # Case 2: All integers (with optional -INF at first start only)
        elif isinstance(first_val, int) or first_val == '-INF':
            # Check first boundary separately
            if first_val == '-INF':
                if not isinstance(boundaries[0][1], int):
                    raise ValueError(
                        f"When using '-INF', all other boundary values must "
                        f"be integers. Found non-integer end value in first "
                        f"boundary: {boundaries[0][1]}")
            else:
                if not isinstance(boundaries[0][1], int):
                    raise ValueError(
                        f"When using integer offsets, all boundaries must "
                        f"be integers. Found non-integer end value in "
                        f"first boundary: {boundaries[0][1]}")

            # Check remaining boundaries
            for idx, (start, end) in enumerate(boundaries[1:], 1):
                if not (isinstance(start, int) and isinstance(end, int)):
                    raise ValueError(
                        f"When using integer offsets, all boundaries must "
                        f"be integers (except optional '-INF' for first "
                        f"start). Found non-integer values in boundary "
                        f"{idx}: ({start}, {end})")

        else:
            raise ValueError(
                f"Boundaries must be either all timestamp strings or all "
                f"integer offsets (with optional '-INF' for first start). "
                f"Found invalid first value: {first_val}")

    def _validate_timestamp_boundaries(self) -> None:
        boundaries = self.split.timestamp_boundaries
        assert boundaries is not None
        self._validate_tuple_structure(boundaries)
        self._validate_boundary_values(boundaries)

        # Convert boundaries to comparable format based on type
        comparable_boundaries = [(pd.Timestamp(start), pd.Timestamp(end))
                                 for start, end in boundaries]
        self._validate_indices_or_boundaries(
            comparable_boundaries)  # type: ignore

    def _validate_offset_boundaries(self) -> None:
        boundaries = self.split.offset_boundaries
        assert boundaries is not None
        self._validate_tuple_structure(boundaries)
        self._validate_boundary_values(boundaries)

        self._validate_indices_or_boundaries(boundaries)  # type: ignore

    def _validate_fractions(self) -> None:
        r"""Validate the fractions list for splitting data.

        This method performs validation checks for fraction-based splits. It
        ensures:
            - The correct number of splits (between 2 and 3).
            - Each fraction is between 0 and 1.
            - Fractions sum to approximately 1.
            - Fractions are in ascending order.

        Raises:
            ValueError: If any validation check fails:
                - Less than 2 or more than 3 splits
                - Fractions outside [0,1] range
                - Fractions don't sum to ~1
            AssertionError: If fractions is None
        """
        assert self.split.fractions is not None
        data = self.split.fractions
        self._validate_num_splits(data)

        # Validate individual fractions
        for idx, frac in enumerate(data):
            if not 0 <= frac <= 1:
                stage = SPLIT_STAGES[idx]
                raise ValueError(
                    f"The fraction ({frac}) for the ({stage.value}) split "
                    f"must be between 0 and 1.")

        # Validate sum of fractions
        total = sum(data)
        if abs(total - 1.0) > 1e-6:  # Allow small floating point differences
            raise ValueError(f"Fractions must sum to 1.0, got {total}")

    def validate(self) -> None:
        # Validate timeframe parameters consistency
        self._validate_timeframe_params()

        # Make sure one and only one field is set
        self._validate_arg_exclusivity()

        # Validate the list argument that was specified
        if self.split.indices is not None:
            self._validate_indices()
        elif self.split.timestamp_boundaries is not None:
            self._validate_timestamp_boundaries()
        elif self.split.offset_boundaries is not None:
            self._validate_offset_boundaries()
        elif self.split.fractions is not None:
            self._validate_fractions()

    def validate_indices_against_num_timestamps(self,
                                                num_timestamps: int) -> None:
        r"""Validate that all indices are within bounds of available
        timestamps.

        Since indices must be strictly decreasing, we only need to check the
        smallest indices values (start and end of first split) to ensure all
        indices are within bounds.

        Args:
            num_timestamps (int): Number of unique timestamps available.

        Raises:
            ValueError: If any index would be out of bounds.
        """
        indices = self.split.indices
        assert indices is not None
        # Get first split's indices (train split)
        start_idx, end_idx = indices[0]

        # Skip validation for '-INF' start index
        if abs(int(end_idx)) > num_timestamps:
            raise ValueError(
                f"End index out of bounds for {num_timestamps} timestamps: "
                f"train split ({start_idx}, {end_idx})")

        if start_idx != '-INF':
            if abs(int(start_idx)) > num_timestamps:
                raise ValueError(
                    f"Start index out of bounds for {num_timestamps} "
                    f"timestamps: train split ({start_idx}, {end_idx})")

    def validate_num_splits_against_num_timestamps(self,
                                                   num_timestamps: int) -> None:
        """Validate that there are enough timestamps for the requested split.

        Args:
            num_timestamps (int): Number of unique timestamps available.

        Raises:
            ValueError: If there aren't enough timestamps for the requested
                split.
        """
        num_splits = self.split.num_splits
        if num_timestamps < num_splits:
            raise ValueError(f"Not enough timestamps ({num_timestamps}) for "
                             f"{num_splits} splits. Each split must contain at "
                             f"least one timestamp.")
