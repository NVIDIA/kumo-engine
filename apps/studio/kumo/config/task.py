# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from dataclasses import dataclass as std_dataclass
from dataclasses import field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from kumo_core.types import TaskType
from kumo_core.types.schema import (
    BaseColumnConfig,
    TableFingerprint,
    TrainingTableFingerprint,
)
from pydantic import field_validator
from torch import Tensor

from io_utils import kumo_dataclass
from kumo.config import ConnectorConfig, GraphConfig, TableConfig
from kumo.config.graph import BoolColumnConfig, ColumnConfig


@kumo_dataclass
class SplitConfig:
    pass


SplitConfigSerializable = Any


@kumo_dataclass
class TrainingTableConfig(TableConfig):
    # Any: Union[ConnectorConfig, dict]
    connector: Optional[Any] = None
    task_name: Optional[str] = None
    task_type: Optional[TaskType] = None
    target_col: Optional[ColumnConfig] = None
    weight_col: Optional[BaseColumnConfig] = None
    entity_include_bit_col: Optional[BoolColumnConfig] = None
    split: Optional[SplitConfigSerializable] = None
    is_pred: bool = False
    job_id: Optional[str] = None
    job_dir: Optional[str] = None

    def as_table_config(self) -> TableConfig:
        return TableConfig(
            table_name=self.table_name,
            task_name=self.task_name,
            task_type=self.task_type,
            source_name=self.source_name,
            data_source=self.data_source,
            data_version=self.data_version,
            pkey=self.pkey,
            fkeys=self.fkeys,
            cols=self.cols + [self.target_col] if self.target_col else [],
            time_col=self.time_col,
        )

    @property
    def all_cols(self) -> List[BaseColumnConfig]:
        all_cols = self.cols.copy()
        if self.target_col:
            all_cols.append(self.target_col)
        if self.weight_col:
            all_cols.append(self.weight_col)
        if self.time_col:
            all_cols.append(self.time_col)
        if self.entity_include_bit_col:
            all_cols.append(self.entity_include_bit_col)
        for fkey in self.fkeys:
            all_cols.append(fkey)
        if self.pkey:
            all_cols.append(self.pkey)
        return all_cols

    def fingerprint(self) -> TableFingerprint:
        if isinstance(self.connector, ConnectorConfig):
            connector_name: str = self.connector.name
        elif isinstance(self.connector, dict):
            # part of a TaskConfig loaded from json
            connector_name: str = self.connector['name']
        else:
            raise ValueError(f"TrainingTableConfig.fingerprint()"
                             f"requires a connector, got {self.connector}")

        return TrainingTableFingerprint(
            connector_name=connector_name,
            source_name=self.table_name,
        )


@kumo_dataclass
class TaskConfig:
    graph: GraphConfig
    task_type: TaskType
    train_table: TrainingTableConfig
    metrics: List[str]
    target_to_entity_paths: List[List[Tuple[str, str,
                                            str]]] = field(default_factory=list)
    target_table_name: Optional[str] = None
    name: Optional[str] = None
    exclude_cols: Dict[str, List[str]] = field(default_factory=dict)
    is_sliding_window: bool = False
    lag_timesteps: Optional[int] = None
    num_targets: Optional[int] = None
    prediction_attrs: Optional[Any] = None
    forecast_length: Optional[int] = None
    unix_start_offset: Optional[int] = None
    unix_end_offset: Optional[int] = None
    unix_step_size: Optional[int] = None
    list_distinct_target_is_array: Optional[bool] = False
    lp_bp_layout: Optional[str] = None
    kumo_version: Optional[str] = None

    use_lp_shallow_rhs: bool = False  # Deprecated.
    use_lp_id_awareness: bool = False  # Deprecated.
    use_lp_hybrid: bool = False  # Deprecated.


def _parse_datetime_str(value: str) -> datetime:
    """Parse a serialized timestamp string into a Python ``datetime``.

    Serialized horizon bounds can land outside pandas' nanosecond
    ``Timestamp`` range (≈ 1677-2262), which is *narrower* than Python's
    ``datetime`` range (year 1-9999), so a ``pd.to_datetime`` fallback is the
    wrong safety net. Two degenerate-but-real sentinels appear in practice:
    a far-future bound (e.g. year 9918) and a near-zero bound (e.g. year 0).
    The producer never re-validated these, so this parser must always
    round-trip ANY serialized string to a valid Python ``datetime`` instead of
    raising.

    Strategy: prefer ``datetime.fromisoformat`` (covers year 1-9999, incl. the
    9918 sentinel and the ISO ``'YYYY-MM-DDTHH:MM:SS'`` serialization form). If
    the year is below ``datetime.min.year`` (e.g. year 0) clamp to
    ``datetime.min``; if above ``datetime.max.year`` clamp to ``datetime.max``.
    Year-0 was already a degenerate value, so clamping preserves round-trip
    without inventing meaningful data. Fall back to ``pd.to_datetime`` only for
    genuinely non-ISO strings, wrapped so an out-of-range bound clamps rather
    than propagating.
    """
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        pass

    # ISO-ish but out of Python's year range (e.g. '0000-...' or '>9999-...').
    # ``fromisoformat`` rejects these with ValueError; clamp by leading year.
    year_match = re.match(r'\s*(-?\d+)-', value)
    if year_match is not None:
        year = int(year_match.group(1))
        if year < datetime.min.year:
            return datetime.min
        if year > datetime.max.year:
            return datetime.max

    # Genuinely non-ISO string: let pandas parse, but never let an
    # out-of-range bound raise -- clamp to ``datetime.min`` (a safe degenerate
    # default) instead. pandas' representable range is a strict subset of the
    # Python ``datetime`` range, so anything pandas can parse is in range here.
    try:
        return pd.to_datetime(value).to_pydatetime()
    except (pd.errors.OutOfBoundsDatetime, OverflowError, ValueError):
        return datetime.min


@kumo_dataclass
class TimePeriod:
    """Represents a range of time for a split of the training table. The
    timestamps should refer to the time_col of the original source table.
    It can be either pseudo time column or standard time column.

    In case of pseudo time column, the start and end are always integers and
    they are inclusive on both ends.

    In case of standard time column, it is undefined whether the start and end
    are inclusive or exclusive. However, in default split algorithm
    `DateOffsetSplit`, the end is exclusive. Start is guaranteed to be equal or
    earlier in time compared to end.
    """
    start: Union[int, datetime]
    end: Union[int, datetime]
    is_pseudo: bool = False  # indicate whether time is pseudo time or not.

    @field_validator('start', 'end', mode='before')
    @classmethod
    def _coerce_serialized_datetime(cls, value: Any) -> Any:
        # Serialization turns a saved datetime into an ISO string. Under
        # pydantic v2 the ``Union[int, datetime]`` is validated before
        # ``__post_init__`` runs, and v2's strict datetime parser handles a
        # different range of dates than ``pd.to_datetime`` does (and rejects
        # year 0). Re-cast string inputs to a Python ``datetime`` here, before
        # the union resolves, so the value matches the historical
        # ``__post_init__`` coercion. Pseudo (integer) times round-trip as
        # ints, so the ``str`` guard leaves them untouched.
        if isinstance(value, str):
            return _parse_datetime_str(value)
        return value

    def __post_init__(self):
        # verify the correctness of value types depending on is_pseudo
        if self.is_pseudo:
            if not (isinstance(self.start, int) and isinstance(self.end, int)):
                raise ValueError(
                    'TimePeriod (is_pseudo=True) requires integer type for '
                    f'both start and end but {type(self.start)} and '
                    f'{type(self.end)} types given')
        else:
            # Serialization would change saved datetime into str.
            # So we re-cast it back to datetime here.
            if isinstance(self.start, str):
                self.start = _parse_datetime_str(self.start)
            if isinstance(self.end, str):
                self.end = _parse_datetime_str(self.end)

            if not (isinstance(self.start, datetime)
                    and isinstance(self.end, datetime)):
                raise ValueError(
                    'TimePeriod (is_pseudo=False) requires datetime type for '
                    f'both start and end but {type(self.start)} and '
                    f'{type(self.end)} types given')

    @classmethod
    def cast(cls, start: Any, end: Any,
             is_pseudo: bool = False) -> 'TimePeriod':
        if is_pseudo:
            start = int(start)
            end = int(end)
        else:
            # Route string inputs through the full-range parser so far-future
            # horizon sentinels (e.g. year 9918) -- valid Python ``datetime``
            # but out of pandas' nanosecond ``Timestamp`` range -- don't
            # overflow as bare ``pd.to_datetime`` would. Non-str inputs (e.g.
            # int epoch values from split callers, or already-``datetime``
            # values) keep the original ``pd.to_datetime`` coercion so a
            # ``datetime``-typed value reaches ``__post_init__``.
            start = (_parse_datetime_str(start)
                     if isinstance(start, str) else pd.to_datetime(start))
            end = (_parse_datetime_str(end)
                   if isinstance(end, str) else pd.to_datetime(end))
        return TimePeriod(start=start, end=end, is_pseudo=is_pseudo)


# Use standard library dataclass since Tensor is not json-serializable
@std_dataclass
class Mask:
    """A bitmap that can be used to mask the training table, in order to get
    the rows that should be included in a particular split of the data.

    index_tensor contains the NODE_IDX_COL_NAME values corresponding to the
    masks's entries, for use filtering the train table when indexing is not
    contiguous

    Additional metadata about the mask (such as time period) is also provided.
    """
    tensor: Tensor
    index_tensor: Tensor

    # Only present for temporal spliting algorithms, (where
    # Split.is_temporal_split is True).  All examples in the tensor are
    # guaranteed to fall within this range of time.
    time_period: Optional[TimePeriod] = None
