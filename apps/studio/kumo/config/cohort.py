# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import field
from typing import Any, Dict, List, Optional

from kumo_core.types import TargetType, TaskType

from io_utils import kumo_dataclass

COHORT_DATAFRAME = Dict[str, Any]


@kumo_dataclass
class DimensionSpec:  # Descriptor for a dimension
    name: str
    # List of all possible discrete values.
    all_values: List[str]

    # TODO: make this an enum of INT,FLOAT,STRING,DATETIME_INTERVAL, etc
    value_type: str = 'str'

    # TODO: should we tell if this dimension is categorical vs quantitative (
    # consecutive intervals)


@kumo_dataclass
class DataCube:
    # Nested json dictionary to represent N-dimensional data cube so that
    # json_data[dim1_val][dim2_val]...[dimN_val] = 1d-array
    json_data: Dict[str, Any] = field(default_factory=dict)

    # N (>=1) dimensions, in the same order as they are nested in json_data.
    # DataCube has N+1 dimension, with an extra, implicit "index" dimension in
    # the 1D array.
    dimensions: List[DimensionSpec] = field(default_factory=list)


def _data_frame_to_cube(col_stats: COHORT_DATAFRAME,
                        is_classification: bool) -> DataCube:
    dimensions = []
    if is_classification:
        dimensions.append(
            DimensionSpec(name='Class Label', all_values=list(col_stats.keys()),
                          value_type='str'))
        # Only support binary classification now
        all_values = list(next(iter(col_stats.values())).keys())
        dimensions.append(
            DimensionSpec(name='Data Field', all_values=all_values,
                          value_type='str'))
    else:
        dimensions.append(
            DimensionSpec(name='Data Field', all_values=list(col_stats.keys()),
                          value_type='str'))
    return DataCube(json_data=col_stats, dimensions=dimensions)


@kumo_dataclass
class ColumnCohort:
    r"""Cohort analysis results for a column."""
    task_type: TaskType = TaskType.BINARY_CLASSIFICATION
    col_stats: COHORT_DATAFRAME = field(default_factory=dict)
    col_value_stats: Dict[str, COHORT_DATAFRAME] = field(default_factory=dict)

    def to_data_cube(self) -> DataCube:
        """
        Helper to migrate ColumnCohort to DataCube schema in /cohort_stats API.
        """
        if self.col_stats:
            return _data_frame_to_cube(
                self.col_stats,
                is_classification=self.task_type.is_classification)

        assert self.col_value_stats
        dimensions = [
            DimensionSpec(name='Categorical Column Value',
                          all_values=list(self.col_value_stats.keys()),
                          value_type='str')
        ]
        json_data = {}
        for col_val, col_stats in self.col_value_stats.items():
            cube = _data_frame_to_cube(
                col_stats, is_classification=self.task_type.is_classification)
            json_data[col_val] = cube.json_data
            if len(dimensions) == 1:
                dimensions.extend(cube.dimensions)
        return DataCube(json_data, dimensions)


@kumo_dataclass
class TableCohort:
    r"""Cohort analysis results for a table. It contains `cols`
    where the key is column name and value is dataclass
    `ColumnCohort`, which holds cohort analysis results for each
    column.
    """
    cols: Dict[str, ColumnCohort] = field(default_factory=dict)


@kumo_dataclass
class TaskCohort:
    r"""Cohort analysis results for a task with a specific task type.
    Current task type only supports `truth` but may add another one
    in the near future. This class also contains `tables` where the
    key is the table name and value is dataclass `TableCohort`, which
    holds cohort analysis results for each column.
    """
    target_type: TargetType = TargetType.TRUTH
    tables: Dict[str, TableCohort] = field(default_factory=dict)
    task_type: TaskType = TaskType.BINARY_CLASSIFICATION

    # Globally aggregated stats (e.g., population average per class label)
    global_stats: Optional[DataCube] = None


@kumo_dataclass(repr=False)
class QuantileStats:
    r"""Quantile statistics for cohort analysis.
    """
    stats: List[float] = field(default_factory=list)

    def to_unique_list(self) -> List[float]:
        r"""Covert quantile stats to list without duplicate values."""
        return sorted(list(set(self.stats)))
