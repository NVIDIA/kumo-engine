# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: MIT

import copy
from dataclasses import field
from typing import Dict, List, Optional, Tuple, Union

from pydantic.dataclasses import dataclass

from kumoapi.task import TaskType
from kumoapi.typing import Stype


@dataclass
class ColumnCohorts:
    # The semantic type of the column. Can be `None` for COUNT cohorts.
    stype: Optional[Stype]
    # The description for each cohort:
    cohorts: List[str]
    # Holds the relative population for each cohort:
    populations: List[float]
    # Holds the averaged prediction values for each cohort:
    predictions: Union[List[float], Dict[str, List[float]]]
    # Holds the averaged ground-truth values for each cohort:
    targets: Optional[Union[List[float], Dict[str, List[float]]]]
    # Holds the weighted standard deviation of predictions across cohorts:
    prediction_score: float
    # Holds the weighted standard deviation of targets across cohorts:
    target_score: Optional[float]
    # The table name of the column.
    table_name: str = '???'
    # The hop for which cohorts are computed.
    hop: int = -1
    # The column name.
    col_name: str = '???'

    @property
    def num_cohorts(self) -> int:
        return len(self.cohorts)


@dataclass
class ColumnAnalysisOutput:
    task_type: TaskType
    cohorts: List[ColumnCohorts] = field(default_factory=list)

    def __getitem__(self, key: Tuple[str, int, str]) -> ColumnCohorts:
        table_name, hop, col_name = key

        for cohort in self.cohorts:
            if (table_name == cohort.table_name and cohort.hop == hop
                    and cohort.col_name == col_name):
                return cohort

        raise KeyError(f"Could not find column analysis for "
                       f"{table_name}[hop={hop}].{col_name}")

    def __setitem__(
        self,
        key: Tuple[str, int, str],
        value: Optional[ColumnCohorts],
    ) -> None:
        table_name, hop, col_name = key

        if value is None:
            return

        value.table_name = table_name
        value.hop = hop
        value.col_name = col_name

        self.cohorts.append(value)

    def normalize(self) -> 'ColumnAnalysisOutput':
        out = copy.deepcopy(self)

        total_prediction_score = total_target_score = 0.0
        for col_cohort in out.cohorts:
            total_prediction_score += col_cohort.prediction_score
            if col_cohort.target_score is not None:
                total_target_score += col_cohort.target_score

        for col_cohort in out.cohorts:
            if total_prediction_score > 0:
                col_cohort.prediction_score /= total_prediction_score
            if col_cohort.target_score is not None and total_target_score > 0:
                col_cohort.target_score /= total_target_score

        return out
