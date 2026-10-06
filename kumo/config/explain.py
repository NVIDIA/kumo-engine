# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os.path as osp
from dataclasses import field
from enum import Enum
from typing import Any, Dict, List, NewType, Optional, Union

import pandas as pd
from kumoapi.explain import EntityMappings as StrategyMappings  # noqa: F401
from kumoapi.typing import Stype

from io_utils import fsutil, kumo_dataclass, logging

logger = logging.getLogger(__name__)

ALL_CLASSES_KEY: str = '__ALL_CLASSES__'


@kumo_dataclass
class ColumnContribution:
    col_name: str
    table_name: str
    contribution: float


@kumo_dataclass
class ValueContributionSummary:
    min: float
    lower_quartile: float
    median: float
    upper_quartile: float
    max: float


ContributionClass = NewType('ContributionClass', str)
ContributionEntityName = NewType('ContributionEntityName', str)
ContributionValueName = NewType('ContributionValueName', str)


@kumo_dataclass
class ClassColumnAndValueContributions:
    column_contributions: List[ColumnContribution]
    # Maps column name to value name to summary of contributions
    categorical_value_contributions: Optional[Dict[ContributionEntityName, Dict[
        ContributionValueName, ValueContributionSummary]]] = None


class ContributionKind(Enum):
    LINKED_FACT_TABLE = 'LINKED_FACT_TABLE'
    LINKED_DIM_TABLE = 'LINKED_DIM_TABLE'
    COLUMN = 'COLUMN'


@kumo_dataclass
class Contribution:
    kind: ContributionKind

    # Value of the contribution
    magnitude: Union[str, float]

    # If kind == Column
    # Maps a value name to a struct containing contribution summary stats for
    # each value in the column
    value_magnitudes: Optional[Dict[ContributionValueName,
                                    ValueContributionSummary]] = None

    # If kind == Column
    column_stype: Optional[Stype] = None


@kumo_dataclass
class ColumnContributionSummary:
    """Stores the column contributions computed during
    `trainer.test` and saved in `GradientExplainer.export`
    as a bunch of dictionaries

    Args:
        global_column_contributions(List[ColumnContribution]): The contribution
            of all columns across the whole test set.
        classes (Dict[str, ClassColumnAndValueContributions]): The contribution
            of columns per class in the test set.
            :obj:`None` for non-classification tasks.
    """
    # All tasks
    global_column_contributions: List[ColumnContribution]

    # Classification
    # Maps class name to contributions
    classes: Optional[Dict[ContributionClass,
                           ClassColumnAndValueContributions]] = None

    @staticmethod
    def _column_contribution_from_df(
            df: pd.DataFrame) -> List[ColumnContribution]:
        """Given  a dataframe `df` with columns `table`, `column` and `score`
        return a list of `ColumnContribution`"""
        cols = []
        for row in df.itertuples():
            cols.append(
                ColumnContribution(col_name=str(row.column),
                                   table_name=str(row.table),
                                   contribution=float(row.score)))
        return cols

    @classmethod
    def empty(cls):
        return cls([], {})

    @property
    def is_empty(self) -> bool:
        return len(self.global_column_contributions) == 0 and len(
            self.classes) == 0

    @classmethod
    def from_trainer(
        cls,
        explain_dir: str,
    ) -> 'ColumnContributionSummary':
        """Loads the contribution csvs saved
        in `explain_dir` during `trainer.test` into
        `ColumnContributionSummary`"""
        if not fsutil.exists(explain_dir) or len(
                fsutil.listdir(explain_dir)) == 0:
            logger.error(f"Error when loading explainer score, that "
                         f"the explain directory {explain_dir} does not "
                         f"exist or is empty.")
            return cls.empty()
        try:
            global_column_contributions_df = pd.read_csv(
                osp.join(explain_dir, 'scores.csv'))
            global_column_contributions = cls._column_contribution_from_df(
                global_column_contributions_df)
            classes = {}
            assert global_column_contributions is not None
            return ColumnContributionSummary(global_column_contributions,
                                             classes)
        except Exception as e:
            logger.exception(e)
            return cls.empty()


@kumo_dataclass
class EntityOutput:
    prediction: float | Dict[str, float] | None = None
    target: float | Dict[str, float] | None = None
    # table name -> column name -> rows of features and importance scores.
    feats: Dict[str, Dict[str, List[Any]]] = field(default_factory=dict)
    scores: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)
