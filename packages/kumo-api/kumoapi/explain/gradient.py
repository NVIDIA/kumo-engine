# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
from dataclasses import field
from typing import Any, Dict, List, Tuple

import pandas as pd
from pydantic.dataclasses import dataclass


@dataclass
class TableGradientScore:
    name: str
    columns: Dict[str, float] = field(default_factory=dict)

    def __getitem__(self, column_name: str) -> float:
        return self.columns[column_name]

    def __setitem__(self, column_name: str, score: float):
        self.columns[column_name] = score

    @property
    def total_score(self) -> float:
        return sum(self.columns.values())


@dataclass
class GraphGradientScore:
    tables: Dict[str, TableGradientScore] = field(default_factory=dict)

    def __getitem__(self, table_name: str) -> TableGradientScore:
        if table_name not in self.tables:
            self.tables[table_name] = TableGradientScore(table_name)
        return self.tables[table_name]

    def print_summary(self, normalize: bool = False) -> None:
        r"""Prints a summary of all column scores across all tables.

        Args:
            normalize: If set to :obj:`True`, will normalize the scores.
        """
        from tabulate import tabulate

        total_score = sum(table.total_score for table in self.tables.values())

        for table_name, table in self.tables.items():
            table_score = table.total_score
            scores: List[Tuple[str, Any]] = sorted(
                table.columns.items(),
                key=lambda x: x[1],
                reverse=True,
            )

            if normalize:
                scores = [(col, score / total_score) for col, score in scores]
                table_score = table_score / total_score

            headers = ['Column', 'Score']
            scores = [(col, f'{score:.4f}') for col, score in scores]
            scores_repr = tabulate(scores, headers=headers, tablefmt='psql')
            print(f'{table_name}: {table_score:.4f}\n{scores_repr}')

    def normalize(self) -> 'GraphGradientScore':
        out = copy.deepcopy(self)

        total_score = 0.0
        for table in out.tables.values():
            for score in table.columns.values():
                total_score += score

        if total_score > 0:
            for table in out.tables.values():
                for col_name, score in table.columns.items():
                    table.columns[col_name] /= total_score

        return out

    def to_pandas(self) -> pd.DataFrame:
        r"""Converts table-column scores into a :class:`pandas.DataFrame`.

        Args:
            normalize: If set to :obj:`True`, will normalize the scores.
        """
        data: List[Tuple[str, str, float]] = []

        for table_name, table in self.tables.items():
            for col_name, score in table.columns.items():
                data.append((table_name, col_name, score))

        df = pd.DataFrame(data, columns=pd.Index(['table', 'column', 'score']))
        df = df.sort_values('score', ascending=False).reset_index(drop=True)

        return df
