# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
from kumoapi.task import TaskType

from .benchmarking_dataset import BenchmarkingDataset, SupervisedTables


class YelpChiBenchmarkingDataset(BenchmarkingDataset):
    URL = 's3://kumo-public-datasets/yelp_chi'

    def __init__(self, **kwargs) -> None:
        pass

    def get_entity_tables(self) -> SupervisedTables:
        df = pd.read_parquet(f'{self.URL}/review.parquet')
        train_df = pd.read_parquet(f'{self.URL}/train.parquet')
        train_mask = train_df['TRAIN_MASK']
        test_mask = train_df['TEST_MASK']

        df = df.drop('REVIEW_ID', axis=1)
        df['TARGET'] = train_df['TARGET']

        return SupervisedTables(
            train=df[train_mask].reset_index(drop=True),
            test=df[test_mask].reset_index(drop=True),
            target_col='TARGET',
            task_type=TaskType.BINARY_CLASSIFICATION,
        )

    def get_joined_tables(self) -> SupervisedTables:
        return self.get_entity_tables()

    def get_pquery(self) -> str:
        raise NotImplementedError

    def get_enterprise_graph(self):
        raise NotImplementedError
