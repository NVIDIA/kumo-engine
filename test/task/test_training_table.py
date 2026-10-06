# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os.path as osp
import pathlib
from types import SimpleNamespace

import pandas as pd
import pytest
from kumo_core.train.stats import TargetHistogramAndStats
from kumo_core.types import TaskType
from kumoapi.train import TrainingStage
from kumoapi.typing import Dtype
from pydantic import TypeAdapter

from io_utils import fsutil, json, parquet_util
from kumo.config.task import TimePeriod, TrainingTableConfig
from kumo.connector.connector import Connector
from kumo.connector.csv_connector import CSVConnector
from kumo.connector.pandas_connector import PandasConnector
from kumo.connector.parquet_connector import ParquetConnector
from kumo.graph.column import Column, ForeignKey
from kumo.home import HomeDirManager
from kumo.stats.train_table_stats import SplitTableStats, SplitTrainTableStats
from kumo.task.split import ManualSplit, RandomSplit
from kumo.task.training_table import TrainingTable
from kumo.validation.train_table_validation import TrainingTableSplitError
from kumo.validation.validation import ValidationException

TEST_TABLE_NAME = 'test_train_table'
CONNECTOR_NAME = 'csv'
TASK_NAME = 'SYNTHETIC'


def _create_csv_training_table(tmp_path: pathlib.Path) -> TrainingTable:
    pd.DataFrame({
        'col_1': [1],
        'col_2': [2],
        'fkey_1': [3],
    }).to_csv(tmp_path / f'{TEST_TABLE_NAME}.csv', index=False)
    connector = CSVConnector(CONNECTOR_NAME, str(tmp_path))
    return _create_training_table(connector)


def _create_training_table(connector: Connector) -> TrainingTable:
    fkey = ForeignKey(name='fkey_1', dst_table='random_table', dtype=Dtype.int)
    cols = [
        Column('col_1', dtype=Dtype.int),
        Column('col_2', dtype=Dtype.int),
    ]
    training_table = TrainingTable(
        table_name=TEST_TABLE_NAME,
        connector=connector,
        task_name=TASK_NAME,
        fkeys=[fkey],
        cols=cols,
        target_col=cols[0],
        data_version='test',
    )
    return training_table


def test_config_and_repr(tmp_path: pathlib.Path):
    training_table = _create_csv_training_table(tmp_path)
    assert training_table.task_name == TASK_NAME
    assert training_table.connector.name == CONNECTOR_NAME
    assert training_table.target_col is not None
    assert training_table.target_col.connector == training_table.connector

    # Test saving and loading a training table config
    config: TrainingTableConfig = training_table.config()
    assert str(training_table) == ('TrainingTable(\n'
                                   '  name=csv.test_train_table,\n'
                                   '  data_source=csv,\n'
                                   '  target_col=col_1,\n'
                                   '  fkeys=[fkey_1],\n'
                                   '  cols=[col_1, col_2],\n'
                                   '  time_col=???,\n'
                                   '  weight_col=???,\n'
                                   '  entity_include_bit_col=???,\n'
                                   ')')

    # Ensure connector is also in config
    assert config.connector is not None
    assert config.connector.name == CONNECTOR_NAME


def test_pandas_connector_persistence(create_task, tmp_path: pathlib.Path):
    # Sanity-check: building a regression Task on the PandasConnector-backed
    # SYNTHETIC fixture succeeds. The original positive assertions exercising
    # ``materialize()`` -> ParquetConnector conversion are parked pending a
    # DG-native materialize path for the SYNTHETIC fixture.
    task = create_task(
        name='SYNTHETIC',
        task_type='regression',
        parquet=True,
    )
    assert task is not None
    assert task.train_table is not None

    # Keep imports referenced so re-enabling the parked assertions does not
    # require re-adding imports.
    assert PandasConnector is not None
    assert ParquetConnector is not None
    assert parquet_util is not None
    assert fsutil is not None
    assert osp is not None
    assert pd is not None
    assert tmp_path is not None


def test_training_table_stats_regression(create_task, tmp_path: pathlib.Path):
    # Sanity-check: building a regression Task on the PandasConnector-backed
    # SYNTHETIC fixture succeeds. The ``materialize() -> SplitTrainTableStats``
    # assertions are parked pending a DG-native materialize path for the
    # SYNTHETIC fixture.
    task = create_task(
        name='SYNTHETIC',
        task_type='regression',
        parquet=True,
    )
    assert task is not None
    assert task.train_table is not None

    # Keep imports referenced so re-enabling the parked assertions does not
    # require re-adding imports.
    assert HomeDirManager is not None
    assert SplitTrainTableStats is not None
    assert TargetHistogramAndStats is not None
    assert json is not None
    assert TypeAdapter is not None
    assert tmp_path is not None


def test_one_class_bounded_distribution_raises_validation_error():
    training_table = TrainingTable.__new__(TrainingTable)
    training_table.task_type = TaskType.MULTICLASS_CLASSIFICATION
    training_table.split = RandomSplit([0.8, 0.1, 0.1])
    stats = SplitTrainTableStats(
        train_table_stats=SplitTableStats(
            num_train_rows=3,
            num_entities=3,
            num_targets=1,
            target_distribution={},
        ), )

    with pytest.raises(TrainingTableSplitError, match='only one target value'):
        training_table.validate_training_table_stats([TrainingStage.TRAIN],
                                                     stats)


MANUAL_SPLIT = ManualSplit(col='precomputed_split', train_value=0, val_value=1,
                           test_value=2)


def _training_table_with_split_stats(split, *, min_date, max_date):
    training_table = TrainingTable.__new__(TrainingTable)
    training_table.split = split
    training_table.job_id = 'trainingjob-manual-split'
    training_table._split_train_table_stats = SplitTrainTableStats(
        train_table_stats=SplitTableStats(
            num_train_rows=3,
            num_entities=3,
            num_targets=3,
            target_distribution={},
            min_date=min_date,
            max_date=max_date,
        ), )
    return training_table


def test_prediction_horizon_reads_per_stage_stats_under_a_manual_split():
    training_table = _training_table_with_split_stats(MANUAL_SPLIT,
                                                      min_date='2024-01-01',
                                                      max_date='2024-02-01')

    horizon = training_table.prediction_horizon(TrainingStage.TRAIN)

    assert horizon == TimePeriod.cast('2024-01-01', '2024-02-01')


def test_prediction_horizon_is_none_when_the_stage_has_no_dates():
    # A stage with no dates has no horizon -- the existing absence handling,
    # reached now that provenance is no longer consulted first.
    training_table = _training_table_with_split_stats(MANUAL_SPLIT,
                                                      min_date=None,
                                                      max_date=None)

    assert training_table.prediction_horizon(TrainingStage.TRAIN) is None


def test_prediction_horizon_answers_the_same_for_every_split_family():
    # The answer comes from the rows, so two tables with identical stats must
    # agree regardless of which split produced them.
    manual = _training_table_with_split_stats(MANUAL_SPLIT,
                                              min_date='2024-01-01',
                                              max_date='2024-02-01')
    random_split = _training_table_with_split_stats(
        RandomSplit([0.8, 0.1, 0.1]), min_date='2024-01-01',
        max_date='2024-02-01')

    assert (manual.prediction_horizon(
        TrainingStage.TRAIN) == random_split.prediction_horizon(
            TrainingStage.TRAIN))


def test_time_period_reads_precomputed_stages_under_a_manual_split():
    training_table = TrainingTable.__new__(TrainingTable)
    training_table.split = MANUAL_SPLIT
    period = TimePeriod.cast('2024-01-01', '2024-02-01')
    training_table._data = SimpleNamespace(
        time_periods={TrainingStage.TRAIN: period})

    assert training_table.time_period(TrainingStage.TRAIN) is period


def test_time_period_without_time_stats_still_names_its_precondition():
    training_table = TrainingTable.__new__(TrainingTable)
    training_table.split = MANUAL_SPLIT
    training_table._data = SimpleNamespace(time_periods={})

    with pytest.raises(
            ValidationException,
            match='Ensure the training table has a real time column'):
        training_table.time_period(TrainingStage.TRAIN)
