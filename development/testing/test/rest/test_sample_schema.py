# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import datetime

from kumo_core.types import TaskType

import rest.config.batch_prediction_job_v2 as bps

sample_data = {
    'ENTITY': [
        12356.0, 12371.0, 12423.0, 12437.0, 12437.0, 12437.0, 12444.0, 12444.0,
        12451.0, 12454.0
    ],
    'TARGET': [120, 36, 24, 120, 48, 36, 12, 24, 120, 288],
    'TIMESTAMP': [
        datetime.datetime(2011, 1, 13, 12, 31),
        datetime.datetime(2011, 10, 10, 12, 31),
        datetime.datetime(2011, 3, 14, 12, 31),
        datetime.datetime(2011, 6, 12, 12, 31),
        datetime.datetime(2011, 5, 13, 12, 31),
        datetime.datetime(2011, 3, 14, 12, 31),
        datetime.datetime(2011, 10, 10, 12, 31),
        datetime.datetime(2011, 9, 10, 12, 31),
        datetime.datetime(2011, 1, 13, 12, 31),
        datetime.datetime(2011, 10, 10, 12, 31)
    ],
    'arange': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
}


def test_suggest_prediction_samples_invalid_data():
    """
    Test by passing invalid data.
    """
    invalid_sample_data = {}
    instance = bps.SuggestPredictionSamples()
    assert instance.verify_input_data(sample_data) is True
    assert instance.verify_input_data(invalid_sample_data) is False


def test_suggest_prediction_samples_regression():
    """
    Test for regression and forecastion tasks.
    """
    instance = bps.SuggestPredictionSamples()
    regression_sample = instance.suggest_col_samples_by_task_type(
        task_type=TaskType.REGRESSION, samples_rows_dict=sample_data)

    assert regression_sample[1].name == "TARGET_PRED"
    assert len(regression_sample[1].sample_rows) == 10


def test_suggest_prediction_samples_bc():
    """
    Test for binary classification tasks.
    """
    instance = bps.SuggestPredictionSamples()
    sample = instance.suggest_col_samples_by_task_type(
        task_type=TaskType.BINARY_CLASSIFICATION, samples_rows_dict=sample_data)

    assert sample[2].name == "False_PROB"
    assert len(sample[2].sample_rows) == 10


def test_suggest_prediction_samples_mc():
    """
    Test for multiclass/multilabel classification tasks.
    """
    instance = bps.SuggestPredictionSamples()
    sample = instance.suggest_col_samples_by_task_type(
        task_type=TaskType.MULTICLASS_CLASSIFICATION,
        samples_rows_dict=sample_data)

    assert sample[1].name == "PREDICTED"
    assert len(sample[2].sample_rows) == 10
    assert sample[2].name == "CLASS"


def test_suggest_prediction_samples_ranking():
    """
    Test for ranking tasks.
    """
    instance = bps.SuggestPredictionSamples()
    sample = instance.suggest_col_samples_by_task_type(
        task_type=TaskType.MULTILABEL_RANKING, samples_rows_dict=sample_data)

    assert sample[1].name == "CLASS"
    assert len(sample[2].sample_rows) == 10
    assert sample[2].name == "SCORE"
