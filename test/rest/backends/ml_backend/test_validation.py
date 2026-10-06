# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os.path as osp
import pathlib
from typing import Literal

import pandas as pd
import pytest
from kumo_core.types import TaskType
from kumoapi.pquery import QueryType

from rest.backends.ml_backend.validation import (
    GeneratePredictionTableValidator,
    GenerateTrainingTableValidator,
    TableValidator,
)


@pytest.mark.parametrize('validator,stage', [
    (GenerateTrainingTableValidator(task_type=TaskType.BINARY_CLASSIFICATION,
                                    query_type=QueryType.STATIC), 'training'),
    (GeneratePredictionTableValidator(task_type=TaskType.BINARY_CLASSIFICATION,
                                      query_type=QueryType.STATIC), 'pred'),
])
def test_non_empty_training_table(
    tmp_path: pathlib.Path,
    validator: TableValidator,
    stage: Literal['pred'] | Literal['training'],
):
    temp_path = tmp_path.resolve().as_posix()
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [True, False, True]})
    path = osp.join(temp_path, 'test.parquet')
    df.to_parquet(path)

    validation_response = validator.post_generation(stage, path)

    assert len(validation_response.errors) == 0


@pytest.mark.parametrize('validator,stage', [
    (GenerateTrainingTableValidator(task_type=TaskType.BINARY_CLASSIFICATION,
                                    query_type=QueryType.STATIC), 'training'),
    (GeneratePredictionTableValidator(task_type=TaskType.BINARY_CLASSIFICATION,
                                      query_type=QueryType.STATIC), 'pred'),
])
def test_empty_training_table(
    tmp_path: pathlib.Path,
    validator: TableValidator,
    stage: Literal['pred'] | Literal['training'],
):
    temp_path = tmp_path.resolve().as_posix()
    df = pd.DataFrame()
    path = osp.join(temp_path, 'test.parquet')
    df.to_parquet(path)

    validation_response = validator.post_generation(stage, path)

    assert len(validation_response.errors) == 1
    if (stage == 'pred' and validator.query_type == QueryType.STATIC
            and validator.task_type.is_node_pred):
        expected_message = 'Your prediction column is completely filled.'
    else:
        expected_message = (
            f'Generated {stage} table is empty. Please modify your query to '
            'make filters less restrictive or add more data that will fit '
            'your desired filters.')
    assert validation_response.errors[0].message.startswith(expected_message)
