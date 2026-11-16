# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest
from kumo_core.train.formatting import prediction_output_columns
from kumo_core.types import TaskType


@pytest.mark.parametrize('task_type, columns', [
    (TaskType.BINARY_CLASSIFICATION,
     ['TARGET_PRED', 'False_PROB', 'True_PROB']),
    (TaskType.REGRESSION, ['TARGET_PRED']),
    (TaskType.FORECASTING, ['TIMESTAMP', 'TARGET_PRED']),
    (TaskType.MULTICLASS_CLASSIFICATION, ['PREDICTED', 'CLASS', 'SCORE']),
    (TaskType.MULTILABEL_CLASSIFICATION, ['PREDICTED', 'CLASS', 'SCORE']),
    (TaskType.MULTILABEL_RANKING, ['CLASS', 'SCORE']),
    (TaskType.TEMPORAL_LINK_PREDICTION, ['CLASS', 'SCORE']),
    (TaskType.STATIC_LINK_PREDICTION, ['CLASS', 'SCORE']),
])
def test_prediction_output_columns(task_type, columns) -> None:
    assert prediction_output_columns(task_type) == columns
