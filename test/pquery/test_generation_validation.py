# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pandas as pd
import pytest
from kumo_core.training.validation import (
    GenerationValidationError,
    validate_anchor_time,
    validate_target_classes,
    validate_train_offsets,
)
from kumoapi.typing import ProblemType

from kumo.pquery.pquery_validators.pre_pred_table_validator import (
    PrePredTableValidator,
)
from kumo.pquery.pquery_validators.pre_train_table_validator import (
    PreTrainTableValidator,
)
from kumo.pquery.pquery_validators.train_table_validator import (
    TrainTableValidator,
)


@pytest.mark.parametrize('start,end', [(10, 60), (60, 60), (60, 10)])
def test_hosted_offsets_use_the_shared_error(start, end):
    start_offset = pd.DateOffset(days=start)
    end_offset = pd.DateOffset(days=end)
    shared = validate_train_offsets(start_offset, end_offset)
    response = PreTrainTableValidator.validate_train_offsets(
        None, start_offset, end_offset)
    assert response.errors == ([] if shared is None else [shared])


@pytest.mark.parametrize('year', [2000, 2024, 2030])
def test_hosted_anchors_use_the_shared_error(year):
    anchor, minimum = pd.Timestamp(f'{year}-01-01'), pd.Timestamp('2024-01-01')
    shared = validate_anchor_time(anchor, minimum, 'orders')
    response = PrePredTableValidator().validate_anchor_time(
        anchor, minimum, 'orders')
    assert response.errors == ([] if shared is None else [shared])


@pytest.mark.parametrize('classes',
                         [[], [False], [True], ['only'], [False, True]])
def test_hosted_label_cardinality_uses_the_shared_error(classes):
    validator = object.__new__(TrainTableValidator)
    validator.query = SimpleNamespace(problem_type=ProblemType.CLASSIFY)
    stats = SimpleNamespace(
        num_targets=max(1, len(classes)), target_distribution={
            'percentage': dict.fromkeys(classes, 1.0),
        })
    expected = validate_target_classes(stats.num_targets, classes,
                                       classify=True)
    assert validator.validate_class_number(stats).errors == expected


def test_hosted_generation_retains_structured_quality_error():
    from kumo.pquery.train_table_generator import (
        _invoke_diskgraph_train_table_seam,
    )
    from kumo.validation.train_table_validation import (
        TrainingTableGenerationError,
    )

    error = validate_target_classes(1, [False], classify=True)[0]

    def generate():
        raise GenerationValidationError(error)

    with pytest.raises(TrainingTableGenerationError) as caught:
        _invoke_diskgraph_train_table_seam(generate, entity_table='users',
                                           timestamp_count=1)
    assert caught.value.as_error() == error
