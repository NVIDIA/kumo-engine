# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import TypeAdapter, ValidationError

from rest.config.training_job_v2 import TrainingJobRequest


def test_training_request_rejects_removed_extra_fields():
    removed_name = 'data_backend_' + 'session_id'
    with pytest.raises(ValidationError) as caught:
        TypeAdapter(TrainingJobRequest).validate_python({removed_name: 'old'})
    assert any(error['loc'] == (removed_name, )
               and error['type'] == 'unexpected_keyword_argument'
               for error in caught.value.errors())
