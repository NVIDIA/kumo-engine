# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: MIT

import pytest
from pydantic import TypeAdapter, ValidationError

from kumoapi.train import TrainingTableSpec


@pytest.mark.parametrize(
    'unit', ['seconds', 'milliseconds', 'microseconds', 'nanoseconds'])
def test_timestamp_unit_round_trip(unit):
    adapter = TypeAdapter(TrainingTableSpec)
    spec = TrainingTableSpec(timestamp_unit=unit)
    assert adapter.validate_json(
        adapter.dump_json(spec)).timestamp_unit == unit


def test_timestamp_unit_default_and_positional_compatibility():
    assert TrainingTableSpec('weight', 'split').timestamp_unit == 'nanoseconds'
    assert TypeAdapter(TrainingTableSpec).validate_python(
        {}).timestamp_unit == 'nanoseconds'


@pytest.mark.parametrize('unit', ['ns', 'days', '', None])
def test_invalid_timestamp_unit(unit):
    with pytest.raises(ValidationError):
        TrainingTableSpec(timestamp_unit=unit)
