# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from kumoapi.common import ValidationError, ValidationWarning

from kumo.validation.validation import ValidationResponse


@pytest.fixture
def validation_response():
    return ValidationResponse()


def test_merge_with_empty_response(validation_response):
    other_response = ValidationResponse()
    validation_response.merge(other_response)
    assert validation_response.errors == []
    assert validation_response.warnings == []


def test_merge_with_errors_with_dedup(validation_response):
    other_response = ValidationResponse()
    other_response.errors.append(ValidationError("Error Title 1", "Error 1"))
    other_response.errors.append(ValidationError("Error Title 1", "Error 1"))
    validation_response.merge(other_response)
    assert validation_response.errors == [
        ValidationError("Error Title 1", "Error 1")
    ]


def test_merge_with_warnings(validation_response):
    other_response = ValidationResponse()
    other_response.warnings.append(
        ValidationWarning("Warning Title 1", "Warning 1"))
    validation_response.merge(other_response)
    assert validation_response.warnings == [
        ValidationWarning("Warning Title 1", "Warning 1")
    ]


def test_merge_with_errors_and_warnings(validation_response):
    other_response = ValidationResponse()
    other_response.errors.append(ValidationError("Error Title 1", "Error 1"))
    other_response.warnings.append(
        ValidationWarning("Warning Title 1", "Warning 1"))
    validation_response.merge(other_response)
    assert validation_response.errors == [
        ValidationError(
            "Error Title 1",
            "Error 1",
        )
    ]
    assert validation_response.warnings == [
        ValidationWarning("Warning Title 1", "Warning 1")
    ]


def test_merge_combines_errors_and_warnings(validation_response):
    validation_response.errors.append(
        ValidationError("Error Title 1", "Error 1"))
    validation_response.errors.append(
        ValidationError("Error Title 2", "Error 2"))
    validation_response.warnings.append(
        ValidationWarning("Warning Title 1", "Warning 1"))
    other_response = ValidationResponse()
    other_response.errors.append(ValidationError("Error Title 1", "Error 1"))
    other_response.warnings.append(
        ValidationWarning("Warning Title 2", "Warning 2"))
    validation_response.merge(other_response)
    assert len(validation_response.errors) == 2
    assert len(validation_response.warnings) == 2
    assert ValidationError(
        "Error Title 1",
        "Error 1",
    ) in validation_response.errors
    assert ValidationError(
        "Error Title 2",
        "Error 2",
    ) in validation_response.errors
    assert ValidationWarning(
        "Warning Title 1",
        "Warning 1",
    ) in validation_response.warnings
    assert ValidationWarning(
        "Warning Title 2",
        "Warning 2",
    ) in validation_response.warnings
