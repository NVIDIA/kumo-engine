# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from kumo_core.diskgraph.runner import DiskgraphNoSpaceError
from kumo_core.errors import PublicError, TrainingValidationError
from kumo_core.public_error import PublicTrainingError

from common_utils.error import ErrorInfo, PublicException
from kumo.platform_errors import as_platform_public_exception
from kumo.validation import ValidationException
from statedb.utils.errors import ProviderBackendError


def test_training_validation_error_restores_legacy_type_and_message():
    error = TrainingValidationError(
        title="GPU out-of-memory error",
        message="Reduce output_embedding_dim.",
    )

    converted = as_platform_public_exception(error)

    assert isinstance(converted, ValidationException)
    assert converted.get_user_message() == (
        "GPU out-of-memory error: Reduce output_embedding_dim.")
    assert converted.title_and_message() == (
        "GPU out-of-memory error - Reduce output_embedding_dim.")


def test_public_training_error_restores_public_exception_contract():
    error = PublicTrainingError("Refit ran out of GPU memory.")

    converted = as_platform_public_exception(error)

    assert type(converted) is PublicException
    assert converted.get_user_message() == "Refit ran out of GPU memory."
    assert converted.non_retryable


def test_public_error_retains_cause_without_exposing_builtin_detail():
    error = PublicError(
        "Could not read source",
        error=RuntimeError("private provider detail"),
    )

    converted = as_platform_public_exception(error)

    assert type(converted) is ProviderBackendError
    assert converted.get_user_message() == "Could not read source"
    assert "private provider detail" not in converted.get_user_message()


def test_public_error_passes_cause_to_platform_classifier(monkeypatch):
    class WarehouseError(Exception):
        pass

    monkeypatch.setattr(
        ProviderBackendError,
        "CLASSIFIERS",
        [("Warehouse", WarehouseError,
          lambda _error: ErrorInfo("Warehouse", "query rejected"))],
    )
    cause = WarehouseError("private provider detail")

    converted = as_platform_public_exception(
        PublicError("Could not read source", error=cause))

    assert type(converted) is ProviderBackendError
    assert converted.get_user_message() == (
        "Could not read source\nWarehouse error:\nquery rejected")
    assert "private provider detail" not in converted.get_user_message()


def test_diskgraph_runner_error_is_not_rewritten():
    error = DiskgraphNoSpaceError("local SSD is full: 17 bytes available")

    assert as_platform_public_exception(error) is None
    assert str(error) == "local SSD is full: 17 bytes available"
    assert error.non_retryable is False
