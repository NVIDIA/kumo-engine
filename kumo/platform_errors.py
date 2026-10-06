# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Translate portable errors at the monolith platform boundary."""

from kumo_core.errors import PublicError, TrainingValidationError

from common_utils.error import PublicException
from kumo.validation import ValidationException
from statedb.utils.errors import ProviderBackendError

CORE_PLATFORM_ERROR_TYPES = (PublicError, TrainingValidationError)


def as_platform_public_exception(
    error: BaseException, ) -> PublicException | None:
    """Return the legacy platform representation of a portable error."""
    if isinstance(error, PublicException):
        return error
    if isinstance(error, TrainingValidationError):
        return ValidationException(
            title=error.title,
            message=error.detail,
        )
    if isinstance(error, PublicError):
        platform_error_type = (ProviderBackendError
                               if error.error is not None else PublicException)
        platform_error = platform_error_type(
            message=error.message,
            error=error.error,
            non_retryable=error.non_retryable,
        )
        platform_error.http_code = error.http_code
        for related_error in error.related_errors:
            converted = as_platform_public_exception(related_error)
            if converted is not None:
                platform_error.add_related_error(converted)
        return platform_error
    return None


__all__ = [
    "CORE_PLATFORM_ERROR_TYPES",
    "as_platform_public_exception",
]
