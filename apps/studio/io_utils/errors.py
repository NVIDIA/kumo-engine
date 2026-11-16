# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lightweight error helpers shared across integration boundaries."""

from contextlib import contextmanager
from typing import Final, Iterator, Optional, TypeAlias

RESTRICTED_LOG_MESSAGE_ATTRIBUTE: Final = '_kumo_restricted_log_message'
SAFE_LOG_MESSAGE_ATTRIBUTE: Final = '_kumo_safe_log_message'
ExceptionTypes: TypeAlias = type[Exception] | tuple[type[Exception], ...]


class RestrictedDiagnosticError(RuntimeError):
    """Fixed-message fallback when a foreign exception rejects metadata."""

    _kumo_restricted_log_message = True

    def __init__(self, original_error_type: str,
                 safe_message: Optional[str] = None) -> None:
        self.original_error_type = original_error_type
        if safe_message:
            setattr(self, SAFE_LOG_MESSAGE_ATTRIBUTE, safe_message)
        super().__init__(safe_message
                         or 'Restricted integration operation failed')


def mark_exception_restricted(
    error: BaseException,
    *,
    safe_message: Optional[str] = None,
) -> BaseException:
    """Mark one boundary-originated exception without changing its type.

    Most Python and extension exceptions accept instance metadata and preserve
    it through pickling. A fixed-message wrapper is returned as a defensive
    fallback for exceptions that reject attribute assignment. Callers must
    raise the returned object rather than assuming it is always ``error``.

    This metadata is passive: only marker-aware diagnostic renderers honor it.
    Arbitrary logging of ``error`` still renders its raw message.
    """
    try:
        setattr(error, RESTRICTED_LOG_MESSAGE_ATTRIBUTE, True)
        if safe_message:
            setattr(error, SAFE_LOG_MESSAGE_ATTRIBUTE, safe_message)
    except (AttributeError, TypeError):
        return RestrictedDiagnosticError(type(error).__name__, safe_message)
    return error


@contextmanager
def restricted_diagnostic_boundary(
    *,
    exception_types: ExceptionTypes,
    safe_message: Optional[str] = None,
) -> Iterator[None]:
    """Mark failures from the smallest practical external operation."""
    try:
        yield
    except exception_types as error:
        marked_error = mark_exception_restricted(
            error,
            safe_message=safe_message,
        )
        if marked_error is error:
            raise
        raise marked_error from None


class WarehouseMisconfigurationError(Exception):
    r"""Thrown when we fail to successfully open a Connector"""


class WarehouseRetriableError(Exception):
    """Base class for retriable errors."""


class WarehouseExtendedRetriableError(WarehouseRetriableError):
    """Error that should be retried with extended retry parameters."""
