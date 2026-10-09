# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Portable runtime errors with no platform or orchestration dependencies."""

from __future__ import annotations

from typing import Any


class PublicError(Exception):
    """Minimal user-facing exception behavior needed by portable callers."""

    non_retryable: bool = True
    http_code: int = 422

    def __init__(
        self,
        message: str = "",
        error: Exception | None = None,
        related_errors: list[Exception] | None = None,
        non_retryable: bool | None = None,
        **_: Any,
    ) -> None:
        super().__init__(message)
        self.message = message
        # Provider-specific classification belongs to the owning adapter, but
        # the adapter still needs the original exception to reproduce its
        # public error contract.
        self.error = error
        self.kwargs = {"message": message}
        self.related_errors = related_errors or []
        if non_retryable is not None:
            self.non_retryable = non_retryable

    def get_user_message(self, indent_level: int = 0) -> str:
        """Return the safe message without exposing a wrapped exception."""
        return f"{'  ' * indent_level}{self.message}"

    def add_related_error(self, error: Exception) -> None:
        self.related_errors.append(error)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the stable public fields used at process boundaries."""
        return {
            "type": self.__class__.__name__,
            "message": self.message,
            "kwargs": self.kwargs,
            "http_code": self.http_code,
            "non_retryable": self.non_retryable,
            "related_errors": [],
        }


class TrainingValidationError(ValueError):
    """Invalid portable training configuration or runtime state."""
    def __init__(self, *, title: str = "", message: str = "") -> None:
        self.title = title
        self.detail = message
        display = (f"{title}: {message}"
                   if title and message else title or message)
        super().__init__(display)


__all__ = [
    "PublicError",
    "TrainingValidationError",
]
