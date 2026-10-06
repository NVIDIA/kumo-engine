# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional

import kumoapi.common as api

from common_utils.error import PublicException
from io_utils import json
from io_utils.dataclasses import kumo_dataclass
from kumo.home import HomeDirManager

ValidationWarning = api.ValidationWarning
ValidationError = api.ValidationError
ValidationInfo = api.ValidationInfo


class ValidationException(PublicException):
    r"""
    Validation exception for all validation failures.

    Can be raised directly in validation logic (connectors, drivers) with title/message,
    or created from ValidationResponse for workflow error propagation.

    Supports conversion to ValidationError/Warning for backward compatibility.
    """
    template = ""  # Empty - we format the message manually

    def __init__(self, title: str = "", message: str = "",
                 id: Optional[str] = None,
                 related_errors: Optional[List[Exception]] = None,
                 **kwargs) -> None:
        self._title = title
        self._message = message
        self._id = id

        # Format display message flexibly
        if title and message:
            display_message = f"{title}: {message}"
        elif message:
            display_message = message
        elif title:
            display_message = title
        else:
            display_message = ""

        super().__init__(related_errors=related_errors, **kwargs)
        self.message = display_message

    def as_error(self) -> ValidationError:
        return ValidationError(self._title, self._message)

    def as_warning(self) -> ValidationWarning:
        return ValidationWarning(self._title, self._message)

    def as_info(self) -> ValidationInfo:
        return ValidationInfo(self._title, self._message, self._id or "")

    def title_and_message(self) -> str:
        # Preserve original format for backward compatibility
        return f"{self._title} - {self._message}"

    def to_dict(self) -> dict:
        """Override to preserve title, message, and id for proper deserialization"""
        result = super().to_dict()
        # Include the explicit constructor parameters in kwargs
        result['kwargs'] = {
            'title': self._title,
            'message': self._message,
            'id': self._id,
            **result['kwargs']
        }
        return result


@kumo_dataclass
class ValidationResponse(api.ValidationResponse):
    def error_message(self) -> str:
        """Serialized error message without the 'Errors:' prefix."""
        if len(self.errors) == 0:
            return ""

        def _fmt(err: api.ValidationError) -> str:
            if err.message:
                return f"{err.title}. {err.message}"
            return err.title

        if len(self.errors) == 1:
            return _fmt(self.errors[0])
        return "\n".join(
            [f"{i}. {_fmt(err)}" for i, err in enumerate(self.errors, 1)])

    def merge(self, other: 'ValidationResponse') -> 'ValidationResponse':
        self.warnings = list(set(self.warnings).union(other.warnings))
        self.errors = list(set(self.errors).union(other.errors))
        self.info_items = list(set(self.info_items).union(other.info_items))
        return self

    def save(self, home_dir: str, unique_id: str) -> None:
        r"""Saves this validation response to `root_path`, under the validation
        directory, as a JSON. `unique_id` must be unique for validations that
        are saved under this root path."""
        path = HomeDirManager(home_dir).validation_path(unique_id)
        json.save_json(self, path)

    @classmethod
    def load(cls, home_dir: str, unique_id: str) -> 'ValidationResponse':
        r"""Loads this validation response from `root_path`."""
        path = HomeDirManager(home_dir).validation_path(unique_id)
        return json.load_json(path, cls)

    @classmethod
    def load_or_create(cls, home_dir: str,
                       unique_id: str) -> 'ValidationResponse':
        r"""Loads existing ValidationResponse or creates empty one if not found."""
        path = HomeDirManager(home_dir).validation_path(unique_id)
        return json.load_json(
            path,
            cls,
            default=cls(),
            raise_on_missing=False,
        )

    def to_exception(self) -> ValidationException:
        """
        Convert validation errors to ValidationException for workflow propagation.

        Only call this when validation.ok is False.

        Returns:
            ValidationException with validation errors in message, warnings as related_errors
        """
        if self.ok:
            raise ValueError(
                "Cannot convert successful validation to exception")

        primary_message = self.error_message()

        exc = ValidationException(message=primary_message)

        # Add warnings as related errors for additional context
        for warning in self.warnings:
            warning_exc = Warning(f"{warning.title}: {warning.message}")
            exc.add_related_error(warning_exc)

        return exc
