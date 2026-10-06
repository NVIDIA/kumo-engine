# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: MIT

from dataclasses import dataclass, field
from enum import Enum
from typing import List


# This is not needed in python 3.11
class StrEnum(str, Enum):
    def __repr__(self) -> str:
        return self.value

    def __str__(self) -> str:
        return self.value


class JobSource(StrEnum):
    SDK = "SDK"
    UI = "UI"
    REST = "REST"
    UNKNOWN = "UNKNOWN"


class JobStatus(StrEnum):
    # Job has not yet been started, e.g., waiting for job precondition
    # (upstream job/data dependency) or the job is not yet scheduled.
    NOT_STARTED = 'NOT_STARTED'

    # Job is enqueued and waiting to acquire necessary resource/quota in order
    # to start execution. For example, after reaching max job concurrency
    # limit, more jobs may be submitted but in QUEUED status initially.
    QUEUED = "QUEUED"

    # Job has been submitted and is currently running.
    RUNNING = 'RUNNING'

    # Terminal status:
    DONE = 'DONE'  # Job has completed successfully
    FAILED = 'FAILED'  # Job has failed due to error.
    CANCELLED = 'CANCELLED'  # Job has been aborted/cancelLed by the customer.

    # Others:
    UNTRAINED = 'UNTRAINED'
    MISSING = 'MISSING'

    @property
    def is_terminal(self) -> bool:
        return self in [JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELLED]

    @property
    def is_submitted(self) -> bool:
        return self in [JobStatus.QUEUED, JobStatus.RUNNING]


@dataclass(frozen=True)
class ValidationInfo:
    """
    Represents non-critical information about automatic actions or decisions
    taken by Kumo that the user should be aware of, but do not require
    immediate attention or action.
    """
    # Short (< 1 line) title
    title: str
    # Precise message, customer-understandable
    message: str
    # Id to match position of info in the frontend
    id: str


@dataclass(frozen=True)
class ValidationWarning:
    # Short (< 1 line) description of what is wrong
    title: str

    # A multi-line, customer-understandable description of the problem, which
    # also contain 1 or more ways for the customer to fix it. It should not
    # contain any internal terminology (eg. only use terms that are consistent
    # with our public docs).
    message: str


@dataclass(frozen=True)
class KumoValidationError:
    title: str
    message: str


# Keep the established Python import while giving OpenAPI generators a unique
# component name that cannot collide with request-validation errors.
ValidationError = KumoValidationError


@dataclass
class ValidationResponse:
    warnings: List[ValidationWarning] = field(default_factory=list)
    errors: List[ValidationError] = field(default_factory=list)
    info_items: List[ValidationInfo] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0

    def empty(self) -> bool:
        return (len(self.errors) == 0 and len(self.warnings) == 0
                and len(self.info_items) == 0)

    def message(self) -> str:
        r"""Returns a simple string message enumerating errors, warnings, and
        info items in this validation response."""
        message = self.error_message()
        if len(self.warnings) > 0:
            if len(self.errors) > 0:
                message += "\n"
            message += "Warnings:\n"
            message += "\n".join([
                f"{i}. {warn.message}"
                for i, warn in enumerate(self.warnings, 1)
            ])

        if self.info_items:
            if message:
                message += "\n"
            message += "Info:\n"
            message += "\n".join([
                f"{i}. {info.title}. {info.message}"
                for i, info in enumerate(self.info_items, 1)
            ])

        return message

    def error_message(self) -> str:
        """
        Serialized, pretty-printed error message from all the errors
        contained in this ValidationResponse.
        """
        message = ""
        if len(self.errors) > 0:
            # message += "Errors:\n"
            if len(self.errors) == 1:
                message += f"{self.errors[0].title}. {self.errors[0].message}"
            else:
                message += "\n".join([
                    f"{i}. {err.title}. {err.message}"
                    for i, err in enumerate(self.errors, 1)
                ])
        return message
