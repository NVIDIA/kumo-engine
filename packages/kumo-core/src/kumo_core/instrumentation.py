# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Service-free observability hooks for portable training."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import wraps
from typing import Any, ContextManager, Protocol, TypeVar

T = TypeVar("T")


class TrainingInstrumentation(Protocol):
    """Observability surface; implementations cannot alter training inputs."""
    def record_metrics_job_name(self, job_name: str) -> None:
        ...

    def span(self, name: str, **fields: Any) -> ContextManager[None]:
        ...

    def timing(self, name: str, **fields: Any) -> ContextManager[None]:
        ...


class NoOpTrainingInstrumentation:
    def record_metrics_job_name(self, job_name: str) -> None:
        return None

    @contextmanager
    def span(self, name: str, **fields: Any) -> Iterator[None]:
        yield

    @contextmanager
    def timing(self, name: str, **fields: Any) -> Iterator[None]:
        yield


_instrumentation: TrainingInstrumentation = NoOpTrainingInstrumentation()


def configure_instrumentation(value: TrainingInstrumentation) -> None:
    global _instrumentation
    _instrumentation = value


def set_metrics_job_name(job_name: str) -> None:
    _instrumentation.record_metrics_job_name(job_name)


def timing(name: str, **fields: Any) -> ContextManager[None]:
    return _instrumentation.timing(name, **fields)


def trace() -> Callable[[Callable[..., T]], Callable[..., T]]:
    def decorate(function: Callable[..., T]) -> Callable[..., T]:
        @wraps(function)
        def wrapped(*args: Any, **kwargs: Any) -> T:
            with _instrumentation.span(function.__qualname__):
                return function(*args, **kwargs)

        return wrapped

    return decorate
