# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public, dependency-light contracts for user-submitted Snowflake ML Jobs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Mapping

from kumoapi.jobs import (
    AutoTrainerProgress,
    JobStatusReport,
    PredictionProgress,
)
from kumoapi.plugin import PublishedModel

_TRANSPORT_KEY = '__kumo_ml_job_transport__'
_PUBLISHED_MODEL_TYPE = 'kumo.PublishedModel'
_PUBLISHED_MODEL_SCHEMA_VERSION = 1


def _transport_node(kind: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Wrap an SDK value without colliding with user-returned mappings."""
    return {
        _TRANSPORT_KEY: {
            'kind': kind,
            'payload': dict(payload),
        }
    }


def _parse_transport_node(
    value: Mapping[str, Any], ) -> tuple[str, Mapping[str, Any]] | None:
    marker = value.get(_TRANSPORT_KEY)
    if marker is None:
        return None
    if not isinstance(marker, Mapping):
        raise ValueError('invalid Kumo ML Job transport marker')
    kind = marker.get('kind')
    payload = marker.get('payload')
    if not isinstance(kind, str) or not isinstance(payload, Mapping):
        raise ValueError('invalid Kumo ML Job transport node')
    return kind, payload


def _published_model_payload(model: PublishedModel) -> dict[str, Any]:
    """Serialize the existing plugin model result for an ML Job boundary."""
    if not isinstance(model, PublishedModel):
        raise TypeError('expected a PublishedModel')
    for name in ('name', 'version', 'uri'):
        value = getattr(model, name)
        if not isinstance(value, str) or not value:
            raise ValueError(f'PublishedModel {name} must be non-empty')
    metadata = dict(model.metadata)
    if any(not isinstance(key, str) or not isinstance(value, str)
           for key, value in metadata.items()):
        raise ValueError('PublishedModel metadata must map strings to strings')
    metrics = dict(model.metrics)
    if any(not isinstance(key, str) or not isinstance(value, (int, float))
           or isinstance(value, bool) or not math.isfinite(value)
           for key, value in metrics.items()):
        raise ValueError('PublishedModel metrics must be finite numbers')
    return {
        'type': _PUBLISHED_MODEL_TYPE,
        'schema_version': _PUBLISHED_MODEL_SCHEMA_VERSION,
        'name': model.name,
        'version': model.version,
        'uri': model.uri,
        'metadata': metadata,
        'metrics': metrics,
    }


def _published_model_from_payload(payload: Mapping[str, Any]) -> PublishedModel:
    """Validate and reconstruct an ML Job transported PublishedModel."""
    if payload.get('type') != _PUBLISHED_MODEL_TYPE:
        raise ValueError('payload is not a PublishedModel')
    if payload.get('schema_version') != _PUBLISHED_MODEL_SCHEMA_VERSION:
        raise ValueError('unsupported PublishedModel transport version')
    model = PublishedModel(
        name=payload.get('name'),
        version=payload.get('version'),
        uri=payload.get('uri'),
        metadata=payload.get('metadata', {}),
        metrics=payload.get('metrics', {}),
    )
    # Use the encoder as the single validation path in both directions.
    _published_model_payload(model)
    return model


class SnowflakeMLJobStatus(str, Enum):
    """Normalized Snowflake ML Job service states."""

    QUEUED = 'QUEUED'
    RUNNING = 'RUNNING'
    SUCCEEDED = 'SUCCEEDED'
    CANCELLING = 'CANCELLING'
    CANCELLED = 'CANCELLED'
    FAILED = 'FAILED'
    DELETED = 'DELETED'

    @property
    def terminal(self) -> bool:
        return self in {
            SnowflakeMLJobStatus.SUCCEEDED,
            SnowflakeMLJobStatus.CANCELLED,
            SnowflakeMLJobStatus.FAILED,
            SnowflakeMLJobStatus.DELETED,
        }


@dataclass(frozen=True)
class SnowflakeMLJobDiagnostic:
    """Serializable diagnostic captured from a remote or Snowflake failure."""

    type: str
    message: str
    code: int | None = None
    sqlstate: str | None = None
    sfqid: str | None = None
    traceback: str | None = None


@dataclass(frozen=True)
class SnowflakeMLJobProgress:
    """Latest structured Kumo progress mirrored by the remote function."""

    execution_id: str
    status_report: JobStatusReport
    updated_at: datetime
    operation_kind: str | None = None
    operation_id: str | None = None
    training: AutoTrainerProgress | None = None
    prediction: PredictionProgress | None = None
    current_metrics: Mapping[str, float] | None = None
    best_metrics: Mapping[str, float] | None = None


@dataclass(frozen=True)
class SnowflakeMLJobResult:
    """Terminal result envelope for one submitted user function.

    ``value`` preserves the function's supported return value. When the
    function returns a top-level :class:`PublishedModel`, ``published_model``
    exposes the same shared transport contract as a convenience sidecar.
    """

    execution_id: str
    service_id: str
    status: SnowflakeMLJobStatus
    value: Any = None
    published_model: PublishedModel | None = None
    progress: AutoTrainerProgress | PredictionProgress | None = None
    status_report: JobStatusReport | None = None
    diagnostic: SnowflakeMLJobDiagnostic | None = None
    finished_at: datetime | None = None

    @property
    def succeeded(self) -> bool:
        return self.status is SnowflakeMLJobStatus.SUCCEEDED

    def raise_for_failure(self) -> 'SnowflakeMLJobResult':
        if not self.succeeded:
            raise SnowflakeMLJobFailed(self)
        return self


class SnowflakeMLJobError(Exception):
    """Base exception for the Kumo Snowflake ML Job integration."""


class SnowflakeMLJobConfigurationError(SnowflakeMLJobError, ValueError):
    """The submission configuration or callable is invalid."""


class SnowflakeMLJobPreflightError(SnowflakeMLJobError):
    """Snowflake resources could not be validated before submission."""
    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


class SnowflakeMLJobPermissionError(
        SnowflakeMLJobPreflightError,
        PermissionError,
):
    """Snowflake rejected a preflight operation for authorization reasons."""


class SnowflakeMLJobTransientError(SnowflakeMLJobPreflightError):
    """A retryable connection, transport, or timeout failure."""


class SnowflakeMLJobSubmissionError(SnowflakeMLJobError):
    """Submission failed after preflight, possibly after service creation."""
    def __init__(
        self,
        message: str,
        *,
        retryable: bool,
        service_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.service_id = service_id


class SnowflakeMLJobFailed(SnowflakeMLJobError, RuntimeError):
    """Raised by :meth:`SnowflakeMLJobResult.raise_for_failure`."""
    def __init__(self, result: SnowflakeMLJobResult) -> None:
        diagnostic = result.diagnostic
        detail = diagnostic.message if diagnostic is not None else result.status.value
        super().__init__(
            f'Snowflake ML Job {result.service_id!r} did not succeed: {detail}')
        self.result = result
