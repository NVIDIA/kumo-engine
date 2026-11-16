# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Serializable run state and explicit paths for native pipeline execution."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import NativeRuntimeConfig

MANIFEST_SCHEMA_VERSION = 1
_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(timezone.utc)


def _format_datetime(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _parse_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(
            f"expected an ISO timestamp, got {type(value).__name__}")
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise ValueError("manifest timestamps must include a timezone")
    return result


class RunStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    DONE = "DONE"
    FAILED = "FAILED"


class StageStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    DONE = "DONE"
    FAILED = "FAILED"


@dataclass(frozen=True)
class RunPaths:
    """All filesystem locations owned by a run."""

    run_id: str
    scratch_dir: Path
    artifact_dir: Path
    manifest_path: Path
    diagnostics_dir: Path

    @classmethod
    def from_config(cls, config: NativeRuntimeConfig,
                    run_id: str) -> "RunPaths":
        if _SAFE_RUN_ID.fullmatch(run_id) is None:
            raise ValueError(
                "run_id must start with an alphanumeric character and contain "
                "only alphanumerics, dots, underscores, or hyphens")
        candidate = Path(run_id)
        if candidate.is_absolute() or len(candidate.parts) != 1:
            raise ValueError("run_id must be a single path-safe component")
        artifact_dir = config.artifact_root / run_id
        return cls(
            run_id=run_id,
            scratch_dir=config.scratch_root / run_id,
            artifact_dir=artifact_dir,
            manifest_path=artifact_dir / config.manifest_filename,
            diagnostics_dir=artifact_dir / "diagnostics",
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "run_id": self.run_id,
            "scratch_dir": str(self.scratch_dir),
            "artifact_dir": str(self.artifact_dir),
            "manifest_path": str(self.manifest_path),
            "diagnostics_dir": str(self.diagnostics_dir),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RunPaths":
        return cls(
            run_id=str(data["run_id"]),
            scratch_dir=Path(data["scratch_dir"]),
            artifact_dir=Path(data["artifact_dir"]),
            manifest_path=Path(data["manifest_path"]),
            diagnostics_dir=Path(data["diagnostics_dir"]),
        )


@dataclass(frozen=True)
class StageError:
    type: str
    message: str
    traceback: str

    def to_dict(self) -> dict[str, str]:
        return {
            "type": self.type,
            "message": self.message,
            "traceback": self.traceback,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StageError":
        return cls(
            type=str(data["type"]),
            message=str(data["message"]),
            traceback=str(data["traceback"]),
        )


@dataclass(frozen=True)
class StageRecord:
    name: str
    status: StageStatus = StageStatus.PENDING
    attempt: int = 0
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_seconds: float | None = None
    outputs: Mapping[str, Any] = field(default_factory=dict)
    output_fingerprint: str | None = None
    error: StageError | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "attempt": self.attempt,
            "started_at": _format_datetime(self.started_at),
            "finished_at": _format_datetime(self.finished_at),
            "duration_seconds": self.duration_seconds,
            "outputs": dict(self.outputs),
            "output_fingerprint": self.output_fingerprint,
            "error": self.error.to_dict() if self.error else None,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StageRecord":
        error = data.get("error")
        outputs = data.get("outputs", {})
        if not isinstance(outputs, Mapping):
            raise ValueError("stage outputs must be a mapping")
        return cls(
            name=str(data["name"]),
            status=StageStatus(str(data["status"])),
            attempt=int(data.get("attempt", 0)),
            started_at=_parse_datetime(data.get("started_at")),
            finished_at=_parse_datetime(data.get("finished_at")),
            duration_seconds=(float(data["duration_seconds"])
                              if data.get("duration_seconds") is not None else
                              None),
            outputs=dict(outputs),
            output_fingerprint=(str(data["output_fingerprint"])
                                if data.get("output_fingerprint") is not None
                                else None),
            error=StageError.from_dict(error)
            if isinstance(error, Mapping) else None,
        )


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    paths: RunPaths
    stages: Sequence[StageRecord]
    status: RunStatus = RunStatus.PENDING
    input_fingerprint: str | None = None
    schema_version: int = MANIFEST_SCHEMA_VERSION
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)

    @classmethod
    def create(
        cls,
        run_id: str,
        paths: RunPaths,
        stage_names: Sequence[str],
        *,
        input_fingerprint: str | None = None,
    ) -> "RunManifest":
        now = utc_now()
        return cls(
            run_id=run_id,
            paths=paths,
            stages=tuple(StageRecord(name=name) for name in stage_names),
            input_fingerprint=input_fingerprint,
            created_at=now,
            updated_at=now,
        )

    def replace_stage(self, index: int, stage: StageRecord,
                      now: datetime) -> "RunManifest":
        stages = list(self.stages)
        stages[index] = stage
        return replace(self, stages=tuple(stages), updated_at=now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "status": self.status.value,
            "input_fingerprint": self.input_fingerprint,
            "paths": self.paths.to_dict(),
            "created_at": _format_datetime(self.created_at),
            "updated_at": _format_datetime(self.updated_at),
            "stages": [stage.to_dict() for stage in self.stages],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RunManifest":
        schema_version = int(data["schema_version"])
        if schema_version != MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported manifest schema version {schema_version}; "
                f"expected {MANIFEST_SCHEMA_VERSION}")
        paths_data = data.get("paths")
        if not isinstance(paths_data, Mapping):
            raise ValueError("manifest paths must be a mapping")
        stage_data = data.get("stages")
        if not isinstance(stage_data, list):
            raise ValueError("manifest stages must be a list")
        created_at = _parse_datetime(data.get("created_at"))
        updated_at = _parse_datetime(data.get("updated_at"))
        if created_at is None or updated_at is None:
            raise ValueError("manifest timestamps are required")
        return cls(
            schema_version=schema_version,
            run_id=str(data["run_id"]),
            status=RunStatus(str(data["status"])),
            input_fingerprint=(str(data["input_fingerprint"])
                               if data.get("input_fingerprint") is not None else
                               None),
            paths=RunPaths.from_dict(paths_data),
            created_at=created_at,
            updated_at=updated_at,
            stages=tuple(StageRecord.from_dict(stage) for stage in stage_data),
        )
