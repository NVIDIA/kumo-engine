# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ordered synchronous execution with durable failure and resume state."""

from __future__ import annotations

import hashlib
import json
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from types import MappingProxyType
from typing import Any

from .artifacts import ArtifactStore, FilesystemArtifactStore
from .config import NativeRuntimeConfig
from .manifest import (
    RunManifest,
    RunPaths,
    RunStatus,
    StageError,
    StageRecord,
    StageStatus,
    utc_now,
)

StageOutput = Mapping[str, Any] | None
StageCallable = Callable[["StageContext"], StageOutput]
StageOutputValidator = Callable[["StageContext"], bool]


class StagePlanMismatchError(ValueError):
    """The stored manifest was created with a different ordered stage plan."""


class RunAlreadyExistsError(RuntimeError):
    """A manifest exists and resume was disabled."""


class InputFingerprintMismatchError(ValueError):
    """Resume was requested with different logical training inputs."""


class StageOutputCorruptionError(ValueError):
    """A completed stage's persisted output no longer matches its digest."""


class StageExecutionError(RuntimeError):
    """A pipeline stage failed after its failure record was persisted."""
    def __init__(
        self,
        run_id: str,
        stage_name: str,
        cause: Exception | None = None,
    ) -> None:
        self.run_id = run_id
        self.stage_name = stage_name
        message = f"run {run_id!r} failed in stage {stage_name!r}"
        if cause is not None:
            message += f": {type(cause).__name__}: {cause}"
        super().__init__(message)


@dataclass(frozen=True)
class Stage:
    name: str
    function: StageCallable
    resume_function: StageCallable | None = None
    output_validator: StageOutputValidator | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("stage name must not be empty")
        if not callable(self.function):
            raise TypeError("stage function must be callable")
        if self.resume_function is not None and not callable(
                self.resume_function):
            raise TypeError("stage resume_function must be callable")
        if self.output_validator is not None and not callable(
                self.output_validator):
            raise TypeError("stage output_validator must be callable")


@dataclass(frozen=True)
class StageContext:
    """Read-only run information passed to an injected stage function."""

    run_id: str
    paths: RunPaths
    stage_name: str
    attempt: int
    completed_outputs: Mapping[str, Mapping[str, Any]]


class StageRunner:
    """Run injected stage functions in order and durably record transitions.

    A stage is considered complete only after its ``DONE`` record is atomically
    persisted. If the process exits between the callable's side effects and
    that write, resume invokes the stage again; stage implementations therefore
    need to be idempotent or commit their own outputs atomically.
    """
    def __init__(
        self,
        config: NativeRuntimeConfig,
        stages: Sequence[Stage],
        artifact_store: ArtifactStore | None = None,
        *,
        input_fingerprint: str | None = None,
        now: Callable[[], datetime] = utc_now,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if not stages:
            raise ValueError("at least one stage is required")
        names = [stage.name for stage in stages]
        if len(names) != len(set(names)):
            raise ValueError("stage names must be unique")
        self._config = config
        self._stages = tuple(stages)
        self._artifact_store = artifact_store or FilesystemArtifactStore()
        self._input_fingerprint = input_fingerprint
        self._now = now
        self._monotonic = monotonic

    def run(self, run_id: str, *, resume: bool | None = None) -> RunManifest:
        paths = RunPaths.from_config(self._config, run_id)
        self._artifact_store.prepare_run(paths)
        manifest = self._artifact_store.load_manifest(paths)
        resumed_existing_run = manifest is not None
        should_resume = self._config.resume if resume is None else resume

        if manifest is None:
            manifest = RunManifest.create(
                run_id,
                paths,
                self.stage_names,
                input_fingerprint=self._input_fingerprint,
            )
            self._artifact_store.write_manifest(paths, manifest)
        else:
            self._validate_existing_manifest(manifest, paths)
            if not should_resume:
                raise RunAlreadyExistsError(
                    f"run {run_id!r} already has a durable manifest")
            if (manifest.status is RunStatus.DONE
                    and not self._requires_resume_checks):
                return manifest

        manifest = replace(manifest, status=RunStatus.RUNNING,
                           updated_at=self._now())
        self._artifact_store.write_manifest(paths, manifest)

        for index, stage in enumerate(self._stages):
            record = manifest.stages[index]
            if record.status is StageStatus.DONE:
                if resumed_existing_run and stage.resume_function is not None:
                    context = self._stage_context(manifest, paths, index)
                    outputs = self._normalize_outputs(
                        stage.resume_function(context))
                    refreshed = replace(
                        record,
                        outputs=outputs,
                        output_fingerprint=self._fingerprint_outputs(outputs),
                    )
                    manifest = manifest.replace_stage(index, refreshed,
                                                      self._now())
                    self._artifact_store.write_manifest(paths, manifest)
                if stage.output_validator is None:
                    continue
                context = self._stage_context(manifest, paths, index)
                validity = stage.output_validator(context)
                if not isinstance(validity, bool):
                    raise TypeError("stage output_validator must return a bool")
                if validity:
                    continue
                manifest = self._invalidate_suffix(manifest, index)
                self._artifact_store.write_manifest(paths, manifest)
            manifest = self._start_stage(manifest, index)
            self._artifact_store.write_manifest(paths, manifest)
            record = manifest.stages[index]
            context = self._stage_context(manifest, paths, index)
            started = self._monotonic()
            try:
                outputs = self._normalize_outputs(stage.function(context))
            except Exception as error:
                duration = max(0.0, self._monotonic() - started)
                manifest = self._fail_stage(manifest, index, error, duration)
                self._artifact_store.write_manifest(paths, manifest)
                raise StageExecutionError(run_id, stage.name, error) from error
            duration = max(0.0, self._monotonic() - started)
            manifest = self._finish_stage(manifest, index, outputs, duration)
            self._artifact_store.write_manifest(paths, manifest)

        manifest = replace(manifest, status=RunStatus.DONE,
                           updated_at=self._now())
        self._artifact_store.write_manifest(paths, manifest)
        return manifest

    @property
    def stage_names(self) -> tuple[str, ...]:
        return tuple(stage.name for stage in self._stages)

    @property
    def _requires_resume_checks(self) -> bool:
        return any(stage.resume_function is not None
                   or stage.output_validator is not None
                   for stage in self._stages)

    def _stage_context(self, manifest: RunManifest, paths: RunPaths,
                       index: int) -> StageContext:
        record = manifest.stages[index]
        return StageContext(
            run_id=paths.run_id,
            paths=paths,
            stage_name=record.name,
            attempt=record.attempt,
            completed_outputs=self._completed_outputs(manifest),
        )

    def _invalidate_suffix(self, manifest: RunManifest,
                           index: int) -> RunManifest:
        stages = list(manifest.stages)
        for suffix_index in range(index, len(stages)):
            previous = stages[suffix_index]
            stages[suffix_index] = StageRecord(
                name=previous.name,
                attempt=previous.attempt,
            )
        return replace(
            manifest,
            stages=tuple(stages),
            status=RunStatus.RUNNING,
            updated_at=self._now(),
        )

    def _validate_existing_manifest(self, manifest: RunManifest,
                                    paths: RunPaths) -> None:
        if manifest.run_id != paths.run_id:
            raise ValueError(
                "stored manifest run_id does not match the requested run")
        if manifest.paths != paths:
            raise ValueError(
                "stored manifest paths do not match runtime configuration")
        if manifest.input_fingerprint != self._input_fingerprint:
            raise InputFingerprintMismatchError(
                "stored manifest input fingerprint does not match the "
                "requested training inputs")
        stored_names = tuple(stage.name for stage in manifest.stages)
        if stored_names != self.stage_names:
            raise StagePlanMismatchError(
                f"stored stage plan {stored_names!r} does not match {self.stage_names!r}"
            )
        found_incomplete = False
        for stage in manifest.stages:
            if stage.status is not StageStatus.DONE:
                found_incomplete = True
            elif found_incomplete:
                raise ValueError(
                    "manifest has a completed stage after an incomplete stage")
            if (stage.status is StageStatus.DONE
                    and stage.output_fingerprint is not None
                    and stage.output_fingerprint != self._fingerprint_outputs(
                        stage.outputs)):
                raise StageOutputCorruptionError(
                    f"stored outputs for stage {stage.name!r} failed "
                    "fingerprint validation")

    def _start_stage(self, manifest: RunManifest, index: int) -> RunManifest:
        previous = manifest.stages[index]
        now = self._now()
        record = StageRecord(
            name=previous.name,
            status=StageStatus.RUNNING,
            attempt=previous.attempt + 1,
            started_at=now,
        )
        return manifest.replace_stage(index, record, now)

    def _finish_stage(
        self,
        manifest: RunManifest,
        index: int,
        outputs: Mapping[str, Any],
        duration: float,
    ) -> RunManifest:
        previous = manifest.stages[index]
        now = self._now()
        record = replace(
            previous,
            status=StageStatus.DONE,
            finished_at=now,
            duration_seconds=duration,
            outputs=outputs,
            output_fingerprint=self._fingerprint_outputs(outputs),
            error=None,
        )
        return manifest.replace_stage(index, record, now)

    def _fail_stage(
        self,
        manifest: RunManifest,
        index: int,
        error: Exception,
        duration: float,
    ) -> RunManifest:
        previous = manifest.stages[index]
        now = self._now()
        failure = StageError(
            type=f"{type(error).__module__}.{type(error).__qualname__}",
            message=str(error),
            traceback="".join(
                traceback.format_exception(type(error), error,
                                           error.__traceback__)),
        )
        record = replace(
            previous,
            status=StageStatus.FAILED,
            finished_at=now,
            duration_seconds=duration,
            outputs={},
            error=failure,
        )
        result = manifest.replace_stage(index, record, now)
        return replace(result, status=RunStatus.FAILED)

    @staticmethod
    def _completed_outputs(
        manifest: RunManifest, ) -> Mapping[str, Mapping[str, Any]]:
        values = {
            stage.name: MappingProxyType(dict(stage.outputs))
            for stage in manifest.stages if stage.status is StageStatus.DONE
        }
        return MappingProxyType(values)

    @staticmethod
    def _normalize_outputs(outputs: StageOutput) -> Mapping[str, Any]:
        if outputs is None:
            return MappingProxyType({})
        if not isinstance(outputs, Mapping):
            raise TypeError("stage output must be a mapping or None")
        normalized = dict(outputs)
        try:
            json.dumps(normalized, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise TypeError("stage output must be JSON serializable") from error
        return MappingProxyType(normalized)

    @staticmethod
    def _fingerprint_outputs(outputs: Mapping[str, Any]) -> str:
        serialized = json.dumps(
            dict(outputs),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(serialized).hexdigest()
