# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from kumo_core.artifacts import FilesystemArtifactStore
from kumo_core.config import NativeRuntimeConfig
from kumo_core.manifest import RunManifest, RunPaths, RunStatus, StageStatus
from kumo_core.pipeline import (
    InputFingerprintMismatchError,
    RunAlreadyExistsError,
    Stage,
    StageExecutionError,
    StageOutputCorruptionError,
    StagePlanMismatchError,
    StageRunner,
)


def _config(tmp_path: Path, *, resume: bool = True) -> NativeRuntimeConfig:
    return NativeRuntimeConfig(
        scratch_root=tmp_path / "scratch",
        artifact_root=tmp_path / "artifacts",
        resume=resume,
    )


def test_runner_executes_ordered_stages_and_persists_outputs(
        tmp_path: Path) -> None:
    calls: list[str] = []

    def ingest(context):
        calls.append(context.stage_name)
        assert context.completed_outputs == {}
        return {"snapshot": "abc"}

    def train(context):
        calls.append(context.stage_name)
        assert context.completed_outputs["ingest"]["snapshot"] == "abc"
        return {"metric": 0.9}

    runner = StageRunner(_config(tmp_path),
                         [Stage("ingest", ingest),
                          Stage("train", train)])

    manifest = runner.run("run-1")

    assert calls == ["ingest", "train"]
    assert manifest.status is RunStatus.DONE
    assert [record.status for record in manifest.stages] == [
        StageStatus.DONE,
        StageStatus.DONE,
    ]
    assert manifest.stages[1].outputs == {"metric": 0.9}
    assert all(record.duration_seconds is not None
               for record in manifest.stages)


def test_runner_records_failure_then_resumes_from_failed_stage(
        tmp_path: Path) -> None:
    calls = {"ingest": 0, "train": 0}

    def ingest(_context):
        calls["ingest"] += 1
        return {"snapshot": "abc"}

    def train(context):
        calls["train"] += 1
        if context.attempt == 1:
            raise RuntimeError("GPU unavailable")
        return {"metric": 0.8}

    config = _config(tmp_path)
    runner = StageRunner(config,
                         [Stage("ingest", ingest),
                          Stage("train", train)])

    with pytest.raises(StageExecutionError) as failure:
        runner.run("resume-me")

    assert failure.value.__cause__ is not None
    assert str(failure.value) == ("run 'resume-me' failed in stage 'train': "
                                  "RuntimeError: GPU unavailable")
    paths = RunPaths.from_config(config, "resume-me")
    failed = FilesystemArtifactStore().load_manifest(paths)
    assert failed is not None
    assert failed.status is RunStatus.FAILED
    assert failed.stages[0].status is StageStatus.DONE
    assert failed.stages[1].status is StageStatus.FAILED
    assert failed.stages[1].error is not None
    assert failed.stages[1].error.message == "GPU unavailable"
    assert "RuntimeError" in failed.stages[1].error.traceback

    completed = runner.run("resume-me")

    assert completed.status is RunStatus.DONE
    assert calls == {"ingest": 1, "train": 2}
    assert completed.stages[1].attempt == 2
    assert completed.stages[1].error is None


def test_resume_hook_rehydrates_completed_stage_before_failed_stage(
        tmp_path: Path) -> None:
    config = _config(tmp_path)
    calls: list[str] = []
    should_fail = True

    def downstream(_context):
        nonlocal should_fail
        if should_fail:
            should_fail = False
            raise RuntimeError("restart")
        calls.append("downstream")

    runner = StageRunner(
        config,
        [
            Stage(
                "ingest", lambda _context: {"index": "/local"},
                resume_function=lambda _context: calls.append(
                    "hydrate") or {"index": "/local"}),
            Stage("downstream", downstream),
        ],
    )
    with pytest.raises(StageExecutionError):
        runner.run("rehydrate")

    runner.run("rehydrate")

    assert calls == ["hydrate", "downstream"]


def test_runner_retries_a_stage_left_running_by_process_interruption(
    tmp_path: Path, ) -> None:
    config = _config(tmp_path)
    calls: list[int] = []

    def stage(context):
        calls.append(context.attempt)

    runner = StageRunner(config, [Stage("interrupted", stage)])
    paths = RunPaths.from_config(config, "interrupted-run")
    store = FilesystemArtifactStore()
    store.prepare_run(paths)
    manifest = RunManifest.create("interrupted-run", paths, ("interrupted", ))
    running_stage = replace(
        manifest.stages[0],
        status=StageStatus.RUNNING,
        attempt=1,
        started_at=manifest.updated_at,
    )
    manifest = manifest.replace_stage(0, running_stage, manifest.updated_at)
    manifest = replace(manifest, status=RunStatus.RUNNING)
    store.write_manifest(paths, manifest)

    completed = runner.run("interrupted-run")

    assert completed.status is RunStatus.DONE
    assert completed.stages[0].attempt == 2
    assert calls == [2]


def test_completed_run_is_idempotent_when_resume_is_enabled(
        tmp_path: Path) -> None:
    calls = 0

    def stage(_context):
        nonlocal calls
        calls += 1

    runner = StageRunner(_config(tmp_path), [Stage("only", stage)])
    first = runner.run("done")
    second = runner.run("done")

    assert first == second
    assert calls == 1


def test_resume_reruns_suffix_when_staged_source_was_deleted(
        tmp_path: Path) -> None:
    calls: list[str] = []
    source_path = tmp_path / "durable" / "source"

    def source(_context):
        calls.append("source")
        source_path.mkdir(parents=True, exist_ok=True)
        return {"path": str(source_path)}

    def downstream(context):
        calls.append(context.stage_name)
        return {"input": context.completed_outputs["source"]["path"]}

    def source_is_valid(context):
        return Path(context.completed_outputs["source"]["path"]).is_dir()

    runner = StageRunner(
        _config(tmp_path),
        [
            Stage("source", source, output_validator=source_is_valid),
            Stage("ingest", downstream),
            Stage("train", downstream),
        ],
    )
    runner.run("deleted-source")
    source_path.rmdir()

    resumed = runner.run("deleted-source")

    assert calls == [
        "source",
        "ingest",
        "train",
        "source",
        "ingest",
        "train",
    ]
    assert [stage.attempt for stage in resumed.stages] == [2, 2, 2]


def test_resume_reruns_materialization_suffix_when_tt_extension_was_deleted(
        tmp_path: Path) -> None:
    calls: list[str] = []
    extension = tmp_path / "scratch" / "graph" / "train_tables" / "tt-1"

    def stage(context):
        calls.append(context.stage_name)
        if context.stage_name == "materialize":
            extension.mkdir(parents=True, exist_ok=True)
        return {"path": str(extension)}

    def extension_is_valid(context):
        path = context.completed_outputs["materialize"]["path"]
        return Path(path).is_dir()

    runner = StageRunner(
        _config(tmp_path),
        [
            Stage("source", stage),
            Stage("ingest", stage),
            Stage("train-table", stage),
            Stage(
                "materialize",
                stage,
                output_validator=extension_is_valid,
            ),
            Stage("train", stage),
        ],
    )
    runner.run("deleted-extension")
    extension.rmdir()

    resumed = runner.run("deleted-extension")

    assert calls == [
        "source",
        "ingest",
        "train-table",
        "materialize",
        "train",
        "materialize",
        "train",
    ]
    assert [stage.attempt for stage in resumed.stages] == [1, 1, 1, 2, 2]


def test_existing_manifest_is_rejected_when_resume_is_disabled(
        tmp_path: Path) -> None:
    config = _config(tmp_path, resume=False)
    runner = StageRunner(config, [Stage("only", lambda _context: None)])
    runner.run("existing")

    with pytest.raises(RunAlreadyExistsError):
        runner.run("existing")


def test_changed_stage_plan_cannot_resume(tmp_path: Path) -> None:
    config = _config(tmp_path)
    StageRunner(config, [Stage("first", lambda _context: None)]).run("plan")

    changed = StageRunner(config, [Stage("renamed", lambda _context: None)])
    with pytest.raises(StagePlanMismatchError):
        changed.run("plan")


def test_changed_input_fingerprint_cannot_resume(tmp_path: Path) -> None:
    config = _config(tmp_path)
    stage = Stage("only", lambda _context: {"artifact": "one"})
    StageRunner(config, [stage], input_fingerprint="input-one").run("inputs")

    changed = StageRunner(config, [stage], input_fingerprint="input-two")
    with pytest.raises(InputFingerprintMismatchError):
        changed.run("inputs")


def test_tampered_completed_output_cannot_resume(tmp_path: Path) -> None:
    config = _config(tmp_path)
    runner = StageRunner(
        config,
        [Stage("only", lambda _context: {"artifact": "original"})],
        input_fingerprint="stable",
    )
    runner.run("tampered")
    paths = RunPaths.from_config(config, "tampered")
    store = FilesystemArtifactStore()
    manifest = store.load_manifest(paths)
    assert manifest is not None
    altered = replace(manifest.stages[0], outputs={"artifact": "changed"})
    store.write_manifest(
        paths,
        manifest.replace_stage(0, altered, manifest.updated_at),
    )

    with pytest.raises(StageOutputCorruptionError):
        runner.run("tampered")


def test_non_json_stage_output_is_recorded_as_failure(tmp_path: Path) -> None:
    runner = StageRunner(
        _config(tmp_path),
        [Stage("bad-output", lambda _context: {"value": object()})])

    with pytest.raises(StageExecutionError):
        runner.run("bad")

    paths = RunPaths.from_config(_config(tmp_path), "bad")
    failed = FilesystemArtifactStore().load_manifest(paths)
    assert failed is not None
    assert failed.stages[0].error is not None
    assert failed.stages[
        0].error.message == "stage output must be JSON serializable"


def test_non_standard_json_number_is_recorded_as_failure(
        tmp_path: Path) -> None:
    runner = StageRunner(
        _config(tmp_path),
        [Stage("bad-number", lambda _context: {"value": float("nan")})])

    with pytest.raises(StageExecutionError):
        runner.run("bad-number")


def test_duplicate_stage_names_are_rejected(tmp_path: Path) -> None:
    duplicate = Stage("duplicate", lambda _context: None)
    with pytest.raises(ValueError, match="unique"):
        StageRunner(_config(tmp_path), [duplicate, duplicate])
