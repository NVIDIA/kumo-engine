# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Durable artifact-store contract and a local filesystem implementation."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    Iterable,
    Mapping,
    Protocol,
    runtime_checkable,
)

from .io.filesystem import atomic_write_bytes, fsync_directory, local_path
from .io.json import load_json, save_json
from .manifest import RunManifest, RunPaths

#: Every engine artifact's manifest file; underscore-prefixed so Parquet
#: readers skip it.
PROVENANCE_FILE = '_provenance.json'


@runtime_checkable
class ArtifactStore(Protocol):
    """Storage operations required by the portable stage runner."""
    def prepare_run(self, paths: RunPaths) -> None:
        """Create the durable and scratch directories for a run."""

    def load_manifest(self, paths: RunPaths) -> RunManifest | None:
        """Load a manifest, or return ``None`` when the run is new."""

    def write_manifest(self, paths: RunPaths, manifest: RunManifest) -> None:
        """Atomically replace the durable manifest."""


class FilesystemArtifactStore:
    """Persist run state on a filesystem supporting atomic rename."""
    def prepare_run(self, paths: RunPaths) -> None:
        paths.scratch_dir.mkdir(parents=True, exist_ok=True)
        paths.artifact_dir.mkdir(parents=True, exist_ok=True)
        paths.diagnostics_dir.mkdir(parents=True, exist_ok=True)

    def load_manifest(self, paths: RunPaths) -> RunManifest | None:
        try:
            payload = paths.manifest_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        data = json.loads(payload)
        if not isinstance(data, dict):
            raise ValueError("run manifest must contain a JSON object")
        return RunManifest.from_dict(data)

    def write_manifest(self, paths: RunPaths, manifest: RunManifest) -> None:
        if manifest.run_id != paths.run_id or manifest.paths != paths:
            raise ValueError(
                "manifest identity and paths must match the write target")
        paths.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            manifest.to_dict(),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n"
        atomic_write_bytes(paths.manifest_path, payload.encode("utf-8"))


@dataclass(frozen=True)
class VitalArtifactSources:
    """Exact training outputs retained in a published model bundle.

    Publication is deliberately role-based instead of scanning a trainer
    output directory. This prevents trial state and diagnostic files from
    becoming durable model artifacts merely because they live beside the
    selected checkpoint.
    """

    selected_checkpoint: Path
    trainer_config: Path
    encoders: Path
    target_encoder: Path
    resolved_model_plan: Path
    graph_manifest: Path
    task_manifest: Path
    input_manifest: Path
    metrics: Path
    prediction_schema: Path

    def __post_init__(self) -> None:
        for role in self.roles():
            object.__setattr__(self, role, Path(getattr(self, role)))

    @classmethod
    def roles(cls) -> tuple[str, ...]:
        return tuple(cls.__dataclass_fields__)

    def as_dict(self) -> dict[str, Path]:
        return {role: getattr(self, role) for role in self.roles()}


@dataclass(frozen=True)
class PublishedArtifact:
    """Integrity metadata for one file in a published model bundle."""

    role: str
    relative_path: str
    size_bytes: int
    sha256: str

    def to_dict(self) -> dict[str, str | int]:
        return {
            "role": self.role,
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class VitalArtifactPublication:
    """Manifest returned after publishing a vital-artifact bundle."""

    destination: Path
    artifacts: tuple[PublishedArtifact, ...]

    schema_version: ClassVar[int] = 1

    def published_paths(self) -> dict[str, str]:
        """Return the durable path for each explicit source role."""

        return {
            artifact.role: str(self.destination / artifact.relative_path)
            for artifact in self.artifacts
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version":
            self.schema_version,
            "policy":
            "vital-training-artifacts-v1",
            "excluded_by_default":
            list(VitalArtifactPublicationPolicy.EXCLUDED_BY_DEFAULT),
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
        }


class VitalArtifactPublicationPolicy:
    """Atomically publish only the files needed to use a trained model."""

    MANIFEST_NAME = "publication-manifest.json"
    EXCLUDED_BY_DEFAULT = (
        "trial_logs",
        "intermediate_checkpoints",
        "tensorboard",
        "analysis",
    )
    _DESTINATIONS = {
        "selected_checkpoint": "model/checkpoint",
        "trainer_config": "model/config.json",
        "encoders": "model/encoders.json",
        "target_encoder": "model/target-encoder.json",
        "resolved_model_plan": "model/resolved-model-plan.json",
        "graph_manifest": "manifests/graph.json",
        "task_manifest": "manifests/task.json",
        "input_manifest": "manifests/input.json",
        "metrics": "metrics.json",
        "prediction_schema": "prediction-schema.json",
    }

    def publish(
        self,
        sources: VitalArtifactSources,
        destination: Path,
    ) -> VitalArtifactPublication:
        """Copy the exact allowlist into a new, atomic publication directory.

        The destination must not already exist. Requiring a new destination
        avoids merging an earlier publication's stale files into this one.
        """

        if not isinstance(sources, VitalArtifactSources):
            raise TypeError("sources must be VitalArtifactSources")
        destination = Path(destination)
        if destination.exists():
            raise FileExistsError(
                f"publication destination already exists: {destination}")

        source_files = sources.as_dict()
        for role, source in source_files.items():
            if not source.is_file():
                raise FileNotFoundError(
                    f"vital artifact {role!r} is not a file: {source}")

        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name}.",
                dir=destination.parent,
            ))
        try:
            artifacts = []
            for role, source in source_files.items():
                relative_path = self._DESTINATIONS[role]
                target = staging / relative_path
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                artifacts.append(
                    PublishedArtifact(
                        role=role,
                        relative_path=relative_path,
                        size_bytes=target.stat().st_size,
                        sha256=self._sha256(target),
                    ))

            publication = VitalArtifactPublication(
                destination=destination,
                artifacts=tuple(artifacts),
            )
            manifest_path = staging / self.MANIFEST_NAME
            manifest_path.write_text(
                json.dumps(
                    publication.to_dict(),
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                ) + "\n",
                encoding="utf-8",
            )
            os.replace(staging, destination)
            fsync_directory(destination.parent)
            return publication
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()


def verify_model_bundle(bundle: Path) -> str:
    """Verify fetched artifacts before parsing or staging a model bundle.

    Returns the bundle id, a SHA-256 over every verified file except
    ``_provenance.json``. Paths must stay inside the bundle after symlink
    resolution; declared sizes and SHA-256 digests must match. Older
    manifests may omit sizes or leave training-spec.json unlisted.
    """
    from .sources import layout
    from .sources.errors import PermanentSourceError

    bundle = Path(bundle).resolve()
    manifest_path = bundle / 'publication-manifest.json'
    if not manifest_path.resolve().is_relative_to(bundle):
        raise ValueError('Publication manifest is not below the bundle')
    if not manifest_path.is_file():
        raise ValueError(
            f"No publication-manifest.json in {bundle}, so what was downloaded cannot "
            f"be checked. It is not a published Kumo model bundle.")
    with open(manifest_path, encoding='utf-8') as handle:
        manifest = json.load(handle)

    artifacts = manifest.get('artifacts')
    problems: list[str] = []
    verified: list[tuple[str, str]] = []
    seen = set()
    claimed = set()
    if not artifacts:
        problems.append('the manifest names no artifacts')
    for entry in artifacts or []:
        relative = entry.get('relative_path')
        if not relative:
            problems.append('an artifact entry has no relative_path')
            continue
        expected = entry.get('sha256')
        if not expected:
            problems.append(f'{relative} has no sha256 to verify against')
            continue
        relative = str(relative)
        if relative.startswith('/') or relative.startswith('\\'):
            problems.append(f'{relative} is an absolute path')
            continue
        try:
            safe = layout.confined(relative)
        except PermanentSourceError:
            problems.append(f'{relative} is not a path below the bundle')
            continue
        if safe in seen:
            problems.append(f'{relative} is named more than once')
            continue
        seen.add(safe)
        role = str(entry.get('role') or '')
        if role and role in claimed:
            problems.append(f'the {role} role is claimed more than once')
            continue
        if role == 'training_spec' and safe != 'training-spec.json':
            problems.append(
                f'the training_spec role names {relative} rather than '
                f'training-spec.json')
            continue
        if role:
            claimed.add(role)
        path = bundle / safe
        if not path.resolve().is_relative_to(bundle):
            problems.append(f'{relative} is not a path below the bundle')
            continue
        if not path.is_file():
            problems.append(f'{relative} is missing')
            continue
        size = entry.get('size_bytes')
        if size is not None and path.stat().st_size != size:
            problems.append(f'{relative} does not match its size')
            continue
        if VitalArtifactPublicationPolicy._sha256(path) != expected:
            problems.append(f'{relative} does not match its digest')
            continue
        verified.append((safe, expected))
    if problems:
        raise ValueError(
            f"The model bundle at {bundle} did not survive the transfer: "
            f"{'; '.join(sorted(problems))}. Import it again.")

    spec_path = bundle / 'training-spec.json'
    if not spec_path.resolve().is_relative_to(bundle):
        raise ValueError('Training specification is not below the bundle')
    if 'training-spec.json' not in seen and spec_path.is_file():
        verified.append(('training-spec.json',
                         VitalArtifactPublicationPolicy._sha256(spec_path)))
    return _bundle_id(verified)


def bundle_id_from_manifest(bundle: Path) -> str:
    """The bundle id from the digests a freshly published bundle lists."""
    manifest_path = Path(bundle) / VitalArtifactPublicationPolicy.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    return _bundle_id((str(entry['relative_path']), str(entry['sha256']))
                      for entry in manifest['artifacts'])


def _bundle_id(pairs: Iterable[tuple[str, str]]) -> str:
    payload = json.dumps(
        sorted(pair for pair in pairs if pair[0] != PROVENANCE_FILE),
        separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def add_bundle_file(bundle_dir: Path, role: str, relative_path: str) -> None:
    """List one file already written in a bundle in its publication manifest."""
    bundle_dir = Path(bundle_dir)
    path = bundle_dir / relative_path
    manifest_path = bundle_dir / VitalArtifactPublicationPolicy.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest['artifacts'] = [
        entry for entry in manifest.get('artifacts') or []
        if entry.get('relative_path') != relative_path
        and entry.get('role') != role
    ] + [
        PublishedArtifact(
            role=role,
            relative_path=relative_path,
            size_bytes=path.stat().st_size,
            sha256=VitalArtifactPublicationPolicy._sha256(path),
        ).to_dict()
    ]
    _write_bundle_manifest(bundle_dir, manifest)


def _write_bundle_manifest(bundle_dir: Path,
                           manifest: Mapping[str, Any]) -> None:
    payload = json.dumps(manifest, indent=2, sort_keys=True) + '\n'
    atomic_write_bytes(
        bundle_dir / VitalArtifactPublicationPolicy.MANIFEST_NAME,
        payload.encode('utf-8'))


@runtime_checkable
class VitalArtifactPublisher(Protocol):
    """Environment publication boundary for one successful training run."""
    def publish(
        self,
        sources: VitalArtifactSources,
    ) -> Mapping[str, str]:
        """Publish explicit source roles and return their durable paths."""


@dataclass(frozen=True)
class FilesystemVitalArtifactPublisher:
    """Publish a vital bundle to one explicit filesystem destination."""

    destination: Path
    policy: VitalArtifactPublicationPolicy = field(
        default_factory=VitalArtifactPublicationPolicy)

    def publish(
        self,
        sources: VitalArtifactSources,
    ) -> Mapping[str, str]:
        publication = self.policy.publish(sources, self.destination)
        return publication.published_paths()


def resolve_vital_artifact_sources(
    training_input: Any,
    auto_trainer: Any,
    best_trainer: Any,
    metrics: Mapping[str, float],
) -> VitalArtifactSources:
    """Resolve the exact portable files required to reload a Kumo model."""

    bundle_dir = Path(training_input.task_dataset_bundle_dir)
    source_dir = Path(training_input.home_dir) / "publication-sources"
    source_dir.mkdir(parents=True, exist_ok=True)
    save_json(best_trainer.target_encoder,
              str(source_dir / "target-encoder.json"))
    from .automl.runtime import inherit_warm_start_model_plan
    from .model_plan import ModelPlan

    resolved_plan = copy.deepcopy(training_input.model_plan)
    selected_plan = load_json(
        str(Path(best_trainer.original_log_dir) / "model_plan.json"), ModelPlan)
    inherit_warm_start_model_plan(resolved_plan, selected_plan)
    save_json(resolved_plan, str(source_dir / "resolved-model-plan.json"))
    save_json(training_input.task.to_dict(), str(source_dir / "input.json"))
    save_json(dict(metrics), str(source_dir / "metrics.json"))
    save_json(
        {
            "task_type": training_input.task.task_type.value,
            "target_column": training_input.task.train_table.target_column,
        },
        str(source_dir / "prediction-schema.json"),
    )
    manifest = json.loads((bundle_dir / "manifest.json").read_text())
    graph_manifest = bundle_dir / manifest["graph_runtime_metadata_path"]
    # Evaluation changes fit_mode; publish the mode hosted prediction and
    # warm start select, rather than whichever mode was evaluated last.
    best_trainer._set_fit_mode_to_use()
    return VitalArtifactSources(
        selected_checkpoint=local_path(best_trainer.best_ckpt),
        trainer_config=Path(best_trainer.original_log_dir) / "config.json",
        encoders=Path(auto_trainer.log_dir) / "encoders.json",
        target_encoder=source_dir / "target-encoder.json",
        resolved_model_plan=source_dir / "resolved-model-plan.json",
        graph_manifest=graph_manifest,
        task_manifest=bundle_dir / "manifest.json",
        input_manifest=source_dir / "input.json",
        metrics=source_dir / "metrics.json",
        prediction_schema=source_dir / "prediction-schema.json",
    )


_BUNDLE_LAYOUT = {
    'trainer_config': 'model/config.json',
    'selected_checkpoint': 'model/checkpoint/step=0.ckpt',
    'encoders': 'model/encoders.json',
    'target_encoder': 'model/target-encoder.json',
    'resolved_model_plan': 'model/resolved-model-plan.json',
    'graph_manifest': 'manifests/graph.json',
    'task_manifest': 'manifests/task.json',
    'input_manifest': 'manifests/input.json',
    'prediction_schema': 'prediction-schema.json',
    'metrics': 'metrics.json',
}


def stage_model_bundle(artifacts: Mapping[str, str], dest: Path) -> Path:
    r"""Copies a trained run's artifacts into the layout scoring expects.

    Args:
        artifacts: Role to path, as a completed training run recorded them.
        dest: Directory to build the bundle under.

    Returns:
        The model root to hand to the trainer.

    Raises:
        ValueError: If the checkpoint is missing, since nothing can be scored
            without it.
    """
    for role, relative in _BUNDLE_LAYOUT.items():
        source = artifacts.get(role)
        if not source or not Path(source).exists():
            continue  # optional roles may be absent
        target = dest / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    model_root = dest / 'model'
    checkpoint = dest / _BUNDLE_LAYOUT['selected_checkpoint']
    if not checkpoint.exists():
        raise ValueError(
            f"The training run recorded no checkpoint, so there is no model "
            f"to score with. Roles present: {sorted(artifacts) or 'none'}.")
    return model_root
