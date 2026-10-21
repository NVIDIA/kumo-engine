# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from kumo_core.artifacts import (
    FilesystemArtifactStore,
    FilesystemVitalArtifactPublisher,
    VitalArtifactPublicationPolicy,
    VitalArtifactSources,
    add_bundle_file,
    bundle_id_from_manifest,
    bundle_task_type,
    resolve_vital_artifact_sources,
    verify_model_bundle,
)
from kumo_core.config import NativeRuntimeConfig
from kumo_core.manifest import RunManifest, RunPaths


def test_filesystem_store_writes_and_replaces_manifest_atomically(
        tmp_path: Path) -> None:
    config = NativeRuntimeConfig(tmp_path / "scratch", tmp_path / "artifacts")
    paths = RunPaths.from_config(config, "test-run")
    store = FilesystemArtifactStore()
    store.prepare_run(paths)
    manifest = RunManifest.create("test-run", paths, ("train", ))

    store.write_manifest(paths, manifest)
    store.write_manifest(paths, manifest)

    assert store.load_manifest(paths) == manifest
    assert json.loads(paths.manifest_path.read_text())["schema_version"] == 1
    assert list(paths.artifact_dir.glob("*.tmp")) == []


def test_store_rejects_manifest_for_a_different_run(tmp_path: Path) -> None:
    config = NativeRuntimeConfig(tmp_path / "scratch", tmp_path / "artifacts")
    paths = RunPaths.from_config(config, "target")
    other_paths = RunPaths.from_config(config, "other")
    manifest = RunManifest.create("other", other_paths, ("train", ))

    with pytest.raises(ValueError, match="identity"):
        FilesystemArtifactStore().write_manifest(paths, manifest)


def _vital_sources(
        tmp_path: Path) -> tuple[
            VitalArtifactSources,
            dict[str, bytes],
        ]:
    source_dir = tmp_path / "trainer-output"
    values = {
        "selected_checkpoint": b"selected checkpoint\n",
        "trainer_config": b'{"trainer": "diskgraph"}\n',
        "encoders": b'{"age": "numerical"}\n',
        "target_encoder": b'{"target": "categorical"}\n',
        "resolved_model_plan": b'{"model": "graph-transformer"}\n',
        "graph_manifest": b'{"tables": ["users"]}\n',
        "task_manifest": b'{"task_type": "binary_classification"}\n',
        "input_manifest": b'{"snapshot": "snapshot-1"}\n',
        "metrics": b'{"accuracy": 0.75}\n',
        "prediction_schema": b'{"prediction": "float"}\n',
    }
    paths = {}
    for role, value in values.items():
        path = source_dir / "vital" / role
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
        paths[role] = path

    # These files deliberately live beside the selected outputs. Publication
    # must not discover them by walking the trainer directory.
    for relative_path in (
            "trials/trial-7.log",
            "checkpoints/epoch-2.ckpt",
            "tensorboard/events.out.tfevents",
            "analysis/feature-importance.json",
    ):
        path = source_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("diagnostic\n", encoding="utf-8")

    return VitalArtifactSources(**paths), values


def test_vital_policy_publishes_only_allowlisted_artifacts(
        tmp_path: Path) -> None:
    sources, values = _vital_sources(tmp_path)
    destination = tmp_path / "published"

    publication = VitalArtifactPublicationPolicy().publish(sources, destination)

    expected_paths = {
        "model/checkpoint",
        "model/config.json",
        "model/encoders.json",
        "model/target-encoder.json",
        "model/resolved-model-plan.json",
        "manifests/graph.json",
        "manifests/task.json",
        "manifests/input.json",
        "metrics.json",
        "prediction-schema.json",
    }
    published_paths = {
        str(path.relative_to(destination))
        for path in destination.rglob("*") if path.is_file()
    }
    assert published_paths == expected_paths | {"publication-manifest.json"}

    artifacts_by_role = {
        artifact.role: artifact
        for artifact in publication.artifacts
    }
    assert set(artifacts_by_role) == set(VitalArtifactSources.roles())
    for role, artifact in artifacts_by_role.items():
        target = destination / artifact.relative_path
        assert target.read_bytes() == values[role]
        assert artifact.size_bytes == len(values[role])
        assert artifact.sha256 == hashlib.sha256(values[role]).hexdigest()

    manifest = json.loads(
        (destination / "publication-manifest.json").read_text())
    assert manifest["schema_version"] == 1
    assert manifest["excluded_by_default"] == [
        "trial_logs",
        "intermediate_checkpoints",
        "tensorboard",
        "analysis",
    ]


def test_filesystem_publisher_returns_exact_published_role_paths(
        tmp_path: Path) -> None:
    sources, _ = _vital_sources(tmp_path)
    destination = tmp_path / "published"

    paths = FilesystemVitalArtifactPublisher(destination).publish(sources)

    assert set(paths) == set(VitalArtifactSources.roles())
    assert paths["selected_checkpoint"] == str(destination / "model/checkpoint")
    assert all(Path(path).is_file() for path in paths.values())


def test_vital_policy_requires_every_allowlisted_file(tmp_path: Path) -> None:
    sources, _ = _vital_sources(tmp_path)
    sources.metrics.unlink()
    destination = tmp_path / "published"

    with pytest.raises(FileNotFoundError, match="'metrics' is not a file"):
        VitalArtifactPublicationPolicy().publish(sources, destination)

    assert not destination.exists()


def test_vital_policy_never_merges_into_existing_publication(
        tmp_path: Path) -> None:
    sources, _ = _vital_sources(tmp_path)
    destination = tmp_path / "published"
    destination.mkdir()
    (destination / "stale-analysis.json").write_text("{}")

    with pytest.raises(FileExistsError, match="already exists"):
        VitalArtifactPublicationPolicy().publish(sources, destination)

    assert (destination / "stale-analysis.json").is_file()


@pytest.mark.parametrize('refit', [False, True, 'full'])
def test_vital_source_resolution_is_core_owned(tmp_path: Path, refit) -> None:
    from kumo_core.train.trainer import Trainer
    from kumo_core.training.config import FitMode

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "graph.json").write_text("{}", encoding="utf-8")
    (bundle / "manifest.json").write_text(
        json.dumps({"graph_runtime_metadata_path": "graph.json"}),
        encoding="utf-8",
    )
    automl = tmp_path / "automl"
    trainer = tmp_path / "trainer"
    automl.mkdir()
    trainer.mkdir()
    (automl / "encoders.json").write_text("{}", encoding="utf-8")
    (trainer / "config.json").write_text("{}", encoding="utf-8")
    from kumo_core.io.json import load_json, save_json
    from kumo_core.model_plan import ModelPlan
    selected_plan = ModelPlan()
    selected_plan.model_architecture.channels = [16]
    save_json(selected_plan, str(trainer / "model_plan.json"))
    checkpoints = {mode: trainer / f"{mode.value}.ckpt" for mode in FitMode}
    for mode, path in checkpoints.items():
        path.write_bytes(mode.value.encode())

    class SourceTrainer(SimpleNamespace):
        _set_fit_mode_to_use = Trainer._set_fit_mode_to_use
        published_fit_mode = Trainer.published_fit_mode

        @property
        def best_ckpt(self):
            return str(checkpoints[self.fit_mode])

    selected_mode = (FitMode.refit_full if refit == 'full' else
                     FitMode.refit_trainval if refit is True else FitMode.fit)
    training_input = SimpleNamespace(
        task_dataset_bundle_dir=str(bundle),
        home_dir=str(tmp_path / "home"),
        model_plan=ModelPlan(),
        task=SimpleNamespace(
            task_type=SimpleNamespace(value="binary_classification"),
            train_table=SimpleNamespace(target_column="TARGET"),
            to_dict=lambda: {"task": "resolved"},
        ),
    )

    sources = resolve_vital_artifact_sources(
        training_input,
        SimpleNamespace(log_dir=str(automl)),
        SourceTrainer(
            fit_mode=FitMode.fit,
            refit_cfg=SimpleNamespace(fitted=True, refitted_trainval=bool(refit),
                                      refitted_full=refit == 'full'),
            original_log_dir=str(trainer),
            target_encoder={"target": "categorical"},
        ),
        {"test_accuracy": 0.8},
    )

    resolved_plan = load_json(str(sources.resolved_model_plan), ModelPlan)
    assert resolved_plan.model_architecture.channels == [16]
    assert training_input.model_plan.model_architecture.channels != [16]
    assert sources.selected_checkpoint == checkpoints[selected_mode]
    assert sources.graph_manifest == bundle / "graph.json"
    assert json.loads(sources.metrics.read_text()) == {"test_accuracy": 0.8}



def test_bundle_id_ignores_provenance_only(tmp_path: Path) -> None:
    sources, _ = _vital_sources(tmp_path)
    bundle = tmp_path / "bundle"
    VitalArtifactPublicationPolicy().publish(sources, bundle)
    bundle_id = verify_model_bundle(bundle)

    for payload in ('{"a": 1}', '{"a": 2}'):
        (bundle / "provenance.json").write_text(payload)
        add_bundle_file(bundle, "provenance", "provenance.json")
        assert verify_model_bundle(bundle) == bundle_id
        assert bundle_id_from_manifest(bundle) == bundle_id
    roles = [
        entry["role"] for entry in json.loads(
            (bundle / "publication-manifest.json").read_text())["artifacts"]
    ]
    assert roles.count("provenance") == 1

    (bundle / "metrics.json").write_text('{"auc": 0}')
    add_bundle_file(bundle, "metrics", "metrics.json")
    assert verify_model_bundle(bundle) != bundle_id


def test_bundle_task_type_reads_the_prediction_schema(tmp_path: Path) -> None:
    schema = tmp_path / "prediction-schema.json"
    schema.write_text('{"task_type": "regression"}')
    assert bundle_task_type({"prediction_schema": str(schema)}).value == (
        "regression")
