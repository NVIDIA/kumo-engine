# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import shutil
from pathlib import Path

import pytest
import torch
import yaml

from tfm.cli.recipe import run_command
from tfm.runtime import common


@pytest.fixture
def config_path(tmp_path):
    pytest.importorskip("kumo_core.modeling.gnn.hetero")
    config = yaml.safe_load(Path("recipes/gnn_encoder/config.smoke.yaml").read_text())
    config["training"]["max_steps"] = 1
    config["sequence_training"]["max_steps"] = 1
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def train(config_path, run_dir):
    return run_command("train", config_path=config_path, run_dir=run_dir, device="cpu")


def test_same_shaped_checkpoint_from_another_seed_is_rejected_before_loading(
    config_path, tmp_path, monkeypatch
):
    first = train(config_path, tmp_path / "first")
    other = yaml.safe_load(config_path.read_text())
    other["seed"] += 1
    second_config = tmp_path / "second.yaml"
    second_config.write_text(yaml.safe_dump(other))
    second = train(second_config, tmp_path / "second")
    a = torch.load(first / "model.pt", weights_only=False)
    b = torch.load(second / "model.pt", weights_only=False)
    assert {k: v.shape for k, v in a["state_dict"].items()} == {
        k: v.shape for k, v in b["state_dict"].items()
    }
    assert a["metadata"]["config_sha256"] != b["metadata"]["config_sha256"]
    original_sidecar = (first / "model.manifest.json").read_bytes()
    shutil.copyfile(second / "model.pt", first / "model.pt")

    def forbidden(*args, **kwargs):
        raise AssertionError("Installed weights before validating checkpoint identity")

    monkeypatch.setattr(torch.nn.Module, "load_state_dict", forbidden)
    with pytest.raises(ValueError, match="config identity"):
        run_command(
            "infer", config_path=config_path, run_dir=first, device="cpu", reuse_checkpoint=True
        )
    assert (first / "model.manifest.json").read_bytes() == original_sidecar
    assert not (first / "embeddings.npz").exists()


@pytest.mark.parametrize("sidecar", ["missing", "corrupt"])
def test_sidecars_are_regenerated_from_the_loaded_weight_metadata(config_path, tmp_path, sidecar):
    out = train(config_path, tmp_path / "run")
    expected = torch.load(out / "model.pt", weights_only=False)["metadata"]
    if sidecar == "missing":
        (out / "model.manifest.json").unlink()
    else:
        (out / "model.manifest.json").write_text("this is not JSON")
    run_command("infer", config_path=config_path, run_dir=out, device="cpu", reuse_checkpoint=True)
    canonical = json.loads(json.dumps(expected))
    assert json.loads((out / "model.manifest.json").read_text()) == canonical
    assert json.loads((out / "checkpoint_manifest.json").read_text()) == canonical


@pytest.mark.parametrize(
    "corruption",
    ["old_format", "schema", "split", "engine", "missing_metadata", "event_key_encoding"],
)
def test_embedded_metadata_is_validated_before_any_state_installation(
    config_path, tmp_path, monkeypatch, corruption
):
    out = train(config_path, tmp_path / "run")
    payload = torch.load(out / "model.pt", weights_only=False)
    if corruption == "old_format":
        payload["metadata"]["format"] = "unsupported-format"
    elif corruption == "schema":
        payload["metadata"]["graph_schema"]["hidden_size"] += 1
    elif corruption == "split":
        payload["metadata"]["split"]["probe_owners"]["train"] = ["wrong_owner"]
    elif corruption == "engine":
        payload["metadata"]["kumo_engine_version"] = "wrong_engine"
    elif corruption == "event_key_encoding":
        payload["metadata"]["split"]["sampling"].pop("event_key_encoding")
    else:
        payload.pop("metadata")
    torch.save(payload, out / "model.pt")

    def forbidden(*args, **kwargs):
        raise AssertionError("Installed weights before validating checkpoint metadata")

    monkeypatch.setattr(torch.nn.Module, "load_state_dict", forbidden)
    with pytest.raises(
        ValueError, match="Unsupported checkpoint format|graph_schema|split|kumo_engine_version"
    ):
        run_command(
            "infer", config_path=config_path, run_dir=out, device="cpu", reuse_checkpoint=True
        )


@pytest.mark.parametrize("has_git", [True, False])
def test_training_saves_from_another_working_directory(config_path, tmp_path, monkeypatch, has_git):
    expected = common.source_provenance()
    if not has_git:
        root = tmp_path / "installed"
        (root / "src/tfm").mkdir(parents=True)
        (root / "pyproject.toml").write_text("")
        monkeypatch.setattr(common, "repo_root", lambda: root)
        expected = {"source_revision": None, "source_dirty": None}
    monkeypatch.chdir(tmp_path)
    out = train(config_path.resolve(), tmp_path / "outside")
    assert (out / "model.pt").is_file()
    metadata = torch.load(out / "model.pt", weights_only=False)["metadata"]
    assert {key: metadata[key] for key in expected} == expected


def test_future_hybrid_retains_historical_and_matched_supervised_controls(tmp_path, monkeypatch):
    pytest.importorskip("kumo_core.modeling.gnn.hetero")
    from tfm.layers.gnn import GNNSequenceModel, SequenceBinaryModel
    from tfm.layers.profile_event_decoder import ProfileEventDecoderModel
    from tfm.runtime import graph as runtime

    config = yaml.safe_load(Path("recipes/gnn_sequence/config.pretrain_smoke.yaml").read_text())
    config["training"]["max_steps"] = config["sequence_training"]["max_steps"] = 1
    path = tmp_path / "future.yaml"
    path.write_text(yaml.safe_dump(config))
    real = runtime.fit_selected
    observed = {}

    def record(model, batches, validation, settings, **kwargs):
        if isinstance(model, GNNSequenceModel):
            observed["hybrid"] = (
                {k: v.clone() for k, v in model.state_dict().items()},
                batches,
                validation,
                settings,
                kwargs["seed"],
            )
        elif isinstance(model, SequenceBinaryModel):
            state, graph_batches, graph_validation, original_settings, seed = observed["hybrid"]
            for k, v in model.state_dict().items():
                torch.testing.assert_close(v, state[k], rtol=0, atol=0)
            assert settings == original_settings and kwargs["seed"] == seed
            for left, right in zip(batches, graph_batches, strict=True):
                assert left.history is right.history and left.labels is right.labels
                assert left.anchors == right.history_anchors
            for left, right in zip(validation, graph_validation, strict=True):
                assert left.history is right.history and left.labels is right.labels
            observed["matched"] = True
        elif isinstance(model, ProfileEventDecoderModel):
            observed["historical"] = True
        return real(model, batches, validation, settings, **kwargs)

    monkeypatch.setattr(runtime, "fit_selected", record)
    out = run_command("smoke", config_path=path, run_dir=tmp_path / "future", device="cpu")
    assert observed["matched"] and observed["historical"]
    metadata = json.loads((out / "model.manifest.json").read_text())
    assert {"sequence", "sequence_baseline", "sequence_supervised"} <= metadata["stages"].keys()
    metrics = json.loads((out / "metrics.json").read_text())
    assert {
        "sequence_only_embedding",
        "sequence_supervised_embedding",
        "sequence_supervised_task_head",
        "task_head",
    } <= metrics.keys()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_future_control_bundle_reuse_places_every_arm_on_cuda(tmp_path):
    import numpy as np

    pytest.importorskip("kumo_core.modeling.gnn.hetero")
    config = yaml.safe_load(Path("recipes/gnn_sequence/config.pretrain_smoke.yaml").read_text())
    config["training"]["max_steps"] = config["sequence_training"]["max_steps"] = 1
    path = tmp_path / "future_cuda.yaml"
    path.write_text(yaml.safe_dump(config))
    out = run_command("smoke", config_path=path, run_dir=tmp_path / "cuda", device="cuda")
    before = dict(np.load(out / "embeddings.npz"))
    run_command("evaluate", config_path=path, run_dir=out, device="cuda", reuse_checkpoint=True)
    for key, value in dict(np.load(out / "embeddings.npz")).items():
        if value.dtype.kind == "f":
            np.testing.assert_allclose(value, before[key], rtol=1e-5, atol=1e-6)
        else:
            np.testing.assert_array_equal(value, before[key])
    assert "sequence_supervised_task_head" in json.loads((out / "metrics.json").read_text())
