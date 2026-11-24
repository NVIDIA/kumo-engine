# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The lab notebooks' assets: smoke mode, local overrides, and verified downloads."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

import numpy as np
import pytest
import torch

from tfm.config import RecipeConfig, load_recipe_config
from tfm.contracts.errors import (
    IncompatibleCompositionError,
    InsufficientLabelSupportError,
    SchemaError,
)
from tfm.data.synthetic.lab_book import lab_book_records
from tfm.evaluation.surprise import extract_surprise
from tfm.evaluation.tabular import evaluate_raw_embedding_arms
from tfm.extras.capability import extra_available
from tfm.inference.embedding import extract_embeddings
from tfm.runtime.common import repo_root
from tfm.runtime.evaluate import evaluate_run
from tfm.runtime.lab import (
    BOOK_ENV,
    CHECKPOINT_ENV,
    LAB_CONFIG,
    LAB_TASKS,
    LAB_XGBOOST_PARAMS,
    SMOKE_ENV,
    SMOKE_USERS,
    LabDownload,
    fetch,
    fetch_lab_assets,
    lab_tasks,
    load_lab_model,
    prepare_lab_data,
)
from tfm.training.checkpointing import save_checkpoint


def test_the_lab_book_is_deterministic() -> None:
    assert lab_book_records(12, seed=3) == lab_book_records(12, seed=3)
    assert lab_book_records(12, seed=3) != lab_book_records(12, seed=4)


@pytest.fixture(scope="module")
def smoke(tmp_path_factory: pytest.TempPathFactory):  # noqa: ANN201
    root = tmp_path_factory.mktemp("lab")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv(SMOKE_ENV, "1")
        patch.delenv(BOOK_ENV, raising=False)
        assets = fetch_lab_assets("t", run_root=root)
        data = prepare_lab_data(assets)
    return assets, data


def test_smoke_mode_runs_every_task_on_cpu_without_a_download(smoke) -> None:  # noqa: ANN001
    assets, data = smoke
    assert assets.smoke and assets.checkpoint_dir is None
    assert assets.device == torch.device("cpu")
    assert "SMOKE MODE" in assets.describe()
    assert len(data.profiles) == SMOKE_USERS
    tasks = lab_tasks(assets, data)
    assert tuple(tasks) == LAB_TASKS
    for task in tasks.values():
        assert 0 < task.y_train.sum() < len(task.y_train)
        assert 0 < task.y_eval.sum() < len(task.y_eval)
        train_users = {s.entity_id for s in task.train_sequences}
        assert train_users.isdisjoint(s.entity_id for s in task.eval_sequences)


def test_smoke_mode_uses_the_lab_checkpoints_config(smoke) -> None:  # noqa: ANN001
    assets, _data = smoke
    lab = load_recipe_config(repo_root() / LAB_CONFIG)
    assert assets.config.model == lab.model
    assert assets.config.fraud_task.finetune.rank == lab.fraud_task.finetune.rank == 16


def test_a_checkpoint_with_a_2_0_0a1_config_loads_migrated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, book = tmp_path / "ckpt", tmp_path / "book"
    bundle.mkdir()
    book.mkdir()
    shutil.copy(
        Path("tests/fixtures/configs/v2.0.0a1_blueprint_v2_lab_tuned.yaml"), bundle / "config.yaml"
    )
    monkeypatch.delenv(SMOKE_ENV, raising=False)
    monkeypatch.setenv(BOOK_ENV, str(book))
    monkeypatch.setenv(CHECKPOINT_ENV, str(bundle))

    config = fetch_lab_assets("t", run_root=tmp_path).config

    candidate = load_recipe_config(repo_root() / LAB_CONFIG)
    assert config.model == candidate.model
    assert config.training.max_steps == candidate.training.max_steps == 3000
    assert config.fraud_task.finetune == candidate.fraud_task.finetune
    assert config.data.path == str(book)


def test_a_task_the_book_does_not_label_is_refused(smoke) -> None:  # noqa: ANN001
    assets, data = smoke
    with pytest.raises(InsufficientLabelSupportError, match="no 'aml' labels"):
        lab_tasks(assets, data, ("aml",))


def test_a_local_checkpoint_restores_its_weights_and_tokenizer(
    smoke,  # noqa: ANN001
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assets, data = smoke
    trained = load_lab_model(assets, data)
    with torch.no_grad():
        for parameter in trained.parameters():
            parameter.add_(0.5)
    bundle = tmp_path / "ckpt"
    bundle.mkdir()
    save_checkpoint(trained, bundle / "model.pt", metadata={})
    (bundle / "tokenizer.json").write_text(json.dumps(data.tokenizer.state_dict()))
    manifest = {"vocab_size": data.tokenizer.vocab_size}
    (bundle / "checkpoint_manifest.json").write_text(json.dumps(manifest))

    monkeypatch.delenv(SMOKE_ENV, raising=False)
    monkeypatch.setenv(BOOK_ENV, str(assets.book))
    monkeypatch.setenv(CHECKPOINT_ENV, str(bundle))
    local = fetch_lab_assets("t", n_users=SMOKE_USERS, run_root=tmp_path)
    assert not local.smoke and local.checkpoint_dir == bundle
    reloaded = prepare_lab_data(local)
    assert reloaded.tokenizer.state_dict() == data.tokenizer.state_dict()
    restored = load_lab_model(local, reloaded).cpu()
    for a, b in zip(restored.state_dict().values(), trained.state_dict().values(), strict=True):
        torch.testing.assert_close(a, b.cpu())
    assert not torch.equal(
        next(load_lab_model(local, reloaded, pretrained=False).cpu().parameters()),
        next(restored.parameters()),
    )

    (bundle / "checkpoint_manifest.json").write_text(json.dumps({"vocab_size": 3}))
    with pytest.raises(IncompatibleCompositionError, match="not saved together"):
        load_lab_model(local, reloaded)


@pytest.mark.skipif(not extra_available("xgboost"), reason="needs the baselines extra")
def test_the_recipes_evaluation_scores_what_the_lab_notebook_scores(smoke) -> None:  # noqa: ANN001
    """`evaluate_run` under the tuned config and notebook 04's own steps give equal arms.

    The architecture is shrunk so the test runs in seconds on CPU; the protocol
    (tasks, readouts, fixed XGBoost setting, fine-tuned tasks) is the tuned config's.
    """
    assets, data = smoke
    raw = assets.config.model_dump(mode="json")
    granularity = raw["model"]["composition"]["granularity"]
    granularity.update(profile_depth=1, event_depth=1)
    granularity["decoder"].update(depth=1, hidden_size=32, num_heads=2)
    raw["fraud_task"]["finetune"].update(max_steps=2, eval_every=1)
    config = RecipeConfig.model_validate(raw)
    assets = dataclasses.replace(assets, config=config)
    model = load_lab_model(assets, data)
    metrics, _manifest, finetune = evaluate_run(model, data, config, assets.device)

    batch_size = config.evaluation.inference_batch_size
    profiles = {p.entity_id: p for p in data.profiles}
    baselines = config.evaluation.tabular_baselines
    compared = 0
    for name, task in lab_tasks(assets, data).items():
        prefix = "history_window" if name == "fraud_txn" else name
        if f"{prefix}_xgboost_raw" not in metrics:
            continue
        sides = [(task.train_batch, "train"), (task.eval_batch, "eval")]
        last = {
            side: extract_embeddings(model, batch, device=assets.device, minibatch_size=batch_size)
            for batch, side in sides
        }
        surprise = {
            side: extract_surprise(
                model, batch, data.tokenizer, device=assets.device, minibatch_size=batch_size
            )
            for batch, side in sides
        }
        views = {
            "": (last["train"].vectors, last["eval"].vectors),
            "surprise": tuple(
                np.hstack([last[side].vectors, surprise[side].vectors])
                for side in ("train", "eval")
            ),
        }
        notebook, _ = evaluate_raw_embedding_arms(
            task.train_sequences,
            task.y_train,
            task.eval_sequences,
            task.y_eval,
            profiles=profiles,
            embedding_views=views,
            xgboost_params=LAB_XGBOOST_PARAMS,
            seed=config.seed,
            top_k=baselines.top_k,
            coded_keys=tuple(config.features.categorical_keys),
            n_jobs=baselines.n_jobs,
            k=config.evaluation.precision_at_k,
        )
        for arm, scored in notebook.items():
            assert metrics[f"{prefix}_{arm}"] == scored, (name, arm)
        compared += 1
    assert compared >= 2
    assert "history_window_xgboost_raw_embedding:surprise" in metrics
    assert "history_window" in finetune["tasks"] and "single_event" not in finetune["tasks"]
    assert f"{finetune['arm']}_history_window" in metrics


def _archive(tmp_path: Path, folder: str) -> Path:
    source = tmp_path / "src" / folder
    source.mkdir(parents=True)
    (source / "payload.txt").write_text("synthetic")
    archive = tmp_path / f"{folder}.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(source, arcname=folder)
    return archive


def test_a_download_is_verified_extracted_and_reused(tmp_path: Path) -> None:
    archive = _archive(tmp_path, "bundle")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    download = LabDownload(url=archive.as_uri(), folder="bundle", sha256=digest)
    target = fetch(download, tmp_path / "cache")
    assert (target / "payload.txt").read_text() == "synthetic"
    archive.unlink()
    assert fetch(download, tmp_path / "cache") == target

    tampered = LabDownload(
        url=_archive(tmp_path / "t", "other").as_uri(), folder="other", sha256="0"
    )
    with pytest.raises(SchemaError, match="SHA-256"):
        fetch(tampered, tmp_path / "cache")
    assert not (tmp_path / "cache" / "other").exists()


def test_an_unhosted_asset_says_how_to_run_without_it(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=SMOKE_ENV):
        fetch(LabDownload(url="https://TODO.example.com/x.tar.gz", folder="x"), tmp_path)
