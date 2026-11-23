# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reload the T01 checkpoint in a fresh interpreter and continue training."""

import subprocess
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pyarrow.parquet as pq
import torch

from e2e.datasets import bind_tables
from e2e.flows.common import BatchDigest
from kumo.pytorch import api
from tests.e2e.streaming_training import CountSAGE, count_loss
from tests.test_inference_lifecycle import ROOT, assert_same


def _train_batches(task):
    return task.batches(split="train", batch_size=1, output_format="model", shuffle=True, seed=17)


def _update(batch, model, optimizer):
    optimizer.zero_grad()
    loss = count_loss(model, batch)
    loss.backward()
    optimizer.step()
    return batch.metadata["sample_ids"], loss.item()


def _finish(batches, model, optimizer):
    initial = torch.nn.utils.parameters_to_vector(model.parameters()).detach()
    remaining = [_update(batch, model, optimizer) for batch in batches]
    assert remaining, "the checkpoint must have a nonempty epoch suffix"
    assert not torch.equal(initial, torch.nn.utils.parameters_to_vector(model.parameters()))
    return (
        remaining,
        model.state_dict(),
        optimizer.state_dict(),
        batches.state_dict(),
        torch.get_rng_state(),
    )


@torch.no_grad()
def _predictions(task, model, *, inference):
    scores, inputs = {}, BatchDigest()
    for batch in task.batches(split=("valid", "test"), batch_size=4, output_format="model"):
        if inference:
            assert not batch.targets
        inputs.update(batch)
        values = model(batch)
        assert values.isfinite().all()
        for sample, score in zip(batch.metadata["sample_ids"], values.tolist(), strict=True):
            assert sample not in scores
            scores[sample] = score
    return scores, inputs.hexdigest()


def run_lifecycle(task, data, table, index, work):
    model = CountSAGE(task)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0)
    batches = _train_batches(task)
    _update(next(batches), model, optimizer)
    task.to_artifact().save(work / "artifact")
    api.save_weights(work / "model.pt", model.state_dict(), task.fitted_state)
    scores, inputs = _predictions(task, model.eval(), inference=False)
    ids = pq.read_table(
        table.source.path, columns=["sample_id"], filters=[("split", "in", ["valid", "test"])]
    )["sample_id"].to_pylist()
    assert scores and scores.keys() == set(ids)
    torch.save(
        dict(
            tables={name: source.path for name, source in data.tables.items()},
            anchors=str(Path(table.source.path).parent),
            index=str(index.resolve()),
            threads=torch.get_num_threads(),
            optimizer=optimizer.state_dict(),
            cursor=batches.state_dict(),
            rng=torch.get_rng_state(),
            predictions=(scores, inputs),
        ),
        work / "checkpoint.pt",
    )
    torch.save(_finish(batches, model.train(), optimizer), work / "expected.pt")
    subprocess.run([sys.executable, "-m", __name__, str(work.resolve())], cwd=ROOT, check=True)


@patch("kumo.pytorch.assembly._fit_inputs", side_effect=AssertionError("input encoders refitted"))
def _reload(work, _fit_inputs):
    checkpoint = torch.load(work / "checkpoint.pt", weights_only=True)
    torch.set_num_threads(checkpoint["threads"])
    artifact = api.Artifact.load(work / "artifact")
    _, data, table = bind_tables(
        Path(checkpoint["anchors"]),
        artifact.contract,
        "viewer_id",
        ("count",),
        source_paths=checkpoint["tables"],
    )
    engine = api.DiskGraphEngine(index=checkpoint["index"], ingest="never")
    fitted = artifact.fitted_state
    weights = api.load_weights(work / "model.pt", fitted)
    with (
        patch("kumo.pytorch.assembly._fit_targets", side_effect=AssertionError("inference fit")),
        api.prepare(artifact.contract, data, engine, inference=True) as plan,
        plan.materialize(replace(table, targets={}), fitted=fitted) as task,
    ):
        model = CountSAGE(task)
        model.load_state_dict(weights)
        assert_same(_predictions(task, model.eval(), inference=True), checkpoint["predictions"])
    expected = torch.load(work / "expected.pt", weights_only=True)
    with api.prepare(artifact.contract, data, engine) as plan:
        for state, split in ((fitted, None), (fitted.inputs, "train")):
            with plan.materialize(table, fitted=state, fit_split=split) as task:
                assert task.fitted_state.inputs.fingerprint == fitted.inputs.fingerprint
                model = CountSAGE(task)
                model.load_state_dict(weights)
                optimizer = torch.optim.AdamW(model.parameters())
                optimizer.load_state_dict(deepcopy(checkpoint["optimizer"]))
                batches = _train_batches(task)
                batches.load_state_dict(checkpoint["cursor"])
                torch.set_rng_state(checkpoint["rng"])
                assert_same(_finish(batches, model, optimizer), expected)


if __name__ == "__main__":
    _reload(Path(sys.argv[1]))
