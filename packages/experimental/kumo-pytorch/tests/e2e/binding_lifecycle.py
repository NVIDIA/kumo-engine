# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reload the package task and optimizer in a fresh interpreter and continue training."""

import json
import os
import subprocess
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pyarrow.parquet as pq
import torch

from e2e.flows.common import BatchDigest
from kumo.pytorch import api
from tests.e2e.binding_model import make_model
from tests.test_inference_lifecycle import ROOT, assert_same


def _train_batches(task):
    return task.batches(split="train", batch_size=1, output_format="model", shuffle=True, seed=17)


def _update(batch, model, optimizer):
    optimizer.zero_grad()
    loss = model.loss(batch)
    loss.backward()
    optimizer.step()
    return batch.metadata["sample_ids"], loss.item()


def _finish(batches, model, optimizer, evidence=None):
    initial = torch.nn.utils.parameters_to_vector(model.parameters()).detach()
    remaining = []
    for batch in batches:
        remaining.append(_update(batch, model, optimizer))
        if evidence is not None:
            evidence.observe(batch)
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
def _predictions(task, model, *, inference, evidence=None):
    scores, inputs = {}, BatchDigest()
    for split in ("valid", "test"):
        for batch in task.batches(split=split, batch_size=4, output_format="model"):
            if inference:
                assert not batch.targets
            inputs.update(batch)
            values = model(batch)
            if evidence is not None:
                evidence.observe(batch)
            assert values.isfinite().all()
            for sample, score in zip(batch.metadata["sample_ids"], values.detach(), strict=True):
                assert sample not in scores
                scores[sample] = score
    return scores, inputs.hexdigest()


def run_lifecycle(task, data, table, index, work, evidence, *, binding=None, raw_table=None):
    raw_table = table if raw_table is None else raw_table
    model = make_model(task, binding)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0)
    batches = _train_batches(task)
    batch = next(batches)
    _update(batch, model, optimizer)
    evidence.observe(batch)
    task.to_artifact().save(work / "artifact")
    api.save_weights(work / "model.pt", model.state_dict(), task.fitted_state)
    scores, inputs = _predictions(task, model.eval(), inference=False, evidence=evidence)
    if binding is not None:
        # Use the same target-free inputs before and after checkpoint reload.
        with api.prepare(
            task.contract, data, api.DiskGraphEngine(index=index, ingest="never"), inference=True
        ) as plan:
            with plan.materialize(
                replace(raw_table, targets={}), fitted=task.fitted_state
            ) as inference:
                scores, inputs = _predictions(inference, model.eval(), inference=True)
    ids = pq.read_table(
        raw_table.source.path, columns=["sample_id"], filters=[("split", "in", ["valid", "test"])]
    )["sample_id"].to_pylist()
    assert scores and scores.keys() == set(ids)
    torch.save(
        dict(
            tables={name: source.path for name, source in data.tables.items()},
            anchors=raw_table.to_dict(),
            binding=binding,
            index=str(index.resolve()),
            threads=torch.get_num_threads(),
            optimizer=optimizer.state_dict(),
            cursor=batches.state_dict(),
            rng=torch.get_rng_state(),
            predictions=(scores, inputs),
        ),
        work / "checkpoint.pt",
    )
    torch.save(_finish(batches, model.train(), optimizer, evidence), work / "expected.pt")
    subprocess.run([sys.executable, "-m", __name__, str(work.resolve())], cwd=ROOT, check=True)
    result = json.loads((work / "reload.json").read_text())
    result["fresh_process"] = result.pop("pid") != os.getpid()
    return result


@patch("kumo.pytorch.assembly._fit_inputs", side_effect=AssertionError("input encoders refitted"))
def _reload(work, _fit_inputs):
    checkpoint = torch.load(work / "checkpoint.pt", weights_only=True)
    torch.set_num_threads(checkpoint["threads"])
    artifact = api.Artifact.load(work / "artifact")
    data = api.Dataset({name: api.Parquet(path) for name, path in checkpoint["tables"].items()})
    columns = checkpoint["anchors"].copy()
    raw_table = api.AnchorTable(api.Parquet(columns.pop("source")["path"]), **columns)
    table = raw_table
    if (work / "values").exists():
        from kumo.pytorch.aggregate_materialization import MaterializedValues

        table = MaterializedValues.open(work / "values").anchors
    engine = api.DiskGraphEngine(index=checkpoint["index"], ingest="never")
    fitted = artifact.fitted_state
    weights = api.load_weights(work / "model.pt", fitted)
    evidence = {"pid": os.getpid()}
    expected = torch.load(work / "expected.pt", weights_only=True)
    with api.prepare(artifact.contract, data, engine) as plan:
        for state, split in ((fitted, None), (fitted.inputs, "train")):
            with plan.materialize(table, fitted=state, fit_split=split) as task:
                assert task.fitted_state.inputs.fingerprint == fitted.inputs.fingerprint
                model = make_model(task, checkpoint["binding"])
                model.load_state_dict(weights)
                if split is None:
                    with (
                        patch(
                            "kumo.pytorch.assembly._fit_targets",
                            side_effect=AssertionError("inference fit"),
                        ),
                        api.prepare(artifact.contract, data, engine, inference=True) as scoring,
                        scoring.materialize(
                            replace(raw_table, targets={}), fitted=fitted
                        ) as inference,
                    ):
                        predictions = _predictions(inference, model.eval(), inference=True)
                        assert_same(predictions, checkpoint["predictions"])
                        evidence["inference"] = len(predictions[0])
                optimizer = torch.optim.AdamW(model.parameters())
                optimizer.load_state_dict(deepcopy(checkpoint["optimizer"]))
                batches = _train_batches(task)
                batches.load_state_dict(checkpoint["cursor"])
                torch.set_rng_state(checkpoint["rng"])
                resumed = _finish(batches, model, optimizer)
                assert_same(resumed, expected)
                evidence["mid_epoch_resume" if split is None else "finetuning"] = len(resumed[0])
    (work / "reload.json").write_text(json.dumps(evidence))


if __name__ == "__main__":
    _reload(Path(sys.argv[1]))
