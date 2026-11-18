# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fresh-process AML finetuning with frozen inputs, followed by optimizer resume."""

import argparse
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch

import kumo.pytorch.api as r

from .anchor_tables import load_probe_anchors, write_probe_anchors
from .involvement import verify_batch
from .lifecycle import forbid_fitting, input_digest
from .models.aml import build_model


def involvement_model(task, config, pretrained_weights=None):
    model = build_model(task, config["selection"], config["family"], **config["model"])
    if pretrained_weights is not None:
        model.load_state_dict(pretrained_weights, strict=True)
    if model.workload != "account-involvement-next24h":
        # Keep the learned backbone; only the downstream head starts fresh.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config["seed"])
            model.head = torch.nn.Linear(config["model"]["dim"], 1)
        model.workload = "account-involvement-next24h"
    return model


def update(task, model, optimizer, requests, config, *, epoch):
    """One deterministic probe batch, not a complete epoch or learning benchmark."""
    batch = next(
        iter(
            task.batches(
                split="train",
                output_format="model",
                batch_size=config["batch_size"],
                seed=config["seed"],
                epoch=epoch,
            )
        )
    )
    inputs = input_digest(batch.inputs)
    assert set(batch.targets) == {"label"}, "Finetuning kept a pretraining target"
    if model.family == "pragma":
        assert not (batch.inputs["history"]["value_ids"] == 2).any(), (
            "Finetuning kept input masking"
        )
    before = input_digest(model.backbone.state_dict())
    model.train()
    optimizer.zero_grad(set_to_none=True)
    result = model.training_step(batch)
    assert result.weight > 0 and torch.isfinite(result.loss)
    verify_batch(
        batch, result, {row.sample_id: row.targets[0].value for row in requests}
    )
    result.loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    optimizer.step()
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert input_digest(model.backbone.state_dict()) != before, (
        "Backbone did not finetune"
    )
    assert input_digest(batch.inputs) == inputs, "Training changed the input batch"
    return {
        "loss": result.loss.item(),
        "target_weight": result.weight,
        "sample_ids": list(batch.metadata["sample_ids"]),
        "inputs_sha256": inputs,
        "history_tokens": int(batch.inputs["history"]["valid"].sum())
        if "history" in batch.inputs
        else None,
        "unknown_tokens": int(
            (
                (batch.inputs["history"]["value_ids"] == 1)
                & batch.inputs["history"]["valid"]
            ).sum()
        )
        if "history" in batch.inputs
        else None,
        "model_sha256": input_digest(model.state_dict()),
        "optimizer_sha256": input_digest(optimizer.state_dict()),
    }


def training_requests(config):
    return tuple(
        r.SampleRequest(
            **{
                **row,
                "targets": tuple(r.TargetValue(**target) for target in row["targets"]),
            }
        )
        for row in config["training_requests"]
    )


def resume(directory):
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())
    torch.set_num_threads(config["threads"])
    assert os.getpid() != config["parent_pid"] and r.__file__ == config["kumo_api"]
    artifact = r.Artifact.load(directory / "artifact")
    checkpoint = torch.load(directory / "optimizer.pt", weights_only=True)
    assert artifact.fitted_state.fingerprint == checkpoint["fitted_state_sha256"]
    requests = training_requests(config)
    dataset = r.Dataset(
        {name: r.Parquet(path) for name, path in config["sources"].items()}
    )
    engine = r.DiskGraphEngine(
        index=Path(config["index"]), ingest="never", num_threads=config["threads"]
    )
    with forbid_fitting(), r.prepare(artifact.contract, dataset, engine) as plan:
        with plan.materialize(
            load_probe_anchors(config["training_anchor_table"]),
            fitted=artifact.fitted_state,
        ) as task:
            model = involvement_model(task, config)
            model.load_state_dict(
                r.load_weights(directory / "weights.pt", task.fitted_state), strict=True
            )
            optimizer = torch.optim.AdamW(model.parameters())
            optimizer.load_state_dict(checkpoint["optimizer"])
            assert input_digest(model.state_dict()) == checkpoint["model_sha256"]
            assert (
                input_digest(optimizer.state_dict()) == checkpoint["optimizer_sha256"]
            )
            torch.set_rng_state(checkpoint["torch_rng"])
            actual = update(
                task, model, optimizer, requests, config, epoch=checkpoint["next_epoch"]
            )
            assert task.fitted_state.fingerprint == checkpoint["fitted_state_sha256"]
    expected = json.loads((directory / "expected.json").read_text())
    assert actual == expected, "Resumed update differs from uninterrupted training"
    result = {
        "status": "passed",
        "pid": os.getpid(),
        "parent_pid": config["parent_pid"],
        "completed_updates": checkpoint["completed_updates"] + 1,
        "comparison": "exact_same_cpu_configuration",
        "next_update": actual,
    }
    (directory / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def check_training(lifecycle_directory, requests):
    """Run finetuning in a fresh interpreter; keep required errors as failures."""
    lifecycle_directory = Path(lifecycle_directory).resolve()
    config = json.loads((lifecycle_directory / "config.json").read_text())
    assert requests and all(row.split == "train" for row in requests)
    config["training_requests"] = [
        {
            "sample_id": row.sample_id,
            "entity_id": row.entity_id,
            "as_of_time": row.as_of_time,
            "split": row.split,
            "metadata": dict(row.metadata),
            "targets": [
                {"name": target.name, "value": target.value} for target in row.targets
            ],
        }
        for row in requests
    ]
    config["finetune_parent_pid"] = os.getpid()
    config["lifecycle_directory"] = str(lifecycle_directory)
    directory = lifecycle_directory.parent / "finetuning"
    directory.mkdir(exist_ok=False)
    config["training_anchor_table"] = write_probe_anchors(
        requests, directory / "anchors.parquet"
    ).to_dict()
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    command = [
        sys.executable,
        "-m",
        "test.kumo_pytorch.training_lifecycle",
        str(directory),
        "--finetune",
    ]
    with (directory / "finetune.log").open("w") as log:
        child = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[2],
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=240,
            check=False,
        )
    if child.returncode:
        raise RuntimeError(
            f"Fresh-process finetuning failed; see {directory / 'finetune.log'}"
        )
    result = json.loads((directory / "report.json").read_text())
    assert result["status"] == "passed"
    assert result["pid"] != os.getpid()
    assert result["parent_pid"] == os.getpid()
    return result


def finetune(directory):
    """Reload only persisted inputs/weights before the first finetuning update."""
    directory = Path(directory).resolve()
    config = json.loads((directory / "config.json").read_text())
    assert os.getpid() != config["finetune_parent_pid"], (
        "Finetuning did not start a new process"
    )
    assert r.__file__ == config["kumo_api"], "Finetuning loaded a different package"
    torch.set_num_threads(config["threads"])
    torch.manual_seed(config["seed"])
    lifecycle_directory = Path(config["lifecycle_directory"])
    requests = training_requests(config)
    assert requests and all(row.split == "train" for row in requests)
    # The subsequent resume process must differ from this finetuning process.
    config["parent_pid"] = os.getpid()
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    artifact = r.Artifact.load(lifecycle_directory / "artifact")
    assert artifact.fitted_state.fingerprint == config["fitted_state_sha256"]
    original = artifact.contract
    contract = replace(
        original,
        sample=replace(
            original.sample, targets={"label": r.ExternalTarget(r.DType("float32"))}
        ),
        batch=replace(
            original.batch, targets={"label": r.LabelTarget("label", r.Raw())}
        ),
    )
    assert contract.input_fingerprint == original.input_fingerprint
    dataset = r.Dataset(
        {name: r.Parquet(path) for name, path in config["sources"].items()}
    )
    engine = r.DiskGraphEngine(
        index=Path(config["index"]), ingest="never", num_threads=config["threads"]
    )
    frozen = artifact.fitted_state.inputs.fingerprint
    input_state = input_digest(artifact.fitted_state.inputs.to_dict())
    # Target fitting is allowed; input fitting must never run even if it would
    # reproduce the same values. The full input fingerprint guards all state.
    with (
        patch(
            "kumo.pytorch.assembly._fit_inputs",
            side_effect=AssertionError("Finetuning attempted input refitting"),
        ),
        r.prepare(contract, dataset, engine) as plan,
    ):
        with plan.materialize(
            load_probe_anchors(config["training_anchor_table"]),
            fitted=artifact.fitted_state.inputs,
            fit_split="train",
        ) as task:
            assert task.fitted_state.inputs.fingerprint == frozen
            assert input_digest(task.fitted_state.inputs.to_dict()) == input_state
            assert (
                task.fitted_state.fit["inputs"] == artifact.fitted_state.fit["inputs"]
            )
            weights = r.load_weights(
                lifecycle_directory / "weights.pt", artifact.fitted_state
            )
            model = involvement_model(task, config, weights)
            assert input_digest(model.backbone.state_dict()) == input_digest(
                {
                    key.removeprefix("backbone."): value
                    for key, value in weights.items()
                    if key.startswith("backbone.")
                }
            ), "Finetuning replaced the learned backbone"
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0)
            first = update(task, model, optimizer, requests, config, epoch=0)
            assert task.fitted_state.inputs.fingerprint == frozen
            assert input_digest(task.fitted_state.inputs.to_dict()) == input_state
            task.to_artifact().save(directory / "artifact")
            r.save_weights(
                directory / "weights.pt", model.state_dict(), task.fitted_state
            )
            checkpoint = {
                "optimizer": optimizer.state_dict(),
                "torch_rng": torch.get_rng_state(),
                "completed_updates": 1,
                "next_epoch": 1,
                "fitted_state_sha256": task.fitted_state.fingerprint,
                "model_sha256": first["model_sha256"],
                "optimizer_sha256": first["optimizer_sha256"],
            }
            torch.save(checkpoint, directory / "optimizer.pt")
            expected = update(task, model, optimizer, requests, config, epoch=1)
            assert task.fitted_state.inputs.fingerprint == frozen
            assert input_digest(task.fitted_state.inputs.to_dict()) == input_state
    (directory / "expected.json").write_text(json.dumps(expected, indent=2) + "\n")
    with (directory / "child.log").open("w") as log:
        child = subprocess.run(
            [
                sys.executable,
                "-m",
                "test.kumo_pytorch.training_lifecycle",
                str(directory),
            ],
            cwd=Path(__file__).resolve().parents[2],
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=120,
            check=False,
        )
    if child.returncode:
        raise RuntimeError(
            f"Fresh-process resume failed; see {directory / 'child.log'}"
        )
    result = {
        "status": "passed",
        "task": "account-involvement-next24h",
        "pid": os.getpid(),
        "parent_pid": config["finetune_parent_pid"],
        "input_state_content_sha256": input_state,
        "split": "forecast_split/train",
        "input_state_sha256": frozen,
        "input_refitting": False,
        "head_replaced": config["selection"]["selector"]["workload_id"]
        != "account-involvement-next24h",
        "first_update": first,
        "resume": json.loads((directory / "result.json").read_text()),
    }

    (directory / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--finetune", action="store_true")
    args = parser.parse_args()
    (finetune if args.finetune else resume)(args.directory)
