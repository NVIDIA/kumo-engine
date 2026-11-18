# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded AML save/load and fresh-process inference using Kumo artifact APIs."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch

import kumo.pytorch.api as r

from .anchor_tables import load_probe_anchors, write_probe_anchors
from .models.aml import build_model


@contextmanager
def forbid_fitting():
    """A frozen-state AML inference run must never enter a fitting path."""
    with ExitStack() as stack:
        for name in (
            "kumo.pytorch.preparation.Plan._fit_selection",
            "kumo.pytorch.preparation.Plan._pinned_from_stats",
            "kumo.pytorch.assembly._fit_inputs",
            "kumo.pytorch.assembly._fit_targets",
        ):
            stack.enter_context(
                patch(name, side_effect=AssertionError("Inference attempted refitting"))
            )
        yield


def input_digest(value):
    """Exact input identity for the same engine/configuration before/after reload."""
    digest = hashlib.sha256()

    def add(item):
        if isinstance(item, torch.Tensor):
            digest.update(str((item.dtype, tuple(item.shape))).encode())
            digest.update(
                item.detach()
                .cpu()
                .contiguous()
                .reshape(-1)
                .view(torch.uint8)
                .numpy()
                .tobytes()
            )
        elif isinstance(item, dict):
            for key in sorted(item):
                digest.update(repr(key).encode())
                add(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(str(len(item)).encode())
            for member in item:
                add(member)
        else:
            digest.update(repr(item).encode())

    add(value)
    return digest.hexdigest()


def snapshot(task, model, requests, *, batch_size, seed):
    model.eval()
    results = []
    with torch.no_grad():
        for batch in task.batches(
            batch_size=batch_size, seed=seed, epoch=0, output_format="model"
        ):
            assert batch.targets == {}, "Inference exposed targets"
            assert not batch.metadata["is_padding"].any()
            if model.family == "pragma":
                model.backbone.validate_kumo_batch(batch, target_name=None)
                ids = batch.inputs["history"]["value_ids"]
                assert not (ids == 2).any(), "Inference still masks PRAGMA values"
                output = model.backbone.forward_kumo(batch)
                # Include the trained MLM head as a persistence check, on every
                # unmasked field. These are not masked-objective loss scores.
                predictions = {
                    "user_repr": output.user_repr,
                    "unmasked_value_logits": model.head(
                        output,
                        model.backbone.embedding_weight,
                        torch.arange(ids.numel()),
                    ),
                }
            else:
                if model.family == "nuformer":
                    model.backbone.validate_kumo_batch(batch, target_name=None)
                predictions = {"logits": model(batch)}
            assert all(torch.isfinite(value).all() for value in predictions.values())
            metadata = {
                key: batch.metadata[key]
                for key in (
                    "sample_ids",
                    "entity_ids",
                    "as_of_time",
                    "request_metadata",
                    "is_padding",
                )
            }
            results.append(
                {
                    "sample_ids": list(batch.metadata["sample_ids"]),
                    "inputs_sha256": input_digest((batch.inputs, metadata)),
                    "predictions": predictions,
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
                }
            )
    assert [sid for batch in results for sid in batch["sample_ids"]] == [
        request.sample_id for request in requests
    ], "Inference delivery differs"
    return results


def compare_snapshots(actual, expected):
    assert len(actual) == len(expected), "Inference batch count changed"
    for observed, original in zip(actual, expected, strict=True):
        for key in ("sample_ids", "inputs_sha256", "history_tokens", "unknown_tokens"):
            assert observed[key] == original[key], f"Inference {key} changed"
        assert observed["predictions"].keys() == original["predictions"].keys()
        for key in original["predictions"]:
            # Same CPU, engine, thread count, batch shape and weights; no
            # cross-engine or performance comparison is implied here.
            torch.testing.assert_close(
                observed["predictions"][key],
                original["predictions"][key],
                rtol=0,
                atol=0,
            )


def reload_and_infer(directory):
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())
    torch.set_num_threads(config["threads"])
    artifact = r.Artifact.load(directory / "artifact")
    assert artifact.fitted_state.fingerprint == config["fitted_state_sha256"]
    dataset = r.Dataset(
        {name: r.Parquet(path) for name, path in config["sources"].items()}
    )
    requests = tuple(r.SampleRequest(**row) for row in config["requests"])
    engine = r.DiskGraphEngine(
        index=Path(config["index"]), ingest="never", num_threads=config["threads"]
    )
    with (
        forbid_fitting(),
        r.prepare(artifact.contract, dataset, engine, inference=True) as plan,
    ):
        with plan.materialize(
            load_probe_anchors(config["anchor_table"]),
            fitted=artifact.fitted_state,
            inference=True,
        ) as task:
            model = build_model(
                task, config["selection"], config["family"], **config["model"]
            )
            model.load_state_dict(
                r.load_weights(directory / "weights.pt", artifact.fitted_state),
                strict=True,
            )
            actual = snapshot(
                task,
                model,
                requests,
                batch_size=config["batch_size"],
                seed=config["seed"],
            )
            assert task.fitted_state.fingerprint == config["fitted_state_sha256"]
    expected = torch.load(directory / "expected.pt", weights_only=True)
    compare_snapshots(actual, expected)
    result = {
        "status": "passed",
        "pid": os.getpid(),
        "parent_pid": config["parent_pid"],
        "kumo_api": r.__file__,
        "fitted_state_sha256": artifact.fitted_state.fingerprint,
        "sample_ids": [sid for batch in actual for sid in batch["sample_ids"]],
        "history_tokens": [batch["history_tokens"] for batch in actual],
        "unknown_tokens": [batch["unknown_tokens"] for batch in actual],
        "no_refitting": True,
        "targets_absent": True,
        "prediction_comparison": "exact_same_cpu_configuration",
    }
    assert result["pid"] != result["parent_pid"]
    assert result["kumo_api"] == config["kumo_api"]
    (directory / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def check_lifecycle(
    plan,
    task,
    model,
    dataset,
    requests,
    selection,
    family,
    *,
    directory,
    index,
    batch_size,
    threads,
    seed,
    max_seq_len,
    dim=16,
):
    """Save the trained state, establish inference output, then check a new process."""
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    fingerprint = task.fitted_state.fingerprint
    task.to_artifact().save(directory / "artifact")
    r.save_weights(directory / "weights.pt", model.state_dict(), task.fitted_state)
    requests = tuple(replace(row, targets=()) for row in requests)
    anchors = write_probe_anchors(requests, directory / "anchors.parquet")
    config = {
        "anchor_table": anchors.to_dict(),
        "selection": selection,
        "family": family,
        "model": {"dim": dim, "max_seq_len": max_seq_len, "seed": seed},
        "sources": {
            name: str(Path(source.path).resolve())
            for name, source in dataset.tables.items()
        },
        "index": str(Path(index).resolve()),
        "batch_size": batch_size,
        "threads": threads,
        "seed": seed,
        "parent_pid": os.getpid(),
        "kumo_api": r.__file__,
        "fitted_state_sha256": fingerprint,
        "requests": [
            {
                "sample_id": row.sample_id,
                "entity_id": row.entity_id,
                "as_of_time": int(row.as_of_time.timestamp())
                if hasattr(row.as_of_time, "timestamp")
                else row.as_of_time,
                "metadata": dict(row.metadata),
                "split": row.split,
            }
            for row in requests
        ],
    }
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    with (
        forbid_fitting(),
        plan.materialize(
            anchors, fitted=task.fitted_state, inference=True
        ) as inference,
    ):
        expected = snapshot(
            inference, model, requests, batch_size=batch_size, seed=seed
        )
        assert inference.fitted_state.fingerprint == fingerprint
    torch.save(expected, directory / "expected.pt")
    command = [sys.executable, "-m", "test.kumo_pytorch.lifecycle", str(directory)]
    with (directory / "child.log").open("w") as log:
        child = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[2],
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=120,
        )
    if child.returncode:
        raise RuntimeError(
            f"Fresh-process inference failed; see {directory / 'child.log'}"
        )
    assert task.fitted_state.fingerprint == fingerprint
    return json.loads((directory / "result.json").read_text())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    reload_and_infer(parser.parse_args().directory)
