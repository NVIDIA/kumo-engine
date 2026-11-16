"""Exact CPU training continuation across a fresh-process, mid-epoch checkpoint."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import kumo.pytorch.api as r
import torch

from .lifecycle import forbid_fitting, input_digest
from .models.aml import build_model
from .training_lifecycle import training_requests


def training_update(model, optimizer, batch):
    """Record complete batch identity and the resulting training state."""
    batch_digest = input_digest((batch.inputs, batch.targets))
    model.train()
    optimizer.zero_grad(set_to_none=True)
    result = model.training_step(batch)
    assert torch.isfinite(result.loss) and torch.isfinite(result.logits).all()
    if result.weight:
        result.loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        optimizer.step()
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert input_digest((batch.inputs, batch.targets)) == batch_digest
    return {
        "sample_ids": list(batch.metadata["sample_ids"]),
        "is_padding": batch.metadata["is_padding"].tolist(),
        "inputs_and_targets_sha256": batch_digest,
        "logits_sha256": input_digest(result.logits),
        "loss": result.loss.item(),
        "target_weight": result.weight,
        "model_sha256": input_digest(model.state_dict()),
        "optimizer_sha256": input_digest(optimizer.state_dict()),
        "torch_rng_sha256": input_digest(torch.get_rng_state()),
    }


def continuation(directory, *, restore):
    directory = Path(directory).resolve()
    config = json.loads((directory / "config.json").read_text())
    if restore:
        assert os.getpid() != config["parent_pid"]
    assert r.__file__ == config["kumo_api"]
    lifecycle = Path(config["lifecycle_directory"])
    artifact = r.Artifact.load(lifecycle / "artifact")
    assert artifact.fitted_state.fingerprint == config["fitted_state_sha256"]
    requests = training_requests(config)
    dataset = r.Dataset({n: r.Parquet(p) for n, p in config["sources"].items()})
    engine = r.DiskGraphEngine(
        index=Path(config["index"]), ingest="never", num_threads=config["threads"]
    )
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(config["threads"])
    try:
        with (
            torch.random.fork_rng(devices=[]),
            forbid_fitting(),
            r.prepare(artifact.contract, dataset, engine) as plan,
            plan.materialize(requests, fitted=artifact.fitted_state) as task,
        ):
            torch.manual_seed(config["seed"])
            model = build_model(
                task, config["selection"], config["family"], **config["model"]
            )
            model.load_state_dict(
                r.load_weights(
                    (directory if restore else lifecycle) / "weights.pt",
                    task.fitted_state,
                ),
                strict=True,
            )
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0)
            batches = task.batches(**config["batches"])
            if restore:
                checkpoint = torch.load(directory / "checkpoint.pt", weights_only=True)
                optimizer.load_state_dict(checkpoint["optimizer"])
                batches.load_state_dict(checkpoint["batches"])
                torch.set_rng_state(checkpoint["torch_rng"])
                prefix = checkpoint["prefix"]
            else:
                prefix = [
                    training_update(model, optimizer, next(batches))
                    for _ in range(config["checkpoint_after"])
                ]
                checkpoint = {
                    "batches": batches.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "torch_rng": torch.get_rng_state(),
                    "prefix": prefix,
                }
                r.save_weights(
                    directory / "weights.pt", model.state_dict(), task.fitted_state
                )
                torch.save(checkpoint, directory / "checkpoint.pt")
            remaining = [training_update(model, optimizer, batch) for batch in batches]
            assert remaining, "Checkpoint must be inside the epoch, not at its end"
            assert any(row["target_weight"] for row in remaining), (
                "Resume must exercise an optimizer update"
            )
            sample_ids = [
                sid for row in prefix + remaining for sid in row["sample_ids"]
            ]
            assert len(sample_ids) == len(requests) and set(sample_ids) == {
                row.sample_id for row in requests
            }, "Resumed epoch duplicated or omitted anchors"
            assert not any(any(row["is_padding"]) for row in prefix + remaining)
            assert task.fitted_state.to_dict() == artifact.fitted_state.to_dict()
            result = {
                "remaining": remaining,
                "final_iterator": batches.state_dict(),
                "fitted_state_sha256": task.fitted_state.fingerprint,
            }
    finally:
        torch.set_num_threads(previous_threads)
    if restore:
        expected = json.loads((directory / "expected.json").read_text())
        assert result == expected, (
            "Mid-epoch continuation differs from uninterrupted training"
        )
        report = {
            "status": "passed",
            "comparison": "exact_same_cpu_configuration",
            "pid": os.getpid(),
            "parent_pid": config["parent_pid"],
            "workload": config["selection"]["selector"]["workload_id"],
            "family": config["family"],
            "checkpoint_after": config["checkpoint_after"],
            "batches": config["batches"],
            "prefix": prefix,
            **result,
        }
        (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    (directory / "expected.json").write_text(json.dumps(result, indent=2) + "\n")


def check_resume(lifecycle_directory, requests, *, checkpoint_after=1):
    """Compare every remaining batch/update, using the workload's original head."""
    lifecycle_directory = Path(lifecycle_directory).resolve()
    assert len(requests) >= 2 and all(row.split == "train" for row in requests), (
        "Mid-epoch resume requires at least two training anchors"
    )
    assert len({row.sample_id for row in requests}) == len(requests)
    config = json.loads((lifecycle_directory / "config.json").read_text())
    config.update(
        lifecycle_directory=str(lifecycle_directory),
        parent_pid=os.getpid(),
        checkpoint_after=checkpoint_after,
        # Keep this bounded probe inside the selected population even when the
        # ordinary model check fits that entire population in a single batch.
        batches={
            "split": "train",
            "batch_size": min(config["batch_size"], max(1, len(requests) // 2)),
            "shuffle": True,
            "seed": config["seed"],
            "epoch": 3,
            "chunk_batches": 1,
            "drop_last": False,
        },
        training_requests=[
            {
                "sample_id": row.sample_id,
                "entity_id": row.entity_id,
                "as_of_time": row.as_of_time,
                "split": row.split,
                "metadata": dict(row.metadata),
                "targets": [{"name": t.name, "value": t.value} for t in row.targets],
            }
            for row in requests
        ],
    )
    directory = lifecycle_directory.parent / "mid-epoch-resume"
    directory.mkdir(exist_ok=False)
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    continuation(directory, restore=False)
    with (directory / "child.log").open("w") as log:
        child = subprocess.run(
            [
                sys.executable,
                "-m",
                "test.kumo_pytorch.checkpoint_resume",
                str(directory),
            ],
            cwd=Path(__file__).resolve().parents[2],
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=120,
            check=False,
        )
    if child.returncode:
        raise RuntimeError(f"Mid-epoch resume failed; see {directory / 'child.log'}")
    report = json.loads((directory / "report.json").read_text())
    assert report["status"] == "passed"
    assert report["parent_pid"] == os.getpid() != report["pid"]
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    continuation(parser.parse_args().directory, restore=True)
