"""Controlled CPU overfitting checks; not full-data learning or throughput tests."""

from collections import Counter
from copy import deepcopy
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform

import pytest
import torch
import kumo.pytorch.api as r

from test.kumo_pytorch.involvement import verify_batch
from test.kumo_pytorch.lifecycle import input_digest
from test.kumo_pytorch.masked_values import (
    FIELDS,
    original_tokens,
    selected_labels,
    verify_loss,
    verify_targets,
)
from test.kumo_pytorch.models.aml import build_model
from .fixtures import build_fixture
from .test_aml import CASES, fit_case


@pytest.fixture(autouse=True)
def deterministic_cpu():
    threads = torch.get_num_threads()
    dtype = torch.get_default_dtype()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.set_num_threads(1)
        torch.set_default_dtype(torch.float32)
        torch.use_deterministic_algorithms(True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            yield
    finally:
        torch.set_num_threads(threads)
        torch.set_default_dtype(dtype)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


@pytest.fixture(params=("combined", "pragma"))
def learning_case(request, tmp_path_factory):
    family = request.param
    selection = next(selection for selection, name in CASES if name == family)
    yield from fit_case(selection, family, "diskgraph", tmp_path_factory)


def masked_labels(task, batch):
    """Reuse the independent source/target oracle before evaluating learning."""
    fixture = build_fixture()
    anchors = {
        row.sample_id: {
            "entity_id": str(row.entity_id),
            "as_of_time": int(row.as_of_time.timestamp()) * 10**9,
        }
        for row in fixture.training
    }
    reference = {
        "events": [
            {
                "id": row["transaction_id"],
                "time": row["event_time_seconds"],
                "accounts": {str(row["sender_id"]), str(row["receiver_id"])},
                "sender": str(row["sender_id"]),
                "receiver": str(row["receiver_id"]),
                "values": {name: row[name] for name in FIELDS if name != "direction"},
            }
            for row in fixture.tables["transactions"]
        ]
    }
    original = original_tokens(
        batch,
        anchors,
        reference,
        task.fitted_state.to_dict(),
        {str(i): i for i in range(1, 5)},
    )
    verify_targets(batch, original)
    return selected_labels(batch)[1]


def measure(model, batch, family, labels):
    result = model.training_step(batch)
    if family == "combined":
        verify_batch(batch, result, {"fit-account-1": 1, "fit-account-2": 0})
        predictions = (result.logits.detach().flatten() >= 0).long().tolist()
    else:
        verify_loss(batch, result)
        predictions = result.logits.detach().argmax(-1).tolist()
    assert result.weight == len(labels) and torch.isfinite(result.logits).all()
    return result, {
        "loss": result.loss.item(),
        "accuracy": sum(
            left == right for left, right in zip(predictions, labels, strict=True)
        )
        / len(labels),
        "predictions": predictions,
        "eligible": result.weight,
    }


def test_controlled_learning(learning_case, tmp_path):
    task, batch, selection, family = learning_case
    steps = 50 if family == "combined" else 100
    model = build_model(task, selection, family, dim=16, max_seq_len=121, seed=17)
    control = deepcopy(model).eval()  # Same initialization, never updated.
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0)
    labels = [1, 0] if family == "combined" else masked_labels(task, batch)
    assert len(set(labels)) > 1, "The learning fixture must not be a single-class task"
    modules = (
        (
            "backbone.graph_encoder",
            "backbone.sequence_encoder",
            "backbone.aggregate_encoder",
            "backbone.head",
        )
        if family == "combined"
        else (
            "backbone.embed",
            "backbone.event_encoder",
            "backbone.history_encoder",
            "head",
        )
    )
    initial = {
        name: input_digest(model.get_submodule(name).state_dict()) for name in modules
    }
    batch_hash = input_digest((batch.inputs, batch.targets, batch.metadata))
    fitted_hash = task.fitted_state.fingerprint
    with torch.no_grad():
        baseline_result, baseline = measure(control, batch, family, labels)
    classes = 2 if family == "combined" else baseline_result.logits.shape[-1]
    smoothing = 0 if family == "combined" else 0.1
    # The smoothed CE optimum is not zero: it is the entropy of the target
    # distribution (0.9 + 0.1/V for the answer, 0.1/V elsewhere).
    correct = 1 - smoothing + smoothing / classes
    other = smoothing / classes
    floor = -correct * math.log(correct) - (
        (classes - 1) * other * math.log(other) if other else 0
    )
    suite = Path(__file__).resolve().parents[1]
    sources = [
        Path(__file__),
        suite / "model_checks/test_aml.py",
        suite / "model_checks/fixtures.py",
        suite / "contracts.py",
        suite / "involvement.py",
        suite / "masked_values.py",
        suite / "reference_tokens.py",
    ] + sorted((suite / "models").rglob("*.py"))
    report = {
        "status": "failed",
        "family": family,
        "steps": steps,
        "completed_updates": 0,
        "model_seed": 17,
        "batch_seed": 7,
        "epoch": 0,
        "batch_size": 2,
        "dim": 16,
        "optimizer": {"name": "AdamW", "lr": 0.01, "weight_decay": 0},
        "threads": 1,
        "device": "cpu",
        "dtype": "float32",
        "deterministic_algorithms": True,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "kumo_api": r.__file__,
            **{
                name: version(name)
                for name in ("torch", "torch-geometric", "pyarrow", "diskgraph")
            },
        },
        "source_sha256": {
            str(path.relative_to(suite)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
        },
        "workspace_lock_sha256": hashlib.sha256(
            (suite.parents[1] / "uv.lock").read_bytes()
        ).hexdigest(),
        "fitted_state_sha256": fitted_hash,
        "batch_sha256": batch_hash,
        "sample_ids": list(batch.metadata["sample_ids"]),
        "labels": labels,
        "loss_floor": floor,
        "uniform_loss": math.log(classes),
        "majority_accuracy": max(Counter(labels).values()) / len(labels),
        "minimum_accuracy": 1.0 if family == "combined" else 0.9,
        "maximum_excess_loss_ratio": 0.2,
        "baseline": baseline,
        "history": [],
    }
    try:
        model.train()
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            result, metrics = measure(model, batch, family, labels)
            result.loss.backward()
            gradients = [
                parameter.grad
                for parameter in model.parameters()
                if parameter.grad is not None
            ]
            assert gradients and all(
                torch.isfinite(gradient).all() for gradient in gradients
            )
            if step == 0:
                for name in modules:
                    assert any(
                        parameter.grad is not None
                        and torch.count_nonzero(parameter.grad)
                        for parameter in model.get_submodule(name).parameters()
                    ), f"No gradient in {name}"
            optimizer.step()
            report["completed_updates"] += 1
            assert all(
                torch.isfinite(parameter).all() for parameter in model.parameters()
            )
            if step == 0:
                for name in modules:
                    assert (
                        input_digest(model.get_submodule(name).state_dict())
                        != initial[name]
                    ), f"No update in {name}"
            report["history"].append({"step": step, **metrics})
        model.eval()
        with torch.no_grad():
            _, final = measure(model, batch, family, labels)
            _, unchanged = measure(control, batch, family, labels)
        report.update(final=final, frozen_control=unchanged)
        assert unchanged == baseline, "The untrained control changed"
        assert final["accuracy"] >= report["minimum_accuracy"], (
            "Insufficient answer recovery"
        )
        assert final["accuracy"] > report["majority_accuracy"], (
            "Only learned a constant answer"
        )
        assert floor - 2e-6 <= final["loss"] < math.log(classes)
        assert final["loss"] - floor < 0.2 * (baseline["loss"] - floor), (
            "Insufficient loss reduction"
        )
        assert input_digest((batch.inputs, batch.targets, batch.metadata)) == batch_hash
        assert task.fitted_state.fingerprint == fitted_hash
        report["status"] = "passed"
    finally:
        (tmp_path / "learning.json").write_text(json.dumps(report, indent=2) + "\n")
