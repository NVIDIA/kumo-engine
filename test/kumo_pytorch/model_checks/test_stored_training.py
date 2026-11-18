# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stored task heads must use independent masks and replay native training."""

from copy import deepcopy
from test.kumo_pytorch.bindings import load_bindings
from test.kumo_pytorch.contracts import binding_contract
from test.kumo_pytorch.lifecycle import input_digest
from test.kumo_pytorch.models import build_model
from test.kumo_pytorch.population_anchors import (
    project_binding,
    projected_rows,
)
from test.kumo_pytorch.targets import verify_stored_loss
from test.kumo_pytorch.training_acceptance import (
    compare_repetitions,
    train_repetition,
)

import pytest
import torch

import kumo.pytorch.api as r

from .fixtures import build_fixture
from .test_aml_bindings import LI_BINDINGS, source_anchors


@pytest.fixture(params=LI_BINDINGS)
def stored_task(tmp_path, request):
    binding = load_bindings()[request.param]
    fixture = build_fixture()
    anchors, report = project_binding(
        source_anchors(tmp_path / "original.parquet"),
        tmp_path / "projected.parquet",
        {"account-0001": 1},
        binding,
    )
    contract = binding_contract(fixture.schema, binding, max_events=11)
    dataset = fixture.parquet_dataset(tmp_path / "source")
    with r.prepare(
        contract, dataset, r.DiskGraphEngine(index=tmp_path / "index", num_threads=1)
    ) as plan:
        task = plan.materialize(
            anchors, fit_split="train", encoder_cutoff=report["encoder_cutoff_seconds"]
        )
        selection = {
            "selector": binding["selector"],
            "models": [binding["representation"]],
        }
        yield (
            task,
            selection,
            list(projected_rows(tmp_path / "projected.parquet", "train")),
        )


def test_stored_model_loss_masks_and_input_preservation(stored_task):
    task, selection, _ = stored_task
    family = selection["models"][0]
    model = build_model(task, selection, family, seed=17)
    batch = next(task.batches(split="train", batch_size=3, output_format="model"))
    digest = input_digest((batch.inputs, batch.targets, batch.metadata))
    result = model.training_step(batch)
    verify_stored_loss(selection["selector"]["workload_id"], batch, result)
    assert result.weight == (4 if family == "combined" else 2)
    assert input_digest((batch.inputs, batch.targets, batch.metadata)) == digest
    # Invalid labels may contain arbitrary fill values; they cannot affect loss.
    invalid = deepcopy(batch)
    for field in invalid.targets.values():
        field["values"][~field["valid"]] = (
            float("nan") if field["values"].is_floating_point() else 999999
        )
    mutated = model.training_step(invalid)
    torch.testing.assert_close(mutated.loss, result.loss, rtol=0, atol=0)
    # Padding is excluded independently for every target.
    invalid.metadata["is_padding"][0] = True
    masked = model.training_step(invalid)
    verify_stored_loss(selection["selector"]["workload_id"], invalid, masked)
    assert masked.weight == result.weight - (2 if family == "combined" else 1)


def test_zero_eligible_stored_targets_do_not_update(stored_task):
    task, selection, _ = stored_task
    model = build_model(task, selection, selection["models"][0], seed=17)
    batch = next(task.batches(split="test", batch_size=3, output_format="model"))
    before = input_digest(model.state_dict())
    result = model.training_step(batch)
    assert result.weight == 0 and result.loss.item() == 0
    result.loss.backward()
    assert all(p.grad is None or not p.grad.any() for p in model.parameters())
    assert input_digest(model.state_dict()) == before


def test_stored_native_training_acceptance_repeats_exactly(stored_task, tmp_path):
    task, selection, rows = stored_task
    reports = [
        train_repetition(
            task,
            None,
            selection,
            selection["models"][0],
            directory=tmp_path / f"repeat-{i}",
            capacity=11,
            seed=17,
            batch_size=3,
            expected_rows=rows,
        )
        for i in range(2)
    ]
    for report in reports:
        assert report["status"] == "passed", report.get(
            "error",
            {
                n: p
                for n, p in report["parameters"].items()
                if not p.get("updated") or not p["nonzero_gradient_steps"]
            },
        )
        assert report["completed_updates"] == 100
        assert report["final_loss"] < report["initial_loss"]
    assert compare_repetitions(*reports)
