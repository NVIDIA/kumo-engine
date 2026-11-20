# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PRAGMA AML task integration; generic masking mechanics stay in package tests."""

import math
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
from test.kumo_pytorch.bindings import selections as catalog_selections
from test.kumo_pytorch.contracts import build_contract
from test.kumo_pytorch.masked_values import (
    FIELDS,
    original_tokens,
    selected_labels,
    verify_loss,
    verify_targets,
)
from test.kumo_pytorch.models import build_model

import pytest
import torch

import kumo.pytorch.api as r

from .fixtures import CUTOFF, build_fixture, probe_anchor_table
from .test_aml import assert_equal

SELECTION = next(
    row
    for row in catalog_selections()
    if row["selector"]["workload_id"] == "masked-value-pretrain"
)


@pytest.fixture(scope="module", autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module", params=("diskgraph",))
def case(request, tmp_path_factory):
    fixture = build_fixture()
    requests = tuple(replace(row, targets=()) for row in fixture.training) + (
        r.SampleRequest("held-out", 4, CUTOFF + timedelta(days=1), split="test"),
        r.SampleRequest("unknown", 1, CUTOFF + timedelta(days=1), split="test"),
    )
    anchors = {
        row.sample_id: {
            "entity_id": str(row.entity_id),
            "as_of_time": int(row.as_of_time.timestamp()) * 10**9,
        }
        for row in requests
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
    root = tmp_path_factory.mktemp("masked-aml")

    def source(data, name):
        return data.parquet_dataset(root / name), r.DiskGraphEngine(
            index=root / f"{name}-index", num_threads=1
        )

    dataset, engine = source(fixture, "original")
    contract = build_contract(fixture.schema, "pragma", native=True, max_events=11)
    policy = contract.batch.targets["masked_values"]
    assert (
        policy.token_probability,
        policy.event_probability,
        policy.key_probability,
        policy.unknown_probability,
    ) == (0.15, 0.10, 0.10, 0.10)
    assert (
        policy.fields == FIELDS
        and policy.combine == "union"
        and policy.ensure_at_least_one
        and not policy.supervise_unknown
    )
    with r.prepare(contract, dataset, engine) as plan:
        fitted = plan.materialize(
            probe_anchor_table(plan, requests[:2]),
            fit_split="train",
            encoder_cutoff=int(CUTOFF.timestamp()) - 1,
        )
        # Fit only training rows, then inspect the complete fixed probe population.
        table = replace(probe_anchor_table(plan, requests), split=None)
        task = plan.materialize(table, fitted=fitted.fitted_state)
        batch = next(task.batches(batch_size=4, seed=7, epoch=0, output_format="model"))
        model = build_model(task, SELECTION, "pragma", dim=8, seed=17).eval()
        yield dict(
            fixture=fixture,
            requests=requests,
            anchors=anchors,
            reference=reference,
            identities={str(i): i for i in range(1, 5)},
            task=task,
            batch=batch,
            model=model,
            contract=contract,
            source=source,
            dataset=dataset,
            engine=engine,
            table=table,
        )


def originals(case, batch):
    return original_tokens(
        batch,
        case["anchors"],
        case["reference"],
        case["task"].fitted_state.to_dict(),
        case["identities"],
    )


def test_source_targets_head_positions_and_smoothed_loss(case):
    batch, model = case["batch"], case["model"]
    original = originals(case, batch)
    assert 1 in original  # Held-out JPY/new_format are train-vocabulary unknowns.
    # The null-valued transaction keeps direction and format; missing fields vanish.
    assert 2 in torch.diff(batch.inputs["history"]["field_offsets"]).tolist()
    verify_targets(batch, original)
    result = model.training_step(batch)
    metrics = verify_loss(batch, result)
    indices, labels, _, _ = selected_labels(batch)
    output = model.backbone.forward_kumo(batch)
    expected_logits = model.head(
        output, model.backbone.embedding_weight, torch.tensor(indices)
    )
    torch.testing.assert_close(result.logits, expected_logits, rtol=0, atol=0)
    assert metrics["eligible"] > 0 and metrics["invalid"] > 0
    result.logits.retain_grad()
    result.loss.backward()
    for row, gradient, label in zip(
        result.logits.detach().tolist(),
        result.logits.grad.tolist(),
        labels,
        strict=True,
    ):
        peak = max(row)
        exp = [math.exp(value - peak) for value in row]
        for index, actual in enumerate(gradient):
            expected = (
                exp[index] / sum(exp) - (0.9 if index == label else 0) - 0.1 / len(row)
            ) / len(labels)
            assert actual == pytest.approx(expected, abs=3e-7)
    model.zero_grad(set_to_none=True)


@pytest.mark.parametrize("unknown_probability", (0.0, 1.0))
def test_known_all_selected_policy_and_unknown_replacement(case, unknown_probability):
    policy = replace(
        case["contract"].batch.targets["masked_values"],
        token_probability=1.0,
        event_probability=0.0,
        key_probability=0.0,
        unknown_probability=unknown_probability,
    )
    contract = replace(
        case["contract"],
        batch=replace(case["contract"].batch, targets={"masked_values": policy}),
    )
    with r.prepare(contract, case["dataset"], case["engine"]) as plan:
        task = plan.materialize(case["table"], fitted=case["task"].fitted_state.inputs)
        assert (
            task.fitted_state.to_dict()["encoders"]
            == case["task"].fitted_state.to_dict()["encoders"]
        )
        batch = next(task.batches(batch_size=4, seed=7, output_format="model"))
        original = originals(case, batch)
        verify_targets(batch, original)
        target = batch.targets["masked_values"]
        if unknown_probability == 0:
            assert target["valid"].all() and target["values"].tolist() == original
            assert (batch.inputs["history"]["value_ids"] == 2).all()
            assert 2 in selected_labels(batch)[1]  # Original UNK remaps to PRAGMA UNK.
        else:
            # All context is corrupted; the guaranteed first field per sample
            # becomes MASK, while other positions become unsupervised UNK.
            offsets = batch.inputs["history"]["field_offsets"].tolist()
            owners = batch.inputs["history"]["owner"].tolist()
            first = [offsets[owners.index(owner)] for owner in range(4)]
            assert target["valid"].nonzero().flatten().tolist() == first
            assert all(
                value == (2 if index in first else 1)
                for index, value in enumerate(
                    batch.inputs["history"]["value_ids"].tolist()
                )
            )
        verify_loss(batch, case["model"].training_step(batch))


def test_padding_invalid_ignore_and_empty_objectives(case):
    batch = deepcopy(case["batch"])
    batch.metadata["is_padding"][1] = True
    target = batch.targets["masked_values"]
    indices = target["valid"].nonzero().flatten().tolist()
    target["valid"][indices[0]] = False
    target["values"][indices[1]] = -100  # Explicit ignore ID is independently excluded.
    metrics = verify_loss(batch, case["model"].training_step(batch))
    assert metrics["eligible"] > 0 and metrics["padding"] > 0
    for mode in ("all_padding", "all_invalid"):
        empty = deepcopy(case["batch"])
        if mode == "all_padding":
            empty.metadata["is_padding"][:] = True
        else:
            empty.targets["masked_values"]["valid"][:] = False
            empty.targets["masked_values"]["values"][:] = -100
        result = case["model"].training_step(empty)
        assert verify_loss(empty, result)["mean_loss"] is None
        result.loss.backward()
        case["model"].zero_grad(set_to_none=True)


def test_future_and_evaluation_fields_do_not_change_training_inputs(case):
    fixture = deepcopy(case["fixture"])
    for table in ("transactions", "account_events"):
        for row in fixture.tables[table]:
            row["is_laundering"] = 99
            if row["event_time"] >= CUTOFF:
                row["payment_format"], row["amount_paid"] = "FUTURE_POISON", 123456789.0
    dataset, engine = case["source"](fixture, "future")
    with r.prepare(case["contract"], case["dataset"], case["engine"]) as original:
        with original.materialize(
            probe_anchor_table(original, case["requests"][:2]),
            fitted=case["task"].fitted_state,
        ) as unchanged:
            expected = next(
                unchanged.batches(batch_size=2, seed=7, output_format="model")
            )
    with r.prepare(case["contract"], dataset, engine) as plan:
        task = plan.materialize(
            probe_anchor_table(plan, case["requests"][:2]),
            fitted=case["task"].fitted_state,
        )
        actual = next(task.batches(batch_size=2, seed=7, output_format="model"))
        assert_equal(actual.inputs, expected.inputs)
        assert_equal(actual.targets, expected.targets)
