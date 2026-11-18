# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AML next-token task and copied nuFormer consumer integration."""

import math
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
from test.kumo_pytorch.bindings import selections as catalog_selections
from test.kumo_pytorch.contracts import build_contract
from test.kumo_pytorch.models import StepResult, build_model
from test.kumo_pytorch.next_token import (
    selected_labels,
    verify_loss,
    verify_targets,
)
from test.kumo_pytorch.reference_tokens import FIELDS

import pytest
import torch

import kumo.pytorch.api as r

from .fixtures import CUTOFF, build_fixture, probe_anchor_table
from .test_aml import assert_equal

SELECTION = next(
    row
    for row in catalog_selections()
    if row["selector"]["workload_id"] == "next-token-pretrain"
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
        r.SampleRequest("one-event", 4, CUTOFF, split="test"),
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
    root = tmp_path_factory.mktemp("next-token-aml")

    def source(data, name):
        return data.parquet_dataset(root / name), r.DiskGraphEngine(
            index=root / f"{name}-index", num_threads=1
        )

    dataset, engine = source(fixture, "original")
    contract = build_contract(fixture.schema, "nuformer", native=True, max_events=11)
    assert contract.batch.targets["next_token"] == r.NextTokenTarget("history")
    with r.prepare(contract, dataset, engine) as plan:
        fitted = plan.materialize(
            probe_anchor_table(plan, requests[:2]),
            fit_split="train",
            encoder_cutoff=int(CUTOFF.timestamp()) - 1,
        )
        task = plan.materialize(
            replace(probe_anchor_table(plan, requests), split=None),
            fitted=fitted.fitted_state,
        )
        batch = next(task.batches(batch_size=4, seed=7, output_format="model"))
        model = build_model(task, SELECTION, "nuformer", dim=8, seed=17).eval()
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
        )


def check_targets(case, batch):
    return verify_targets(
        batch,
        case["anchors"],
        case["reference"],
        case["task"].fitted_state.to_dict(),
        case["identities"],
    )


def test_source_alignment_separator_unknown_and_terminal_targets(case):
    batch = case["batch"]
    sequences = check_targets(case, batch)
    assert [len(row) for row in sequences] == [76, 65, 10, 98]
    assert 1 in sequences[-1]  # Future-at-fit JPY/new_format are now historical UNK.
    assert all(row[-1] != 5 for row in sequences)  # No trailing separator.
    result = case["model"].training_step(batch)
    metrics = verify_loss(batch, result)
    assert metrics["eligible"] == sum(len(row) - 1 for row in sequences)
    assert (
        metrics["invalid"]
        == batch.targets["next_token"]["valid"].numel() - metrics["eligible"]
    )
    # The terminal *original* token is the final answer, not an input with a
    # made-up successor. Cross-event separators and UNK are ordinary targets.
    for owner, row in enumerate(sequences):
        assert (
            batch.targets["next_token"]["values"][owner, len(row) - 2].item() == row[-1]
        )
    result.logits.retain_grad()
    result.loss.backward()
    positions, labels, _, _ = selected_labels(batch)
    for (owner, index), label in zip(positions, labels, strict=True):
        row = result.logits[owner, index].detach().tolist()
        peak = max(row)
        exp = [math.exp(value - peak) for value in row]
        gradient = result.logits.grad[owner, index].tolist()
        for token, actual in enumerate(gradient):
            expected = (exp[token] / sum(exp) - int(token == label)) / len(labels)
            assert actual == pytest.approx(expected, abs=3e-7)
    assert (result.logits.grad[~batch.targets["next_token"]["valid"]] == 0).all()
    case["model"].zero_grad(set_to_none=True)


def check_causality(case):
    batch = case["batch"]
    altered = deepcopy(batch)
    ids = altered.inputs["history"]["value_ids"]
    end = int(altered.inputs["history"]["valid"][0].sum())
    start = 12
    ids[0, start:end] = (ids[0, start:end] + 1 - 6) % (
        case["model"].backbone.config.vocab_size - 6
    ) + 6
    with torch.no_grad():
        original = case["model"](batch)
        changed = case["model"](altered)
    torch.testing.assert_close(original[0, :start], changed[0, :start], rtol=0, atol=0)
    torch.testing.assert_close(original[1:], changed[1:], rtol=0, atol=0)
    assert not torch.equal(original[0, start:end], changed[0, start:end]), (
        "Suffix perturbation had no effect"
    )


def test_causal_prefix_and_example_isolation(case):
    check_causality(case)


def test_causal_check_rejects_bidirectional_attention(case, monkeypatch):
    attention = torch.nn.functional.scaled_dot_product_attention

    def bidirectional(*args, **kwargs):
        kwargs["is_causal"] = False
        return attention(*args, **kwargs)

    monkeypatch.setattr(
        torch.nn.functional, "scaled_dot_product_attention", bidirectional
    )
    with pytest.raises(AssertionError):
        check_causality(case)


def test_target_values_and_padding_ids_cannot_change_predictions(case):
    batch = case["batch"]
    changed = deepcopy(batch)
    changed.targets["next_token"]["values"][:] = 1
    changed.inputs["history"]["value_ids"][~changed.inputs["history"]["valid"]] = 1
    with torch.no_grad():
        torch.testing.assert_close(
            case["model"](changed), case["model"](batch), rtol=0, atol=0
        )


def test_loss_excludes_padding_invalid_ignore_and_empty_objectives(case):
    batch = deepcopy(case["batch"])
    batch.metadata["is_padding"][1] = True
    batch.targets["next_token"]["valid"][0, 0] = False
    batch.targets["next_token"]["values"][0, 1] = -100
    result = case["model"].training_step(batch)
    metrics = verify_loss(batch, result)
    assert (
        metrics["eligible"] > 0
        and metrics["padding"] == batch.inputs["history"]["valid"].shape[1]
    )
    result.logits.retain_grad()
    result.loss.backward()
    assert (result.logits.grad[1] == 0).all() and (result.logits.grad[0, :2] == 0).all()
    case["model"].zero_grad(set_to_none=True)
    for mode in ("all_padding", "all_invalid"):
        empty = deepcopy(case["batch"])
        if mode == "all_padding":
            empty.metadata["is_padding"][:] = True
        else:
            empty.targets["next_token"]["valid"][:] = False
            empty.targets["next_token"]["values"][:] = -100
        result = case["model"].training_step(empty)
        assert verify_loss(empty, result)["mean_loss"] is None
        result.loss.backward()
        case["model"].zero_grad(set_to_none=True)


@pytest.mark.parametrize(
    "damage",
    (
        "double_shift",
        "target",
        "mask",
        "history",
        "cutoff",
        "leakage",
        "loss",
        "weight",
    ),
)
def test_bad_outputs_are_detected(case, damage):
    batch = deepcopy(case["batch"])
    result = case["model"].training_step(batch)
    target = batch.targets["next_token"]
    if damage == "double_shift":
        target["values"] = target["values"].roll(-1, dims=1)
    elif damage == "target":
        target["values"][0, 0] = 0
    elif damage == "mask":
        target["valid"][0, 0] = False
    elif damage == "history":
        batch.inputs["history"]["value_ids"][0, 0] = 0
    elif damage == "cutoff":
        batch.metadata["as_of_time"][0] += 1
    elif damage == "leakage":
        batch.inputs["target_values"] = target
    elif damage == "loss":
        result.loss = result.loss + 0.1
    else:
        result.weight += 1
    with pytest.raises(AssertionError):
        check_targets(case, batch)
        verify_loss(batch, result)


def test_future_and_evaluation_fields_do_not_change_training_inputs(case):
    fixture = deepcopy(case["fixture"])
    for table in ("transactions", "account_events"):
        for row in fixture.tables[table]:
            row["is_laundering"] = 99
            if row["event_time"] >= CUTOFF:
                row["amount_paid"], row["payment_format"] = 123456789.0, "FUTURE_POISON"
    dataset, engine = case["source"](fixture, "future")
    with r.prepare(case["contract"], case["dataset"], case["engine"]) as original:
        with original.materialize(
            probe_anchor_table(original, case["requests"][:2]),
            fitted=case["task"].fitted_state,
        ) as unchanged:
            expected = next(unchanged.batches(batch_size=2, output_format="model"))
    with r.prepare(case["contract"], dataset, engine) as plan:
        task = plan.materialize(
            probe_anchor_table(plan, case["requests"][:2]),
            fitted=case["task"].fitted_state,
        )
        actual = next(task.batches(batch_size=2, output_format="model"))
        assert_equal(actual.inputs, expected.inputs)
        assert_equal(actual.targets, expected.targets)


def test_missing_required_field_fails_explicitly(case):
    # This declared layout uses missing='error'; do not silently drop source fields.
    request = r.SampleRequest(
        "missing-fields", 4, CUTOFF + timedelta(days=1), split="test"
    )
    with r.prepare(case["contract"], case["dataset"], case["engine"]) as plan:
        with plan.materialize(
            probe_anchor_table(plan, [request]), fitted=case["task"].fitted_state
        ) as task:
            with pytest.raises(ValueError, match="NULL.*missing='error'"):
                next(task.batches(output_format="model"))


def test_scalar_loss_known_values_and_no_second_shift():
    from types import SimpleNamespace

    batch = SimpleNamespace(
        inputs={"history": {"valid": torch.tensor([[True, True]])}},
        targets={
            "next_token": {
                "values": torch.tensor([[1, 0]]),
                "valid": torch.tensor([[True, True]]),
            }
        },
        metadata={"is_padding": torch.tensor([False])},
    )
    # Aligned answers have probabilities 3/4 and 1/2 respectively.
    expected = (-math.log(0.75) - math.log(0.5)) / 2
    result = StepResult(
        torch.tensor([[[0.0, math.log(3)], [0.0, 0.0]]]), torch.tensor(expected), 2
    )
    assert verify_loss(batch, result)["mean_loss"] == pytest.approx(expected)
