# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task checks across three models and both engines, using source-level goldens."""

import math
from copy import deepcopy
from dataclasses import replace
from test.kumo_pytorch.bindings import selections as catalog_selections
from test.kumo_pytorch.contracts import build_contract
from test.kumo_pytorch.involvement import (
    expected_anchor,
    verify_batch,
    verify_inputs,
)
from test.kumo_pytorch.models import build_model

import pytest
import torch

import kumo.pytorch.api as r

from .fixtures import CUTOFF, build_fixture, probe_anchor_table

T = int(CUTOFF.timestamp())
RECORDS = catalog_selections()


@pytest.fixture(scope="module", autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module", params=("diskgraph",))
def backend(request):
    return request.param


@pytest.fixture(scope="module", params=("graphsage", "nuformer", "combined"))
def task_case(request, backend, tmp_path_factory):
    family = request.param
    fixture = build_fixture()
    for row in fixture.tables["transactions"]:
        row["is_laundering"] = int(row["transaction_id"] in (106, 110))
    reference = {
        "events": [
            {
                "id": row["transaction_id"],
                "time": row["event_time_seconds"],
                "accounts": {str(row["sender_id"]), str(row["receiver_id"])},
                "sender": str(row["sender_id"]),
                "receiver": str(row["receiver_id"]),
                "laundering": bool(row["is_laundering"]),
            }
            for row in fixture.tables["transactions"]
        ],
        "start": T - 40 * 86400,
        "cuts": [T + 2 * 86400, T + 4 * 86400],
        "end": T + 6 * 86400,
    }
    identities = {str(index): index for index in range(1, 5)}
    rows = [
        {"sample_id": f"a{index}", "entity_id": str(index), "as_of_time": T * 10**9}
        for index in range(1, 5)
    ]
    expected = {row["sample_id"]: expected_anchor(row, reference)[1] for row in rows}
    assert expected == {"a1": 1, "a2": 0, "a3": 1, "a4": 1}
    # A separately supplied missing external label tests masking, not anchor truth.
    rows.append({**rows[0], "sample_id": "missing-label"})
    expected["missing-label"] = None
    anchors = {row["sample_id"]: row for row in rows}
    graph = family != "nuformer"
    requests = tuple(
        r.SampleRequest(
            row["sample_id"],
            identities[row["entity_id"]],
            T - int(graph),
            targets=(r.TargetValue("label", expected[row["sample_id"]]),),
            split="train",
            metadata={"prediction_time_ns": T * 10**9} if graph else {},
        )
        for row in rows
    )
    contract = build_contract(fixture.schema, family, max_events=11)
    selection = next(
        record
        for record in RECORDS
        if family in record["models"]
        and record["selector"]["workload_id"] == "account-involvement-next24h"
    )
    directory = tmp_path_factory.mktemp(f"involvement-{family}-{backend}")

    def source(data, name):
        return data.parquet_dataset(directory / name), r.DiskGraphEngine(
            index=directory / f"{name}-index", num_threads=1
        )

    dataset, engine = source(fixture, "original")
    with r.prepare(contract, dataset, engine) as plan:
        fitted = plan.materialize(
            probe_anchor_table(plan, requests[:2]),
            fit_split="train",
            encoder_cutoff=T - 1,
        )
        task = plan.materialize(
            probe_anchor_table(plan, requests), fitted=fitted.fitted_state
        )
        batch = next(task.batches(batch_size=8, output_format="model"))
        model = build_model(task, selection, family, dim=8, seed=17).eval()
        yield {
            "fixture": fixture,
            "reference": reference,
            "identities": identities,
            "anchors": anchors,
            "rows": rows,
            "expected": expected,
            "family": family,
            "requests": requests,
            "contract": contract,
            "task": task,
            "batch": batch,
            "model": model,
            "source": source,
        }


def check_inputs(case, batch):
    verify_inputs(
        batch,
        case["family"],
        case["anchors"],
        case["reference"],
        case["identities"],
        case["rows"][:2],
    )


def equal_tree(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0, equal_nan=True)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal_tree(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            equal_tree(a, b)
    else:
        assert left == right


def test_targets_inputs_loss_and_evaluation(task_case):
    case = task_case
    batch = case["batch"]
    check_inputs(case, batch)
    result = case["model"].training_step(batch)
    reference = verify_batch(batch, result, case["expected"])
    assert (
        reference["eligible"] == 4
        and reference["invalid"] == 1
        and reference["padding"] == 0
    )
    # Independent derivative: excluded logits must receive exactly zero gradient.
    result.logits.retain_grad()
    result.loss.backward()
    gradients = result.logits.grad.flatten().tolist()
    for index, sid in enumerate(batch.metadata["sample_ids"]):
        label = case["expected"][sid]
        score = result.logits.flatten()[index].item()
        expected = 0 if label is None else (1 / (1 + math.exp(-score)) - label) / 4
        assert gradients[index] == pytest.approx(expected, abs=2e-7)
    case["model"].zero_grad(set_to_none=True)


def test_distributed_padding_and_no_eligible_examples(task_case):
    case = task_case
    batch = next(
        case["task"].batches(batch_size=8, rank=1, world_size=2, output_format="model")
    )
    assert batch.metadata["is_padding"].any()
    # Existing test_padding_identity_native owns distributed identity/delivery.
    # Here only the AML model's loss exclusion is under test.
    batch.targets["label"]["values"][batch.metadata["is_padding"]] = float("nan")
    result = case["model"].training_step(batch)
    metrics = verify_batch(batch, result, case["expected"])
    assert metrics["padding"] == 1
    for mode in ("all_padding", "all_invalid"):
        empty = deepcopy(case["batch"])
        labels = case["expected"]
        if mode == "all_padding":
            empty.metadata["is_padding"][:] = True
        else:
            labels = dict.fromkeys(labels)
            empty.targets["label"]["valid"][:] = False
            empty.targets["label"]["values"][:] = float("nan")
        result = case["model"].training_step(empty)
        reference = verify_batch(empty, result, labels)
        assert (
            reference["eligible"] == 0
            and reference["mean_loss"] is None
            and result.loss.item() == 0
        )
        result.loss.backward()
        case["model"].zero_grad(set_to_none=True)


def test_future_and_label_changes_cannot_change_inputs(task_case):
    case = task_case
    fixture = deepcopy(case["fixture"])
    for table in ("transactions", "account_events"):
        for row in fixture.tables[table]:
            row["is_laundering"] = 99  # Evaluation-only; not a permitted feature.
            if row["event_time_seconds"] >= T:
                row["amount_paid"], row["payment_format"] = 987654321.0, "FUTURE_POISON"
    dataset, engine = case["source"](fixture, "future-poison")
    changed_labels = tuple(
        replace(a, targets=(r.TargetValue("label", 0),)) for a in case["requests"]
    )
    with r.prepare(case["contract"], dataset, engine) as plan:
        task = plan.materialize(
            probe_anchor_table(plan, changed_labels), fitted=case["task"].fitted_state
        )
        batch = next(task.batches(batch_size=8, output_format="model"))
        equal_tree(batch.inputs, case["batch"].inputs)


def test_historical_change_is_observable_at_the_correct_token_position(task_case):
    case = task_case
    fixture = deepcopy(case["fixture"])
    for table in ("transactions", "account_events"):
        for row in fixture.tables[table]:
            if row["transaction_id"] == 111:
                row["payment_format"] = "PAST_CHANGE"
    dataset, engine = case["source"](fixture, "past-change")
    with r.prepare(case["contract"], dataset, engine) as plan:
        task = plan.materialize(
            probe_anchor_table(plan, case["requests"]), fitted=case["task"].fitted_state
        )
        batch = next(task.batches(batch_size=8, output_format="model"))
        check_inputs(case, batch)
        if case["family"] != "graphsage":
            changed = (
                batch.inputs["history"]["value_ids"]
                != case["batch"].inputs["history"]["value_ids"]
            )
            # Last event in each chronological stream; payment_format is field 6.
            assert changed[0].nonzero().flatten().tolist() == [6 * 11 + 6]
            assert changed[1].nonzero().flatten().tolist() == [5 * 11 + 6]
        else:
            before = case["batch"].inputs["graph"]["nodes"]["transactions"]["features"][
                "payment_format"
            ]["values"]
            after = batch.inputs["graph"]["nodes"]["transactions"]["features"][
                "payment_format"
            ]["values"]
            assert not torch.equal(before, after)
