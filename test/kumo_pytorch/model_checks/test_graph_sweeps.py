# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AML graph fan-outs with even-degree and hot-account source fixtures."""

from dataclasses import replace
from datetime import timedelta
from test.kumo_pytorch.contracts import build_contract
from test.kumo_pytorch.involvement import history, verify_inputs
from test.kumo_pytorch.models.aml import build_model
from test.kumo_pytorch.reference_tokens import history_tokens

import pytest
import torch

import kumo.pytorch.api as r

from .fixtures import (
    CUTOFF,
    VALUE_FIELDS,
    AMLFixture,
    build_fixture,
    probe_anchor_table,
)
from .test_aml import CASES

T = int(CUTOFF.timestamp())
GRAPH_CASES = [
    (selection, family)
    for selection, family in CASES
    if family in {"graphsage", "combined"}
]


@pytest.fixture(scope="module", autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module", params=("uniform", "skewed"))
def source(request, tmp_path_factory):
    degrees = (192, 192) if request.param == "uniform" else (384, 3)
    base = build_fixture()
    transactions, touches, events = [], [], []
    for root, degree in enumerate(degrees, 1):
        # T-1 is eligible under graph v2; T and T+1 must be excluded.
        ages = [1 + 2000 * i for i in range(degree)] + [0, -1]
        for position, age in enumerate(ages):
            identity = 10000 * root + position
            sender, receiver = ((root, 3), (3, root), (root, root))[position % 3]
            stamp = CUTOFF - timedelta(seconds=age)
            values = dict(
                amount_paid=float(position % 97 - 20),
                amount_received=float(position % 89),
                payment_currency=("USD", "EUR")[position % 2],
                receiving_currency="USD",
                payment_format="FUTURE_ONLY" if age <= 0 else "wire",
            )
            transactions.append(
                dict(
                    transaction_id=identity,
                    sender_id=sender,
                    receiver_id=receiver,
                    event_time=stamp,
                    event_time_seconds=T - age,
                    **values,
                )
            )
            roles = (
                [(root, "self")]
                if sender == receiver
                else [(sender, "outgoing"), (receiver, "incoming")]
            )
            for i, (account, direction) in enumerate(roles):
                touches.append(
                    dict(
                        event_id=2 * identity + i,
                        transaction_id=identity,
                        account_id=account,
                        direction=direction,
                        event_time=stamp,
                        event_time_seconds=T - age,
                        **values,
                    )
                )
            events.append(
                dict(
                    id=identity,
                    time=T - age,
                    sender=str(sender),
                    receiver=str(receiver),
                    accounts={str(sender), str(receiver)},
                    values={name: values[name] for name in VALUE_FIELDS},
                )
            )
    tables = dict(
        accounts=[
            dict(
                account_id=i,
                first_observed_time=min(
                    row["event_time"]
                    for row in transactions
                    if i in (row["sender_id"], row["receiver_id"])
                ),
            )
            for i in (1, 2, 3)
        ],
        transactions=transactions[::-1],
        account_events=touches[::-1],
    )
    fixture = AMLFixture(tables, base.schema, base.training)
    directory = tmp_path_factory.mktemp(f"aml-graph-{request.param}")
    return (
        fixture,
        fixture.parquet_dataset(directory / "sources"),
        directory / "index",
        {"events": events},
        degrees,
    )


@pytest.mark.parametrize("fanout", (2, 32, 128))
@pytest.mark.parametrize(
    "selection,family", GRAPH_CASES, ids=[family for _, family in GRAPH_CASES]
)
def test_aml_graph_degree_and_fanout(source, selection, family, fanout):
    fixture, dataset, index, reference, degrees = source
    anchors = {
        row.sample_id: dict(
            sample_id=row.sample_id, entity_id=str(row.entity_id), as_of_time=T * 10**9
        )
        for row in fixture.training
    }
    requests = tuple(
        replace(
            row,
            as_of_time=CUTOFF - timedelta(seconds=1),
            metadata={"prediction_time_ns": T * 10**9},
        )
        for row in fixture.training
    )
    base = build_contract(fixture.schema, family, max_events=max(degrees))
    neighborhood = base.sample.neighborhoods["history_graph"]
    contract = replace(
        base,
        sample=replace(
            base.sample,
            neighborhoods={
                "history_graph": replace(
                    neighborhood,
                    layers=(
                        replace(
                            neighborhood.layers[0],
                            samples=tuple(
                                replace(sample, fanout=fanout)
                                for sample in neighborhood.layers[0].samples
                            ),
                        ),
                        *neighborhood.layers[1:],
                    ),
                )
            },
        ),
    )
    # Only first-hop per-role fan-outs change. Sequence/count declarations and
    # endpoint expansion retain the published two-layer AML semantics.
    assert contract.batch == base.batch
    assert contract.sample.events == base.sample.events
    assert contract.sample.entity_values == base.sample.entity_values
    selected = set()
    for anchor, degree in zip(anchors.values(), degrees, strict=True):
        past = history(anchor, reference)
        assert len(past) == degree
        for role in ("sender", "receiver"):
            candidates = [row for row in past if row[role] == anchor["entity_id"]]
            selected.update(row["id"] for row in candidates[-fanout:])
    expected_graph = {
        "events": [row for row in reference["events"] if row["id"] in selected]
    }
    identities = {str(i): i for i in (1, 2, 3)}
    engine = r.DiskGraphEngine(index=index, num_threads=1)
    with (
        r.prepare(contract, dataset, engine) as plan,
        plan.materialize(
            probe_anchor_table(plan, requests), fit_split="train", encoder_cutoff=T - 1
        ) as task,
    ):
        fingerprint = task.fitted_state.fingerprint
        state = task.fitted_state.to_dict()
        assert all(
            "FUTURE_ONLY" not in dict(encoder.get("entries", ()))
            for encoder in state["encoders"].values()
        )
        batches = list(task.batches(batch_size=2, seed=7, output_format="model"))
        assert len(batches) == 1
        batch = batches[0]
        assert list(batch.metadata["sample_ids"]) == list(anchors)
        verify_inputs(
            batch,
            family,
            anchors,
            reference,
            identities,
            list(anchors.values()),
            graph_reference=expected_graph,
        )
        assert batch.targets["label"]["valid"].all()
        assert batch.targets["label"]["values"].flatten().tolist() == [1.0, 0.0]
        if len(selected) < sum(degrees):
            # A full-history expectation must reject this intentionally
            # sampled graph; sequence/count expectations still remain full.
            with pytest.raises(AssertionError, match="Graph history differs"):
                verify_inputs(
                    batch,
                    family,
                    anchors,
                    reference,
                    identities,
                    list(anchors.values()),
                )
        if family == "combined":
            stream = batch.inputs["history"]
            for owner, anchor in enumerate(anchors.values()):
                tokens = history_tokens(anchor, history(anchor, reference), state)
                assert (
                    stream["value_ids"][owner][stream["valid"][owner]].tolist()
                    == tokens
                )
        model = build_model(
            task, selection, family, dim=8, max_seq_len=11 * max(degrees), seed=17
        )
        model.backbone.validate_kumo_batch(batch)
        assert task.fitted_state.fingerprint == fingerprint
