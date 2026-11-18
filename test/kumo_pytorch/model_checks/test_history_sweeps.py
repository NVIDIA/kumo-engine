# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AML history sizes and explicit selection variants through DiskGraph.

These are batch correctness/input compatibility checks, not long-sequence
training benchmarks. Published full-history workload declarations stay intact.
"""

from dataclasses import replace
from datetime import timedelta
from test.kumo_pytorch.contracts import build_contract, window
from test.kumo_pytorch.involvement import history, verify_inputs
from test.kumo_pytorch.masked_values import original_tokens
from test.kumo_pytorch.masked_values import verify_targets as verify_masked
from test.kumo_pytorch.models.aml import build_model
from test.kumo_pytorch.next_token import verify_targets as verify_next
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

CASES = [(selection, family) for selection, family in CASES if family != "graphsage"]
PROFILES = (
    "full",
    "count_limit",
    "time_window",
    "two_tables",
    "two_tables_window_limit",
)
T = int(CUTOFF.timestamp())


@pytest.fixture(scope="module", autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module", params=(16, 128, 1024))
def source(request, tmp_path_factory):
    length = request.param
    base = build_fixture()
    transactions, touches, reference = [], [], []
    for root, count in ((1, length), (2, length - 1)):
        # Unequal histories exercise padding. Rows at T and T+1 must be excluded.
        for index, age in enumerate(range(count, -2, -1)):
            identity = root * 10000 + index
            sender, receiver = ((root, 3), (3, root), (root, root))[index % 3]
            stamp = CUTOFF - timedelta(seconds=age)
            row = dict(
                transaction_id=identity,
                sender_id=sender,
                receiver_id=receiver,
                event_time=stamp,
                event_time_seconds=T - age,
                amount_paid=float(index % 97 - 20),
                amount_received=float(index % 89),
                payment_currency=("USD", "EUR")[index % 2],
                receiving_currency="USD",
                payment_format="FUTURE_ONLY"
                if age <= 0
                else ("wire", "card", "cash")[index % 3],
            )
            transactions.append(row)
            roles = (
                [(root, "self")]
                if sender == receiver
                else [(sender, "outgoing"), (receiver, "incoming")]
            )
            for role_index, (account, direction) in enumerate(roles):
                touches.append(
                    dict(
                        event_id=2 * identity + role_index,
                        account_id=account,
                        transaction_id=identity,
                        event_time=stamp,
                        event_time_seconds=T - age,
                        direction=direction,
                        **{name: row[name] for name in VALUE_FIELDS},
                    )
                )
            reference.append(
                dict(
                    id=identity,
                    time=T - age,
                    accounts={str(sender), str(receiver)},
                    sender=str(sender),
                    receiver=str(receiver),
                    values={name: row[name] for name in VALUE_FIELDS},
                )
            )
    tables = {
        "accounts": [
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
        # Deliberately reverse physical order; expected histories use source time.
        "transactions": transactions[::-1],
        "account_events": touches[::-1],
        **{
            f"account_events_{parity}": [
                row for row in reversed(touches) if row["transaction_id"] % 2 == parity
            ]
            for parity in (0, 1)
        },
    }
    event_table = next(
        table for table in base.schema.tables if table.name == "account_events"
    )
    schema = replace(
        base.schema,
        tables=base.schema.tables
        + tuple(replace(event_table, name=f"account_events_{i}") for i in (0, 1)),
        relations=base.schema.relations
        + tuple(
            r.RelationSpec(
                f"touches_{i}",
                "accounts",
                "account_id",
                f"account_events_{i}",
                "account_id",
            )
            for i in (0, 1)
        ),
    )
    fixture = AMLFixture(tables, schema, base.training)
    directory = tmp_path_factory.mktemp(f"aml-histories-{length}")
    return dict(
        length=length,
        fixture=fixture,
        dataset=fixture.parquet_dataset(directory / "sources"),
        index=directory / "index",
        reference={"events": reference},
        anchors={
            row.sample_id: {
                "sample_id": row.sample_id,
                "entity_id": str(row.entity_id),
                "as_of_time": T * 10**9,
            }
            for row in fixture.training
        },
    )


def history_variant(contract, profile, length, graph):
    """Only the named history selection changes; graph/count declarations do not."""
    events = dict(contract.sample.events)
    source = contract.batch.inputs["history"].source
    if profile.startswith("two_tables"):
        event = events["transactions"]
        events = {
            f"part_{i}": replace(
                event, rows=replace(event.rows, path=(f"touches_{i}",))
            )
            for i in (0, 1)
        }
        source = replace(source, events=tuple(events))
    if profile in ("time_window", "two_tables_window_limit"):
        events = {
            name: replace(
                event,
                rows=replace(
                    event.rows, time=window("event_time", -length // 2, inclusive=graph)
                ),
            )
            for name, event in events.items()
        }
    if profile in ("count_limit", "two_tables_window_limit"):
        limit = length - 3 if profile == "count_limit" else length // 4
        source = replace(source, max_events=limit, keep="last")
    return replace(
        contract,
        sample=replace(contract.sample, events=events),
        batch=replace(
            contract.batch,
            inputs={
                **contract.batch.inputs,
                "history": replace(contract.batch.inputs["history"], source=source),
            },
        ),
    )


def selected_reference(source, profile):
    """Select expected rows independently of Kumo's contract/selection helpers."""
    length = source["length"]
    selected = []
    for anchor in source["anchors"].values():
        rows = history(anchor, source["reference"])
        if profile in ("time_window", "two_tables_window_limit"):
            rows = [row for row in rows if T - length // 2 <= row["time"] < T]
        if profile == "count_limit":
            rows = rows[-(length - 3):]
        elif profile == "two_tables_window_limit":
            rows = rows[-(length // 4):]
        selected.extend(rows)
    return {"events": selected}


@pytest.mark.parametrize(
    "selection,family",
    CASES,
    ids=[
        f"{selection['selector']['workload_id']}-{family}"
        for selection, family in CASES
    ],
)
@pytest.mark.parametrize("profile", PROFILES)
def test_history_sizes_and_selection(source, selection, family, profile):
    fixture, length = source["fixture"], source["length"]
    graph = family == "combined"
    native = selection["selector"]["workload_id"] != "account-involvement-next24h"
    requests = tuple(
        replace(
            row,
            as_of_time=CUTOFF - timedelta(seconds=int(graph)),
            targets=() if native else row.targets,
            metadata={"prediction_time_ns": T * 10**9} if graph else {},
        )
        for row in fixture.training
    )
    base = build_contract(fixture.schema, family, native=native, max_events=length)
    contract = history_variant(base, profile, length, graph)
    assert contract.sample.neighborhoods == base.sample.neighborhoods
    assert contract.sample.entity_values == base.sample.entity_values
    expected = selected_reference(source, profile)
    counts = [len(history(anchor, expected)) for anchor in source["anchors"].values()]
    expected_counts = {
        "full": [length, length - 1],
        "two_tables": [length, length - 1],
        "count_limit": [length - 3] * 2,
        "time_window": [length // 2] * 2,
        "two_tables_window_limit": [length // 4] * 2,
    }[profile]
    assert counts == expected_counts
    engine = r.DiskGraphEngine(index=source["index"], num_threads=1)
    with r.prepare(contract, source["dataset"], engine) as plan:
        with plan.materialize(
            probe_anchor_table(plan, requests), fit_split="train", encoder_cutoff=T - 1
        ) as task:
            fingerprint = task.fitted_state.fingerprint
            batches = list(
                task.batches(batch_size=2, seed=7, epoch=0, output_format="model")
            )
            assert len(batches) == 1
            batch = batches[0]
            assert list(batch.metadata["sample_ids"]) == list(source["anchors"])
            state = task.fitted_state.to_dict()
            assert all(
                "FUTURE_ONLY" not in dict(encoder.get("entries", ()))
                for encoder in state["encoders"].values()
            )
            identities = {str(i): i for i in (1, 2, 3)}
            model = build_model(
                task, selection, family, dim=8, max_seq_len=11 * length, seed=17
            )
            if family == "pragma":
                event_families = (
                    {
                        event["id"]: f"part_{event['id'] % 2}"
                        for event in expected["events"]
                    }
                    if profile.startswith("two_tables")
                    else None
                )
                original = original_tokens(
                    batch,
                    source["anchors"],
                    expected,
                    state,
                    identities,
                    event_families=event_families,
                )
                if event_families is not None:
                    with pytest.raises(
                        AssertionError, match="Source fields/order differ"
                    ):
                        original_tokens(
                            batch,
                            source["anchors"],
                            expected,
                            state,
                            identities,
                            event_families={
                                event["id"]: f"part_{1 - event['id'] % 2}"
                                for event in expected["events"]
                            },
                        )
                assert len(original) == 6 * sum(counts)
                verify_masked(batch, original)
                model.backbone.validate_kumo_batch(batch, target_name="masked_values")
                assert (
                    batch.inputs["history"]["sequence"]["valid"].sum(1).tolist()
                    == counts
                )
            elif native:
                sequences = verify_next(
                    batch, source["anchors"], expected, state, identities
                )
                assert [len(row) for row in sequences] == [
                    11 * count - 1 for count in counts
                ]
                assert batch.targets["next_token"]["valid"].sum(1).tolist() == [
                    11 * count - 2 for count in counts
                ]
                model.backbone.validate_kumo_batch(batch, target_name="next_token")
            else:
                verify_inputs(
                    batch,
                    family,
                    source["anchors"],
                    source["reference"],
                    identities,
                    list(source["anchors"].values()),
                    history_reference=expected,
                )
                stream = batch.inputs["history"]
                for owner, anchor in enumerate(source["anchors"].values()):
                    tokens = history_tokens(anchor, history(anchor, expected), state)
                    assert (
                        stream["value_ids"][owner][stream["valid"][owner]].tolist()
                        == tokens
                    )
                    assert not stream["value_ids"][owner][~stream["valid"][owner]].any()
                assert batch.targets["label"]["valid"].all()
                assert batch.targets["label"]["values"].flatten().tolist() == [1.0, 0.0]
                model.backbone.validate_kumo_batch(batch)
            assert task.fitted_state.fingerprint == fingerprint
