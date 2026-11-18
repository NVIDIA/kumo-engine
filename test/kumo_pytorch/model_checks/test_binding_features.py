# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""L01's declared features: independent source answers and native model batches."""

import hashlib
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from test.kumo_pytorch.bindings import contract_for, load_bindings
from test.kumo_pytorch.data import prepare_binding, schema_for
from test.kumo_pytorch.involvement import combined_features
from test.kumo_pytorch.nightly import run_case
from test.kumo_pytorch.population import verify_batch_for
from test.kumo_pytorch.population_anchors import projected_rows
from test.kumo_pytorch.targets import (
    fitted_target_encoder,
    verify_next_outgoing,
)

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

import kumo.pytorch.api as r

from .test_aml_source import account, source_files, transaction

FEATURES = ("F01", "F02", "F03", "F04", "F05", "F07", "F08", "F10", "F11", "F14")
DAY = 86400


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    root = tmp_path_factory.mktemp("binding-features")
    return prepare_features(root)


def prepare_features(root):
    cutoff = datetime(2024, 3, 1, tzinfo=timezone.utc)
    members = [account(i, f"000{i}", "li-medium") for i in range(1, 5)]
    # Immutable source order breaks forecast-target ties. Incoming and self events,
    # inclusive lower window bounds, cutoff and upper forecast bound are explicit.
    events = [
        (-40 * DAY, 1, 2, "-100", "USD"),
        (-30 * DAY, 1, 2, "30", "USD"),
        (-7 * DAY, 1, 1, "0", "EUR"),
        (-DAY, 2, 1, "40", "EUR"),
        (-3600, 1, 3, "9", "USD"),
        (-3600, 3, 1, "-5", "EUR"),
        (-60, 1, 2, "12.1234567890", "USD"),
        (-60, 4, 3, "3", "USD"),
        (0, 3, 1, "80", "USD"),
        (0, 1, 2, "60", "JPY"),
        (0, 1, 3, "70", "EUR"),
        (60, 1, 1, "100", "USD"),
        (DAY, 1, 2, "900", "JPY"),
        (DAY, 4, 3, "999", "JPY"),
    ]
    tx, accounts, _ = source_files(
        root,
        [
            transaction(
                timestamp=(cutoff + timedelta(seconds=offset)).strftime(
                    "%Y/%m/%d %H:%M"
                ),
                from_bank=str(sender),
                from_account=f"000{sender}",
                to_bank=str(receiver),
                to_account=f"000{receiver}",
                amount_paid=amount,
                payment_currency=currency,
                payment_format="FUTURE_ONLY" if offset >= 0 else "Wire",
            )
            for offset, sender, receiver, amount, currency in events
        ],
        members,
    )
    stamp = int(cutoff.timestamp()) * 10**9
    records = [
        {
            "sample_id": f"{split}-{i}",
            "entity_id": member,
            "as_of_time": stamp,
            "graph_as_of_time": stamp - 10**9,
            "forecast_split": split,
            "next_amount": Decimal("60")
            if i == 1
            else Decimal("80")
            if i == 3
            else None,
            "next_currency": "JPY" if i == 1 else "USD" if i == 3 else None,
        }
        for split in ("train", "valid", "test")
        for i, member in enumerate(members, 1)
    ]
    schema = pa.schema(
        [
            ("sample_id", pa.string()),
            ("entity_id", pa.string()),
            ("as_of_time", pa.timestamp("ns")),
            ("graph_as_of_time", pa.timestamp("ns")),
            ("forecast_split", pa.string()),
            ("next_amount", pa.decimal128(38, 10)),
            ("next_currency", pa.string()),
        ]
    )
    anchor_path = root / "anchors.parquet"
    pq.write_table(pa.Table.from_pylist(records, schema=schema), anchor_path)
    paths = {
        "datasets/aml-data/v8/li-medium/transactions.parquet": tx,
        "anchors/aml-data/v8/li-medium/_bindings/v1/accounts.parquet": accounts,
        "anchors/aml-data/v8/li-medium/account-daily/v2/anchors.parquet": anchor_path,
    }
    files = {}
    for key, path in paths.items():
        destination = root / key
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
        files[key] = {
            "key": key,
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    binding = load_bindings()["L01"]
    output = root / "prepared"
    output.mkdir()
    data = prepare_binding(root, output, binding, files, limit=4)
    return binding, output, data


@pytest.fixture(scope="module")
def case(prepared):
    binding, output, (dataset, anchors, population, mapping, capacity) = prepared
    contract = contract_for(schema_for(dataset, binding), binding, max_events=capacity)
    with r.prepare(
        contract,
        dataset,
        r.DiskGraphEngine(index=output / "probe-index", num_threads=1),
    ) as plan:
        task = plan.materialize(
            anchors,
            fit_split="train",
            encoder_cutoff=population["encoder_cutoff_seconds"],
        )
        state = task.fitted_state.to_dict()
        projection = output / "anchors.parquet"
        check = combined_features(binding, projection, mapping, state, mode="smoke")
        batch = next(
            task.batches(
                split="train",
                epoch=0,
                batch_size=4,
                shuffle=False,
                output_format="model",
            )
        )
        yield binding, batch, state, check, list(projected_rows(projection, "train"))


def test_l01_required_correctness_records_every_feature_and_split(prepared):
    binding, output, (dataset, anchors, population, mapping, capacity) = prepared
    assert tuple(binding["feature_checks"]["features"]) == FEATURES
    report = run_case(
        dataset,
        anchors,
        population,
        binding,
        output,
        mode="smoke",
        mapping=mapping,
        capacity=capacity,
        batch_size=2,
    )
    assert report["status"] == "passed", report.get("traceback", report)
    assert report["feature_coverage"] == {
        f: dict.fromkeys(("train", "valid", "test"), 4) for f in FEATURES
    }
    assert not report["full_population_verified"]


@pytest.mark.parametrize("reverse", (False, True))
def test_l01_equal_time_history_accepts_either_order(case, reverse):
    binding, original, state, check, rows = case
    batch = deepcopy(original)
    if reverse:
        values = batch.inputs["history"]["value_ids"][0]
        values[44:54], values[55:65] = values[55:65].clone(), values[44:54].clone()
    verify_batch_for(binding, batch, iter(rows), state)
    check(batch)


def test_l01_future_only_values_cannot_enter_input_fitting(case):
    _, _, state, _, _ = case
    encoders = [value for key, value in state["encoders"].items() if '"inputs"' in key]
    assert encoders
    for encoder in encoders:
        values = {value for value, _ in encoder.get("entries", ())}
        assert not values & {"FUTURE_ONLY", "JPY"}, (
            "Future values leaked into input fitting"
        )
    # The future currency remains a valid supervised target, independently fitted.
    target = fitted_target_encoder(state, "currency")
    assert "JPY" in dict(target["entries"])


@pytest.mark.parametrize(
    "fault",
    (
        "identity",
        "graph-member",
        "edge",
        "count",
        "derived-token",
        "event-order",
        "tie-value-mix",
        "tie-duplicate",
        "graph-number",
        "graph-number-ulp",
        "graph-category",
        "layout",
        "extra-sample",
        "aggregate-dtype",
        "aggregate-mask",
        "source-index",
        "source-index-in-range",
        "extra-account",
        "target-leak",
        "amount",
        "currency",
    ),
)
def test_l01_feature_oracles_detect_corrupted_native_output(case, fault):
    binding, original, state, check, rows = case
    check(original)
    verify_batch_for(binding, original, iter(rows), state)
    batch = deepcopy(original)
    stream = batch.inputs["history"]
    nodes = batch.inputs["graph"]["nodes"]["transactions"]
    if fault == "identity":
        batch.metadata["entity_ids"][0] += 1
    elif fault == "graph-member":
        nodes["ids"][0] = 999
    elif fault == "edge":
        batch.inputs["graph"]["edges"]["sender"]["index"][0, 0] = 999
    elif fault == "count":
        batch.inputs["aggregates"]["24h_total_touching_count"]["values"][0] += 1
    elif fault == "derived-token":
        stream["value_ids"][0, 0] += 1
    elif fault == "event-order":
        stream["value_ids"][0, :10], stream["value_ids"][0, 11:21] = (
            stream["value_ids"][0, 11:21].clone(),
            stream["value_ids"][0, :10].clone(),
        )
    elif fault == "tie-value-mix":
        values = stream["value_ids"][0]
        values[46], values[57] = values[57].clone(), values[46].clone()
    elif fault == "tie-duplicate":
        values = stream["value_ids"][0]
        values[44:54] = values[55:65].clone()
    elif fault == "graph-number":
        nodes["features"]["amount_paid"]["values"][0] += 1
    elif fault == "graph-number-ulp":
        values = nodes["features"]["amount_paid"]["values"]
        values[0] = torch.nextafter(values[0], torch.tensor(float("inf")))
    elif fault == "graph-category":
        nodes["features"]["payment_currency"]["values"][0] += 1
    elif fault == "layout":
        stream["value_ids"] = stream["value_ids"].to(torch.float32)
    elif fault == "extra-sample":
        for field, value in stream.items():
            stream[field] = torch.cat((value, value[:1]), dim=0)
    elif fault == "aggregate-dtype":
        field = batch.inputs["aggregates"]["all_history_total_touching_count"]
        field["values"] = field["values"].to(torch.float64)
    elif fault == "aggregate-mask":
        field = batch.inputs["aggregates"]["all_history_total_touching_count"]
        field["valid"] = field["valid"].to(torch.int64)
    elif fault == "source-index":
        indices = stream["source_event_indices"]
        indices[indices >= 0] += 1_000_000
    elif fault == "source-index-in-range":
        indices = stream["source_event_indices"][0]
        indices[indices >= 0] += 1
    elif fault == "extra-account":
        accounts = batch.inputs["graph"]["nodes"]["accounts"]
        accounts["ids"] = torch.cat((accounts["ids"], torch.tensor([999])))
        accounts["batch"] = torch.cat((accounts["batch"], torch.tensor([0])))
    elif fault == "target-leak":
        batch.inputs["currency"] = batch.targets["currency"]
    else:
        batch.targets[fault]["values"][0] += 1
    with pytest.raises((AssertionError, IndexError)):
        verify_batch_for(binding, batch, iter(rows), state)
        check(batch)


def test_next_outgoing_source_oracle_window_ties_and_independent_nulls():
    def event(identity, stamp, sender, amount, currency):
        return {
            "id": identity,
            "time": stamp,
            "sender": sender,
            "values": {"amount_paid": amount, "payment_currency": currency},
        }

    row = {
        "sample_id": "a",
        "prediction_time_ns": 100 * 10**9,
        "registered_entity_id": "root",
        "amount": Decimal("2.0000000001"),
        "currency": "EUR",
    }
    events = [
        event(1, 99, "root", "9", "USD"),
        event(2, 100, "other", "8", "USD"),
        event(4, 100, "root", "3", "JPY"),
        event(3, 100, "root", "2.0000000001", "EUR"),
        event(5, 100 + DAY, "root", "100", "GBP"),
    ]
    verify_next_outgoing([row], {"events": events})
    for bad in ({"amount": Decimal("3")}, {"currency": "JPY"}):
        with pytest.raises(AssertionError, match="earliest outgoing"):
            verify_next_outgoing([{**row, **bad}], {"events": events})
    verify_next_outgoing(
        [{**row, "amount": None, "currency": None}], {"events": [events[-1]]}
    )
    for amount, currency in ((None, "EUR"), ("0", None)):
        verify_next_outgoing(
            [
                {
                    **row,
                    "amount": None if amount is None else Decimal(amount),
                    "currency": currency,
                }
            ],
            {"events": [event(0, 100, "root", amount, currency)]},
        )
