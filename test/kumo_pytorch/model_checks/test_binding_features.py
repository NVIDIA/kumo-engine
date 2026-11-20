# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""L01's declared features: independent source answers and native model batches."""

import hashlib
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from test.kumo_pytorch.bindings import contract_for, load_bindings
from test.kumo_pytorch.data import prepare_binding, schema_for
from test.kumo_pytorch.involvement import combined_features
from test.kumo_pytorch.nightly import run_case
from test.kumo_pytorch.population_anchors import projected_rows
from test.kumo_pytorch.targets import fitted_target_encoder

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import kumo.pytorch.api as r

from .fixtures import account, source_files, transaction

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
        (0, 1, 2, "60.1234567890", "JPY"),
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
            "next_amount": Decimal("60.1234567890")
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
    from test.kumo_pytorch.feature_checks import validate_evidence

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
    validate_evidence(report, binding["feature_checks"], "smoke")
    assert report["feature_coverage"] == {
        f: dict.fromkeys(("train", "valid", "test"), 4) for f in FEATURES
    }
    assert not report["full_population_verified"]
    assertions = report["feature_assertions"]
    assert set(assertions) == set(binding["feature_checks"]["assertions"])
    for row in assertions.values():
        assert row["status"] == "passed"
        assert row["checked_anchors"]["population"] == dict.fromkeys(
            ("train", "valid", "test"), 4
        )


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


def test_l01_raw_forecast_oracle_preserves_decimal_precision(prepared, case, tmp_path):
    from test.kumo_pytorch.targets import (
        source_target_rows,
        verify_target_projection,
    )

    binding, output, (_, _, _, mapping, _) = prepared
    _, batch, state, _, projected = case
    originals = source_target_rows(binding, mapping["anchor_source"]["path"], projected)
    row = projected[0]
    original = originals[row["anchor_row_position"]]
    assert original["amount"] == Decimal("60.1234567890")
    assert row["amount"] != original["amount"]
    verify_target_projection(binding, row, original)
    # A raw-label change smaller than float32 precision must still fail the
    # source forecast oracle, even though its projected model label is unchanged.
    table = pq.read_table(mapping["anchor_source"]["path"])
    records = table.to_pylist()
    records[0]["next_amount"] += Decimal("0.0000000001")
    path = tmp_path / "changed-anchors.parquet"
    pq.write_table(pa.Table.from_pylist(records, schema=table.schema), path)
    mapping = {**mapping, "anchor_source": {"path": str(path)}}
    check = combined_features(
        binding, output / "anchors.parquet", mapping, state, mode="smoke"
    )
    with pytest.raises(AssertionError, match="forecast_targets"):
        check(batch)
    assert check.evidence["stored_targets"]["status"] == "passed"
    assert check.evidence["forecast_targets"]["status"] == "failed"
