# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Combined graph/history/aggregate answers over original relational rows."""

import hashlib
from test.kumo_pytorch.bindings import contract_for
from test.kumo_pytorch.data import prepare_binding, schema_for
from test.kumo_pytorch.history_features import combined_features
from test.kumo_pytorch.nightly import run_case

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import kumo.pytorch.api as r

from .test_relational_history import STAMP, inputs


def prepared(tmp_path, *, mode_tie=False):
    root, files, binding = inputs(tmp_path, binding_ref="S04")
    key = next(key for key in files if "/card_txns/" in key)
    # Asymmetric groups distinguish nested grouping from a total count. Boundary
    # rows include both sides of every window, plus cutoff/future poison values.
    offsets = [
        -30,
        -20,
        -10,
        -86400,
        -86401,
        -604800,
        -604801,
        -2592000,
        -2592001,
        -15552000,
        -15552001,
        -1,
        0,
        1,
        0,
    ]
    table = pa.table(
        {
            "txn_id": list(range(1, 16)),
            "event_time": [STAMP + value for value in offsets],
            "account_id": [100, 101] * 7 + [200],
            "customer_id": [10] * 14 + [20],
            "amount": [
                2.0,
                4.0,
                8.0,
                10.0,
                12.0,
                14.0,
                16.0,
                20.0,
                100.0,
                200.0,
                400.0,
                24.0,
                9999.0,
                9999.0,
                9999.0,
            ],
            "mcc": [10, 10, 10, 20, 20, 30, 30, 30, 99, 99, 99, 20, 999, 999, 999],
            "channel": [
                "web",
                "web",
                "store",
                "web",
                "store",
                "store",
                "app",
                "web",
                "outside",
                "outside",
                "outside",
                "web",
                "POISON",
                "POISON",
                "POISON",
            ],
        }
    )
    # Use the declared physical source clock, independently of mapped names.
    spec = binding["dataset"]["source_options"]["events"]["card_txns"]
    table = table.rename_columns(
        [spec["time"] if name == "event_time" else name for name in table.column_names]
    )
    if mode_tie:
        # Nine eligible records: app/store/web each appear three times.
        channels = table["channel"].to_pylist()
        for index, value in zip(
            [0, 1, 2, 3, 4, 5, 6, 7, 11], ["web", "app", "store"] * 3, strict=True
        ):
            channels[index] = value
        table = table.set_column(
            table.schema.get_field_index("channel"), "channel", pa.array(channels)
        )
    pq.write_table(table, root / key)
    files[key].update(
        bytes=(root / key).stat().st_size,
        sha256=hashlib.sha256((root / key).read_bytes()).hexdigest(),
    )
    output = tmp_path / "output"
    output.mkdir()
    values = prepare_binding(root, output, binding, files, limit=None)
    return binding, output, values


@pytest.mark.parametrize("mode", ["smoke", "training", "full", "lifecycle"])
def test_combined_history_binding_campaigns(tmp_path, mode):
    binding, output, (dataset, anchors, population, mapping, capacity) = prepared(
        tmp_path
    )
    report = run_case(
        dataset,
        anchors,
        population,
        binding,
        output,
        mode=mode,
        capacity=capacity,
        mapping=mapping,
        batch_size=2,
    )
    assert report["status"] == "passed", report.get("traceback", report)
    if mode in {"smoke", "training"} and report.get("outcome") != "refused":
        assert all(
            row["status"] == "passed" for row in report["feature_assertions"].values()
        )
        assert report["feature_coverage"]["F06"] == {"train": 3, "valid": 1, "test": 1}


def combined_batches(tmp_path, *, mode_tie=False):
    binding, output, (dataset, anchors, population, mapping, capacity) = prepared(
        tmp_path, mode_tie=mode_tie
    )
    contract = contract_for(schema_for(dataset, binding), binding, max_events=capacity)
    with r.prepare(
        contract,
        dataset,
        r.DiskGraphEngine(index=output / "probe-index", num_threads=1),
    ) as plan:
        with plan.materialize(
            anchors,
            fit_split="train",
            encoder_cutoff=population["encoder_cutoff_seconds"],
        ) as task:
            batch = next(
                iter(
                    task.batches(
                        split="train",
                        epoch=0,
                        shuffle=False,
                        batch_size=2,
                        output_format="model",
                    )
                )
            )
            features = combined_features(
                binding,
                output / "anchors.parquet",
                mapping,
                task.fitted_state.to_dict(),
                mode="smoke",
            )
            features(batch)
            yield batch, features


@pytest.fixture
def native_combined(tmp_path):
    yield from combined_batches(tmp_path)


def test_native_mode_ties_follow_declared_value_order(tmp_path):
    for batch, features in combined_batches(tmp_path, mode_tie=True):
        # The original three-way tie selects app. The train-only vocabulary
        # assertion in combined_features independently requires app at ID 2.
        assert batch.inputs["aggregate_checks"]["card_channel_mode"][
            "values"
        ].tolist() == [2, 0]
        features(batch)
