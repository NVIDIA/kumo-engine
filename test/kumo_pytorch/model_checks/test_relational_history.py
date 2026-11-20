# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Owned-entity history answers from tiny original public tables."""

import hashlib
from copy import deepcopy
from test.kumo_pytorch.bindings import (
    dataset_definition,
    load_bindings,
    profile,
)
from test.kumo_pytorch.data import prepare_binding
from test.kumo_pytorch.nightly import run_case

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

STAMP = 1735689600 + 100 * 86400


def inputs(tmp_path, *, homogeneous=False, binding_ref="S02"):
    binding = load_bindings()[binding_ref]
    definition = dataset_definition(binding)
    binding["dataset"] = definition
    root = tmp_path / "data"
    root.mkdir()
    files = {}

    def write(key, values):
        path = root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table(values), path)
        files[key] = dict(
            key=key,
            bytes=path.stat().st_size,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )

    prefix = "datasets/pragma-synthetic/medium/v1/"
    write(
        prefix + "customers/part-00000.parquet",
        {
            "customer_id": [10, 20],
            "onboarded_at": [
                STAMP
                - profile(binding, "anchor").get("root_time_offset_seconds", 86400)
            ]
            * 2,
            "risk_tier": [999, 999],
        },
    )
    write(
        prefix + "accounts/part-00000.parquet",
        {
            "account_id": [100, 101, 200],
            "customer_id": [10, 10, 20],
            "opened_at": [STAMP - 86400] * 3,
        },
    )
    defaults = dict(
        amount=12.5,
        mcc=10,
        channel="web",
        transfer_type="bank",
        cross_border=False,
        ip_region=2,
        success=True,
        change_type="address",
        reason="unknown purchase",
        event_type="move",
        plan="basic",
    )
    for family, spec in definition["source_options"]["events"].items():
        values = {
            spec["key"]: [1, 2, 3, 4, 5],
            spec["time"]: [STAMP - 30, STAMP - 20, STAMP, STAMP + 1, STAMP - 10],
            **{c: [defaults[c]] * 5 for c in spec["fields"].values()},
        }
        if family in {"card_txns", "disputes"}:
            values.update(
                account_id=[100, 101, 100, 101, 200], customer_id=[10, 10, 10, 10, 20]
            )
        elif family == "transfers":
            values.update(
                src_account_id=[100, 101, 100, None, 200],
                dst_account_id=[101, 200, 200, 100, None],
            )
        else:
            values["customer_id"] = [10, 10, 10, 10, 20]
        values["final_risk"] = [999] * 5
        write(prefix + family + "/part-00000.parquet", values)
    anchor = definition["anchor_path"].format(
        population=binding["selector"]["population"],
        version=binding["selector"]["anchor_version"],
    )
    write(
        anchor,
        {
            "sample_id": ["tr-a", "tr-b", "tr-c", "va-a", "te-b"],
            "entity_id": ["100", "101", "100", "200", "200"]
            if binding_ref.startswith("S01")
            else ["10", "20", "10", "10", "20"],
            "as_of_time": pa.array([STAMP * 10**9] * 5, pa.timestamp("ns")),
            "split": ["train", "train", "train", "valid", "test"],
            **(
                {
                    "graph_as_of_time": pa.array(
                        [(STAMP - 1) * 10**9] * 5, pa.timestamp("ns")
                    ),
                    "label": [True, False, True, True, False],
                    "gross_profit": [10.0, -5.0, 10.0, 10.0, -5.0],
                }
                if binding_ref == "S04"
                else {}
            ),
        },
    )
    if homogeneous:
        # A separately named supported fixture exercises the shared oracle and
        # training runner. The required registered S02 test keeps every field.
        binding["ref"] = "fixture-owned-history"
        binding["binding_id"] = "fixture-owned-history"
        selected = deepcopy(profile(binding, "input"))
        selected["families"] = [
            f for f in selected["families"] if f[0] in {"card_txns", "transfers"}
        ]
        for f in selected["families"]:
            f[3] = [field for field in f[3] if field[0] in {"source_kind", "amount"}]
        binding["inputs"] = selected
    return root, files, binding


def prepared(tmp_path, *, homogeneous=False, limit=None, binding_ref="S02"):
    root, files, binding = inputs(
        tmp_path, homogeneous=homogeneous, binding_ref=binding_ref
    )
    output = tmp_path / "output"
    output.mkdir()
    values = prepare_binding(root, output, binding, files, limit=limit)
    return binding, output, values


@pytest.mark.parametrize("mode", ["smoke", "training", "full", "lifecycle"])
def test_registered_s02_requires_execution_or_declared_early_refusal(tmp_path, mode):
    binding, output, values = prepared(tmp_path)
    dataset, anchors, population, mapping, capacity = values
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
    assert report["status"] == "passed", report


@pytest.mark.parametrize("mode", ["smoke", "training"])
def test_shared_owned_history_native_runner_and_features(tmp_path, mode):
    binding, output, values = prepared(tmp_path, homogeneous=True)
    dataset, anchors, population, mapping, capacity = values
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
    assert report["status"] == "passed", report
    assert all(r["status"] == "passed" for r in report["feature_assertions"].values())
    assert report["feature_coverage"]["F13"] == {"train": 3, "valid": 1, "test": 1}


@pytest.fixture
def native_history(tmp_path):
    from test.kumo_pytorch.bindings import contract_for
    from test.kumo_pytorch.data import schema_for
    from test.kumo_pytorch.history_features import sequence_features

    import kumo.pytorch.api as r

    binding, output, (dataset, anchors, population, mapping, capacity) = prepared(
        tmp_path
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
            features = sequence_features(
                binding,
                output / "anchors.parquet",
                mapping,
                task.fitted_state.to_dict(),
                mode="smoke",
            )
            features(batch)
            yield batch, features
