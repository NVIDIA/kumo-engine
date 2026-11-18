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
from test.kumo_pytorch.history_features import source_events
from test.kumo_pytorch.nightly import run_case

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

STAMP = 1735689600 + 100 * 86400


def inputs(tmp_path, *, homogeneous=False):
    binding = load_bindings()["S02"]
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
            "onboarded_at": [STAMP - 86400] * 2,
            "risk_tier": [999, 999],
        },
    )
    definition["identity_path"] = prefix + "customers/part-00000.parquet"
    write(
        prefix + "accounts/part-00000.parquet",
        {"account_id": [100, 101, 200], "customer_id": [10, 10, 20]},
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
        population="weekly-customer", version="v1"
    )
    write(
        anchor,
        {
            "sample_id": ["tr-a", "tr-b", "tr-c", "va-a", "te-b"],
            "entity_id": ["10", "20", "10", "10", "20"],
            "as_of_time": pa.array([STAMP * 10**9] * 5, pa.timestamp("ns")),
            "split": ["train", "train", "train", "valid", "test"],
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


def prepared(tmp_path, *, homogeneous=False, limit=None):
    root, files, binding = inputs(tmp_path, homogeneous=homogeneous)
    output = tmp_path / "output"
    output.mkdir()
    values = prepare_binding(root, output, binding, files, limit=limit)
    return binding, output, values


def test_owned_history_mapping_preserves_physical_events_and_source_cutoffs(tmp_path):
    binding, output, (dataset, _, _, mapping, capacity) = prepared(tmp_path)
    cards = pq.read_table(dataset.tables["card_txns"].path).to_pylist()
    assert [(r["root_id"], r["source_id"]) for r in cards] == [
        (10, 1),
        (10, 2),
        (10, 3),
        (10, 4),
        (20, 5),
    ]
    assert {r["event_time_seconds"] for r in cards} == {
        STAMP - 30,
        STAMP - 20,
        STAMP - 10,
        STAMP,
        STAMP + 1,
    }
    transfers = pq.read_table(dataset.tables["transfers"].path).to_pylist()
    assert {(r["root_id"], r["source_id"], r["direction"]) for r in transfers} == {
        (10, 1, "self"),
        (10, 2, "outgoing"),
        (20, 2, "incoming"),
        (10, 3, "outgoing"),
        (20, 3, "incoming"),
        (10, 4, "incoming"),
        (20, 5, "outgoing"),
    }
    assert len({r["event_id"] for r in transfers}) == len(transfers) == 7
    assert capacity == 28
    for table in dataset.tables.values():
        assert not {"final_risk", "risk_tier", "split", "label"} & set(
            pq.read_schema(table.path).names
        )
    originals = source_events(binding, mapping, {10, 20})
    assert len(originals[10]) == 28 and len(originals[20]) == 9
    assert len([e for e in originals[10] if e["time"] < STAMP]) == 14
    assert [
        (e["source_id"], e["values"]["direction"])
        for e in originals[10]
        if e["family"] == "transfers"
    ] == [(1, "self"), (2, "outgoing"), (3, "outgoing"), (4, "incoming")]
    assert pq.read_table(output / "anchors.parquet")["entity_id"].to_pylist() == [
        10,
        20,
        10,
        10,
        20,
    ]


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


@pytest.mark.parametrize(
    "fault",
    [
        "value",
        "calendar",
        "age",
        "target",
        "terminal",
        "validity",
        "source-owner",
        "separator",
        "identity",
        "cutoff",
        "dtype",
    ],
)
def test_owned_history_oracle_rejects_corruption(native_history, fault):
    import torch

    batch, features = native_history
    stream = batch.inputs["history"]
    labels = batch.targets["next_token"]
    if fault == "value":
        stream["value_ids"][0, 2] += 1
    elif fault == "calendar":
        stream["value_ids"][0, 6] += 1
    elif fault == "age":
        stream["value_ids"][0, 0] += 1
    elif fault == "target":
        labels["values"][0, 2] += 1
    elif fault == "terminal":
        labels["values"][0, int(labels["valid"][0].sum()) - 1] += 1
    elif fault == "validity":
        labels["valid"][0, 0] = False
    elif fault == "source-owner":
        stream["source_event_indices"][0, 0] = 999
    elif fault == "separator":
        stream["value_ids"][0, 8] = 999
    elif fault == "identity":
        batch.metadata["entity_ids"][0] = 20
    elif fault == "cutoff":
        batch.metadata["as_of_time"][0] += 1
    elif fault == "dtype":
        stream["value_ids"] = stream["value_ids"].to(torch.float32)
    with pytest.raises(AssertionError, match="Feature assertions failed"):
        features(batch)
    assert any(v["status"] == "failed" for v in features.evidence.values())


def _reorder_records(batch, order):
    import torch

    stream, target = batch.inputs["history"], batch.targets["next_token"]
    valid = stream["valid"][0]
    complete = (
        stream["value_ids"][0][valid].tolist()
        + target["values"][0][valid].tolist()[-1:]
    )
    records = [[]]
    for value in complete:
        if value == 5:
            records.append([])
        else:
            records[-1].append(value)
    reordered = [records[i] for i in order]
    values, indices = [], []
    for i, record in enumerate(reordered):
        if i:
            values.append(5)
            indices.append(-1)
        values.extend(record)
        indices.extend([i] * len(record))
    stream["value_ids"][0][valid] = torch.tensor(values[:-1])
    stream["source_event_indices"][0][valid] = torch.tensor(indices[:-1])
    target["values"][0][valid] = torch.tensor(values[1:])


@pytest.mark.parametrize("cross_time", [False, True])
def test_owned_history_ties_can_permute_but_chronology_cannot(
    native_history, cross_time
):
    batch, features = native_history
    order = list(range(14))
    other = 7 if cross_time else 1
    order[0], order[other] = order[other], order[0]
    _reorder_records(batch, order)
    if cross_time:
        with pytest.raises(AssertionError, match="Feature assertions failed"):
            features(batch)
    else:
        features(batch)


@pytest.mark.parametrize(
    "fault",
    [
        "foreign-owner",
        "unknown-endpoint",
        "duplicate-root",
        "negative-root",
        "large-event-id",
    ],
)
def test_source_identity_errors_fail_before_engine_open(tmp_path, fault):
    root, files, binding = inputs(tmp_path)
    table_name = {
        "foreign-owner": "card_txns",
        "unknown-endpoint": "transfers",
        "duplicate-root": "customers",
        "negative-root": "customers",
        "large-event-id": "card_txns",
    }[fault]
    key = next(key for key in files if f"/{table_name}/" in key)
    table = pq.read_table(root / key).to_pydict()
    if fault == "foreign-owner":
        table["customer_id"][0] = 20
    elif fault == "unknown-endpoint":
        table["dst_account_id"][0] = 999
    elif fault == "duplicate-root":
        table["customer_id"][0] = 20
    elif fault == "negative-root":
        table["customer_id"][0] = -10
    else:
        table["txn_id"][0] = 2**62
    pq.write_table(pa.table(table), root / key)
    files[key].update(
        bytes=(root / key).stat().st_size,
        sha256=hashlib.sha256((root / key).read_bytes()).hexdigest(),
    )
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(ValueError):
        prepare_binding(root, output, binding, files, limit=None)
    assert not (output / "index").exists()


@pytest.mark.parametrize("boundary", ["empty-family", "max-event-id"])
def test_source_mapping_preserves_schema_and_int64_boundary(tmp_path, boundary):
    from test.kumo_pytorch.relational_sources import prepare_sources

    root, files, binding = inputs(tmp_path)
    key = next(key for key in files if "/card_txns/" in key)
    original = pq.read_table(root / key)
    if boundary == "empty-family":
        table = original.slice(0, 0)
    else:
        values = original.to_pydict()
        values["txn_id"][0] = 2**62 - 1
        table = pa.table(values)
    pq.write_table(table, root / key)
    files[key].update(
        bytes=(root / key).stat().st_size,
        sha256=hashlib.sha256((root / key).read_bytes()).hexdigest(),
    )
    dataset, _ = prepare_sources(
        root, tmp_path / "mapped", dataset_definition(binding), files, None
    )
    events = pq.read_table(dataset.tables["card_txns"].path)
    assert events.schema.field("event_id").type == pa.int64()
    if boundary == "empty-family":
        assert events.num_rows == 0
        # Parquet represents second-resolution Arrow timestamps as milliseconds.
        assert events.schema.field("event_time").type == pa.timestamp("ms")
    else:
        assert events["event_id"][0].as_py() == 2**63 - 2


def test_bounded_source_mapping_keeps_all_times_for_selected_owner(tmp_path):
    from test.kumo_pytorch.relational_sources import prepare_sources

    root, files, binding = inputs(tmp_path)
    dataset, mapping = prepare_sources(
        root, tmp_path / "bounded", dataset_definition(binding), files, {"10"}
    )
    assert mapping["scope"] == {
        "kind": "selected_roots",
        "complete_selected_root_histories": True,
        "time_filter": None,
    }
    for source in dataset.tables.values():
        assert set(pq.read_table(source.path)["root_id"].to_pylist()) == {10}
    transfers = pq.read_table(dataset.tables["transfers"].path).to_pylist()
    assert {r["source_id"] for r in transfers} == {1, 2, 3, 4}
    assert {r["event_time_seconds"] for r in transfers} == {
        STAMP - 30,
        STAMP - 20,
        STAMP,
        STAMP + 1,
    }
