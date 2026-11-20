# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Graph/count forecasts with exact original-row boundary and mutation answers."""

import hashlib
from copy import deepcopy
from test.kumo_pytorch.bindings import load_bindings, selection
from test.kumo_pytorch.contracts import binding_contract
from test.kumo_pytorch.data import prepare_binding, schema_for
from test.kumo_pytorch.forecast_features import (
    count_features,
    future_counts,
    source_reference,
)
from test.kumo_pytorch.models import build_model
from test.kumo_pytorch.nightly import run_case
from test.kumo_pytorch.population_anchors import projected_rows

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest
import torch

import kumo.pytorch.api as r

STAMP = 1643673600 * 10**9
DAY = 86400 * 10**9


def fixture(tmp_path):
    from test.kumo_pytorch.bindings import dataset_definition

    binding = load_bindings()["M04"]
    definition = dataset_definition(binding)
    definition.pop("anchor_manifest_format")
    binding["dataset"] = definition
    root = tmp_path / "source"
    files = {}

    def write(key, table):
        path = root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path)
        files[key] = {
            "key": key,
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    keys, eventkeys, anchors = [], [], []
    opts = definition["source_options"]
    spec = opts["events"]["transactions"]
    # File order deliberately differs from UTF-8 identity order.
    for fold, client in enumerate(
        ["z-client", "a-client", "m-client", "v-client", "t-client"]
    ):
        key = f"clients/fold={fold}/part.parquet"
        keys.append(key)
        write(key, pa.table({"client_id": [client], "is_balanced": [1.0]}))
        # Start included, end excluded, nanosecond neighbors, duplicate timestamps,
        # future poison values, and a covered empty future window.
        offsets = [-2 * DAY, -2 * 10**9, -(10**9), -1, 30 * DAY, 31 * DAY]
        if fold != 1:
            offsets += [0, 1, 30 * DAY - 1]
        count = 0 if fold == 1 else 3
        time = [STAMP + x for x in offsets]
        data = {
            "client_id": [client] * len(time),
            "fold": [fold] * len(time),
            "event_time": pa.array(time, pa.timestamp("ns")),
            "amount": [1.0 + fold, 2.0, 3.0, None, 99999.0, 99999.0]
            + ([99999.0] * 3 if fold != 1 else []),
        }
        for field in spec["categorical"]:
            data[field] = pa.array(
                [float(fold + 1), 2.0, None, float("nan"), 999.0, 999.0]
                + ([999.0] * 3 if fold != 1 else []),
                pa.float64(),
            )
        eventkey = f"transactions/fold={fold}/part.parquet"
        eventkeys.append(eventkey)
        write(eventkey, pa.table(data))
        anchors.append(
            {
                "sample_id": f"s{fold}",
                "entity_id": client,
                "as_of_time": STAMP,
                "graph_as_of_time": STAMP - 10**9,
                "split": opts["splits"][str(fold)],
                "future_count": count,
            }
        )
    definition["identity_paths"] = keys
    opts["tables"] = {"clients": {"paths": keys}, "transactions": {"paths": eventkeys}}
    definition["anchor_path"] = "anchors.parquet"
    table = pa.Table.from_pylist(anchors)
    for name in ("as_of_time", "graph_as_of_time"):
        table = table.set_column(
            table.schema.get_field_index(name),
            name,
            pc.cast(table[name], pa.timestamp("ns")),
        )
    write("anchors.parquet", table)
    return root, files, binding


def prepared(tmp_path):
    root, files, binding = fixture(tmp_path)
    out = tmp_path / "output"
    out.mkdir()
    return binding, out, prepare_binding(root, out, binding, files, limit=None)


@pytest.mark.parametrize("mode", ["smoke", "training", "full", "lifecycle"])
def test_count_forecast_campaigns(tmp_path, mode):
    binding, out, (dataset, anchors, population, mapping, capacity) = prepared(tmp_path)
    report = run_case(
        dataset,
        anchors,
        population,
        binding,
        out,
        mode=mode,
        capacity=capacity,
        mapping=mapping,
        batch_size=2,
    )
    assert report["status"] == "passed", report.get("traceback", report)
    if mode in {"smoke", "training"} and report.get("outcome") != "refused":
        assert report["feature_coverage"]["F15"] == {"train": 3, "valid": 1, "test": 1}
        assert all(
            v["status"] == "passed" for v in report["feature_assertions"].values()
        )


@pytest.fixture
def native(tmp_path):
    binding, out, (dataset, anchors, population, mapping, capacity) = prepared(tmp_path)
    schema = schema_for(dataset, binding)
    contract = binding_contract(schema, binding, max_events=capacity)
    with r.prepare(
        contract, dataset, r.DiskGraphEngine(index=out / "probe-index", num_threads=1)
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
                        batch_size=2,
                        shuffle=False,
                        output_format="model",
                    )
                )
            )
            yield binding, out, mapping, task, batch


def test_generated_count_is_check_only_not_model_loss(native):
    binding, _, _, task, batch = native
    model = build_model(task, selection(binding), "graphsage")
    initial = model.training_step(batch)
    changed = deepcopy(batch)
    changed.targets["generated_count"]["values"] += 100
    torch.testing.assert_close(
        model.training_step(changed).loss, initial.loss, rtol=0, atol=0
    )
    assert set(initial.logits) == {"count"}


def test_empty_selected_histories_keep_covered_zero_count(tmp_path):
    root, files, binding = fixture(tmp_path)
    options = binding["dataset"]["source_options"]

    def replace(key, table):
        pq.write_table(table, root / key)
        files[key].update(
            bytes=(root / key).stat().st_size,
            sha256=hashlib.sha256((root / key).read_bytes()).hexdigest(),
        )

    key = options["tables"]["clients"]["paths"][0]
    replace(
        key,
        pa.table(
            {"client_id": ["z-client", "cover-client"], "is_balanced": [1.0, 0.0]}
        ),
    )
    for i, key in enumerate(options["tables"]["transactions"]["paths"]):
        table = pq.ParquetFile(root / key).read()
        if i == 0:
            table = table.set_column(
                table.schema.get_field_index("client_id"),
                "client_id",
                pa.array(["cover-client"] * len(table)),
            )
        else:
            table = table.slice(0, 0)
        replace(key, table)
    table = pq.read_table(root / "anchors.parquet")
    replace(
        "anchors.parquet",
        table.set_column(
            table.schema.get_field_index("future_count"),
            "future_count",
            pa.array([0] * len(table), pa.int64()),
        ),
    )
    out = tmp_path / "output"
    out.mkdir()
    _, _, _, mapping, _ = prepare_binding(root, out, binding, files, limit=None)
    rows = [
        row
        for split in ("train", "valid", "test")
        for row in projected_rows(out / "anchors.parquet", split)
    ]
    anchors = {row["sample_id"]: row for row in rows}
    events, spans = source_reference(binding, mapping, anchors)
    assert not any(events.values())
    from test.kumo_pytorch.targets import generated_checks

    answers = future_counts(anchors, events, spans, generated_checks(binding))
    assert all(value["generated_count"] == 0 for value in answers.values())
    # Building the oracle must not invent a positive-history admission rule.
    features = count_features(
        binding, out / "anchors.parquet", mapping, {}, mode="smoke"
    )
    assert set(features.checks) == set(binding["feature_checks"]["assertions"])
