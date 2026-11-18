# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Combined graph/history/aggregate answers over original relational rows."""

import hashlib
import json
import math
from test.kumo_pytorch.aggregate_features import (
    aggregate_answers,
    aggregate_check,
    reduce_rows,
)
from test.kumo_pytorch.bindings import contract_for, profile
from test.kumo_pytorch.data import prepare_binding, schema_for
from test.kumo_pytorch.history_features import combined_features, source_events
from test.kumo_pytorch.nightly import run_case
from test.kumo_pytorch.population_anchors import projected_rows
from test.kumo_pytorch.targets import (
    source_target_rows,
    verify_target_projection,
)
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

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


def test_grouped_aggregate_answers_match_hand_calculated_boundaries(tmp_path):
    binding, output, (_, _, _, mapping, _) = prepared(tmp_path)
    rows = list(projected_rows(output / "anchors.parquet", "train"))
    events = source_events(binding, mapping, {10, 20})
    answers = aggregate_answers(profile(binding, "input"), events, rows[0])
    for name, expected in {
        "all_history_card_txns_count": 12,
        "24h_card_txns_count": 5,
        "7d_card_txns_count": 7,
        "30d_card_txns_count": 9,
        "180d_card_txns_count": 11,
        "card_amount_sum": 110,
        "card_amount_mean": 110 / 9,
        "card_amount_min": 2,
        "card_amount_max": 24,
        "card_distinct_mcc": 3,
        "card_distinct_channel": 3,
        "card_channel_mode": "web",
        "card_recency_at_execution": 0,
        "card_recency_seconds": 1,
        "recent_card_share": 5 / 9,
        "card_mcc_groups": 3,
        "card_mcc_max_count": 3,
        "card_channel_groups": 3,
        "card_channel_max_count": 5,
        "card_mcc_channel_groups": 7,
        "card_mcc_channel_max_count": 2,
        "card_mean_channels_per_mcc": 7 / 3,
    }.items():
        assert answers[name] == expected, name
    variance = (
        sum((value - 110 / 9) ** 2 for value in [2, 4, 8, 10, 12, 14, 16, 20, 24]) / 9
    )
    assert math.isclose(answers["card_amount_var_population"], variance)
    assert math.isclose(answers["card_amount_std_population"], math.sqrt(variance))
    empty = aggregate_answers(profile(binding, "input"), events, rows[1])
    assert empty["card_amount_sum"] == empty["card_mcc_groups"] == 0
    assert all(
        empty[name] is None
        for name in (
            "card_amount_mean",
            "card_recency_seconds",
            "recent_card_share",
            "card_mcc_max_count",
        )
    )


@pytest.mark.parametrize(
    "values,operation,expected",
    [
        ([], "sum", 0),
        ([None], "sum", None),
        ([None, 4.0, 8.0], "mean", 6.0),
        ([None, 4.0, 8.0], "std_population", 2.0),
        ([None, 4.0, 8.0], "var_population", 4.0),
        ([None, "web", "app", "web", "app"], "mode", "app"),
        ([None, "web", "web"], "count_distinct", 1),
    ],
)
def test_aggregate_missing_and_mode_rules(values, operation, expected):
    assert (
        reduce_rows([{"value": value} for value in values], operation, "value", 0)
        == expected
    )


@pytest.mark.parametrize(
    "profit,label,valid",
    [
        (0.0, False, True),
        (-1.0, False, True),
        (1.0, True, True),
        (1.0, False, False),
        (float("nan"), False, False),
    ],
)
def test_threshold_targets_are_checked_before_float32_conversion(
    tmp_path, profit, label, valid
):
    binding, output, (_, _, _, mapping, _) = prepared(tmp_path)
    projected = list(projected_rows(output / "anchors.parquet", "train"))[0]
    original = source_target_rows(
        binding, mapping["anchor_source"]["path"], [projected]
    )[projected["anchor_row_position"]]
    original.update(gross_profit=profit, label=label)
    projected["label"] = float(label)
    if valid:
        verify_target_projection(binding, projected, original)
    else:
        with pytest.raises(AssertionError):
            verify_target_projection(binding, projected, original)


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


@pytest.mark.parametrize(
    "fault",
    [
        "raw-count",
        "normalized",
        "group",
        "ratio",
        "aggregate-mask",
        "graph-value",
        "graph-edge",
        "graph-identity",
        "history",
        "target",
    ],
)
def test_combined_source_oracle_rejects_corruption(native_combined, fault):
    batch, features = native_combined
    if fault in {"raw-count", "group", "ratio"}:
        field = {
            "raw-count": "all_history_card_txns_count",
            "group": "card_mcc_groups",
            "ratio": "recent_card_share",
        }[fault]
        batch.inputs["aggregate_checks"][field]["values"][0] += 1
    elif fault == "normalized":
        batch.inputs["aggregates"]["all_history_card_txns_count"]["values"][0] += 1
    elif fault == "aggregate-mask":
        batch.inputs["aggregate_checks"]["card_amount_mean"]["valid"][0] = False
    elif fault == "graph-value":
        batch.inputs["graph"]["nodes"]["card_txns"]["features"]["amount"]["values"][
            0
        ] += 1
    elif fault == "graph-edge":
        edge = batch.inputs["graph"]["edges"]["root_card_txns"]
        edge["index"] = edge["index"][:, :-1]
    elif fault == "graph-identity":
        batch.inputs["graph"]["nodes"]["card_txns"]["ids"][0] = 9999
    elif fault == "history":
        batch.inputs["history"]["value_ids"][0, 0] += 1
    else:
        batch.targets["label"]["values"][0] = 0
    with pytest.raises(AssertionError, match="Feature assertions failed"):
        features(batch)


@pytest.mark.parametrize("size", [0, 1, 3])
def test_batch_digest_handles_broadcast_tensors_without_changing_values(size):
    from test.kumo_pytorch.lifecycle import input_digest

    broadcast = torch.tensor(7).expand(size)
    expected = torch.full((size,), 7, dtype=torch.int64)
    assert input_digest(broadcast) == input_digest(expected)
    assert torch.equal(broadcast, expected)


@pytest.mark.parametrize("kind", ["count", "distinct", "grouped", "float"])
def test_aggregate_dtype_does_not_trust_the_fitted_state(kind):
    inputs = {"families": [["events", ["events"], "clock", []]], "aggregates": []}
    source = {"family": "events"}
    if kind == "count":
        inputs["aggregates"] = [["n", ["events"], "clock", None]]
    elif kind == "distinct":
        inputs["aggregate_queries"] = {
            "n": {"source": source, "operation": "count_distinct", "column": "key"}
        }
    elif kind == "float":
        inputs["aggregate_queries"] = {
            "n": {
                "source": source,
                "operation": "recency",
                "column": "event_time",
                "dtype": "float64",
            }
        }
    else:
        source = {
            "source": source,
            "keys": ["key"],
            "values": {"n": {"operation": "count"}},
        }
        inputs["aggregate_queries"] = {
            "n": {"source": source, "operation": "max", "column": "n"}
        }
    anchors = {"a": {"entity_id": 1, "as_of_time": 10, "split": "train"}}
    events = {1: [{"family": "events", "time": 9, "values": {"key": "k"}}]}
    wrong_dtype = "float32" if kind == "float" else "float64"
    state = {
        "encoders": {
            json.dumps(
                ["inputs", "aggregates", "fields", "n"], separators=(",", ":")
            ): {"kind": "zscore", "mean": 1.0, "scale": 1.0},
            json.dumps(
                ["inputs", "aggregate_checks", "fields", "n"], separators=(",", ":")
            ): {"kind": "raw", "dtype": {"name": wrong_dtype}},
        }
    }
    batch = SimpleNamespace(
        metadata={"sample_ids": ["a"]},
        inputs={
            "aggregates": {
                "n": {"values": torch.tensor([0.0]), "valid": torch.tensor([True])}
            },
            "aggregate_checks": {
                "n": {
                    "values": torch.tensor(
                        [1.0 + 1e-13], dtype=getattr(torch, wrong_dtype)
                    ),
                    "valid": torch.tensor([True]),
                }
            },
        },
    )
    check = aggregate_check(inputs, events, anchors, state)
    with pytest.raises(AssertionError):
        check(batch)


def test_combined_graph_accepts_remapped_node_and_edge_order(native_combined):
    batch, features = native_combined
    graph = batch.inputs["graph"]
    nodes = graph["nodes"]["card_txns"]
    order = torch.arange(len(nodes["ids"]) - 1, -1, -1)
    nodes["ids"], nodes["batch"] = nodes["ids"][order], nodes["batch"][order]
    for item in nodes["features"].values():
        for name in ("values", "valid"):
            item[name] = item[name][order]
    edge = graph["edges"]["root_card_txns"]
    edge["index"][1] = len(order) - 1 - edge["index"][1]
    edge["index"] = edge["index"].flip(1)
    features(batch)


@pytest.mark.parametrize("cross_time", [False, True])
def test_combined_sequence_allows_only_equal_time_record_permutations(
    native_combined, cross_time
):
    batch, features = native_combined
    stream = batch.inputs["history"]
    mask = stream["valid"][0]
    records = [[]]
    for value in stream["value_ids"][0][mask].tolist():
        if value == 5:
            records.append([])
        else:
            records[-1].append(value)
    # Eight older card records precede the -30s group of seven event families.
    # The final -1s card is strictly newer than that group.
    left = 8
    right = len(records) - 1 if cross_time else left + 1
    records[left], records[right] = records[right], records[left]
    values, indices = [], []
    for index, record in enumerate(records):
        if index:
            values.append(5)
            indices.append(-1)
        values.extend(record)
        indices.extend([index] * len(record))
    stream["value_ids"][0][mask] = torch.tensor(values)
    stream["source_event_indices"][0][mask] = torch.tensor(indices)
    if cross_time:
        with pytest.raises(AssertionError, match="Feature assertions failed"):
            features(batch)
    else:
        features(batch)
