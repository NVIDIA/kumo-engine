# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Selected widths/cardinalities/lengths; existing goldens own basic semantics."""

from bisect import bisect_right
from collections import Counter
from dataclasses import replace

import pytest

from conformance.assertions import bound_tables
from conformance.cases import discover_cases
from kumo.pytorch import api as r
from tests.test_graph_time_and_reductions import ROWS, SCHEMA


@pytest.mark.parametrize("engine", ("in_memory", "diskgraph"))
@pytest.mark.parametrize("width", (8, 32, 128))
def test_mixed_feature_widths(engine, width):
    # Reuse the reduction fixture: values 10 at t=10, 30 at t=40, and a
    # future value 999 at t=900. Arithmetic goldens remain in their own tests.
    source = r.Rows(("user_events",), r.TimeWindow("occurred_at", stop=0))
    operations = (
        "count",
        "sum",
        "min",
        "max",
        "mean",
        "var_population",
        "std_population",
        "recency",
        "count_distinct",
    )
    values = {
        op: r.Aggregate(
            source,
            op,
            None if op == "count" else "occurred_at" if op == "recency" else "amount",
            dtype=r.DType("int64" if op in {"count", "count_distinct"} else "float64"),
        )
        for op in operations
    }
    values["grouped_count"] = r.Aggregate(
        r.GroupBy(source, ("amount",), {"n": r.Reduce("count", dtype=r.DType("int64"))}),
        "sum",
        "n",
        dtype=r.DType("int64"),
    )
    base_names = (*operations, "grouped_count")
    # Exact answers from the same two source windows, not another engine run.
    expected_base = (
        (2, 40, 10, 30, 20, 100, 10, 60, 2, 2),
        (1, 10, 10, 10, 10, 0, 0, 10, 1, 1),
    )
    fields, expected = [], {}
    for i in range(width):
        name = f"feature_{i:03}"
        values[f"offset_{i}"] = r.Constant(float(i), r.DType("float64"))
        values[name] = r.Derived("add", (base_names[i % len(base_names)], f"offset_{i}"))
        raw = [row[i % len(base_names)] + i for row in expected_base]
        if i % 3 == 0:
            encoder, answers = r.Raw(), raw
        elif i % 3 == 1:
            encoder = r.ZScore(dtype="float64")
            mean = sum(raw) / 2
            scale = abs(raw[0] - raw[1]) / 2 or 1.0
            answers = [(value - mean) / scale for value in raw]
        else:
            encoder = r.Buckets(boundaries=(10.0, 30.0, 50.0))
            answers = [1 + bisect_right((10, 30, 50), value) for value in raw]
        fields.append(r.Encode(name, encoder))
        expected[name] = answers
    contract = r.DataLoaderSpec(
        SCHEMA,
        r.SampleSpec(r.EntitySpec("users"), entity_values=values),
        r.BatchSpec({"entity": r.Input(r.EntitySource(), r.Fields(tuple(fields)), r.Separate())}),
    )
    requests = (
        r.SampleRequest("two-events", 1, 100, split="train"),
        r.SampleRequest("one-event", 1, 20, split="train"),
    )
    with bound_tables(contract, ROWS, engine) as (dataset, backend):
        with r.prepare(contract, dataset, backend) as plan:
            with plan.materialize(requests, fit_split="train") as task:
                fingerprint = task.fitted_state.fingerprint
                batches = list(
                    task.batches(
                        requests,
                        batch_size=2,
                        output_format="model" if engine == "diskgraph" else "inspectable",
                    )
                )
                assert len(batches) == 1
                batch = batches[0]
                assert list(batch.metadata["sample_ids"]) == [row.sample_id for row in requests]
                assert set(batch.inputs["entity"]) == set(expected)
                assert len(batch.inputs["entity"]) == width
                for name, answers in expected.items():
                    field = batch.inputs["entity"][name]
                    assert field["valid"].all()
                    assert field["values"].tolist() == pytest.approx(answers)
                assert task.fitted_state.fingerprint == fingerprint


@pytest.mark.parametrize("engine", ("in_memory", "diskgraph"))
@pytest.mark.parametrize("cardinality", (16, 256, 4096))
@pytest.mark.parametrize("shared", (False, True))
def test_vocabulary_cardinality_and_shared_state(engine, cardinality, shared):
    columns = {"id": r.DType("int64"), "at": r.DType("timestamp", unit="s", timezone="UTC")}
    columns.update({name: r.DType("string") for name in ("left", "right")})
    rows = [
        dict(id=i, at=0, left=f"c{i:05}", right=f"c{i + cardinality // 2:05}")
        for i in range(cardinality)
    ]
    rows += [
        dict(id=cardinality, at=200, left="unseen-left", right="unseen-right"),
        dict(id=cardinality + 1, at=200, left=None, right=None),
    ]
    contract = r.DataLoaderSpec(
        r.DataSchema((r.TableSpec("items", "id", columns, time_column="at"),)),
        r.SampleSpec(
            r.EntitySpec("items"), entity_values={name: r.Field(name) for name in ("left", "right")}
        ),
        r.BatchSpec(
            {
                "entity": r.Input(
                    r.EntitySource(),
                    r.Fields(
                        tuple(
                            r.Encode(
                                name, r.UseEncoding("categories") if shared else r.Vocabulary()
                            )
                            for name in ("left", "right")
                        )
                    ),
                    r.Separate(),
                )
            },
            encodings={"categories": r.Vocabulary()} if shared else {},
        ),
    )
    requests = tuple(
        r.SampleRequest(
            f"row-{row['id']}",
            row["id"],
            200 if row["at"] else 100,
            split="test" if row["at"] else "train",
        )
        for row in rows
    )
    expected = {}
    for name in ("left", "right"):
        names = ("left", "right") if shared else (name,)
        counts = Counter(row[field] for row in rows[:cardinality] for field in names)
        ordered = sorted(counts, key=lambda value: (-counts[value], value))
        expected[name] = {value: i + 2 for i, value in enumerate(ordered)}
    with bound_tables(contract, {"items": rows}, engine) as (dataset, backend):
        with r.prepare(contract, dataset, backend) as plan:
            with plan.materialize(requests, fit_split="train", encoder_cutoff=100) as task:
                fingerprint = task.fitted_state.fingerprint
                for name in ("left", "right"):
                    encoder = task.output("entity").field(name=name).encoder
                    assert dict(encoder.entries) == expected[name]
                    assert len(encoder.entries) == cardinality * (3 if shared else 2) // 2
                seen = []
                for batch in task.batches(
                    requests,
                    batch_size=128,
                    output_format="model" if engine == "diskgraph" else "inspectable",
                ):
                    ids = list(batch.metadata["sample_ids"])
                    seen.extend(ids)
                    indices = [int(sid.removeprefix("row-")) for sid in ids]
                    for name in ("left", "right"):
                        raw = [rows[i][name] for i in indices]
                        field = batch.inputs["entity"][name]
                        assert field["values"].tolist() == [
                            0 if value is None else expected[name].get(value, 1) for value in raw
                        ]
                        assert field["valid"].tolist() == [value is not None for value in raw]
                assert seen == [row.sample_id for row in requests]
                assert task.fitted_state.fingerprint == fingerprint


@pytest.mark.parametrize("engine", ("in_memory", "diskgraph"))
@pytest.mark.parametrize("length", (16, 128, 1024))
def test_controlled_bpe_text_lengths(engine, length):
    # Reuse the golden's BPE merges/IDs rather than training an expected answer.
    golden = discover_cases()["C-ENCODING-003"]
    rows = [
        dict(id=1, description=" ".join(["red", "bike"] * (length // 2))),
        dict(id=2, description=""),
        dict(id=3, description=None),
        dict(id=4, description="?"),
        dict(id=5, description=" ".join(["red", "bike"] * (length // 4))),
    ]
    requests = tuple(
        r.SampleRequest(f"text-{row['id']}", row["id"], golden.requests[0].as_of_time)
        for row in rows
    )
    expected = [8, 11] * (length // 2) + [12, 0] + [8, 11] * (length // 4)
    # Only fixture lengths change: tokenizer/layout/special IDs stay canonical.
    contract = replace(
        golden.contract, batch=replace(golden.contract.batch, batching=r.Batching(max_samples=5))
    )
    with bound_tables(contract, {"users": rows}, engine) as (dataset, backend):
        with r.prepare(contract, dataset, backend) as plan:
            with plan.materialize(requests) as task:
                fingerprint = task.fitted_state.fingerprint
                batches = list(
                    task.batches(
                        requests,
                        output_format="model" if engine == "diskgraph" else "inspectable",
                    )
                )
                assert len(batches) == 1
                batch = batches[0]
                assert list(batch.metadata["sample_ids"]) == [row.sample_id for row in requests]
                field = batch.inputs["entity"]["description"]
                assert field["values"].tolist() == expected
                assert field["piece_offsets"].tolist() == [
                    0,
                    length,
                    length,
                    length + 1,
                    length + 2,
                    length + 2 + length // 2,
                ]
                assert field["valid"].tolist() == [True, True, False, True, True]
                assert task.fitted_state.fingerprint == fingerprint
