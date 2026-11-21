# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Empty, high-degree and duplicate-valued histories share native graph batches."""

from collections import Counter
from types import SimpleNamespace

import pytest
import torch

import kumo.pytorch.api as r
from tests import graph_semantics
from tests.conformance.fixtures import FLOAT, INT, TIME, batches, fields, materialized, record


def fixture(layout, length_input):
    names = ("purchases", "visits")
    tables = [r.TableSpec("users", "id", {"id": INT}, time_column=r.NON_TEMPORAL)]
    rows = {"users": [{"id": n} for n in (1, 2, 3)]}
    events = {}
    for component, name in enumerate(names):
        columns = {
            "id": INT,
            "owner": INT,
            "at": TIME,
            "clock": INT,
            "value": FLOAT,
            "hidden": INT,
        }
        tables.append(r.TableSpec(name, "id", columns, time_column="at"))
        rows[name] = [
            {
                "id": 0,
                "owner": 1,
                "at": 101,
                "clock": 101,
                "value": 1e30,
                "hidden": 999,
            },
            *[
                {
                    "id": i + 1,
                    "owner": 2,
                    "at": 80 + i % 20,
                    "clock": 80 + i % 20,
                    "value": float(i),
                    "hidden": i + 1000,
                }
                for i in range(257)
            ],
            {"id": 300, "owner": 3, "at": 99, "clock": 99, "value": 5.0, "hidden": 10},
            {"id": 301, "owner": 3, "at": 99, "clock": 99, "value": 5.0, "hidden": 11},
            {
                "id": 302,
                "owner": 3,
                "at": 100,
                "clock": 100,
                "value": None,
                "hidden": None,
            },
        ]
        events[name] = r.EventSpec(
            r.Rows((name,), r.TimeWindow("at", stop=0)),
            {
                "identity": r.Field("id"),
                "component": r.Constant(component, INT),
                "clock": r.Field("clock"),
                "value": r.Field("value"),
                "hidden": r.Field("hidden"),
                "age": r.EventTimeDelta(dtype=FLOAT),
            },
        )
    schema = r.DataSchema(
        tuple(tables),
        tuple(r.RelationSpec(name, "users", "id", name, "owner") for name in names),
    )
    graph = r.NeighborhoodSpec((r.GraphLayer(tuple(r.GraphSample(name, 4) for name in names)),))
    contract = r.DataLoaderSpec(
        schema,
        r.SampleSpec(r.EntitySpec("users"), events=events, neighborhoods={"g": graph}),
        r.BatchSpec(
            {
                "history": r.Input(
                    r.EventSource(names, max_events=6, keep="last"),
                    fields("identity", "component", "clock", "value"),
                    layout,
                ),
                "ages": r.Input(r.EventsOf("history"), fields("age"), layout),
                "graph": r.Input(
                    r.NeighborhoodSource("g"),
                    r.Graph({name: fields("value", "clock") for name in names}),
                    None,
                ),
            },
            targets={"aligned": r.AlignedEventTarget("history", (r.Encode("hidden", r.Raw()),))},
            batching=r.Batching(
                max_samples=3,
                max_tokens=12,
                length_input=length_input,
                bucket_by_length=True,
            ),
        ),
    )
    return contract, rows


def verify_history(batch, rows, expected):
    output, labels = batch.inputs["history"], batch.targets["aligned"]["hidden"]
    ids = list(batch.metadata["sample_ids"])
    assert output["valid"].dtype == torch.bool
    assert batch.inputs["ages"]["valid"].dtype == torch.bool
    for name, dtype in {
        "identity": torch.int64,
        "component": torch.int64,
        "clock": torch.int64,
        "value": torch.float64,
    }.items():
        assert output[name]["values"].dtype == dtype
        assert output[name]["valid"].dtype == torch.bool
    assert labels["values"].dtype == torch.int64
    assert labels["valid"].dtype == torch.bool
    assert batch.inputs["ages"]["age"]["values"].dtype == torch.float64
    assert batch.inputs["ages"]["age"]["valid"].dtype == torch.bool
    assert len(ids) <= 3
    assert output["valid"].numel() <= 12
    assert batch.inputs["ages"]["valid"].numel() <= 12
    if "offsets" in output:
        offsets = output["offsets"].tolist()
        assert offsets[0] == 0 and offsets[-1] == len(output["identity"]["values"])
        assert batch.targets["aligned"]["offsets"].tolist() == offsets
        assert batch.inputs["ages"]["offsets"].tolist() == offsets
        assert output["valid"].all()
        assert batch.inputs["ages"]["valid"].all()
    for i, sample in enumerate(ids):
        root, cutoff = expected[sample]
        eligible = {
            (component, row["id"]): row
            for component, name in enumerate(("purchases", "visits"))
            for row in rows[name]
            if row["owner"] == root and row["at"] <= cutoff
        }
        count = min(6, len(eligible))
        if "offsets" in output:
            select = slice(offsets[i], offsets[i + 1])
            assert offsets[i + 1] - offsets[i] == count
            values = {
                name: field["values"][select].tolist()
                for name, field in output.items()
                if isinstance(field, dict)
            }
            validity = output["value"]["valid"][select].tolist()
            target, valid = (
                labels["values"][select].tolist(),
                labels["valid"][select].tolist(),
            )
        else:
            assert output["valid"][i].tolist() == [True] * count + [False] * (
                output["valid"].shape[1] - count
            )
            values = {
                name: field["values"][i, :count].tolist()
                for name, field in output.items()
                if isinstance(field, dict)
            }
            validity = output["value"]["valid"][i, :count].tolist()
            target, valid = (
                labels["values"][i, :count].tolist(),
                labels["valid"][i, :count].tolist(),
            )
            assert not labels["valid"][i, count:].any()
            assert batch.inputs["ages"]["valid"][i].tolist() == output["valid"][i].tolist()
        observed = list(zip(values["component"], values["identity"], strict=True))
        assert len(set(observed)) == len(observed) == count
        assert set(observed) <= eligible.keys(), f"{sample}: wrong owner or leaked future row"
        expected_times = sorted(row["at"] for row in eligible.values())[-count:] if count else []
        assert values["clock"] == expected_times
        age_index = select if "offsets" in output else (i, slice(0, count))
        assert batch.inputs["ages"]["age"]["values"][age_index].tolist() == [
            cutoff - t for t in values["clock"]
        ]
        for j, key in enumerate(observed):
            row = eligible[key]
            assert all(
                output[name]["valid"][age_index][j] for name in ("identity", "component", "clock")
            )
            assert batch.inputs["ages"]["age"]["valid"][age_index][j]
            assert values["clock"][j] == row["at"]
            assert validity[j] == (row["value"] is not None)
            if validity[j]:
                assert values["value"][j] == row["value"]
            assert valid[j] == (row["hidden"] is not None)
            if valid[j]:
                assert target[j] == row["hidden"]
        if root == 3:
            assert (
                Counter(value for value, ok in zip(values["value"], validity, strict=True) if ok)[
                    5.0
                ]
                == 4
            ), "Distinct duplicate-valued events were lost"


@pytest.mark.parametrize("length_input", ["history", "ages"])
@pytest.mark.parametrize("seed", [0, 17])
@pytest.mark.parametrize("layout", [r.Padded(), r.Packed()], ids=["padded", "packed"])
def test_mixed_histories_graphs_and_withheld_labels(tmp_path, seed, layout, length_input):
    contract, rows = fixture(layout, length_input)
    expected = {
        "empty": (1, 100),
        "large": (2, 100),
        "duplicates": (3, 100),
        "earlier": (2, 90),
    }
    records = [record(name, entity, at=cutoff) for name, (entity, cutoff) in expected.items()]
    with materialized(contract, rows, records, tmp_path, seed=seed) as (task, _, _):
        oracle = graph_semantics
        source = oracle.SourceRows(contract.schema, rows)
        delivered = []
        for batch in batches(task, batch_size=3):
            verify_history(batch, rows, expected)
            requests = [
                SimpleNamespace(sample_id=s, entity_id=expected[s][0], as_of_time=expected[s][1])
                for s in batch.metadata["sample_ids"]
            ]
            oracle.assert_graph(
                batch.inputs["graph"],
                contract,
                "graph",
                requests,
                source,
                task.fitted_state.to_dict(),
            )
            delivered.extend(batch.metadata["sample_ids"])
        assert sorted(delivered) == sorted(expected)  # Length bucketing may reorder anchors.
