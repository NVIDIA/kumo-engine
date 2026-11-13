# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model retrieval may be bounded without narrowing fitting or raw samples."""

import json
from dataclasses import replace

import pytest

from kumo.pytorch import api, assembly
from tests.test_preview import _equal as assert_equal


def _sample():
    history = api.Rows(("orders",), api.TimeWindow("at", stop=0))
    return api.SampleSpec(
        api.EntitySpec("users"),
        events={
            "orders": api.EventSpec(
                history,
                {
                    "id": api.Field(column="id"),
                    "amount": api.Field(column="amount"),
                    "kind": api.Field(column="kind"),
                },
            )
        },
        targets={
            "future": api.Aggregate(
                api.Rows(("orders",), api.TimeWindow("at", 0, 3)),
                "sum",
                "amount",
                dtype=api.DType("float64"),
            )
        },
    )


def _input(names=("orders",), limit=2, keep="last", layout=None):
    return api.Input(
        api.EventSource(names, max_events=limit, keep=keep),
        api.Fields(
            fields=(
                api.Encode(field="id", encoding=api.Raw()),
                api.Encode(field="amount", encoding=api.Raw()),
                api.Encode(field="kind", encoding=api.Vocabulary()),
            )
        ),
        layout or api.Padded(),
    )


@pytest.mark.parametrize("keep,order", [("first", "ascending"), ("last", "descending")])
def test_model_retrieval_uses_input_limit_without_mutating_sample(keep, order):
    sample = _sample()
    bounded = assembly._model_retrieval_sample(
        sample, api.BatchSpec({"history": _input(keep=keep)})
    )
    assert bounded.events["orders"].limit == 2
    assert bounded.events["orders"].order == order
    assert sample.events["orders"].limit is None
    assert bounded.targets == sample.targets
    assert bounded.events["orders"].rows == sample.events["orders"].rows


@pytest.mark.parametrize(
    "case", ["unbounded", "shared", "merged", "merged_consumer", "explicit", "descending"]
)
def test_ambiguous_or_already_bounded_requests_are_not_rewritten(case):
    sample = _sample()
    inputs = {"history": _input()}
    if case == "unbounded":
        inputs["history"] = _input(limit=None)
    elif case == "shared":
        inputs["all"] = _input(limit=None)
    elif case in {"merged", "merged_consumer"}:
        sample = replace(sample, events={**sample.events, "other": sample.events["orders"]})
        inputs["merged"] = _input(("orders", "other"))
        if case == "merged":
            del inputs["history"]
    elif case in {"explicit", "descending"}:
        event = replace(
            sample.events["orders"],
            **({"limit": 4} if case == "explicit" else {"order": "descending"}),
        )
        sample = replace(sample, events={"orders": event})
    bounded = assembly._model_retrieval_sample(sample, api.BatchSpec(inputs))
    assert bounded == sample


@pytest.fixture(scope="module")
def history_data(tmp_path_factory):
    pytest.importorskip("diskgraph")
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    root = tmp_path_factory.mktemp("model_history_limits")
    rows = {
        "users": [{"id": i} for i in (1, 2, 3)],
        "orders": [
            dict(zip(("id", "owner", "at", "amount", "kind"), row))
            for row in [
                (40, 1, 1, 10.0, "old"),
                (10, 1, 2, 20.0, "tie-a"),
                (30, 1, 2, None, "tie-b"),
                (20, 1, 3, 40.0, "middle"),
                (50, 1, 4, 50.0, "last"),
                (60, 1, 5, 60.0, "cutoff"),
                (70, 1, 6, 70.0, "future"),
                (80, 3, 2, 80.0, "other"),
            ]
        ],
    }
    sources = {}
    for name, values in rows.items():
        table = pa.Table.from_pylist(values)
        if name == "orders":
            table = table.set_column(
                table.schema.get_field_index("at"),
                "at",
                pa.array(table["at"].to_pylist(), type=pa.timestamp("s")),
            )
        path = root / f"{name}.parquet"
        pq.write_table(table, path)
        sources[name] = api.Parquet(str(path))
    schema = api.DataSchema(
        (
            api.TableSpec("users", "id", {"id": api.DType("int64")}, time_column=api.NON_TEMPORAL),
            api.TableSpec(
                "orders",
                "id",
                {
                    "id": api.DType("int64"),
                    "owner": api.DType("int64"),
                    "at": api.DType("timestamp", unit="s", timezone="UTC"),
                    "amount": api.DType("float64"),
                    "kind": api.DType("string"),
                },
                time_column="at",
            ),
        ),
        (api.RelationSpec("orders", "users", "id", "orders", "owner"),),
    )
    requests = tuple(
        api.SampleRequest(i, owner, cutoff, split="train")
        for i, (owner, cutoff) in enumerate([(1, 5), (2, 5), (1, 3), (3, 5)])
    )
    return (
        schema,
        api.Dataset(sources),
        api.Dataset({name: api.InMemoryRows(values) for name, values in rows.items()}),
        root / "index",
        requests,
    )


class _Queries:
    def __init__(self, sampler):
        self.sampler, self.calls = sampler, []

    def __getattr__(self, name):
        return getattr(self.sampler, name)

    def query_temporal(self, nodes, cutoffs, plan, **kwargs):
        self.calls.append(json.loads(plan))
        return self.sampler.query_temporal(nodes, cutoffs, plan, **kwargs)


@pytest.mark.parametrize("keep", ["first", "last"])
@pytest.mark.parametrize("layout", [api.Padded(), api.Packed()])
@pytest.mark.parametrize("aggregate", [False, True])
@pytest.mark.parametrize("label", [False, True])
@pytest.mark.parametrize("limit", [2, 2**64, True])
def test_real_model_requests_are_bounded_but_preview_and_labels_are_not(
    history_data, keep, layout, aggregate, label, limit
):
    schema, parquet, memory, index, requests = history_data
    sample = _sample()
    inputs = {"history": _input(keep=keep, layout=layout, limit=limit)}
    if aggregate:
        sample = replace(
            sample,
            entity_values={
                "count": api.Aggregate(
                    sample.events["orders"].rows, "count", dtype=api.DType("int64")
                )
            },
        )
        inputs["entity"] = api.Input(
            api.EntitySource(),
            api.Fields(fields=(api.Encode(field="count", encoding=api.Raw()),)),
            api.Separate(),
        )
    contract = api.DataLoaderSpec(
        schema,
        sample,
        api.BatchSpec(inputs, {"future": api.LabelTarget("future", api.Raw())} if label else {}),
    )
    fingerprint = contract.fingerprint
    with api.prepare(contract, memory, api.InMemoryEngine()) as reference:
        expected = next(
            reference.materialize(requests, fit_split="train").batches(requests, batch_size=4)
        )
    with api.prepare(contract, parquet, api.DiskGraphEngine(index=index, num_threads=2)) as plan:
        queries = _Queries(plan._source._sampler)
        plan._source._sampler = queries
        with plan.materialize(requests, fit_split="train") as task:
            assert not queries.calls
            fitted = task.bound_contract().to_json()
            actual = next(task.batches(requests, batch_size=4, output_format="model"))
            pushed = type(limit) is int and limit == 2
            assert queries.calls[0]["sequences"][0]["limit"] == (2 if pushed else None)
            assert queries.calls[0]["sequences"][0]["order"] == (
                "descending" if pushed and keep == "last" else "ascending"
            )
            assert_equal(actual.inputs, expected.inputs)
            assert_equal(actual.targets, expected.targets)
            if label:
                # (5,8]: 70; (3,6]: 50+60+70. Other owners have no future rows.
                assert actual.targets["future"]["values"].tolist() == [70.0, 0.0, 180.0, 0.0]
            if aggregate:
                assert actual.inputs["entity"]["count"]["values"].tolist() == [6, 0, 4, 1]
            queries.calls.clear()
            preview = task.preview(requests)
            assert len(preview.samples[0]["events"]["orders"]) == 6
            assert all(call["sequences"][0]["limit"] is None for call in queries.calls)
            assert task.bound_contract().to_json() == fitted
            assert contract.fingerprint == fingerprint
