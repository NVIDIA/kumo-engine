# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Event retention and model-facing time direction are independent choices."""

from dataclasses import replace

import pytest

from kumo.pytorch import api
from tests.native_fixtures import prepare_rows, request_anchor_table
from tests.test_model_history_limits import _assert_history, _Queries, _sample
from tests.test_model_history_limits import history_data as history_data
from tests.test_preview import _equal as assert_equal


def contract(schema, *, order="ascending", keep="last", layout=None, shared=False):
    def history(direction):
        return api.Input(
            api.EventSource(("orders",), max_events=2, keep=keep, order=direction),
            api.Fields(
                fields=(
                    api.Encode(field="id", encoding=api.Raw()),
                    api.Encode(field="kind", encoding=api.Vocabulary()),
                )
            ),
            layout or api.Padded(),
        )

    inputs = {"history": history(order)}
    if shared:
        inputs["opposite"] = history("descending" if order == "ascending" else "ascending")
    return api.DataLoaderSpec(schema, _sample(), api.BatchSpec(inputs))


@pytest.mark.parametrize("order", ["ascending", "descending"])
@pytest.mark.parametrize("keep", ["first", "last"])
@pytest.mark.parametrize("layout", [api.Padded(), api.Packed()])
@pytest.mark.parametrize("shared", [False, True])
def test_declared_order_matches_both_engines_and_formats(history_data, order, keep, layout, shared):
    schema, parquet, memory, index, requests = history_data
    spec = contract(schema, order=order, keep=keep, layout=layout, shared=shared)
    restored = api.DataLoaderSpec.from_json(spec.to_json())
    assert restored == spec
    assert restored.fingerprint == spec.fingerprint
    with api.prepare(spec, parquet, api.DiskGraphEngine(index=index, num_threads=2)) as plan:
        queries = _Queries(plan._source._sampler)
        plan._source._sampler = queries
        with plan.materialize(requests, fit_split="train") as task:
            queries.calls.clear()
            actual = next(task.batches(requests, batch_size=4, output_format="model"))
            assert set(actual.inputs) == ({"history", "opposite"} if shared else {"history"})
            for name, history in actual.inputs.items():
                direction = spec.batch.inputs[name].source.order
                _assert_history(
                    history, memory, requests, layout, keep, 2, direction, fields=("id", "kind")
                )
            requested = {
                seq["output_order"]
                for seq in queries.calls[-1]["sequences"]
                if seq["name"].startswith("event")
            }
            assert requested == ({"ascending", "descending"} if shared else {order})
            assert not hasattr(plan._source, "_ascending_indices")
            inspected = next(task.batches(requests, batch_size=4, output_format="inspectable"))
            assert_equal(inspected.inputs, actual.inputs)
            preview = task.preview(requests)
            assert len(preview.samples[0]["events"]["orders"]) == 6


def test_event_direction_does_not_replace_explicit_sample_selection(history_data):
    schema, parquet, memory, index, requests = history_data
    spec = contract(schema, order="ascending", keep="first")
    spec = replace(
        spec,
        sample=replace(
            spec.sample,
            events={"orders": replace(spec.sample.events["orders"], order="descending", limit=3)},
        ),
    )
    with api.prepare(spec, parquet, api.DiskGraphEngine(index=index)) as plan:
        with plan.materialize(requests, fit_split="train") as task:
            batch = next(
                task.batches(
                    requests,
                    batch_size=4,
                    output_format="model",
                )
            )
            assert batch.inputs["history"]["id"]["values"][0].tolist() == [20, 50]


def test_invalid_direction_is_rejected():
    with pytest.raises(ValueError, match="order"):
        api.EventSource(("orders",), order="backwards")


@pytest.mark.parametrize("order", ["ascending", "descending"])
def test_existing_model_source_signature_still_works(history_data, order):
    schema, parquet, _memory, index, requests = history_data
    spec = contract(schema, order=order)
    with api.prepare(spec, parquet, api.DiskGraphEngine(index=index)) as plan:
        source = plan._source

        class ExistingSource:
            def build_model(self, spec, requests):
                return source.build_model(spec, requests)

            def __getattr__(self, name):
                if name == "build_model_ordered":
                    raise AttributeError(name)
                return getattr(source, name)

        plan._source = ExistingSource()
        with plan.materialize(requests, fit_split="train") as task:
            actual = next(task.batches(requests, batch_size=4, output_format="model"))
            expected = next(task.batches(requests, batch_size=4, output_format="inspectable"))
            assert_equal(actual.inputs, expected.inputs)


def test_native_order_aliases_cannot_collide_with_component_names(history_data):
    schema, parquet, _memory, index, requests = history_data
    spec = contract(schema, shared=True)
    alias = "orders:descending"
    spec = replace(
        spec,
        sample=replace(
            spec.sample, events={**spec.sample.events, alias: spec.sample.events["orders"]}
        ),
        batch=replace(
            spec.batch,
            inputs={
                **spec.batch.inputs,
                "alias": replace(spec.batch.inputs["history"], source=api.EventSource((alias,))),
            },
        ),
    )
    with api.prepare(spec, parquet, api.DiskGraphEngine(index=index)) as plan:
        queries = _Queries(plan._source._sampler)
        plan._source._sampler = queries
        with plan.materialize(requests, fit_split="train") as task:
            queries.calls.clear()
            next(task.batches(requests, batch_size=4, output_format="model"))
            names = [seq["name"] for seq in queries.calls[-1]["sequences"]]
            assert len(names) == len(set(names))


@pytest.mark.parametrize("layout", [api.Padded(), api.Packed()])
def test_merged_components_follow_direction_and_reverse_equal_time_ties(layout):
    from types import SimpleNamespace

    import pyarrow as pa

    from kumo.pytorch.assembly import _column_output, _model_rows, _PreparedInput
    from kumo.pytorch.data.model import ModelSequence
    from kumo.pytorch.encoding import OutputBinding, fit_encoder

    values = SimpleNamespace(
        batch_size=2,
        events={
            "orders": ModelSequence(
                pa.array([0, 2, 2]), pa.array([1, 3]), {"id": pa.array([10, 30])}
            ),
            "payments": ModelSequence(
                pa.array([0, 2, 2]), pa.array([2, 3]), {"id": pa.array([20, 31])}
            ),
        },
    )
    for order, expected in [("ascending", [20, 30, 31]), ("descending", [31, 30, 20])]:
        declarations = {
            "id": (
                (("events", "orders", "id"), api.DType("int64"), "scalar"),
                (("events", "payments", "id"), api.DType("int64"), "scalar"),
            )
        }
        prepared = _PreparedInput(
            spec=api.Input(
                api.EventSource(("orders", "payments"), max_events=3, order=order),
                api.Fields(fields=(api.Encode(field="id", encoding=api.Raw()),)),
                layout,
            ),
            declarations=declarations,
            state={
                "id": OutputBinding(encoder=fit_encoder("id", api.Raw(), api.DType("int64"), ()))
            },
        )
        actual = _column_output(
            prepared, _model_rows(prepared.spec.source, values, declarations), model=True
        )
        if isinstance(layout, api.Padded):
            assert actual["id"]["values"].tolist() == [expected, [0, 0, 0]]
        else:
            assert actual["id"]["values"].tolist() == expected


@pytest.mark.parametrize(
    "representation",
    [
        api.Fields(
            fields=(api.Encode(field="kind", encoding=api.Vocabulary(), id_space="tokens"),),
            row=api.Tokens(),
            history=api.Flatten(max_tokens=100),
        ),
        api.Fields(
            fields=(api.Encode(field="kind", encoding=api.Vocabulary(), id_space="tokens"),),
            row=api.Tokens(max_tokens=3),
        ),
    ],
)
def test_token_inputs_share_the_same_direction_on_both_paths(history_data, representation):
    schema, parquet, memory, index, requests = history_data
    layout = (
        api.Nested(fields=api.Packed(), events=api.Padded())
        if isinstance(representation.history, api.Grouped)
        else api.Padded()
    )
    for order in ("ascending", "descending"):
        spec = api.DataLoaderSpec(
            schema,
            _sample(),
            api.BatchSpec(
                {
                    "history": api.Input(
                        api.EventSource(("orders",), max_events=2, order=order),
                        representation,
                        layout,
                    )
                },
                id_spaces={
                    "tokens": api.IdSpace(
                        specials={"pad": "pad", "unknown": "unknown", "missing": "missing"}
                    )
                },
            ),
        )
        with api.prepare(spec, parquet, api.DiskGraphEngine(index=index)) as plan:
            with plan.materialize(requests, fit_split="train") as task:
                actual = next(task.batches(requests, batch_size=4, output_format="model"))
                inspected = next(task.batches(requests, batch_size=4, output_format="inspectable"))
                assert_equal(actual.inputs, inspected.inputs)
                # After three specials come cutoff, last, middle, old, other, tie-a and tie-b.
                expected = [[4, 3], [], [9, 5], [7]]
                if order == "descending":
                    expected = [row[::-1] for row in expected]
                history = actual.inputs["history"]
                if isinstance(layout, api.Padded):
                    assert [
                        values[mask].tolist()
                        for values, mask in zip(history["value_ids"], history["valid"], strict=True)
                    ] == expected
                    assert history["position_ids"][history["valid"]].tolist() == [0] * 5
                    assert history["source_event_indices"][history["valid"]].tolist() == list(
                        range(5)
                    )
                else:
                    assert history["valid"].all()
                    assert history["owner"].tolist() == [0, 0, 2, 2, 3]
                    assert [
                        history["value_ids"][history["owner"] == i].tolist()
                        for i in range(len(requests))
                    ] == expected
                    assert history["history_position"].tolist() == [0, 1, 0, 1, 0]
                    assert history["sequence"]["event_indices"].tolist() == [
                        [0, 1],
                        [-1, -1],
                        [2, 3],
                        [4, -1],
                    ]
                    assert history["sequence"]["valid"].tolist() == [
                        [True, True],
                        [False, False],
                        [True, True],
                        [True, False],
                    ]
                    assert (
                        history["field_offsets"].tolist()
                        == history["field_cu_seqlens"].tolist()
                        == list(range(6))
                    )
                    assert history["field_max_seqlen"] == 1


def test_fitting_is_independent_of_delivery_direction(history_data):
    schema, _parquet, memory, _index, requests = history_data
    states = []
    for order in ("ascending", "descending"):
        spec = contract(schema, order=order)
        spec = replace(
            spec,
            batch=replace(
                spec.batch,
                inputs={
                    "history": replace(
                        spec.batch.inputs["history"],
                        representation=api.Fields(
                            fields=(
                                api.Encode(field="amount", encoding=api.ZScore()),
                                api.Encode(field="kind", encoding=api.Vocabulary()),
                            )
                        ),
                    )
                },
            ),
        )
        with prepare_rows(spec, memory) as plan:
            with plan.materialize(
                request_anchor_table(requests, schema=plan.contract.schema), fit_split="train"
            ) as task:
                states.append(task.bound_contract().batch.inputs["history"].representation)
    assert states[0] == states[1]


@pytest.mark.parametrize("built_samples", [False, True])
def test_aligned_target_fitting_keeps_canonical_reduction_order(built_samples):
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
                    "label": api.DType("float64"),
                },
                time_column="at",
            ),
        ),
        (api.RelationSpec("orders", "users", "id", "orders", "owner"),),
    )
    rows = {
        "users": [{"id": 1}],
        "orders": [
            {"id": i, "owner": 1, "at": i, "label": label}
            for i, label in enumerate([1e16, -1e16, 1.0], start=1)
        ],
    }
    sample = api.SampleSpec(
        api.EntitySpec("users"),
        events={
            "orders": api.EventSpec(
                api.Rows(("orders",), api.TimeWindow("at", stop=0)),
                {
                    "id": api.Field(column="id"),
                    "label": api.Field(column="label"),
                },
            )
        },
    )
    request = (api.SampleRequest(1, 1, 4),)
    states = []
    for order in ("ascending", "descending"):
        spec = api.DataLoaderSpec(
            schema,
            sample,
            api.BatchSpec(
                {
                    "history": api.Input(
                        api.EventSource(("orders",), order=order),
                        api.Fields(fields=(api.Encode(field="id", encoding=api.Raw()),)),
                        api.Padded(),
                    )
                },
                targets={
                    "labels": api.AlignedEventTarget(
                        "history", (api.Encode(field="label", encoding=api.ZScore()),)
                    )
                },
            ),
        )
        if built_samples:
            from tests.native_fixtures import NativeSamples

            with api.prepare_batch_spec(
                sample,
                spec.batch,
                NativeSamples(schema, rows).build(sample, request),
                contract=spec,
            ) as plan:
                encoder = plan.target(name="labels").field(name="label").encoder
                states.append((encoder.mean, encoder.scale))
        else:
            dataset = api.Dataset({name: api.InMemoryRows(value) for name, value in rows.items()})
            with prepare_rows(spec, dataset) as plan:
                with plan.materialize(
                    request_anchor_table(request, schema=plan.contract.schema)
                ) as task:
                    encoder = task.target(name="labels").field(name="label").encoder
                    states.append((encoder.mean, encoder.scale))
    assert states[0][0] == 1 / 3
    assert states[1] == states[0]
