# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit field sequences determine token positions, not map placement."""

from dataclasses import replace

import pytest
import torch

from kumo.pytorch import api
from tests.helpers import reorder_dicts

INTEGER, STRING = api.DType("int64"), api.DType("string")
SCHEMA = api.DataSchema(
    (
        api.TableSpec(
            "users",
            "id",
            {"id": INTEGER, "first": STRING, "second": STRING},
            time_column=api.NON_TEMPORAL,
        ),
        api.TableSpec(
            "events",
            "id",
            {
                "id": INTEGER,
                "user": INTEGER,
                "at": api.DType("timestamp", unit="s", timezone="UTC"),
                "first": STRING,
                "second": STRING,
            },
            time_column="at",
        ),
    ),
    (api.RelationSpec("history", "users", "id", "events", "user"),),
)
FIELDS = {"a_second": api.Field(column="second"), "z_first": api.Field(column="first")}
SAMPLE = api.SampleSpec(
    api.EntitySpec("users"),
    entity_values=FIELDS,
    events={"history": api.EventSpec(api.Rows(("history",)), FIELDS)},
)
ROWS = {
    "users": [{"id": 1, "first": "US", "second": "GB"}],
    "events": [
        {"id": 10, "user": 1, "at": 4, "first": "US", "second": "GB"},
        {"id": 11, "user": 1, "at": 6, "first": "US", "second": None},
    ],
}
REQUESTS = (api.SampleRequest("one", 1, 10),)


@pytest.fixture(params=("in_memory", "diskgraph"))
def binding(request, tmp_path):
    if request.param == "in_memory":
        return api.Dataset(
            {name: api.InMemoryRows(rows) for name, rows in ROWS.items()}
        ), api.InMemoryEngine()
    pytest.importorskip("diskgraph")
    import pyarrow as pa
    import pyarrow.parquet as pq

    sources = {}
    for name, rows in ROWS.items():
        table = pa.Table.from_pylist(rows)
        if name == "events":
            table = table.set_column(
                table.schema.get_field_index("at"), "at", pa.array([4, 6], type=pa.timestamp("s"))
            )
        path = tmp_path / f"{name}.parquet"
        pq.write_table(table, path)
        sources[name] = api.Parquet(str(path))
    return api.Dataset(sources), api.DiskGraphEngine(
        index=tmp_path / "index", num_threads=1, num_shards=1
    )


def _batch(source, representation, layout, *, encodings=None):
    return api.BatchSpec(
        inputs={"tokens": api.Input(source=source, representation=representation, layout=layout)},
        encodings=encodings or {"values": api.Vocabulary()},
        id_spaces={
            "tokens": api.IdSpace(
                specials={"pad": "pad", "unknown": "unknown", "missing": "missing"}
            )
        },
        field_names={"fields": api.FieldNames()},
    )


def _tokens(fields, *, keep="first", limit=8, text=False):
    return api.Fields(
        fields=tuple(
            api.Encode(
                field=name,
                encoding=api.Text(vocabulary_size=1)
                if text and name == "z_first"
                else api.UseEncoding(name="values"),
                id_space="tokens",
            )
            for name in fields
        ),
        row=api.Tokens(max_tokens=limit, keep=keep),
        field_names="fields",
        field_space="tokens",
    )


def _field_id(task, scope, name):
    identity = ("entity", name) if scope == "entity" else ("events", "history", name)
    entries = task._assembler._input_state.field_names["fields"]
    if identity not in entries and scope == "entity":
        identity = ("dated_entity", name)
    local = entries[identity]
    return task.spaces["tokens"].members['["fields","fields"]'][local]


@pytest.mark.parametrize("scope", ("entity", "event"))
@pytest.mark.parametrize("packed", (False, True))
@pytest.mark.parametrize("fields", (("z_first", "a_second"), ("a_second", "z_first")))
@pytest.mark.parametrize("keep,limit", (("first", 8), ("first", 1), ("last", 1)))
def test_field_order_precedes_truncation(binding, monkeypatch, scope, packed, fields, keep, limit):
    dataset, engine = binding
    layout = api.Packed() if packed else api.Padded()
    source = api.EntitySource() if scope == "entity" else api.EventSource(("history",))
    if scope == "event":
        layout = api.Nested(fields=layout, events=api.Padded())
    representation = _tokens(fields, keep=keep, limit=limit)
    contract = api.DataLoaderSpec(SCHEMA, SAMPLE, _batch(source, representation, layout))
    swapped = replace(
        contract,
        batch=replace(
            contract.batch,
            inputs={
                "tokens": api.Input(
                    source,
                    replace(representation, fields=tuple(reversed(representation.fields))),
                    layout,
                )
            },
        ),
    )
    assert swapped.fingerprint != contract.fingerprint
    fitted_fingerprint = None
    for variant in (contract, api.DataLoaderSpec.from_dict(reorder_dicts(contract.to_dict()))):
        assert variant.fingerprint == contract.fingerprint
        with api.prepare(variant, dataset, engine) as plan:
            with plan.materialize(REQUESTS) as task:
                bound = task.bound_contract()
                if fitted_fingerprint is not None:
                    assert bound.fingerprint == fitted_fingerprint
                fitted_fingerprint = bound.fingerprint
                batch = next(task.batches(REQUESTS, batch_size=1))
                tokens = batch.inputs["tokens"]
                expected_rows = [{"z_first": "US", "a_second": "GB"}]
                if scope == "event":
                    expected_rows.append({"z_first": "US"})
                expected = []
                for row in expected_rows:
                    pairs = [(name, row[name]) for name in fields if name in row]
                    expected.extend(pairs[:limit] if keep == "first" else pairs[-limit:])
                assert tokens["field_ids"][tokens["valid"]].tolist() == [
                    _field_id(task, scope, name) for name, _ in expected
                ]
                assert tokens["value_ids"][tokens["valid"]].tolist() == [
                    task.output("tokens").field(name=name).encode(value=value)
                    for name, value in expected
                ]
                assert not tokens["position_ids"][tokens["valid"]].any()
                if engine.name == "diskgraph":
                    with monkeypatch.context() as patcher:
                        patcher.setattr(
                            plan._source,
                            "_adapt_temporal_result",
                            lambda *_: pytest.fail("model output constructed readable records"),
                        )
                        model = next(task.batches(REQUESTS, batch_size=1, output_format="model"))
                    torch.testing.assert_close(model.inputs, batch.inputs)


@pytest.mark.parametrize("shared", (False, True))
def test_composite_components_preserve_order_and_supplied_mappings(shared):
    dataset = api.Dataset({name: api.InMemoryRows(rows) for name, rows in ROWS.items()})
    configs = {
        "z_first": api.Vocabulary(values=(("US", 9),)),
        "a_second": api.Vocabulary(values=(("GB", 3),)),
    }
    fields = tuple(
        api.Encode(field=name, encoding=api.UseEncoding(name=name) if shared else encoding)
        for name, encoding in configs.items()
    )
    representation = api.Fields(fields=fields, row=api.Composite(top_k=8))
    contract = api.DataLoaderSpec(
        SCHEMA,
        SAMPLE,
        api.BatchSpec(
            inputs={
                "tokens": api.Input(api.EventSource(("history",)), representation, api.Padded())
            },
            encodings=configs if shared else {},
        ),
    )
    swapped = replace(
        contract,
        batch=replace(
            contract.batch,
            inputs={
                "tokens": api.Input(
                    api.EventSource(("history",)),
                    replace(representation, fields=tuple(reversed(fields))),
                    api.Padded(),
                )
            },
        ),
    )
    assert swapped.fingerprint != contract.fingerprint
    for variant in (contract, swapped):
        restored = api.DataLoaderSpec.from_json(variant.to_json())
        with api.prepare(restored, dataset, api.InMemoryEngine()) as plan:
            with plan.materialize(REQUESTS) as task:
                expected = task.preview(REQUESTS).batch.inputs
                bound = task.bound_contract()
                component_ids = tuple(
                    task.output("tokens")
                    .field(name=use.field)
                    .encode(value="US" if use.field == "z_first" else "GB")
                    for use in variant.batch.inputs["tokens"].representation.fields
                )
                keys = {
                    key for key, _ in bound.batch.inputs["tokens"].representation.row.vocabulary
                }
                assert tuple(("Vocabulary", value) for value in component_ids) in keys
                assert len(keys) == 2
                assert task.output("tokens").field(name="z_first").encoder.entries == (("US", 9),)
                assert task.output("tokens").field(name="a_second").encoder.entries == (("GB", 3),)
        permuted = api.DataLoaderSpec.from_dict(reorder_dicts(bound.to_dict()))
        assert permuted.fingerprint == bound.fingerprint
        torch.testing.assert_close(
            api.preview(permuted, dataset, api.InMemoryEngine(), REQUESTS).batch.inputs, expected
        )


@pytest.mark.parametrize("fields", ({"a": None}, {"a"}, "a", (), ((None, None),)))
def test_fields_require_ordered_encode_declarations(fields):
    with pytest.raises(ValueError):
        api.Fields(fields=fields)


def test_fields_require_unique_names_and_explicit_encoders():
    with pytest.raises(ValueError, match="same field"):
        api.Fields(
            fields=(
                api.Encode(field="a", encoding=api.Raw()),
                api.Encode(field="a", encoding=api.Raw()),
            )
        )
    with pytest.raises(ValueError, match="explicit"):
        api.Encode(field="a", encoding=None)


@pytest.mark.parametrize("chronological", (False, True))
def test_dated_occurrence_order_is_explicit_and_keeps_equal_time_field_order(chronological):
    sample = api.SampleSpec(
        SAMPLE.entity,
        entity_values={"static": FIELDS["a_second"]},
        dated_entity_values={
            name: api.DatedEntityValue(rows=api.Rows(path=("history",)), field=field)
            for name, field in FIELDS.items()
        },
    )
    group = _tokens(("z_first", "static", "a_second"))
    group = replace(
        group,
        row=replace(group.row, occurrence_order="chronological" if chronological else "fields"),
    )
    contract = api.DataLoaderSpec(SCHEMA, sample, _batch(api.EntitySource(), group, api.Padded()))
    dataset = api.Dataset({name: api.InMemoryRows(rows) for name, rows in ROWS.items()})
    for variant in (contract, api.DataLoaderSpec.from_dict(reorder_dicts(contract.to_dict()))):
        with api.prepare(variant, dataset, api.InMemoryEngine()) as plan:
            with plan.materialize(REQUESTS) as task:
                tokens = task.preview(REQUESTS).batch.inputs["tokens"]
                names = (
                    ("static", "z_first", "a_second", "z_first")
                    if chronological
                    else ("z_first", "z_first", "static", "a_second")
                )
                assert tokens["field_ids"][tokens["valid"]].tolist() == [
                    _field_id(task, "entity", name) for name in names
                ]
                assert "elapsed_seconds" not in tokens


@pytest.mark.parametrize("scope", ("entity", "event"))
def test_text_pieces_keep_positions_within_field_order(binding, scope):
    dataset, engine = binding
    source = api.EntitySource() if scope == "entity" else api.EventSource(("history",))
    layout = (
        api.Padded() if scope == "entity" else api.Nested(fields=api.Packed(), events=api.Padded())
    )
    contract = api.DataLoaderSpec(
        SCHEMA, SAMPLE, _batch(source, _tokens(("z_first", "a_second"), text=True), layout)
    )
    with api.prepare(contract, dataset, engine) as plan:
        with plan.materialize(REQUESTS) as task:
            for output_format in (
                ("inspectable", "model") if engine.name == "diskgraph" else ("inspectable",)
            ):
                tokens = next(
                    task.batches(REQUESTS, batch_size=1, output_format=output_format)
                ).inputs["tokens"]
                names = ("z_first", "z_first", "a_second") + (
                    ("z_first", "z_first") if scope == "event" else ()
                )
                assert tokens["field_ids"][tokens["valid"]].tolist() == [
                    _field_id(task, scope, name) for name in names
                ]
                assert tokens["position_ids"][tokens["valid"]].tolist() == [0, 1, 0] + (
                    [0, 1] if scope == "event" else []
                )
