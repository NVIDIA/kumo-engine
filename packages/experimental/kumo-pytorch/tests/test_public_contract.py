# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The public declarations share one serializable, engine-independent contract."""

from __future__ import annotations

import dataclasses
import json
import pathlib
from dataclasses import replace

import pytest

import kumo.pytorch
from kumo.pytorch.api import (
    NON_TEMPORAL,
    Aggregate,
    BatchSpec,
    Buckets,
    Constant,
    DataLoaderSpec,
    DataSchema,
    DType,
    Encode,
    EntitySource,
    EntitySpec,
    EventSource,
    EventSpec,
    Field,
    FieldNames,
    Fields,
    Graph,
    GraphLayer,
    GraphSample,
    IdSpace,
    Input,
    MaskedValuesTarget,
    NeighborhoodSource,
    NeighborhoodSpec,
    Nested,
    Packed,
    Padded,
    Rows,
    SampleRequest,
    SampleSpec,
    TableSpec,
    TimeWindow,
    Tokens,
    UseEncoding,
    Vocabulary,
)
from kumo.pytorch.contract import batch, sample, schema
from kumo.pytorch.engines.in_memory.samples import InMemorySampleBuilder
from tests.helpers import Plans

I64, F64, STRING = DType("int64"), DType("float64"), DType("string")
TIME = DType("timestamp", unit="s", timezone="UTC")


def _data():
    schema = DataSchema(
        tables=(
            TableSpec(
                "users", "user_id", {"user_id": I64, "country": STRING}, time_column=NON_TEMPORAL
            ),
            TableSpec(
                "transactions",
                "transaction_id",
                {
                    "transaction_id": I64,
                    "user_id": I64,
                    "occurred_at": TIME,
                    "amount": F64,
                    "description": STRING,
                },
                time_column="occurred_at",
            ),
        ),
        relations=(
            kumo.pytorch.RelationSpec(
                "user_transactions", "users", "user_id", "transactions", "user_id"
            ),
        ),
    )
    tables = {
        "users": [{"user_id": 1, "country": "US"}],
        "transactions": [
            {
                "transaction_id": 10,
                "user_id": 1,
                "occurred_at": 10,
                "amount": 5.0,
                "description": "coffee",
            },
            {
                "transaction_id": 11,
                "user_id": 1,
                "occurred_at": 110,
                "amount": 8.0,
                "description": "future coffee",
            },
        ],
    }
    return schema, tables


def _sample():
    return SampleSpec(
        entity=EntitySpec("users", entity_type="user"),
        entity_values={"country": Field(column="country")},
        events={
            "payments": EventSpec(
                rows=Rows(path=("user_transactions",), time=TimeWindow(column="occurred_at")),
                fields={
                    "amount": Field(column="amount"),
                    "description": Field(column="description"),
                },
            ),
            "noise": EventSpec(
                rows=Rows(path=("user_transactions",)),
                fields={"noise": Constant(value="ignore-me", dtype=STRING)},
            ),
        },
        neighborhoods={
            "local_graph": NeighborhoodSpec(
                layers=(
                    GraphLayer(
                        samples=(
                            GraphSample(relation="user_transactions", fanout=1, temporal="static"),
                        )
                    ),
                )
            )
        },
        targets={
            "future_count": Aggregate(
                source=Rows(
                    path=("user_transactions",),
                    time=TimeWindow(column="occurred_at", start=0, stop=20),
                ),
                operation="count",
                dtype=I64,
            )
        },
    )


def _tokens(*, profile_layout=Padded(), field_layout=Packed(), event_layout=Padded()):
    return BatchSpec(
        encodings={"country": Vocabulary()},
        id_spaces={"model": IdSpace(specials={"pad": "[PAD]", "mask": "[MASK]"})},
        field_names={"fields": FieldNames()},
        inputs={
            "entity": Input(
                source=EntitySource(),
                representation=Fields(
                    fields=(
                        Encode(
                            field="country", encoding=UseEncoding(name="country"), id_space="model"
                        ),
                    ),
                    row=Tokens(max_tokens=4),
                    field_names="fields",
                    field_space="model",
                ),
                layout=profile_layout,
            ),
            "events": Input(
                source=EventSource(events=("payments",), max_events=8),
                representation=Fields(
                    fields=(Encode(field="amount", encoding=Buckets(bins=4), id_space="model"),),
                    row=Tokens(max_tokens=4),
                    field_names="fields",
                    field_space="model",
                ),
                layout=Nested(fields=field_layout, events=event_layout),
            ),
        },
        targets={
            "masked_values": MaskedValuesTarget(
                input="events",
                token_probability=0.15,
                event_probability=0.1,
                key_probability=0.1,
            )
        },
    )


def test_package_root_exposes_data_loader_api_not_model_construction():
    assert kumo.pytorch.DataLoaderSpec is DataLoaderSpec
    assert kumo.pytorch.BatchSpec is BatchSpec
    assert kumo.pytorch.__version__ == "0.1.0a1"
    assert not hasattr(kumo.pytorch, "ArchitectureConfig")


def test_public_objects_have_one_owning_module():
    assert schema.TableSpec is kumo.pytorch.TableSpec
    assert sample.EventTimeDelta is kumo.pytorch.EventTimeDelta
    assert batch.Fields is kumo.pytorch.Fields
    assert batch.EventsOf is kumo.pytorch.EventsOf
    assert batch.CandidateTarget is kumo.pytorch.CandidateTarget


def test_contract_and_assembler_do_not_depend_on_any_engine():
    from kumo.pytorch import assembly

    for module in (schema, sample, batch, assembly):
        assert "kumo.pytorch.engines" not in pathlib.Path(module.__file__).read_text()


def test_input_sources_name_components_not_selection_recipes():
    assert _tokens().inputs["events"].source.events == ("payments",)
    graph = Input(
        source=NeighborhoodSource(name="local_graph"), representation=Graph(), layout=None
    )
    assert graph.source.name == "local_graph"
    assert not hasattr(graph.representation, "layers")


def test_unknown_component_is_refused_before_preparing():
    schema, tables = _data()
    spec = _tokens()
    bad = replace(
        spec,
        inputs={
            **spec.inputs,
            "events": replace(
                spec.inputs["events"],
                source=EventSource(events=("missing",)),
            ),
        },
    )
    with pytest.raises(ValueError, match="unknown event component"):
        Plans(schema, tables).prepare(_sample(), bad, [SampleRequest("u1", 1, 100)])


def test_only_declared_components_contribute_to_fitted_encoders():
    schema, tables = _data()
    task = Plans(schema, tables).prepare(_sample(), _tokens(), [SampleRequest("u1", 1, 100)])
    assert task.encoders["country"].entries == (("US", 2),)
    assert task.output("events").field(name="amount").encoder.boundaries == (5.0,)
    assert all("noise" not in identity for identity in task.fitted_state["encoders"])
    output = next(task.batches([SampleRequest("u1", 1, 100)], batch_size=1))
    assert tuple(output.inputs) == ("entity", "events")
    assert "field_ids" in output.inputs["entity"]
    assert "field_ids" in output.inputs["events"]
    assert "event_indices" in output.inputs["events"]["sequence"]
    assert "masked_values" in output.targets


def test_profile_fields_and_history_layouts_are_independent():
    schema, tables = _data()
    task = Plans(schema, tables).prepare(
        _sample(),
        _tokens(profile_layout=Packed(), field_layout=Padded(), event_layout=Packed()),
        [SampleRequest("u1", 1, 100)],
    )
    output = next(task.batches([SampleRequest("u1", 1, 100)], batch_size=1))
    assert output.inputs["entity"]["offsets"].tolist() == [0, 1]
    assert output.inputs["events"]["field_ids"].shape == (1, 1)
    assert output.inputs["events"]["sequence"]["offsets"].tolist() == [0, 1]


def test_relational_target_is_raw_sample_data():
    schema, tables = _data()
    built = InMemorySampleBuilder(schema, tables).build(_sample(), [SampleRequest("u1", 1, 100)])[0]
    assert {target.name: target.value for target in built.targets} == {"future_count": 1}
    assert type(built.targets[0].value) is int


def test_fitted_state_is_written_inspectably_in_the_artifact(tmp_path):
    schema, tables = _data()
    task = Plans(schema, tables).prepare(_sample(), _tokens(), [SampleRequest("u1", 1, 100)])
    task.to_artifact().save(tmp_path)
    written = json.loads((tmp_path / "fitted-state.json").read_text())
    fit, lengths, fitted_for = written.pop("fit"), written.pop("lengths"), written.pop("fitted_for")
    assert written == task.fitted_state
    assert fit == task.fitted_state.fit
    assert lengths == task.manifest["lengths"]
    assert fitted_for == {
        "inputs": task.contract.input_fingerprint,
        "contract": task.contract.fingerprint,
    }
    assert task.fitted_state["format_version"] == 2
    assert task.spaces["model"].size > 2
    assert task.manifest["fitted_state_fingerprint"].startswith("sha256:")


def test_public_contract_dataclasses_are_deserializable():
    from kumo.pytorch.spec import _known_types

    known = _known_types()
    for name in kumo.pytorch.__all__:
        declaration = getattr(kumo.pytorch, name)
        if isinstance(declaration, type) and dataclasses.is_dataclass(declaration):
            if declaration.__module__.startswith("kumo.pytorch.contract"):
                assert known[declaration.__name__] is declaration
    schema, _ = _data()
    contract = DataLoaderSpec(schema=schema, sample=_sample(), batch=_tokens())
    assert DataLoaderSpec.from_json(contract.to_json()) == contract
