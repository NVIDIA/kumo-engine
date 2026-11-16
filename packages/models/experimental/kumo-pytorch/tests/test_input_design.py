# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import replace

import pytest

from kumo.pytorch.api import (
    NON_TEMPORAL,
    BatchSpec,
    Buckets,
    CalendarPart,
    Composite,
    DataLoaderSpec,
    DataSchema,
    Derived,
    DType,
    EntitySource,
    EntitySpec,
    EventSource,
    EventSpec,
    Field,
    FieldNames,
    Flatten,
    Grouped,
    IdSpace,
    Input,
    MaskedValuesTarget,
    Nested,
    NextTokenTarget,
    Packed,
    Padded,
    RelationSpec,
    Rows,
    SampleRequest,
    SampleSpec,
    Separate,
    TableSpec,
    Text,
    TimeWindow,
    Tokens,
    UseEncoding,
    Vocabulary,
)
from kumo.pytorch.contract.batch import (
    Encode,
    Fields,
)
from tests.helpers import Plans as _Plans


def _data():
    schema = DataSchema(
        tables=(
            TableSpec(
                "users",
                "user_id",
                {
                    "user_id": DType("int64"),
                    "country": DType("string"),
                    "home_currency": DType("string"),
                },
                time_column=NON_TEMPORAL,
            ),
            TableSpec(
                "transactions",
                "transaction_id",
                {
                    "transaction_id": DType("int64"),
                    "user_id": DType("int64"),
                    "at": DType("timestamp", unit="s", timezone="UTC"),
                    "amount": DType("float64"),
                    "currency": DType("string"),
                    "channel": DType("string"),
                    "description": DType("string"),
                },
                time_column="at",
            ),
        ),
        relations=(
            RelationSpec("user_transactions", "users", "user_id", "transactions", "user_id"),
        ),
    )
    sample = SampleSpec(
        entity=EntitySpec("users", entity_type="user"),
        entity_values={
            "country": Field(column="country"),
            "home_currency": Field(column="home_currency"),
        },
        events={
            "transactions": EventSpec(
                Rows(path=("user_transactions",), time=TimeWindow("at", stop=0)),
                fields={
                    "amount": Field(column="amount"),
                    "currency": Field(column="currency"),
                    "channel": Field(column="channel"),
                    "description": Field(column="description"),
                },
            )
        },
    )
    tables = {
        "users": [
            {"user_id": 1, "country": "US", "home_currency": "USD"},
            {"user_id": 2, "country": "GB", "home_currency": "GBP"},
        ],
        "transactions": [
            {
                "transaction_id": 10,
                "user_id": 1,
                "at": 10,
                "amount": 20.0,
                "currency": "USD",
                "channel": "store",
                "description": "coffee",
            },
            {
                "transaction_id": 11,
                "user_id": 1,
                "at": 20,
                "amount": 100.0,
                "currency": "EUR",
                "channel": "online",
                "description": "airline ticket",
            },
            {
                "transaction_id": 12,
                "user_id": 2,
                "at": 15,
                "amount": 15.0,
                "currency": "GBP",
                "channel": "store",
                "description": "lunch",
            },
        ],
    }
    requests = (SampleRequest("user-1", 1, 30), SampleRequest("user-2", 2, 30))
    return (schema, sample, tables, requests)


def _resources():
    return {
        "id_spaces": {
            "tokens": IdSpace(
                specials={
                    name: f"[{name.upper()}]"
                    for name in ("pad", "unknown", "missing", "mask", "separator", "bos", "eos")
                }
            )
        },
        "field_names": {"fields": FieldNames()},
    }


def _tokens(*uses, history=None):
    return Fields(
        fields=uses,
        row=Tokens(max_tokens=8),
        history=Grouped() if history is None else history,
        field_names="fields",
        field_space="tokens",
    )


def _run(batch, *, seed=0):
    schema, sample, tables, requests = _data()
    plan = _Plans(schema, tables).prepare(
        DataLoaderSpec(schema=schema, sample=sample, batch=batch), requests
    )
    return (plan, next(plan.batches(requests, batch_size=2, seed=seed)))


def test_shared_vocabulary_is_fitted_once_across_entity_and_event_fields():
    batch = BatchSpec(
        encodings={"currency": Vocabulary()},
        inputs={
            "entity": Input(
                source=EntitySource(),
                representation=Fields(
                    fields=(Encode(field="home_currency", encoding=UseEncoding(name="currency")),)
                ),
                layout=Separate(),
            ),
            "events": Input(
                source=EventSource(events=("transactions",)),
                representation=Fields(
                    fields=(Encode(field="currency", encoding=UseEncoding(name="currency")),)
                ),
                layout=Padded(),
            ),
        },
    )
    plan, result = _run(batch)
    assert (
        result.inputs["entity"]["home_currency"]["values"][0]
        == result.inputs["events"]["currency"]["values"][0, 0]
    )
    assert {value for value, _index in plan.encoders["currency"].entries} == {"EUR", "GBP", "USD"}


def test_token_event_history_is_one_nested_input():
    batch = BatchSpec(
        **_resources(),
        inputs={
            "entity": Input(
                source=EntitySource(),
                representation=_tokens(
                    Encode(field="country", encoding=Vocabulary(), id_space="tokens")
                ),
                layout=Padded(),
            ),
            "history": Input(
                source=EventSource(events=("transactions",), max_events=8),
                representation=_tokens(
                    Encode(field="amount", encoding=Buckets(bins=4), id_space="tokens"),
                    Encode(field="channel", encoding=Vocabulary(), id_space="tokens"),
                    Encode(
                        field="description", encoding=Text(vocabulary_size=32), id_space="tokens"
                    ),
                ),
                layout=Nested(fields=Packed(), events=Padded()),
            ),
        },
    )
    _plan, result = _run(batch)
    assert tuple(result.inputs) == ("entity", "history")
    assert "field_offsets" in result.inputs["history"]
    assert result.inputs["history"]["sequence"]["valid"].shape == (2, 2)


def test_sample_calculations_and_explicit_encoders_define_flattened_field_order():
    schema, sample, tables, requests = _data()
    event = sample.events["transactions"]
    sample = replace(
        sample,
        events={
            "transactions": replace(
                event,
                fields={
                    **event.fields,
                    "time": Field(column="at"),
                    "magnitude": Derived(
                        operation="abs", inputs=("amount",), dtype=DType(name="float64")
                    ),
                    "hour": Derived(
                        operation=CalendarPart(part="hour"),
                        inputs=("time",),
                        dtype=DType(name="int64"),
                    ),
                },
            )
        },
    )
    batch = BatchSpec(
        **_resources(),
        inputs={
            "history": Input(
                source=EventSource(events=("transactions",), max_events=8),
                representation=_tokens(
                    Encode(field="channel", encoding=Vocabulary(), id_space="tokens"),
                    Encode(field="magnitude", encoding=Buckets(bins=4), id_space="tokens"),
                    Encode(field="hour", encoding=Vocabulary(), id_space="tokens"),
                    Encode(
                        field="description", encoding=Text(vocabulary_size=32), id_space="tokens"
                    ),
                    history=Flatten(max_tokens=64, separator="separator"),
                ),
                layout=Padded(),
            )
        },
    )
    plan = _Plans(schema, tables).prepare(
        DataLoaderSpec(schema=schema, sample=sample, batch=batch), requests
    )
    result = next(plan.batches(requests, batch_size=2))
    assert result.inputs["history"]["value_ids"].shape[0] == 2
    assert result.inputs["history"]["valid"].any()
    first = result.inputs["history"]["value_ids"][0, 0].item()
    assert first == plan.output(name="history").field(name="channel").encode(value="store")


def test_composite_rows_declare_each_component_encoding():
    batch = BatchSpec(
        **_resources(),
        inputs={
            "history": Input(
                source=EventSource(events=("transactions",), max_events=8),
                representation=Fields(
                    fields=(
                        Encode(field="channel", encoding=Vocabulary()),
                        Encode(field="amount", encoding=Buckets(bins=4)),
                    ),
                    row=Composite(
                        top_k=16, required_fields=("channel", "amount"), id_space="tokens"
                    ),
                ),
                layout=Padded(),
            )
        },
    )
    _plan, result = _run(batch)
    assert result.inputs["history"]["value_ids"].shape == (2, 2)


def test_nested_tokens_mask_values_without_splitting_history():
    batch = BatchSpec(
        **_resources(),
        inputs={
            "history": Input(
                source=EventSource(events=("transactions",)),
                representation=_tokens(
                    Encode(field="amount", encoding=Buckets(bins=4), id_space="tokens"),
                    Encode(field="channel", encoding=Vocabulary(), id_space="tokens"),
                ),
                layout=Nested(fields=Padded(), events=Packed()),
            )
        },
        targets={
            "masked": MaskedValuesTarget(
                input="history", token_probability=1.0, event_probability=0.0, key_probability=0.0
            )
        },
    )
    plan, result = _run(batch, seed=7)
    assert "offsets" in result.inputs["history"]["sequence"]
    assert result.targets["masked"]["valid"].all()
    assert result.targets["masked"]["corruption"].all()
    assert (
        result.inputs["history"]["value_ids"] == plan.spaces["tokens"].special(role="mask")
    ).all()


def test_flattened_tokens_emit_shifted_next_token_supervision():
    batch = BatchSpec(
        **_resources(),
        inputs={
            "history": Input(
                source=EventSource(events=("transactions",)),
                representation=_tokens(
                    Encode(field="channel", encoding=Vocabulary(), id_space="tokens"),
                    Encode(field="amount", encoding=Buckets(bins=4), id_space="tokens"),
                    history=Flatten(max_tokens=32),
                ),
                layout=Packed(),
            )
        },
        targets={"next": NextTokenTarget(input="history")},
    )
    _plan, result = _run(batch)
    assert result.inputs["history"]["offsets"].shape == (3,)
    assert result.targets["next"]["offsets"].shape == (3,)
    assert result.targets["next"]["valid"].all()


def test_explicit_input_contract_round_trips_through_json():
    schema, sample, _tables, _requests = _data()
    batch = BatchSpec(
        encodings={"currency": Vocabulary(values=(("USD", 2), ("GBP", 3)))},
        inputs={
            "history": Input(
                source=EventSource(events=("transactions",)),
                representation=Fields(
                    fields=(Encode(field="currency", encoding=UseEncoding(name="currency")),)
                ),
                layout=Packed(),
            )
        },
    )
    contract = DataLoaderSpec(schema=schema, sample=sample, batch=batch)
    restored = DataLoaderSpec.from_json(contract.to_json())
    assert restored == contract
    assert restored.fingerprint == contract.fingerprint


def test_nested_outputs_emit_only_declared_values_and_layout_structure():
    batch = BatchSpec(
        **_resources(),
        inputs={
            "entity": Input(
                source=EntitySource(),
                representation=_tokens(
                    Encode(field="country", encoding=Vocabulary(), id_space="tokens")
                ),
                layout=Padded(),
            ),
            "history": Input(
                source=EventSource(events=("transactions",), max_events=8),
                representation=_tokens(
                    Encode(field="amount", encoding=Buckets(bins=4), id_space="tokens"),
                    Encode(field="channel", encoding=Vocabulary(), id_space="tokens"),
                ),
                layout=Nested(fields=Packed(), events=Padded()),
            ),
        },
    )
    _plan, result = _run(batch)
    for name in ("field_ids", "value_ids", "position_ids", "valid"):
        assert name in result.inputs["entity"]
        assert name in result.inputs["history"]
    assert "field_offsets" in result.inputs["history"]
    assert set(result.inputs["history"]["sequence"]) == {"event_indices", "valid"}
    assert "timestamps" not in result.inputs["history"]


def test_supplied_text_and_bucket_state_runs_without_training_a_tokenizer(monkeypatch):
    batch = BatchSpec(
        **_resources(),
        inputs={
            "history": Input(
                source=EventSource(events=("transactions",)),
                representation=_tokens(
                    Encode(field="channel", encoding=Vocabulary(), id_space="tokens"),
                    Encode(field="amount", encoding=Buckets(bins=4), id_space="tokens"),
                    Encode(
                        field="description", encoding=Text(vocabulary_size=32), id_space="tokens"
                    ),
                    history=Flatten(max_tokens=32),
                ),
                layout=Padded(),
            )
        },
    )
    schema, sample, tables, requests = _data()
    engine = _Plans(schema, tables)
    fitted = engine.prepare(DataLoaderSpec(schema=schema, sample=sample, batch=batch), requests)
    original = next(fitted.batches(requests, batch_size=2))
    bound = fitted.bound_contract()
    import tokenizers.trainers

    monkeypatch.setattr(
        tokenizers.trainers,
        "BpeTrainer",
        lambda *args, **kwargs: pytest.fail("supplied Text state must not train"),
    )
    restored = next(engine.prepare(bound, requests).batches(requests, batch_size=2))
    assert original.inputs["history"]["value_ids"].equal(restored.inputs["history"]["value_ids"])


def test_supplied_shared_encoders_preserve_local_and_final_mappings():
    batch = BatchSpec(
        **_resources(),
        encodings={"amounts": Buckets(bins=4), "channels": Vocabulary()},
        inputs={
            "history": Input(
                source=EventSource(events=("transactions",)),
                representation=_tokens(
                    Encode(field="amount", encoding=UseEncoding(name="amounts"), id_space="tokens"),
                    Encode(
                        field="channel", encoding=UseEncoding(name="channels"), id_space="tokens"
                    ),
                ),
                layout=Nested(fields=Padded(), events=Padded()),
            )
        },
    )
    schema, sample, tables, requests = _data()
    engine = _Plans(schema, tables)
    fitted = engine.prepare(DataLoaderSpec(schema=schema, sample=sample, batch=batch), requests)
    original = next(fitted.batches(requests, batch_size=2))
    restored_plan = engine.prepare(fitted.bound_contract(), requests)
    restored = next(restored_plan.batches(requests, batch_size=2))
    assert original.inputs["history"]["field_ids"].equal(restored.inputs["history"]["field_ids"])
    assert original.inputs["history"]["value_ids"].equal(restored.inputs["history"]["value_ids"])
    assert fitted.spaces["tokens"].state_dict() == restored_plan.spaces["tokens"].state_dict()
