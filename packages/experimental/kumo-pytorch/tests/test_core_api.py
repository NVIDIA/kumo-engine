# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace

import pytest

from kumo.pytorch.api import (
    Buckets,
    Columns,
    FieldNames,
    Flatten,
    Graph,
    IdSpace,
    NeighborhoodSource,
    Tokens,
)
from kumo.pytorch.assembly import prepare_batch_spec
from kumo.pytorch.contract.batch import (
    AlignedEventTarget,
    Batching,
    BatchSpec,
    Encode,
    EntitySource,
    EventSource,
    Fields,
    Input,
    LabelTarget,
    MaskedValuesTarget,
    Nested,
    NextTokenTarget,
    Packed,
    Padded,
    Raw,
    Separate,
    StructuredTarget,
    UseEncoding,
    Vocabulary,
    ZScore,
)
from kumo.pytorch.contract.dtypes import DType
from kumo.pytorch.contract.sample import (
    Aggregate,
    Compare,
    EntitySpec,
    EventSpec,
    Field,
    GraphLayer,
    GraphSample,
    GroupBy,
    NeighborhoodSpec,
    Reduce,
    Rows,
    RowTarget,
    SampleRequest,
    SampleSpec,
    TimeWindow,
)
from kumo.pytorch.contract.schema import NON_TEMPORAL, DataSchema, RelationSpec, TableSpec
from kumo.pytorch.data.records import EntityHistory, EntityState, Event, FieldValue
from kumo.pytorch.engines.in_memory.samples import InMemorySampleBuilder
from kumo.pytorch.spec import DataLoaderSpec, ValidationError
from tests.helpers import Plans as _Plans


def _schema() -> DataSchema:
    return DataSchema(
        tables=(
            TableSpec(
                "users",
                "user_id",
                columns={"user_id": DType("int64"), "country": DType("string")},
                time_column=NON_TEMPORAL,
            ),
            TableSpec(
                "accounts",
                "account_id",
                columns={
                    "account_id": DType("int64"),
                    "user_id": DType("int64"),
                    "opened_at": DType("timestamp", unit="s", timezone="UTC"),
                },
                time_column="opened_at",
            ),
            TableSpec(
                "transactions",
                "transaction_id",
                columns={
                    "transaction_id": DType("int64"),
                    "account_id": DType("int64"),
                    "occurred_at": DType("timestamp", unit="s", timezone="UTC"),
                    "amount": DType("float64"),
                    "channel": DType("string"),
                },
                time_column="occurred_at",
            ),
        ),
        relations=(
            RelationSpec("user_accounts", "users", "user_id", "accounts", "user_id"),
            RelationSpec(
                "account_transactions", "accounts", "account_id", "transactions", "account_id"
            ),
        ),
    )


def _tables():
    return {
        "users": [{"user_id": 1, "country": "US"}],
        "accounts": [
            {"account_id": 10, "user_id": 1, "opened_at": 1},
            {"account_id": 11, "user_id": 1, "opened_at": 2},
        ],
        "transactions": [
            {
                "transaction_id": 100,
                "account_id": 10,
                "occurred_at": 80,
                "amount": 1.0,
                "channel": "card",
            },
            {
                "transaction_id": 101,
                "account_id": 10,
                "occurred_at": 90,
                "amount": 2.0,
                "channel": "cash",
            },
            {
                "transaction_id": 102,
                "account_id": 11,
                "occurred_at": 95,
                "amount": 3.0,
                "channel": "card",
            },
            {
                "transaction_id": 103,
                "account_id": 11,
                "occurred_at": 110,
                "amount": 4.0,
                "channel": "web",
            },
            {
                "transaction_id": 104,
                "account_id": 10,
                "occurred_at": 120,
                "amount": 5.0,
                "channel": "card",
            },
        ],
    }


def _history_rows(*, start: int | None = None) -> Rows:
    return Rows(
        path=("user_accounts", "account_transactions"),
        time=TimeWindow("occurred_at", start=start, stop=0),
    )


def _future_rows(*, stop: int | None) -> Rows:
    return Rows(
        path=("user_accounts", "account_transactions"),
        time=TimeWindow("occurred_at", start=0, stop=stop),
    )


def _sample_spec() -> SampleSpec:
    account_counts = GroupBy(
        source=_history_rows(),
        keys=("account_id",),
        values={"transaction_count": Reduce("count", dtype=DType("int64"))},
    )
    return SampleSpec(
        entity=EntitySpec("users", entity_type="user"),
        entity_values={
            "country": Field(column="country"),
            "transaction_count": Aggregate(_history_rows(), "count", dtype=DType("int64")),
            "mean_transactions_per_account": Aggregate(
                account_counts, "mean", column="transaction_count", dtype=DType("float64")
            ),
        },
        events={
            "transactions": EventSpec(
                rows=_history_rows(),
                fields={"amount": Field(column="amount"), "channel": Field(column="channel")},
            )
        },
        targets={
            "spend_next_30s": Aggregate(
                _future_rows(stop=30), "sum", column="amount", dtype=DType("float64")
            ),
            "next_transaction": RowTarget(
                rows=_future_rows(stop=None),
                fields={"amount": Field(column="amount"), "channel": Field(column="channel")},
                selection="first",
            ),
        },
    )


def _batch_spec() -> BatchSpec:
    return BatchSpec(
        inputs={
            "entity": Input(
                source=EntitySource(),
                representation=Fields(
                    fields=(
                        Encode(field="country", encoding=Vocabulary(values=(("US", 2),))),
                        Encode(field="transaction_count", encoding=Raw()),
                        Encode(field="mean_transactions_per_account", encoding=Raw()),
                    ),
                    row=Columns(),
                ),
                layout=Separate(),
            ),
            "events": Input(
                source=EventSource(events=("transactions",), max_events=16),
                representation=Fields(
                    fields=(
                        Encode(field="amount", encoding=Raw()),
                        Encode(
                            field="channel", encoding=Vocabulary(values=(("card", 2), ("cash", 3)))
                        ),
                    ),
                    row=Columns(),
                ),
                layout=Padded(),
            ),
        },
        targets={
            "spend": LabelTarget(source="spend_next_30s", encoding=Raw()),
            "next_transaction": StructuredTarget(
                source="next_transaction",
                fields=(
                    Encode(field="amount", encoding=Raw()),
                    Encode(field="channel", encoding=Vocabulary(values=(("web", 2), ("card", 3)))),
                ),
            ),
        },
        batching=Batching(max_samples=8, length_input="events", bucket_by_length=True),
    )


def test_one_aggregate_expression_works_for_context_and_future_targets() -> None:
    spec = _sample_spec()
    assert isinstance(spec.entity_values["transaction_count"], Aggregate)
    assert isinstance(spec.targets["spend_next_30s"], Aggregate)
    sample = InMemorySampleBuilder(_schema(), _tables()).build(
        spec, [SampleRequest("user-1-at-100", entity_id=1, as_of_time=100, split="train")]
    )[0]
    entity_values = {field.key: field.value for field in sample.history.entity.values}
    targets = {target.name: target.value for target in sample.targets}
    assert entity_values["transaction_count"] == 3.0
    assert targets["spend_next_30s"] == 9.0
    assert all(event.timestamp < 100 for event in sample.history.events)
    assert sample.metadata["split"] == "train"


def test_aggregate_comparisons_round_trip_with_the_contract() -> None:
    sample = _sample_spec()
    compared = replace(
        sample,
        entity_values={
            **sample.entity_values,
            "has_transactions": Compare(
                left=Aggregate(_history_rows(), "count", dtype=DType("int64")),
                operation="greater_than",
                right=0,
                dtype=DType("bool"),
            ),
        },
        targets={
            **sample.targets,
            "will_transact": Compare(
                left=Aggregate(_future_rows(stop=30), "count", dtype=DType("int64")),
                operation="greater_than",
                right=0,
                dtype=DType("bool"),
            ),
        },
    )
    contract = DataLoaderSpec(schema=_schema(), sample=compared, batch=_batch_spec())
    contract.validate()
    restored = DataLoaderSpec.from_json(contract.to_json())
    assert restored == contract
    assert "entity value has_transactions: count" in contract.explain()
    assert "greater_than 0" in contract.explain()


def test_grouped_aggregate_supports_mean_of_per_account_counts() -> None:
    sample = InMemorySampleBuilder(_schema(), _tables()).build(
        _sample_spec(), [SampleRequest("sample", entity_id=1, as_of_time=100)]
    )[0]
    entity_values = {field.key: field.value for field in sample.history.entity.values}
    assert entity_values["mean_transactions_per_account"] == 1.5


def test_next_row_target_is_after_cutoff_and_never_enters_history() -> None:
    sample = InMemorySampleBuilder(_schema(), _tables()).build(
        _sample_spec(), [SampleRequest("sample", entity_id=1, as_of_time=100)]
    )[0]
    target = next(item.value for item in sample.targets if item.name == "next_transaction")
    assert isinstance(target, Event)
    assert target.timestamp == 110
    assert {field.key: field.value for field in target.fields} == {"amount": 4.0, "channel": "web"}
    assert 110 not in {event.timestamp for event in sample.history.events}


def test_contract_validation_reports_the_exact_bad_reference() -> None:
    event = _sample_spec().events["transactions"]
    broken = replace(
        _sample_spec(),
        events={"transactions": replace(event, fields={"amount": Field(column="ammount")})},
    )
    contract = DataLoaderSpec(schema=_schema(), sample=broken, batch=_batch_spec())
    with pytest.raises(ValidationError) as error:
        contract.validate()
    assert "sample.events.transactions.fields.amount.column" in str(error.value)
    assert "transactions.ammount" in str(error.value)


def test_an_unknown_entity_value_is_reported_once() -> None:
    batch = replace(
        _batch_spec(),
        inputs={
            **_batch_spec().inputs,
            "entity": Input(
                source=EntitySource(),
                representation=Fields(
                    fields=(Encode(field="missing", encoding=Raw()),), row=Columns()
                ),
                layout=Separate(),
            ),
        },
    )
    contract = DataLoaderSpec(schema=_schema(), sample=_sample_spec(), batch=batch)
    with pytest.raises(ValidationError) as error:
        contract.validate()
    assert [issue.path for issue in error.value.issues].count(
        "batch.inputs.entity.representation.fields[0]"
    ) == 1


def test_future_rows_cannot_be_declared_as_context() -> None:
    sample = replace(
        _sample_spec(),
        events={
            "transactions": replace(
                _sample_spec().events["transactions"], rows=_future_rows(stop=30)
            )
        },
    )
    with pytest.raises(ValidationError, match="context window must end at or before"):
        DataLoaderSpec(schema=_schema(), sample=sample, batch=_batch_spec()).validate()


def test_complete_contract_round_trips_and_explains_itself() -> None:
    contract = DataLoaderSpec(schema=_schema(), sample=_sample_spec(), batch=_batch_spec())
    contract.validate()
    restored = DataLoaderSpec.from_json(contract.to_json())
    assert restored == contract
    assert restored.fingerprint == contract.fingerprint
    explanation = contract.explain()
    assert explanation.splitlines()[0] == f"Data-loader specification v{contract.version}"
    assert "entity: users.user_id (user)" in explanation
    assert "entity value transaction_count: count" in explanation
    assert "target next_transaction: first future row" in explanation
    assert "batch input events: EventSource -> Columns, Padded" in explanation


def test_structured_future_target_uses_declared_field_encodings() -> None:
    contract = DataLoaderSpec(schema=_schema(), sample=_sample_spec(), batch=_batch_spec())
    requests = (SampleRequest("sample", 1, 100, split="train"),)
    plan = _Plans(_schema(), _tables()).prepare(contract, requests)
    batch = next(plan.batches(requests))
    assert batch.targets["next_transaction"]["amount"]["values"].tolist() == [4.0]
    assert batch.targets["next_transaction"]["channel"]["values"].tolist() == [2]
    assert batch.metadata["splits"] == ("train",)
    assert batch.metadata["contract_fingerprint"] == contract.fingerprint


def test_current_event_is_explicit_and_can_also_appear_in_history() -> None:
    current = EventSpec(
        rows=Rows(
            path=("user_accounts", "account_transactions"),
            time=TimeWindow("occurred_at", start=-1, stop=0),
        ),
        fields={"amount": Field(column="amount"), "channel": Field(column="channel")},
        limit=1,
    )
    sample_spec = replace(_sample_spec(), current_events={"transaction": current})
    batch_spec = BatchSpec(
        inputs={
            "current_transaction": Input(
                source=EventSource(events=("transaction",), max_events=1, collection="current"),
                representation=Fields(
                    fields=(Encode(field="amount", encoding=Raw()),), row=Columns()
                ),
                layout=Padded(),
            )
        },
        targets={
            "channel": AlignedEventTarget(
                input="current_transaction",
                fields=(Encode(field="channel", encoding=Vocabulary(values=(("card", 2),))),),
            )
        },
        batching=Batching(max_samples=1),
    )
    tables = _tables()
    tables["transactions"].append(
        {
            "transaction_id": 105,
            "account_id": 10,
            "occurred_at": 100,
            "amount": 7.0,
            "channel": "card",
        }
    )
    contract = DataLoaderSpec(_schema(), sample_spec, batch_spec)
    request = SampleRequest("scored-transaction", 1, 100)
    built = InMemorySampleBuilder(_schema(), tables).build(sample_spec, [request])[0]
    assert all(event.timestamp <= 100 for event in built.history.events)
    assert any(event.timestamp == 100 for event in built.history.events)
    assert built.current_events["transaction"][0].timestamp == 100
    plan = _Plans(_schema(), tables).prepare(contract, [request])
    batch = next(plan.batches([request]))
    assert batch.inputs["current_transaction"]["amount"]["values"].tolist() == [[7.0]]
    assert batch.targets["channel"]["channel"]["values"].tolist() == [[2]]


def test_length_bucketing_and_token_budget_control_batch_composition() -> None:
    sample = SampleSpec(
        entity=EntitySpec("users", entity_type="user"),
        events={
            "transactions": EventSpec(
                rows=Rows(time=TimeWindow("occurred_at")), fields={"amount": Field(column="amount")}
            )
        },
    )
    spec = BatchSpec(
        inputs={
            "events": Input(
                source=EventSource(events=("transactions",)),
                representation=Fields(
                    fields=(Encode(field="amount", encoding=Raw()),), row=Columns()
                ),
                layout=Padded(),
            )
        },
        batching=Batching(
            max_samples=8, max_tokens=6, length_input="events", bucket_by_length=True
        ),
    )

    def history(identifier: int, length: int) -> EntityHistory:
        events = tuple(
            Event(
                "transactions",
                index + 1,
                (FieldValue("amount", float(index), DType(name="float64")),),
            )
            for index in range(length)
        )
        return EntityHistory(identifier, 100, EntityState((), ()), events)

    histories = [history(1, 5), history(2, 1), history(3, 3)]
    plan = prepare_batch_spec(sample, spec, histories)
    batches = list(plan.batches(histories))
    composition = [batch.metadata["entity_ids"].tolist() for batch in batches]
    assert composition == [[2, 3], [1]]


def _drop_last_plan(batching: Batching, lengths: Sequence[int]) -> Callable[..., list[list[int]]]:
    sample = SampleSpec(
        entity=EntitySpec("users", entity_type="user"),
        events={
            "transactions": EventSpec(
                rows=Rows(time=TimeWindow("occurred_at")), fields={"amount": Field(column="amount")}
            )
        },
    )
    spec = BatchSpec(
        inputs={
            "events": Input(
                source=EventSource(events=("transactions",)),
                representation=Fields(
                    fields=(Encode(field="amount", encoding=Raw()),), row=Columns()
                ),
                layout=Padded(),
            )
        },
        batching=batching,
    )
    histories = [
        EntityHistory(
            identifier,
            100,
            EntityState((), ()),
            tuple(
                Event(
                    "transactions",
                    index + 1,
                    (FieldValue("amount", float(index), DType(name="float64")),),
                )
                for index in range(length)
            ),
        )
        for identifier, length in enumerate(lengths, start=1)
    ]
    plan = prepare_batch_spec(sample, spec, histories)

    def composition(**options: object) -> list[list[int]]:
        return [
            batch.metadata["entity_ids"].tolist() for batch in plan.batches(histories, **options)
        ]

    return composition


def test_drop_last_under_a_token_budget_drops_only_the_final_short_batch() -> None:
    composition = _drop_last_plan(
        Batching(max_samples=8, max_tokens=6, length_input="events"), [3, 3, 3, 3, 1]
    )
    assert composition() == [[1, 2], [3, 4], [5]]
    assert composition(drop_last=True) == [[1, 2], [3, 4]]


def test_drop_last_judges_the_final_batch_by_sample_count_not_tokens() -> None:
    composition = _drop_last_plan(
        Batching(max_samples=8, max_tokens=6, length_input="events"), [3, 3, 3, 3]
    )
    assert composition() == [[1, 2], [3, 4]]
    assert composition(drop_last=True) == [[1, 2]]
    unlimited = _drop_last_plan(Batching(max_tokens=6, length_input="events"), [3, 3, 3, 3, 1])
    assert unlimited(drop_last=True) == unlimited() == [[1, 2], [3, 4], [5]]


def test_drop_last_under_a_token_budget_keeps_a_full_final_batch() -> None:
    composition = _drop_last_plan(
        Batching(max_samples=2, max_tokens=6, length_input="events", drop_last=True),
        [1, 1, 3, 3],
    )
    assert composition() == [[1, 2], [3, 4]]
    assert composition(drop_last=False) == [[1, 2], [3, 4]]


def test_drop_last_without_a_token_budget_drops_the_final_short_batch() -> None:
    composition = _drop_last_plan(Batching(max_samples=2), [1, 2, 3, 1, 2])
    assert composition() == [[1, 2], [3, 4], [5]]
    assert composition(drop_last=True) == [[1, 2], [3, 4]]
    assert composition(batch_size=5, drop_last=True) == [[1, 2, 3, 4, 5]]


def test_token_budget_rejects_one_example_that_cannot_fit() -> None:
    sample = SampleSpec(
        entity=EntitySpec("users", entity_type="user"),
        events={
            "transactions": EventSpec(
                rows=Rows(time=TimeWindow("occurred_at")), fields={"amount": Field(column="amount")}
            )
        },
    )
    spec = BatchSpec(
        inputs={
            "events": Input(
                source=EventSource(events=("transactions",)),
                representation=Fields(
                    fields=(Encode(field="amount", encoding=Raw()),), row=Columns()
                ),
                layout=Padded(),
            )
        },
        batching=Batching(max_samples=8, max_tokens=2, length_input="events"),
    )
    history = EntityHistory(
        1,
        100,
        EntityState((), ()),
        tuple(
            Event(
                "transactions",
                index,
                (FieldValue("amount", float(index), DType(name="float64")),),
            )
            for index in range(3)
        ),
    )
    with pytest.raises(ValueError, match="exceeds Batching.max_tokens"):
        next(prepare_batch_spec(sample, spec, [history]).batches([history]))


def test_graph_references_and_traversal_direction_are_validated_before_fitting() -> None:
    sample = replace(
        _sample_spec(),
        neighborhoods={
            "activity": NeighborhoodSpec(
                layers=(
                    GraphLayer(
                        (
                            GraphSample(
                                "user_accounts",
                                fanout=2,
                                direction="reverse",
                                time_column="opend_at",
                            ),
                        )
                    ),
                )
            )
        },
    )
    batch = BatchSpec(
        inputs={
            "graph": Input(
                source=NeighborhoodSource(name="activity"),
                representation=Graph(
                    node_fields={
                        "accounts": Fields(fields=(Encode(field="balnce", encoding=Raw()),))
                    }
                ),
                layout=None,
            )
        },
        batching=Batching(max_samples=1),
    )
    with pytest.raises(ValidationError) as error:
        DataLoaderSpec(_schema(), sample, batch).validate()
    message = str(error.value)
    assert "sample.neighborhoods.activity.layers[0].samples[0].direction" in message
    assert "sample.neighborhoods.activity.layers[0].samples[0].time_column" in message
    valid_sample = replace(
        _sample_spec(),
        neighborhoods={
            "activity": NeighborhoodSpec(
                layers=(GraphLayer(samples=(GraphSample(relation="user_accounts", fanout=2),)),)
            )
        },
    )
    with pytest.raises(ValidationError) as field_error:
        DataLoaderSpec(schema=_schema(), sample=valid_sample, batch=batch).validate()
    assert "batch.inputs.graph.node_fields.accounts.fields[0]" in str(field_error.value)
    assert "balnce" in str(field_error.value)


def test_supplied_field_encoder_state_reproduces_local_and_final_ids():
    batch = BatchSpec(
        encodings={"categories": Vocabulary()},
        id_spaces={
            "tokens": IdSpace(
                specials={
                    "pad": "[PAD]",
                    "unknown": "[UNK]",
                    "missing": "[MISSING]",
                    "mask": "[MASK]",
                }
            )
        },
        field_names={"fields": FieldNames()},
        inputs={
            "entity": Input(
                source=EntitySource(),
                representation=Fields(
                    fields=(
                        Encode(
                            field="country",
                            encoding=UseEncoding(name="categories"),
                            id_space="tokens",
                        ),
                    ),
                    row=Tokens(max_tokens=8),
                    field_names="fields",
                    field_space="tokens",
                ),
                layout=Padded(),
            ),
            "events": Input(
                source=EventSource(events=("transactions",), max_events=16),
                representation=Fields(
                    fields=(
                        Encode(field="amount", encoding=Buckets(bins=4), id_space="tokens"),
                        Encode(
                            field="channel",
                            encoding=UseEncoding(name="categories"),
                            id_space="tokens",
                        ),
                    ),
                    row=Tokens(max_tokens=4),
                    field_names="fields",
                    field_space="tokens",
                ),
                layout=Nested(fields=Packed(), events=Padded()),
            ),
        },
        targets={
            "masked": MaskedValuesTarget(
                input="events", token_probability=1.0, event_probability=0.0, key_probability=0.0
            )
        },
        batching=Batching(max_samples=1),
    )
    request = SampleRequest(sample_id="sample", entity_id=1, as_of_time=100)
    engine = _Plans(_schema(), _tables())
    fitted = engine.prepare(
        DataLoaderSpec(schema=_schema(), sample=_sample_spec(), batch=batch), [request]
    )
    supplied = engine.prepare(fitted.bound_contract(), [request])
    before = next(fitted.batches([request], seed=7)).inputs["events"]
    after = next(supplied.batches([request], seed=7)).inputs["events"]
    assert after["field_ids"].equal(before["field_ids"])
    assert after["value_ids"].equal(before["value_ids"])
    assert supplied.encoders["categories"].entries == fitted.encoders["categories"].entries
    assert supplied.spaces["tokens"].state_dict() == fitted.spaces["tokens"].state_dict()


def test_supplied_normalization_state_is_used_without_refitting() -> None:
    batch = BatchSpec(
        inputs={
            "entity": Input(
                source=EntitySource(),
                representation=Fields(
                    fields=(
                        Encode(field="transaction_count", encoding=ZScore(mean=10.0, scale=2.0)),
                    ),
                    row=Columns(),
                ),
                layout=Separate(),
            )
        },
        batching=Batching(max_samples=1),
    )
    contract = DataLoaderSpec(_schema(), _sample_spec(), batch)
    request = SampleRequest("sample", 1, 100)
    encoded = next(_Plans(_schema(), _tables()).prepare(contract, [request]).batches([request]))
    assert encoded.inputs["entity"]["transaction_count"]["values"].tolist() == [-3.5]


def test_a_forbidden_field_is_rejected_at_declaration():
    with pytest.raises(ValueError, match="forbidden"):
        Fields(
            fields=(Encode(field="amount", encoding=Buckets(bins=4), id_space="tokens"),),
            row=Tokens(max_tokens=4),
            forbidden_fields=("amount",),
        )


def test_reserved_token_roles_are_explicit_contract_configuration():
    batch = BatchSpec(
        id_spaces={
            "tokens": IdSpace(
                specials={"pad": "<pad>", "unknown": "<unknown>", "separator": "<boundary>"}
            )
        },
        inputs={
            "history": Input(
                source=EventSource(events=("transactions",)),
                representation=Fields(
                    fields=(Encode(field="amount", encoding=Buckets(bins=4), id_space="tokens"),),
                    row=Tokens(),
                    history=Flatten(max_tokens=16),
                ),
                layout=Padded(),
            )
        },
        targets={"next": NextTokenTarget(input="history")},
        batching=Batching(max_samples=1),
    )
    contract = DataLoaderSpec(schema=_schema(), sample=_sample_spec(), batch=batch)
    plan = _Plans(_schema(), _tables()).prepare(
        contract, [SampleRequest(sample_id="sample", entity_id=1, as_of_time=100)]
    )
    assert dict(plan.spaces["tokens"].roles) == {"pad": 0, "unknown": 1, "separator": 2}
    assert contract.batch.id_spaces["tokens"].specials["separator"] == "<boundary>"
