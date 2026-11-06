# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Batch declarations preserve typed inputs, targets, and relational selection."""

from dataclasses import replace

import pytest
import torch

from kumo.pytorch import api
from kumo.pytorch.assembly import prepare_batch_spec
from kumo.pytorch.engines.in_memory.samples import InMemorySampleBuilder
from tests.helpers import plan_in_memory

INT, FLOAT, STRING = api.DType("int64"), api.DType("float64"), api.DType("string")
TIME = api.DType("timestamp", unit="s", timezone="UTC")


def _columns(**encodings):
    return api.Fields(
        fields=tuple(
            api.Encode(field=name, encoding=encoding) for name, encoding in encodings.items()
        )
    )


def _fixture():
    schema = api.DataSchema(
        tables=(
            api.TableSpec(
                "users",
                "id",
                {"id": INT, "country": STRING, "score": FLOAT},
                time_column=api.NON_TEMPORAL,
            ),
            api.TableSpec(
                "events",
                "id",
                {
                    "id": INT,
                    "owner": INT,
                    "at": TIME,
                    "amount": FLOAT,
                    "channel": STRING,
                    "status": STRING,
                },
                time_column="at",
            ),
        ),
        relations=(api.RelationSpec("history", "users", "id", "events", "owner"),),
    )
    rows = {
        "users": [
            {"id": 1, "country": "C001", "score": 4.0},
            {"id": 2, "country": "C001", "score": 2.0},
        ],
        "events": [
            {
                "id": owner * 10 + index,
                "owner": owner,
                "at": index,
                "amount": amount,
                "channel": "online",
                "status": "approved",
            }
            for owner in (1, 2)
            for index, amount in ((1, 12.5), (2, 13.0))
        ],
    }
    selected = api.Rows(path=("history",), time=api.TimeWindow(column="at"))
    sample = api.SampleSpec(
        entity=api.EntitySpec(table="users"),
        entity_values={"country": api.Field(column="country"), "score": api.Field(column="score")},
        events={
            "payments": api.EventSpec(
                rows=selected,
                fields={
                    "amount": api.Field(column="amount"),
                    "channel": api.Field(column="channel"),
                    "status": api.Field(column="status"),
                },
            )
        },
    )
    requests = (
        api.SampleRequest(sample_id="one", entity_id=1, as_of_time=3),
        api.SampleRequest(sample_id="two", entity_id=2, as_of_time=3),
    )
    return schema, sample, rows, requests


def _tokens(*, flatten=False, composite=False):
    fields = (
        api.Encode(
            field="amount",
            encoding=api.Buckets(boundaries=(10.0, 20.0)),
            id_space=None if composite else "tokens",
        ),
        api.Encode(
            field="channel", encoding=api.Vocabulary(), id_space=None if composite else "tokens"
        ),
    )
    return api.Fields(
        fields=fields,
        row=api.Composite(id_space="tokens") if composite else api.Tokens(),
        history=api.Flatten(max_tokens=16) if flatten else api.Grouped(),
    )


def _spaces():
    return {
        "tokens": api.IdSpace(
            specials={"pad": "pad", "unknown": "unknown", "missing": "missing", "mask": "mask"}
        )
    }


def test_batch_spec_defines_data_outputs_without_model_construction():
    spec = api.BatchSpec(
        inputs={
            "history": api.Input(
                source=api.EventSource(events=("payments",)),
                representation=_tokens(),
                layout=api.Nested(fields=api.Packed(), events=api.Padded()),
            )
        },
        id_spaces=_spaces(),
    )
    assert spec.inputs
    assert not hasattr(spec, "encoder")
    assert not hasattr(spec, "model")
    assert not hasattr(spec, "objectives")


def test_supplied_label_vocabulary_produces_embedding_indices():
    schema, sample, rows, requests = _fixture()
    sample = replace(sample, targets={"churn": api.ExternalTarget(dtype=STRING)})
    batch = api.BatchSpec(
        inputs={
            "entity": api.Input(
                source=api.EntitySource(),
                representation=_columns(score=api.Raw()),
                layout=api.Separate(),
            )
        },
        targets={
            "churn": api.LabelTarget(
                source="churn", encoding=api.Vocabulary(values=(("no", 2), ("yes", 3)))
            )
        },
    )
    contract = api.DataLoaderSpec(schema, sample, batch)
    requests = tuple(
        replace(request, targets=(api.TargetValue(name="churn", value=label),))
        for request, label in zip(requests, ("no", "yes"), strict=True)
    )
    built = InMemorySampleBuilder(schema, rows).build(sample, requests)
    plan = prepare_batch_spec(sample, batch, built, contract=contract)
    output = next(plan.batches(built, batch_size=2))
    assert output.targets["churn"]["values"].tolist() == [2, 3]
    assert output.targets["churn"]["valid"].tolist() == [True, True]
    assert plan.manifest["targets"]["churn"]["value"]["vocabulary_size"] == 4
    assert plan.manifest == prepare_batch_spec(sample, batch, built, contract=contract).manifest
    changed = replace(
        batch,
        targets={
            "churn": api.LabelTarget(
                source="churn", encoding=api.Vocabulary(values=(("no", 3), ("yes", 2)))
            )
        },
    )
    alternate = prepare_batch_spec(
        sample, changed, built, contract=replace(contract, batch=changed)
    )
    assert alternate.manifest["artifact_fingerprint"] != plan.manifest["artifact_fingerprint"]


def test_packed_and_padded_sequences_contain_the_same_tokens():
    schema, sample, rows, requests = _fixture()
    outputs = []
    for layout in (api.Padded(), api.Packed()):
        batch = api.BatchSpec(
            inputs={
                "history": api.Input(
                    source=api.EventSource(events=("payments",)),
                    representation=_tokens(flatten=True),
                    layout=layout,
                )
            },
            id_spaces=_spaces(),
        )
        task = plan_in_memory(schema, rows, api.DataLoaderSpec(schema, sample, batch), requests)
        outputs.append(next(task.batches(requests, batch_size=2)).inputs["history"])
    padded, packed = outputs
    assert torch.equal(padded["value_ids"][padded["valid"]], packed["value_ids"])
    assert packed["offsets"][-1] == packed["value_ids"].numel()


def test_typed_entity_and_event_inputs_are_separate():
    schema, sample, rows, requests = _fixture()
    spec = api.BatchSpec(
        inputs={
            "entity": api.Input(
                source=api.EntitySource(),
                representation=_columns(country=api.Vocabulary(values=(("C001", 7),))),
                layout=api.Separate(),
            ),
            "events": api.Input(
                source=api.EventSource(events=("payments",)),
                representation=_columns(
                    amount=api.Raw(), channel=api.Vocabulary(values=(("online", 8),))
                ),
                layout=api.Padded(),
            ),
        },
        targets={
            "current_event": api.AlignedEventTarget(
                input="events",
                fields=(
                    api.Encode(field="status", encoding=api.Vocabulary(values=(("approved", 9),))),
                ),
            )
        },
    )
    task = plan_in_memory(schema, rows, api.DataLoaderSpec(schema, sample, spec), requests)
    batch = next(task.batches(requests, batch_size=2))
    assert tuple(batch.inputs) == ("entity", "events")
    assert batch.inputs["entity"]["country"]["values"].tolist() == [7, 7]
    assert batch.inputs["events"]["channel"]["values"].unique().tolist() == [8]
    assert batch.targets["current_event"]["status"]["values"].unique().tolist() == [9]


def test_typed_event_fields_work_without_a_prediction_target():
    schema, sample, rows, requests = _fixture()
    spec = api.BatchSpec(
        inputs={
            "events": api.Input(
                source=api.EventSource(events=("payments",)),
                representation=_columns(amount=api.Raw()),
                layout=api.Padded(),
            )
        }
    )
    task = plan_in_memory(schema, rows, api.DataLoaderSpec(schema, sample, spec), requests)
    batch = next(task.batches(requests, batch_size=2))
    assert tuple(batch.inputs) == ("events",)
    assert not batch.targets


def test_current_event_target_must_be_withheld_from_its_input():
    with pytest.raises(ValueError, match="aligned target fields.*event input"):
        api.BatchSpec(
            inputs={
                "events": api.Input(
                    source=api.EventSource(events=("payments",)),
                    representation=_columns(status=api.Vocabulary()),
                    layout=api.Padded(),
                )
            },
            targets={
                "label": api.AlignedEventTarget(
                    input="events", fields=(api.Encode(field="status", encoding=api.Vocabulary()),)
                )
            },
        )


def test_packed_prefixed_targets_keep_separate_offsets():
    schema, sample, rows, requests = _fixture()
    spec = api.BatchSpec(
        inputs={
            "entity": api.Input(
                source=api.EntitySource(),
                representation=_columns(country=api.Vocabulary()),
                layout=api.Separate(),
            ),
            "history": api.Input(
                source=api.EventSource(events=("payments",)),
                representation=_tokens(flatten=True, composite=True),
                layout=api.Packed(),
            ),
        },
        id_spaces=_spaces(),
        targets={"next_behavior": api.NextTokenTarget(input="history", prefix_input="entity")},
    )
    task = plan_in_memory(schema, rows, api.DataLoaderSpec(schema, sample, spec), requests)
    batch = next(task.batches(requests, batch_size=2))
    assert torch.equal(
        batch.targets["next_behavior"]["offsets"].diff(),
        batch.inputs["history"]["offsets"].diff() + 1,
    )


def test_graph_neighborhood_is_an_input_shape_over_the_same_relations():
    schema = api.DataSchema(
        tables=(
            api.TableSpec(
                "users",
                "user_id",
                {"user_id": INT, "country": STRING},
                time_column=api.NON_TEMPORAL,
            ),
            api.TableSpec(
                "accounts",
                "account_id",
                {"account_id": INT, "user_id": INT, "balance": FLOAT},
                time_column=api.NON_TEMPORAL,
            ),
            api.TableSpec(
                "transactions",
                "transaction_id",
                {"transaction_id": INT, "account_id": INT, "occurred_at": TIME, "amount": FLOAT},
                time_column="occurred_at",
            ),
        ),
        relations=(
            api.RelationSpec("user_accounts", "users", "user_id", "accounts", "user_id"),
            api.RelationSpec(
                "account_transactions", "accounts", "account_id", "transactions", "account_id"
            ),
        ),
    )
    rows = {
        "users": [{"user_id": 1, "country": "US"}],
        "accounts": [
            {"account_id": 10, "user_id": 1, "balance": 10.0},
            {"account_id": 11, "user_id": 1, "balance": 30.0},
        ],
        "transactions": [
            {"transaction_id": 100, "account_id": 10, "occurred_at": 70, "amount": 1.0},
            {"transaction_id": 101, "account_id": 10, "occurred_at": 90, "amount": 2.0},
            {"transaction_id": 102, "account_id": 10, "occurred_at": 100, "amount": 3.0},
            {"transaction_id": 103, "account_id": 11, "occurred_at": 80, "amount": 4.0},
        ],
    }
    sample = api.SampleSpec(
        entity=api.EntitySpec(table="users"),
        entity_values={"country": api.Field(column="country")},
        events={
            "transactions": api.EventSpec(
                rows=api.Rows(
                    path=("user_accounts", "account_transactions"),
                    time=api.TimeWindow(column="occurred_at"),
                ),
                fields={"amount": api.Field(column="amount")},
            )
        },
        neighborhoods={
            "activity": api.NeighborhoodSpec(
                layers=(
                    api.GraphLayer(
                        samples=(
                            api.GraphSample(
                                relation="user_accounts",
                                fanout=2,
                                strategy="uniform",
                                temporal="static",
                            ),
                        )
                    ),
                    api.GraphLayer(
                        samples=(
                            api.GraphSample(
                                relation="account_transactions",
                                fanout=1,
                                strategy="recent",
                                time_column="occurred_at",
                            ),
                        )
                    ),
                )
            )
        },
        targets={"label": api.ExternalTarget(dtype=INT)},
    )
    spec = api.BatchSpec(
        id_spaces=_spaces(),
        inputs={
            "entity": api.Input(
                source=api.EntitySource(),
                representation=_columns(country=api.Raw()),
                layout=api.Separate(),
            ),
            "profile_tokens": api.Input(
                source=api.EntitySource(),
                representation=api.Fields(
                    fields=(
                        api.Encode(field="country", encoding=api.Vocabulary(), id_space="tokens"),
                    ),
                    row=api.Tokens(),
                ),
                layout=api.Padded(),
            ),
            "event_tokens": api.Input(
                source=api.EventSource(events=("transactions",), max_events=8),
                representation=api.Fields(
                    fields=(
                        api.Encode(field="amount", encoding=api.Buckets(bins=4), id_space="tokens"),
                    ),
                    row=api.Tokens(),
                ),
                layout=api.Nested(fields=api.Packed(), events=api.Padded()),
            ),
            "neighborhood": api.Input(
                source=api.NeighborhoodSource(name="activity"),
                representation=api.Graph(
                    node_fields={
                        "users": _columns(country=api.Vocabulary(values=(("US", 2),))),
                        "accounts": _columns(balance=api.ZScore()),
                    }
                ),
                layout=None,
            ),
        },
        targets={
            "label": api.LabelTarget(source="label", encoding=api.Raw()),
            "masked": api.MaskedValuesTarget(
                input="event_tokens",
                token_probability=0.15,
                event_probability=0.10,
                key_probability=0.10,
            ),
        },
    )
    requests = [
        api.SampleRequest(
            sample_id=name,
            entity_id=1,
            as_of_time=100,
            targets=(api.TargetValue(name="label", value=1),),
        )
        for name in ("u1", "u1-again")
    ]
    task = plan_in_memory(schema, rows, api.DataLoaderSpec(schema, sample, spec), requests)
    batch = next(task.batches(requests, batch_size=2, seed=7))
    graph = batch.inputs["neighborhood"]
    assert graph["nodes"]["users"]["ids"].tolist() == [1, 1]
    assert graph["nodes"]["users"]["batch"].tolist() == [0, 1]
    assert graph["nodes"]["accounts"]["ids"].tolist() == [10, 11, 10, 11]
    assert graph["nodes"]["transactions"]["ids"].tolist() == [102, 103, 102, 103]
    assert graph["nodes"]["users"]["features"]["country"]["values"].tolist() == [2, 2]
    assert graph["edges"]["account_transactions"]["index"].shape == (2, 4)
    assert 101 not in graph["nodes"]["transactions"]["ids"].tolist()
    assert batch.inputs["entity"]["country"]["values"].to_pylist() == ["US", "US"]
    assert batch.inputs["event_tokens"]["field_offsets"].tolist() == list(range(9))


def test_entity_input_can_expose_calculated_values():
    schema, sample, rows, requests = _fixture()
    sample = replace(
        sample,
        entity_values={
            **sample.entity_values,
            "count": api.Aggregate(
                source=sample.events["payments"].rows, operation="count", dtype=INT
            ),
        },
    )
    spec = api.BatchSpec(
        inputs={
            "entity": api.Input(
                source=api.EntitySource(),
                representation=_columns(
                    country=api.Vocabulary(values=(("C001", 2),)), count=api.ZScore()
                ),
                layout=api.Separate(),
            )
        }
    )
    task = plan_in_memory(schema, rows, api.DataLoaderSpec(schema, sample, spec), requests)
    batch = next(task.batches(requests, batch_size=2))
    assert batch.inputs["entity"]["country"]["values"].tolist() == [2, 2]
    assert batch.inputs["entity"]["count"]["values"].shape == (2,)


def test_entity_values_can_mix_aggregates_and_derived_values():
    schema, sample, rows, requests = _fixture()
    sample = replace(
        sample,
        entity_values={
            "count": api.Aggregate(
                source=sample.events["payments"].rows, operation="count", dtype=INT
            ),
            "log_count": api.Derived(operation="log1p", inputs=("count",), dtype=FLOAT),
        },
    )
    built = InMemorySampleBuilder(schema, rows).build(sample, requests)[0]
    values = {field.key: field.value for field in built.history.entity.values}
    assert values["count"] == 2
    assert values["log_count"] == pytest.approx(torch.log1p(torch.tensor(2.0)).item())


@pytest.mark.parametrize("boolean", (False, True))
def test_relational_label_uses_a_separate_future_window(boolean):
    schema, sample, rows, _ = _fixture()
    future = api.Aggregate(
        source=api.Rows(path=("history",), time=api.TimeWindow(column="at", start=0, stop=20)),
        operation="count",
        dtype=INT,
    )
    label = (
        api.Compare(left=future, operation="greater_than", right=0, dtype=api.DType("bool"))
        if boolean
        else future
    )
    sample = replace(sample, targets={"future": label})
    rows["events"] = [
        {"id": i, "owner": 1, "at": time, "amount": 1.0, "channel": "online", "status": "approved"}
        for i, time in enumerate((99, 100, 119, 120), 1)
    ]
    spec = api.BatchSpec(
        inputs={
            "entity": api.Input(
                source=api.EntitySource(),
                representation=_columns(score=api.Raw()),
                layout=api.Separate(),
            )
        },
        targets={"future": api.LabelTarget(source="future", encoding=api.Raw())},
    )
    requests = [
        api.SampleRequest(sample_id="positive", entity_id=1, as_of_time=100),
        api.SampleRequest(sample_id="negative", entity_id=2, as_of_time=100),
    ]
    task = plan_in_memory(schema, rows, api.DataLoaderSpec(schema, sample, spec), requests)
    result = next(task.batches(requests, batch_size=2)).targets["future"]
    assert result["values"].tolist() == ([True, False] if boolean else [2, 0])
    assert result["values"].dtype == (torch.bool if boolean else torch.int64)
    assert result["valid"].tolist() == [True, True]
