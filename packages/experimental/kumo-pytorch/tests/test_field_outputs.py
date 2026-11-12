# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common encoders and shared event outputs through actual preparation."""

import random
from dataclasses import replace
from decimal import Decimal, localcontext

import pyarrow as pa
import pytest
import torch

from kumo.pytorch.assembly import fit_plan
from kumo.pytorch.contract.batch import (
    Batching,
    BatchSpec,
    Buckets,
    Composite,
    Encode,
    EventsOf,
    EventSource,
    FieldNames,
    Fields,
    Flatten,
    Grouped,
    IdSpace,
    Input,
    MaskedValuesTarget,
    Nested,
    Packed,
    Padded,
    Raw,
    Tokens,
    UseEncoding,
    Vocabulary,
    ZScore,
)
from kumo.pytorch.contract.dtypes import DType, arrow_float64
from kumo.pytorch.contract.sample import (
    EntitySpec,
    EventSpec,
    EventTimeDelta,
    Field,
    Rows,
    SampleRequest,
    SampleSpec,
    TimeWindow,
)
from kumo.pytorch.contract.schema import NON_TEMPORAL, DataSchema, RelationSpec, TableSpec
from kumo.pytorch.encoding import fit_encoder, typed_key
from kumo.pytorch.engines.in_memory.source import InMemorySource
from kumo.pytorch.spec import DataLoaderSpec


def _tables():
    return {
        "users": [{"id": 1}, {"id": 2}],
        "events": [
            {"id": 1, "user": 1, "time": 10, "amount": 5.0, "currency": "USD"},
            {"id": 2, "user": 1, "time": 20, "amount": 20.0, "currency": None},
            {"id": 3, "user": 1, "time": 30, "amount": 80.0, "currency": "EUR"},
        ],
    }


def _contract(*, flat=False, composite=False, masking=False):
    integer, number, text = DType("int64"), DType("float64"), DType("string")
    clock = DType("timestamp", unit="s", timezone="UTC")
    schema = DataSchema(
        tables=(
            TableSpec(
                name="users", primary_key="id", columns={"id": integer}, time_column=NON_TEMPORAL
            ),
            TableSpec(
                name="events",
                primary_key="id",
                columns={
                    "id": integer,
                    "user": integer,
                    "time": clock,
                    "amount": number,
                    "currency": text,
                },
                time_column="time",
            ),
        ),
        relations=(
            RelationSpec(
                name="history",
                left_table="users",
                left_key="id",
                right_table="events",
                right_key="user",
            ),
        ),
    )
    sample = SampleSpec(
        entity=EntitySpec(table="users"),
        events={
            "payments": EventSpec(
                rows=Rows(path=("history",), time=TimeWindow(column="time")),
                fields={
                    "amount": Field(column="amount"),
                    "currency": Field(column="currency"),
                    "age": EventTimeDelta(dtype=number),
                },
            )
        },
    )
    row = (
        Composite(id_space="tokens", required_fields=("currency",))
        if composite
        else Tokens(missing="skip_event")
    )
    use_space = None if composite else "tokens"
    group = Fields(
        fields=(
            Encode(field="amount", encoding=Buckets(boundaries=(10.0, 50.0)), id_space=use_space),
            Encode(field="currency", encoding=UseEncoding(name="currencies"), id_space=use_space),
        ),
        row=row,
        history=Flatten(max_tokens=20) if flat else Grouped(),
        field_names=None if composite else "fields",
    )
    inputs = {
        "history": Input(
            source=EventSource(events=("payments",)),
            representation=group,
            layout=Packed() if flat or composite else Nested(fields=Packed(), events=Padded()),
        )
    }
    if not flat:
        inputs["timing"] = Input(
            source=EventsOf(input="history"),
            representation=Fields(fields=(Encode(field="age", encoding=Raw()),)),
            layout=Padded(),
        )
    targets = (
        {
            "masked": MaskedValuesTarget(
                input="history", token_probability=1.0, event_probability=0.0, key_probability=0.0
            )
        }
        if masking
        else {}
    )
    return DataLoaderSpec(
        schema=schema,
        sample=sample,
        batch=BatchSpec(
            inputs=inputs,
            targets=targets,
            encodings={"currencies": Vocabulary()},
            id_spaces={
                "tokens": IdSpace(
                    specials={
                        "pad": "pad",
                        "unknown": "unknown",
                        "missing": "missing",
                        "mask": "mask",
                    }
                )
            },
            field_names={} if composite else {"fields": FieldNames()},
        ),
    )


def _source(contract):
    return InMemorySource(contract.schema, _tables())


def _plan(contract):
    source = _source(contract)
    requests = [
        SampleRequest(sample_id="a", entity_id=1, as_of_time=40),
        SampleRequest(sample_id="b", entity_id=2, as_of_time=40),
    ]
    return fit_plan(contract.sample, contract.batch, requests, source, contract=contract), requests


def test_aligned_columns_follow_retained_events_and_empty_history():
    plan, requests = _plan(_contract())
    batch = next(plan.batches(requests, batch_size=2))
    assert batch.inputs["history"]["field_offsets"].tolist() == [0, 2, 4]
    assert batch.inputs["history"]["sequence"]["event_indices"].tolist() == [[0, 1], [-1, -1]]
    assert batch.inputs["timing"]["age"]["values"].tolist() == [[30.0, 10.0], [0.0, 0.0]]
    assert batch.inputs["timing"]["age"]["valid"].tolist() == [[True, True], [False, False]]
    assert "timestamps" not in batch.inputs["history"]
    assert "elapsed_seconds" not in batch.inputs["history"]["sequence"]


def test_flat_and_grouped_share_token_values_but_not_boundaries():
    grouped, requests = _plan(_contract())
    flattened, _ = _plan(_contract(flat=True))
    left = next(grouped.batches(requests, batch_size=2)).inputs["history"]
    right = next(flattened.batches(requests, batch_size=2)).inputs["history"]
    assert left["value_ids"].tolist() == right["value_ids"].tolist()
    assert right["offsets"].tolist() == [0, 4, 4]
    assert left["field_offsets"].tolist() == [0, 2, 4]


def test_masked_targets_copy_final_ids_before_corruption():
    original, requests = _plan(_contract())
    masked, _ = _plan(_contract(masking=True))
    before = next(original.batches(requests, batch_size=2)).inputs["history"]["value_ids"]
    batch = next(masked.batches(requests, batch_size=2))
    assert (
        batch.inputs["history"]["value_ids"].tolist()
        == [masked.spaces["tokens"].special(role="mask")] * 4
    )
    assert torch.equal(batch.targets["masked"]["values"], before)
    assert batch.targets["masked"]["valid"].all()


@pytest.mark.parametrize(
    "dtype,values",
    [
        (DType("float64"), [1.0, 1.5, None]),
        (DType("uint64"), [2**64 - 1, 2**63, None]),
        (DType("bytes"), [b"a", b"b", None]),
        (
            DType("decimal256", precision=60, scale=2),
            [
                Decimal("123456789012345678901234567890.12"),
                Decimal("123456789012345678901234567890.13"),
                None,
            ],
        ),
    ],
)
def test_vocabulary_exact_typed_python_and_arrow_agreement(dtype, values):
    encoder = fit_encoder("example", Vocabulary(), dtype, values)
    local = [encoder.encode(value=value) for value in values]
    arrow = encoder.encode_arrow(pa.array(values, type=dtype.to_arrow()))["values"].tolist()
    assert local == arrow
    assert local[0] != local[1]


def test_reuse_does_not_refit_and_keeps_final_ids():
    contract = _contract()
    plan, requests = _plan(contract)
    reused = fit_plan(
        contract.sample,
        contract.batch,
        [],
        _source(contract),
        contract=contract,
        input_state=plan._fitted_input_state(),
        target_encoders=plan._target_encoders,
    )
    assert reused.fitted_state == plan.fitted_state
    assert reused.output(name="history").field(name="currency").encode(value="USD") == plan.output(
        name="history"
    ).field(name="currency").encode(value="USD")


def test_composite_preserves_component_order_and_retention():
    plan, requests = _plan(_contract(composite=True))
    batch = next(plan.batches(requests, batch_size=2))
    assert batch.inputs["history"]["valid"].tolist() == [True, True]
    assert batch.inputs["history"]["offsets"].tolist() == [0, 2, 2]


def test_signed_zero_one_category_and_null_not_literal():
    encoder = fit_encoder("number", Vocabulary(), DType("float64"), [0.0, -0.0])
    assert encoder.encode(value=0.0) == encoder.encode(value=-0.0)
    assert (
        encoder.encode_arrow(pa.array([0.0, -0.0]))["values"].tolist()
        == [encoder.encode(value=0.0)] * 2
    )
    string = fit_encoder("string", Vocabulary(), DType("string"), ["[MISSING]", "None"])
    assert string.encode(value="[MISSING]") != string.encode(value=None)
    assert typed_key(
        DType("decimal256", precision=60, scale=0), Decimal("123456789012345678901234567890")
    ) != typed_key(
        DType("decimal256", precision=60, scale=0), Decimal("123456789012345678901234567891")
    )


@pytest.mark.parametrize("packed", (False, True))
def test_grouped_token_budget_counts_materialized_token_slots(packed):
    contract = _contract()
    history = contract.batch.inputs["history"]
    history = replace(
        history,
        representation=replace(
            history.representation, row=replace(history.representation.row, missing="omit")
        ),
        layout=Nested(fields=Packed() if packed else Padded(), events=Padded()),
    )
    budget = 5 if packed else 6
    contract = replace(
        contract,
        batch=replace(
            contract.batch,
            inputs={"history": history},
            batching=Batching(max_tokens=budget, length_input="history"),
        ),
    )
    plan, requests = _plan(contract)
    values = [requests[0], replace(requests[0], sample_id="again"), requests[1]]
    batches = list(plan.batches(values))
    assert len(batches) == 2
    assert all(batch.inputs["history"]["value_ids"].numel() <= budget for batch in batches)


def test_single_grouped_history_exceeding_padding_budget_is_refused():
    contract = _contract()
    history = contract.batch.inputs["history"]
    history = replace(
        history,
        representation=replace(
            history.representation, row=replace(history.representation.row, missing="omit")
        ),
        layout=Nested(fields=Padded(), events=Padded()),
    )
    contract = replace(
        contract,
        batch=replace(
            contract.batch,
            inputs={"history": history},
            batching=Batching(max_tokens=5, length_input="history"),
        ),
    )
    plan, requests = _plan(contract)
    with pytest.raises(ValueError, match="exceeds.*max_tokens"):
        next(plan.batches(requests))


def test_columns_field_ids_distinguish_declared_nulls_from_padding():
    contract = _contract()
    history = contract.batch.inputs["history"]
    history = replace(
        history,
        representation=replace(
            history.representation,
            field_space="tokens",
            row=replace(history.representation.row, missing="omit"),
        ),
    )
    details = Input(
        source=EventsOf(input="history"),
        representation=Fields(
            fields=(
                Encode(field="amount", encoding=Raw()),
                Encode(field="currency", encoding=Raw()),
            ),
            field_names="fields",
            field_space="tokens",
        ),
        layout=Padded(),
    )
    contract = replace(
        contract, batch=replace(contract.batch, inputs={"history": history, "details": details})
    )
    plan, requests = _plan(contract)
    currency = next(plan.batches(requests, batch_size=2)).inputs["details"]["currency"]
    assert currency["values"].to_pylist() == [["USD", None, "EUR"], [None, None, None]]
    assert currency["valid"].tolist() == [[True, False, True], [False, False, False]]
    assert currency["field_ids"].tolist() == [[5, 5, 5], [0, 0, 0]]
    assert currency["field_valid"].tolist() == [[True, True, True], [False, False, False]]


@pytest.mark.parametrize(
    "dtype,values",
    [
        (DType("uint64"), [2**64 - 1, 2**64 - 2]),
        (
            DType("decimal256", precision=60, scale=2),
            [
                Decimal("123456789012345678901234567890.12"),
                Decimal("123456789012345678901234567890.13"),
            ],
        ),
    ],
)
@pytest.mark.parametrize(
    "config", [ZScore(mean=0.0, scale=2.0, dtype="float64"), Buckets(boundaries=(float(2**63),))]
)
def test_numeric_binary64_working_domain_matches_scalar_and_arrow(dtype, values, config):
    encoder = fit_encoder("wide", config, dtype, values)
    expected = [encoder.encode(value=value) for value in values]
    assert (
        encoder.encode_arrow(pa.array(values, type=dtype.to_arrow()))["values"].tolist() == expected
    )


@pytest.mark.parametrize(
    "boundary,values,expected",
    [
        (906.91645, ["906.916450", "906.916449", "906.916451", None], [2, 1, 2, 0]),
        (0.1, ["0.100000", "0.099999", "-0.100000"], [2, 1, 1]),
        (-123.456789, ["-123.456789", "-123.456790", "-123.456788"], [2, 1, 2]),
    ],
)
def test_decimal_boundary_buckets_agree_on_row_and_batch_paths(boundary, values, expected):
    dtype = DType("decimal128", precision=38, scale=6)
    decimals = [None if value is None else Decimal(value) for value in values]
    encoder = fit_encoder("amount", Buckets(boundaries=(boundary,)), dtype, [])
    rows = [encoder.encode(value=value) for value in decimals]
    batch = encoder.encode_arrow(pa.array(decimals, type=dtype.to_arrow()))["values"].tolist()
    assert rows == batch == expected


@pytest.mark.parametrize(
    "dtype",
    [
        DType("decimal128", precision=38, scale=6),
        DType("decimal256", precision=76, scale=30),
        DType("decimal128", precision=18, scale=-4),
    ],
)
def test_decimal_float64_conversion_is_correctly_rounded(dtype):
    generator = random.Random(dtype.scale)
    with localcontext() as context:
        context.prec = 100
        decimals = [
            Decimal(generator.randrange(-(10**dtype.precision) + 1, 10**dtype.precision)).scaleb(
                -dtype.scale
            )
            for _ in range(4_000)
        ]
    converted = arrow_float64(pa.array([*decimals, None], type=dtype.to_arrow())).to_pylist()
    assert converted == [*(float(value) for value in decimals), None]


def test_zscore_refuses_emitted_float32_overflow_consistently():
    encoder = fit_encoder("huge", ZScore(mean=0.0, scale=1e-30), DType("float64"), [])
    with pytest.raises(ValueError, match="overflow float32"):
        encoder.encode(value=1e30)
    with pytest.raises(ValueError, match="overflow float32"):
        encoder.encode_arrow(pa.array([1e30]))


def test_layout_metadata_names_require_qualified_column_selectors():
    with pytest.raises(ValueError, match="qualify the field"):
        Input(
            source=EventSource(events=("events",)),
            representation=Fields(fields=(Encode(field="valid", encoding=Raw()),)),
            layout=Padded(),
        )
    Input(
        source=EventSource(events=("events",)),
        representation=Fields(fields=(Encode(field="events/events/valid", encoding=Raw()),)),
        layout=Padded(),
    )


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"row": "tokens"}, "row requires"),
        ({"history": "flatten"}, "history requires"),
    ],
)
def test_invalid_arrangement_refuses_when_declared(changes, message):
    with pytest.raises(ValueError, match=message):
        Fields(fields=(Encode(field="amount", encoding=Raw()),), **changes)


def test_supplied_composite_mapping_has_unique_keys_and_matches_component_count():
    with pytest.raises(ValueError, match="keys must be unique"):
        Composite(vocabulary=(((("Vocabulary", 2),), 2), ((("Vocabulary", 2),), 3)))
    with pytest.raises(ValueError, match="selected field count"):
        Fields(
            fields=(Encode(field="amount", encoding=Raw()), Encode(field="count", encoding=Raw())),
            row=Composite(vocabulary=(((("int64", 1),), 2),)),
        )


def test_threshold_labels_refuse_nonfinite_cut_points():
    from kumo.pytorch.contract.batch import ThresholdEventLabel

    with pytest.raises(ValueError, match="cut point must be finite"):
        ThresholdEventLabel(
            name="risk",
            field="amount",
            dtype=DType("string"),
            encoding=Vocabulary(),
            threshold=float("inf"),
        )


@pytest.mark.parametrize("engine_name", ("in_memory", "diskgraph"))
@pytest.mark.parametrize(
    "dtype,values",
    [
        (DType("uint64"), [2**64 - 2, 2**64 - 1]),
        (
            DType("decimal256", precision=60, scale=2),
            [
                Decimal("123456789012345678901234567890.12"),
                Decimal("123456789012345678901234567890.13"),
            ],
        ),
    ],
)
def test_qualified_entity_numeric_outputs_preserve_engine_and_delivery_parity(
    engine_name, dtype, values
):
    if engine_name == "diskgraph":
        pytest.importorskip("diskgraph")
    from conformance.assertions import bound_tables
    from kumo.pytorch.api import EntitySource, Separate, prepare

    contract = DataLoaderSpec(
        schema=DataSchema(
            tables=(
                TableSpec(
                    name="users",
                    primary_key="id",
                    time_column=NON_TEMPORAL,
                    columns={"id": DType("int64"), "value": dtype},
                ),
            )
        ),
        sample=SampleSpec(
            entity=EntitySpec(table="users"), entity_values={"v": Field(column="value")}
        ),
        batch=BatchSpec(
            inputs={
                name: Input(
                    source=EntitySource(),
                    representation=Fields(fields=(Encode(field="entity/v", encoding=encoding),)),
                    layout=Separate(),
                )
                for name, encoding in {
                    "zscore": ZScore(mean=0.0, scale=2.0, dtype="float64"),
                    "buckets": Buckets(boundaries=(float(2**63),)),
                    "categories": Vocabulary(),
                }.items()
            }
        ),
    )
    anchors = [SampleRequest(sample_id=str(i), entity_id=i, as_of_time=0) for i in (1, 2)]
    with bound_tables(
        contract, {"users": [{"id": i, "value": v} for i, v in enumerate(values, 1)]}, engine_name
    ) as (dataset, engine):
        with prepare(contract, dataset, engine) as plan, plan.materialize(anchors) as task:
            outputs = [next(task.batches(anchors, batch_size=2))]
            if engine_name == "diskgraph":
                outputs.append(next(task.batches(anchors, batch_size=2, output_format="model")))
            for output in outputs:
                assert output.inputs["zscore"]["entity/v"]["values"].tolist() == [
                    float(v) / 2 for v in values
                ]
                assert output.inputs["buckets"]["entity/v"]["values"].tolist() == [2, 2]
                assert output.inputs["categories"]["entity/v"]["values"].tolist() == [2, 3]


@pytest.mark.parametrize("target_space", (None, "labels"))
def test_full_and_input_only_reuse_keep_independent_target_resources_separate(target_space):
    from kumo.pytorch.api import (
        Dataset,
        EntitySource,
        ExternalTarget,
        InMemoryEngine,
        InMemoryRows,
        LabelTarget,
        Separate,
        TargetValue,
        prepare,
    )

    contract = _contract()
    sample = replace(
        contract.sample,
        entity_values={"id": Field(column="id")},
        targets={"label": ExternalTarget(dtype=DType("string"))},
    )
    spaces = (
        {"labels": IdSpace(specials={"pad": "pad", "unknown": "unknown", "missing": "missing"})}
        if target_space
        else {}
    )
    batch = BatchSpec(
        inputs={
            "entity": Input(
                source=EntitySource(),
                representation=Fields(fields=(Encode(field="id", encoding=Vocabulary()),)),
                layout=Separate(),
            )
        },
        targets={
            "label": LabelTarget(source="label", encoding=Vocabulary(), id_space=target_space)
        },
        id_spaces=spaces,
    )
    contract = replace(contract, sample=sample, batch=batch)
    dataset = Dataset({"users": InMemoryRows([{"id": 1}, {"id": 2}]), "events": InMemoryRows([])})
    requests = [
        SampleRequest(
            sample_id=name,
            entity_id=index + 1,
            as_of_time=40,
            targets=(TargetValue(name="label", value=label),),
        )
        for index, (name, label) in enumerate((("a", "a"), ("b", "b")))
    ]
    with prepare(contract, dataset, InMemoryEngine()) as plan:
        with plan.materialize(requests) as task:
            saved = task.fitted_state
    with prepare(contract, dataset, InMemoryEngine()) as plan:
        with plan.materialize(requests, fitted=saved) as task:
            assert task.fitted_state.to_dict() == saved.to_dict()
    changed = replace(
        contract,
        batch=replace(
            batch,
            targets={
                "label": LabelTarget(
                    source="label", encoding=Vocabulary(max_size=1), id_space=target_space
                )
            },
        ),
    )
    assert changed.input_fingerprint == contract.input_fingerprint
    with prepare(changed, dataset, InMemoryEngine()) as plan:
        with plan.materialize(requests, fitted=saved.inputs) as task:
            result = next(task.batches(requests, batch_size=2))
            binding = task.target("label").field(name="value")
            assert result.targets["label"]["values"].tolist() == [
                binding.encode(value="a"),
                binding.encode(value="b"),
            ]
            assert binding.encoder.encode(value="b") == binding.encoder.roles["unknown"]
            assert (
                task.output("entity").field(name="id").encoder.entries
                == saved._input_state.prepared_inputs["entity"].state["id"].encoder.entries
            )
