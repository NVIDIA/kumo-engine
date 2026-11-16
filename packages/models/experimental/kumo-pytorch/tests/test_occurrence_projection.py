# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dated values and their explicit clocks follow exactly the retained pieces."""

from dataclasses import replace

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from kumo.pytorch.api import (
    NON_TEMPORAL,
    BatchSpec,
    Columns,
    DataLoaderSpec,
    DataSchema,
    Dataset,
    DatedEntityValue,
    DType,
    Encode,
    EntitySource,
    EntitySpec,
    EventTimeDelta,
    Field,
    Fields,
    IdSpace,
    InMemoryEngine,
    InMemoryRows,
    Input,
    OccurrencesOf,
    Packed,
    Padded,
    Raw,
    RelationSpec,
    Rows,
    SampleRequest,
    SampleSpec,
    TableSpec,
    Text,
    TimeWindow,
    Tokens,
    Vocabulary,
    prepare,
)


def _contract(order="fields", keep="first", maximum=None, layout=Padded()):
    integer, string = DType("int64"), DType("string")
    clock = DType("timestamp", unit="s", timezone="UTC")
    tokenizer = Tokenizer(
        models.BPE(
            vocab={"[MISSING]": 0, "[UNK]": 1, "a": 2, "b": 3},
            merges=[],
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    return DataLoaderSpec(
        schema=DataSchema(
            tables=(
                TableSpec(
                    "users", "id", {"id": integer, "country": string}, time_column=NON_TEMPORAL
                ),
                TableSpec(
                    "facts",
                    "id",
                    {"id": integer, "user": integer, "at": clock, "tag": string},
                    time_column="at",
                ),
            ),
            relations=(RelationSpec("facts", "users", "id", "facts", "user"),),
        ),
        sample=SampleSpec(
            entity=EntitySpec("users"),
            entity_values={"country": Field(column="country")},
            dated_entity_values={
                "tag": DatedEntityValue(
                    rows=Rows(path=("facts",), time=TimeWindow("at")),
                    field=Field(column="tag"),
                    first_per_value=True,
                    fields={"age": EventTimeDelta(unit="seconds", dtype=DType("float64"))},
                )
            },
        ),
        batch=BatchSpec(
            id_spaces={
                "model": IdSpace(
                    specials={"pad": "[PAD]", "missing": "[MISSING]", "unknown": "[UNK]"}
                )
            },
            inputs={
                "profile": Input(
                    source=EntitySource(),
                    representation=Fields(
                        fields=(
                            Encode(
                                field="tag",
                                encoding=Text(state=tokenizer.to_str()),
                                id_space="model",
                            ),
                            Encode(
                                field="country",
                                encoding=Vocabulary(values=(("US", 2),)),
                                id_space="model",
                            ),
                        ),
                        row=Tokens(max_tokens=maximum, keep=keep, occurrence_order=order),
                    ),
                    layout=layout,
                ),
                "ages": Input(
                    source=OccurrencesOf(input="profile"),
                    representation=Fields(
                        fields=(Encode(field="age", encoding=Raw()),),
                        row=Columns(),
                    ),
                    layout=layout,
                ),
            },
        ),
    )


@pytest.mark.parametrize("layout", [Padded(), Packed()])
@pytest.mark.parametrize(
    ("order", "keep", "maximum", "ages", "valid"),
    [
        ("fields", "first", None, [90.0, 90.0, 70.0, 70.0, 0.0], [True, True, True, True, False]),
        (
            "chronological",
            "first",
            None,
            [0.0, 90.0, 90.0, 70.0, 70.0],
            [False, True, True, True, True],
        ),
        ("chronological", "first", 3, [0.0, 90.0, 90.0], [False, True, True]),
        ("chronological", "last", 2, [70.0, 70.0], [True, True]),
    ],
)
def test_projection_repeats_explicit_age_per_piece_and_follows_truncation(
    layout, order, keep, maximum, ages, valid
):
    contract = _contract(order=order, keep=keep, maximum=maximum, layout=layout)
    dataset = Dataset(
        tables={
            "users": InMemoryRows(rows=[{"id": 1, "country": "US"}, {"id": 2, "country": "US"}]),
            "facts": InMemoryRows(
                rows=[
                    {"id": 1, "user": 1, "at": 10, "tag": "ab"},
                    {"id": 2, "user": 1, "at": 20, "tag": "ab"},
                    {"id": 3, "user": 1, "at": 30, "tag": "ba"},
                    {"id": 4, "user": 1, "at": 110, "tag": "ab"},
                ]
            ),
        }
    )
    requests = (SampleRequest("one", 1, 100), SampleRequest("empty", 2, 100))
    with prepare(contract=contract, dataset=dataset, engine=InMemoryEngine()) as plan:
        task = plan.materialize(requests)
        batch = next(task.batches(batch_size=2))
        values = batch.inputs["ages"]["age"]
        if isinstance(layout, Packed):
            assert batch.inputs["profile"]["offsets"].tolist() == [0, len(ages), len(ages) + 1]
            assert batch.inputs["ages"]["offsets"].tolist() == [0, len(ages), len(ages) + 1]
            assert values["values"].tolist() == ages + [0.0]
            assert values["valid"].tolist() == valid + [False]
        else:
            assert values["values"][0].tolist() == ages
            assert values["valid"][0].tolist() == valid
            assert (
                batch.inputs["ages"]["valid"].tolist() == batch.inputs["profile"]["valid"].tolist()
            )
        assert str(values["values"].dtype) == "torch.float64"
        assert not {"elapsed_seconds", "calendar", "timestamps"} & set(batch.inputs["profile"])
        assert DataLoaderSpec.from_json(contract.to_json()) == contract


def test_occurrence_sources_refuse_chains_missing_parents_and_learned_encoders():
    contract = _contract()
    with pytest.raises(ValueError, match="another declared input"):
        replace(
            contract.batch,
            inputs={
                **contract.batch.inputs,
                "ages": replace(
                    contract.batch.inputs["ages"],
                    source=OccurrencesOf(input="ages"),
                ),
            },
        )
    with pytest.raises(ValueError, match="Raw"):
        Input(
            source=OccurrencesOf(input="profile"),
            representation=Fields(fields=(Encode(field="age", encoding=Vocabulary()),)),
            layout=Padded(),
        )
