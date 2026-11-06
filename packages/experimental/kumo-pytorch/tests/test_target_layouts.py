# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Supervision follows retained source positions without re-fitting frozen inputs."""

from dataclasses import replace

import pytest
import torch

from conformance.assertions import bound_tables
from kumo.pytorch.api import (
    AlignedEventTarget,
    BatchSpec,
    CandidateRequest,
    CandidateSetSpec,
    CandidateSource,
    CandidateTarget,
    DType,
    Encode,
    EntitySource,
    EntitySpec,
    ExternalTarget,
    Field,
    Fields,
    Flatten,
    IdSpace,
    Input,
    LabelTarget,
    NextTokenTarget,
    Packed,
    Padded,
    Raw,
    SampleRequest,
    SampleSpec,
    Separate,
    TargetValue,
    Text,
    Vocabulary,
    prepare,
)
from tests.test_field_outputs import _contract, _tables


def _anchors():
    return [
        SampleRequest(sample_id="a", entity_id=1, as_of_time=40),
        SampleRequest(sample_id="empty", entity_id=2, as_of_time=40),
    ]


@pytest.mark.parametrize("engine", ("in_memory", "diskgraph"))
@pytest.mark.parametrize("packed", (False, True))
@pytest.mark.parametrize("markers", (False, True))
def test_masked_flattened_answers_follow_clipping_markers_and_empty_samples(
    engine, packed, markers
):
    if engine == "diskgraph":
        pytest.importorskip("diskgraph")
    contract = _contract(flat=True, masking=True)
    history = contract.batch.inputs["history"]
    flatten = Flatten(
        max_tokens=6 if markers else 3,
        keep="last",
        before="bos" if markers else None,
        after="eos" if markers else None,
        separator="separator" if markers else None,
        separator_placement="after_each",
    )
    roles = {"pad": "pad", "unknown": "unknown", "missing": "missing", "mask": "mask"}
    if markers:
        roles.update(separator="separator", bos="bos", eos="eos")
    contract = replace(
        contract,
        batch=replace(
            contract.batch,
            inputs={
                "history": replace(
                    history,
                    representation=replace(history.representation, history=flatten),
                    layout=Packed() if packed else Padded(),
                )
            },
            encodings={"currencies": Vocabulary(values=(("USD", 2), ("EUR", 3)))},
            id_spaces={"tokens": IdSpace(specials=roles)},
        ),
    )
    expected_ids = [5, 3, 3, 4, 6] if markers else [3, 3]
    expected_answers = [-100, 9, 11, -100, -100] if markers else [6, 8]
    expected_valid = [False, True, True, False, False] if markers else [True, True]
    empty_ids = [5, 6] if markers else []
    anchors = _anchors()
    with bound_tables(contract, _tables(), engine) as (dataset, backend):
        with prepare(contract, dataset, backend) as plan, plan.materialize(anchors) as task:
            for delivery in ("inspectable", "model") if engine == "diskgraph" else ("inspectable",):
                batch = next(task.batches(anchors, batch_size=2, output_format=delivery))
                source, target = batch.inputs["history"], batch.targets["masked"]
                if packed:
                    assert source["value_ids"].tolist() == expected_ids + empty_ids
                    assert target["values"].tolist() == expected_answers + [-100] * len(empty_ids)
                    assert target["valid"].tolist() == expected_valid + [False] * len(empty_ids)
                    assert target["corruption"].tolist() == target["valid"].tolist()
                    assert torch.equal(target["offsets"], source["offsets"])
                else:
                    width = len(expected_ids)
                    assert source["value_ids"].tolist() == [
                        expected_ids,
                        empty_ids + [0] * (width - len(empty_ids)),
                    ]
                    assert target["values"].tolist() == [expected_answers, [-100] * width]
                    assert target["valid"].tolist() == [expected_valid, [False] * width]
                    assert target["corruption"].tolist() == target["valid"].tolist()


def test_one_input_cannot_mix_masked_and_next_token_objectives():
    contract = _contract(flat=True, masking=True)
    with pytest.raises(ValueError, match="only one masked or next-token"):
        replace(
            contract.batch,
            targets={**contract.batch.targets, "next": NextTokenTarget(input="history")},
        )


@pytest.mark.parametrize("packed", (False, True))
def test_ensure_masked_supervision_uses_retained_positions_after_history_clipping(packed):
    contract = _contract(flat=True, masking=True)
    history = contract.batch.inputs["history"]
    contract = replace(
        contract,
        batch=replace(
            contract.batch,
            inputs={
                "history": replace(
                    history,
                    representation=replace(history.representation, history=Flatten(max_tokens=3)),
                    layout=Packed() if packed else Padded(),
                )
            },
            targets={
                "masked": replace(
                    contract.batch.targets["masked"],
                    token_probability=0.0,
                    ensure_at_least_one=True,
                )
            },
        ),
    )
    with bound_tables(contract, _tables(), "in_memory") as (dataset, backend):
        with prepare(contract, dataset, backend) as plan, plan.materialize(_anchors()) as task:
            batch = next(task.batches(_anchors(), batch_size=2))
            assert batch.targets["masked"]["valid"].sum().item() == 1
            assert batch.targets["masked"]["corruption"].sum().item() == 1


@pytest.mark.parametrize("engine", ("in_memory", "diskgraph"))
def test_input_only_reuse_resolves_new_aligned_label_fields_without_refitting(engine):
    if engine == "diskgraph":
        pytest.importorskip("diskgraph")
    contract = _contract()
    history = contract.batch.inputs["history"]
    contract = replace(
        contract,
        batch=BatchSpec(
            inputs={
                "history": replace(
                    history,
                    representation=Fields(fields=(Encode(field="amount", encoding=Raw()),)),
                    layout=Padded(),
                )
            }
        ),
    )
    anchors = _anchors()
    with bound_tables(contract, _tables(), engine) as (dataset, backend):
        with prepare(contract, dataset, backend) as plan, plan.materialize(anchors) as task:
            frozen = task.fitted_state.inputs
        changed = replace(
            contract,
            batch=replace(
                contract.batch,
                targets={
                    "heldout": AlignedEventTarget(
                        input="history", fields=(Encode(field="currency", encoding=Vocabulary()),)
                    ),
                },
            ),
        )
        assert changed.input_fingerprint == contract.input_fingerprint
        with (
            prepare(changed, dataset, backend) as plan,
            plan.materialize(anchors, fitted=frozen) as task,
        ):
            assert (
                task.output("history").field(name="amount").encoder
                is frozen._input_state.prepared_inputs["history"].state["amount"].encoder
            )
            for delivery in ("inspectable", "model") if engine == "diskgraph" else ("inspectable",):
                target = next(task.batches(anchors, batch_size=2, output_format=delivery)).targets[
                    "heldout"
                ]["currency"]
                assert target["values"].tolist() == [[3, 0, 2], [0, 0, 0]]
                assert target["valid"].tolist() == [[True, False, True], [False, False, False]]


def _text():
    from tokenizers import Tokenizer, models, pre_tokenizers

    tokenizer = Tokenizer(
        models.BPE(vocab={"[MISSING]": 0, "[UNK]": 1, "a": 2, "b": 3}, merges=[], unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    return Text(state=tokenizer.to_str())


@pytest.mark.parametrize("engine", ("in_memory", "diskgraph"))
@pytest.mark.parametrize("packed", (False, True))
def test_aligned_text_targets_keep_piece_and_sample_boundaries(engine, packed):
    if engine == "diskgraph":
        pytest.importorskip("diskgraph")
    contract = _contract()
    history = contract.batch.inputs["history"]
    contract = replace(
        contract,
        batch=BatchSpec(
            inputs={
                "history": replace(
                    history,
                    representation=Fields(fields=(Encode(field="amount", encoding=Raw()),)),
                    layout=Packed() if packed else Padded(),
                )
            },
            targets={
                "heldout": AlignedEventTarget(
                    input="history", fields=(Encode(field="currency", encoding=_text()),)
                )
            },
        ),
    )
    tables = _tables()
    for row, text in zip(tables["events"], ("a b", None, ""), strict=True):
        row["currency"] = text
    anchors = _anchors()
    with bound_tables(contract, tables, engine) as (dataset, backend):
        with prepare(contract, dataset, backend) as plan, plan.materialize(anchors) as task:
            for delivery in ("inspectable", "model") if engine == "diskgraph" else ("inspectable",):
                target = next(task.batches(anchors, batch_size=2, output_format=delivery)).targets[
                    "heldout"
                ]["currency"]
                assert target["values"].tolist() == [2, 3, 0]
                assert target["piece_offsets"].tolist() == [0, 2, 3, 3]
                assert target["valid"].tolist() == [True, False, True]
                assert target["offsets"].tolist() == [0, 3, 3]


@pytest.mark.parametrize("packed", (False, True))
def test_candidate_text_targets_keep_piece_and_sample_boundaries(packed):
    contract = _contract()
    contract = replace(
        contract,
        sample=SampleSpec(
            entity=EntitySpec(table="users"),
            candidate_sets={
                "choices": CandidateSetSpec(
                    entity=EntitySpec(table="users"),
                    values={"id": Field(column="id")},
                    targets={"description": ExternalTarget(dtype=DType("string"))},
                )
            },
        ),
        batch=BatchSpec(
            inputs={
                "choices": Input(
                    source=CandidateSource(name="choices"),
                    representation=Fields(fields=(Encode(field="id", encoding=Raw()),)),
                    layout=Packed() if packed else Padded(),
                )
            },
            targets={
                "description": CandidateTarget(
                    input="choices", source="description", encoding=_text()
                ),
            },
        ),
    )
    anchors = [
        replace(
            _anchors()[0],
            candidates={
                "choices": tuple(
                    CandidateRequest(
                        entity_id=index, targets=(TargetValue(name="description", value=value),)
                    )
                    for index, value in enumerate(("a b", None, ""), 1)
                )
            },
        ),
        _anchors()[1],
    ]
    tables = _tables()
    tables["users"].append({"id": 3})
    with bound_tables(contract, tables, "in_memory") as (dataset, backend):
        with prepare(contract, dataset, backend) as plan, plan.materialize(anchors) as task:
            batch = next(task.batches(anchors, batch_size=2))
            target = batch.targets["description"]
            assert target["values"].tolist() == [2, 3, 0]
            assert target["piece_offsets"].tolist() == [0, 2, 3, 3]
            assert target["valid"].tolist() == [True, False, True]
            assert target["offsets"].tolist() == [0, 3, 3]
            assert torch.equal(target["candidate_valid"], batch.inputs["choices"]["valid"])


@pytest.mark.parametrize("kind", (Vocabulary, Text))
@pytest.mark.parametrize("role", ("missing_id", "unknown_id"))
def test_local_special_ids_must_fit_int64(kind, role):
    with pytest.raises(ValueError, match="IDs must be distinct"):
        kind(**{role: 2**63})


def test_bound_contract_pins_target_members_of_input_used_spaces_only():
    contract = _contract()
    contract = replace(
        contract,
        sample=replace(
            contract.sample,
            entity_values={"id": Field(column="id")},
            targets={"label": ExternalTarget(dtype=DType("string"))},
        ),
        batch=BatchSpec(
            inputs={
                "entity": Input(
                    source=EntitySource(),
                    representation=Fields(
                        fields=(Encode(field="id", encoding=Vocabulary(), id_space="model"),)
                    ),
                    layout=Separate(),
                )
            },
            targets={"label": LabelTarget(source="label", encoding=Vocabulary(), id_space="model")},
            id_spaces={
                "model": IdSpace(
                    specials={"pad": "pad", "missing": "missing", "unknown": "unknown"}
                )
            },
        ),
    )
    anchors = [
        replace(anchor, targets=(TargetValue(name="label", value=label),))
        for anchor, label in zip(_anchors(), ("a", "b"), strict=True)
    ]
    with bound_tables(contract, _tables(), "in_memory") as (dataset, backend):
        with prepare(contract, dataset, backend) as plan, plan.materialize(anchors) as task:
            bound = task.bound_contract()
            frozen_space = task.spaces["model"].state_dict()
            frozen_ids = next(task.batches(anchors, batch_size=2)).inputs["entity"]["id"]["values"]
        assert bound.batch.targets["label"].encoding.values == (("a", 2), ("b", 3))
        changed = [
            replace(anchor, targets=(TargetValue(name="label", value="c"),)) for anchor in anchors
        ]
        with prepare(bound, dataset, backend) as plan, plan.materialize(changed) as task:
            assert task.spaces["model"].state_dict() == frozen_space
            result = next(task.batches(changed, batch_size=2))
            assert torch.equal(result.inputs["entity"]["id"]["values"], frozen_ids)
            assert result.targets["label"]["values"].tolist() == [1, 1]


def test_standalone_typed_records_respect_qualified_event_selector():
    from kumo.pytorch.api import EventSource, prepare_batch_spec
    from kumo.pytorch.data.records import EntityHistory, EntityState, Event, FieldValue

    contract = _contract()
    definition = contract.sample.events["payments"]
    sample = replace(contract.sample, events={"a": definition, "b": definition})
    batch = BatchSpec(
        inputs={
            "history": Input(
                source=EventSource(events=("a", "b")),
                representation=Fields(fields=(Encode(field="events/a/amount", encoding=Raw()),)),
                layout=Padded(),
            )
        }
    )
    values = [
        EntityHistory(
            1,
            40,
            EntityState((), ()),
            events=(
                Event("a", 10, (FieldValue("amount", 1.0, DType("float64")),)),
                Event("b", 20, (FieldValue("amount", 2.0, DType("float64")),)),
            ),
        )
    ]
    plan = prepare_batch_spec(sample=sample, spec=batch, fitting_values=values)
    result = next(plan.batches(values, batch_size=1)).inputs["history"]["events/a/amount"]
    assert result["values"].tolist() == [[1.0, 0.0]]
    assert result["valid"].tolist() == [[True, False]]
