# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Callers choose the final IDs of an ID space's special roles (G-IDSPACE-001)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pytest
import torch

from kumo.pytorch import api
from kumo.pytorch.contract.batch.tokens import MAX_SPECIAL_ID
from kumo.pytorch.spec import _encode

LABELS: dict[str, str] = {
    "pad": "[PAD]",
    "mask": "[MASK]",
    "unknown": "[UNK]",
    "entity": "[USR]",
    "event": "[EVT]",
    "missing": "[MISSING]",
}
BLUEPRINT: dict[str, int] = {
    "pad": 0,
    "mask": 1,
    "unknown": 2,
    "entity": 3,
    "event": 4,
    "missing": 5,
}
GAPPED_LABELS: dict[str, str] = {
    "pad": "[PAD]",
    "unknown": "[UNK]",
    "missing": "[MISSING]",
    "mask": "[MASK]",
}
GAPPED: dict[str, int] = {"pad": 0, "unknown": 1, "missing": 2, "mask": 10}


def _contract(space: api.IdSpace) -> api.DataLoaderSpec:
    schema = api.DataSchema(
        (
            api.TableSpec(
                "users",
                "id",
                {"id": api.DType("int64"), "country": api.DType("string")},
                time_column=api.NON_TEMPORAL,
            ),
        ),
        (),
    )
    sample = api.SampleSpec(
        api.EntitySpec("users"), entity_values={"country": api.Field(column="country")}
    )

    def tokens(missing: str) -> api.Input:
        return api.Input(
            source=api.EntitySource(),
            representation=api.Fields(
                fields=(api.Encode(field="country", encoding=api.Vocabulary(), id_space="tokens"),),
                row=api.Tokens(missing=missing),
                field_names="fields",
                field_space="tokens",
            ),
            layout=api.Padded(),
        )

    return api.DataLoaderSpec(
        schema,
        sample,
        api.BatchSpec(
            inputs={"emitted": tokens("emit"), "omitted": tokens("omit")},
            id_spaces={"tokens": space},
            field_names={"fields": api.FieldNames()},
        ),
    )


ROWS: list[dict[str, object]] = [
    {"id": 1, "country": "US"},
    {"id": 2, "country": "DE"},
    {"id": 3, "country": None},
]


def _requests() -> list[api.SampleRequest]:
    return [
        api.SampleRequest(sample_id=f"u{index}", entity_id=index, as_of_time=100)
        for index in (1, 2, 3)
    ]


def _dataset(rows: list[dict[str, object]]) -> api.Dataset:
    return api.Dataset({"users": api.InMemoryRows(rows)})


def _blueprint() -> api.IdSpace:
    return api.IdSpace(specials=LABELS, special_ids=BLUEPRINT)


def _batch(contract: api.DataLoaderSpec, fitted: api.FittedState | None = None) -> api.Batch:
    with api.prepare(contract, _dataset(ROWS), api.InMemoryEngine()) as plan:
        with plan.materialize(_requests(), fitted=fitted) as task:
            return next(task.batches(batch_size=3))


def _state_space(contract: api.DataLoaderSpec, name: str = "tokens") -> Mapping[str, object]:
    with api.prepare(contract, _dataset(ROWS), api.InMemoryEngine()) as plan:
        with plan.materialize(_requests()) as task:
            return task.fitted_state.to_dict()["spaces"][name]


@pytest.mark.parametrize(
    ("special_ids", "message"),
    [
        ({"pad": 0, "mask": 1}, "missing"),
        ({**BLUEPRINT, "extra": 9}, "extra"),
        ({**BLUEPRINT, "mask": 0}, "cannot share special ID 0"),
        ({**BLUEPRINT, "mask": True}, "integer"),
        ({**BLUEPRINT, "mask": 1.0}, "integer"),
        ({**BLUEPRINT, "mask": -1}, "integer"),
        ({**BLUEPRINT, "mask": MAX_SPECIAL_ID + 1}, str(MAX_SPECIAL_ID)),
    ],
)
def test_invalid_special_ids_are_refused_at_declaration(
    special_ids: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        api.IdSpace(specials=LABELS, special_ids=special_ids)  # type: ignore[arg-type]


def test_the_largest_special_id_is_accepted() -> None:
    space = api.IdSpace(specials={"pad": "[PAD]"}, special_ids={"pad": MAX_SPECIAL_ID})
    assert space.special_ids == {"pad": MAX_SPECIAL_ID}


def test_omitted_special_ids_keep_the_default_numbering() -> None:
    contract = _contract(api.IdSpace(specials=GAPPED_LABELS))
    assert _state_space(contract)["roles"] == {"pad": 0, "unknown": 1, "missing": 2, "mask": 3}
    assert "special_ids" not in json.dumps(_encode(contract))


def test_declared_special_ids_change_the_fingerprint() -> None:
    plain = _contract(api.IdSpace(specials=LABELS))
    chosen = _contract(_blueprint())
    assert plain.fingerprint != chosen.fingerprint
    assert _encode(chosen)["batch"]["id_spaces"]["tokens"]["special_ids"] == BLUEPRINT


def test_blueprint_layout_numbers_roles_then_ordinary_entries() -> None:
    space = _state_space(_contract(_blueprint()))
    assert space["roles"] == BLUEPRINT
    members = space["members"]
    assert isinstance(members, dict)
    finals = sorted(final for pairs in members.values() for _, final in pairs)
    ordinary = [final for final in finals if final not in BLUEPRINT.values()]
    assert ordinary and min(ordinary) == 6
    assert space["size"] == 6 + len(set(ordinary))


def test_gapped_layout_reserves_ids_between_roles() -> None:
    space = _state_space(_contract(api.IdSpace(specials=GAPPED_LABELS, special_ids=GAPPED)))
    assert space["roles"] == GAPPED
    members = space["members"]
    assert isinstance(members, dict)
    finals = {final for pairs in members.values() for _, final in pairs}
    ordinary = sorted(finals - set(GAPPED.values()))
    assert ordinary[0] == 11
    assert ordinary == list(range(11, 11 + len(ordinary)))
    assert space["size"] == 11 + len(ordinary)


def test_batches_use_the_explicit_ids() -> None:
    batch = _batch(_contract(_blueprint()))
    emitted = batch.inputs["emitted"]
    omitted = batch.inputs["omitted"]
    assert emitted["value_ids"][2].tolist() == [BLUEPRINT["missing"]]
    assert omitted["value_ids"][2].tolist() == [BLUEPRINT["pad"]]
    assert omitted["valid"][2].tolist() == [False]
    assert emitted["field_ids"].min() >= 6
    assert (omitted["field_ids"][2] == BLUEPRINT["pad"]).all()
    assert emitted["value_ids"][:2].min() >= 6


def test_field_ids_pad_with_the_declared_pad_role() -> None:
    space = api.IdSpace(specials=LABELS, special_ids={**BLUEPRINT, "pad": 5, "missing": 0})
    batch = _batch(_contract(space))
    omitted = batch.inputs["omitted"]
    assert omitted["value_ids"][2].tolist() == [5]
    assert omitted["field_ids"][2].tolist() == [5]
    assert batch.inputs["emitted"]["value_ids"][2].tolist() == [0]


def test_save_load_and_inference_keep_the_explicit_ids(tmp_path: Path) -> None:
    contract = _contract(_blueprint())
    with api.prepare(contract, _dataset(ROWS), api.InMemoryEngine()) as plan:
        with plan.materialize(_requests()) as task:
            task.to_artifact().save(tmp_path)
            trained = next(task.batches(batch_size=3))
    artifact = api.Artifact.load(tmp_path)
    assert artifact.contract.batch.id_spaces["tokens"].special_ids == BLUEPRINT
    assert artifact.fitted_state.to_dict()["spaces"]["tokens"]["roles"] == BLUEPRINT

    unseen = [{"id": 1, "country": "FR"}, {"id": 2, "country": "US"}, {"id": 3, "country": None}]
    with api.prepare(artifact.contract, _dataset(unseen), api.InMemoryEngine()) as plan:
        with plan.materialize(_requests(), fitted=artifact.fitted_state, inference=True) as task:
            batch = next(task.batches(batch_size=3))
    emitted = batch.inputs["emitted"]["value_ids"]
    assert emitted[0].tolist() == [BLUEPRINT["unknown"]]
    assert emitted[2].tolist() == [BLUEPRINT["missing"]]
    assert batch.inputs["omitted"]["value_ids"][2].tolist() == [BLUEPRINT["pad"]]
    assert torch.equal(emitted[1], trained.inputs["emitted"]["value_ids"][0])


def test_masked_values_use_the_explicit_mask_id() -> None:
    contract = _contract(_blueprint())
    contract = api.DataLoaderSpec(
        contract.schema,
        contract.sample,
        api.BatchSpec(
            inputs=contract.batch.inputs,
            targets={
                "masked": api.MaskedValuesTarget(
                    input="emitted",
                    token_probability=1.0,
                    event_probability=0.0,
                    key_probability=0.0,
                )
            },
            id_spaces=contract.batch.id_spaces,
            field_names=contract.batch.field_names,
        ),
    )
    batch = _batch(contract)
    ids = batch.inputs["emitted"]["value_ids"]
    assert (ids[:2] == BLUEPRINT["mask"]).all()
