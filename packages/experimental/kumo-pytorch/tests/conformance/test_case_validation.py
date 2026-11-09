# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A malformed or unasserted example must not quietly count as coverage."""

import json
import shutil
from pathlib import Path

import pytest

from conformance.cases import GUARANTEES, discover_cases, load_case
from kumo.pytorch.api import DataLoaderSpec

CASE = GUARANTEES / "G-ROWS-001/golden/C-ROWS-001.json"
CONTRACT_CASE = GUARANTEES / "G-CONTRACT-001/golden/C-CONTRACT-001.json"


@pytest.mark.parametrize(
    "mutation,reason",
    [
        (lambda p: p.pop("reason"), "fields"),
        (lambda p: p.update(reason=" "), "missing reason"),
        (lambda p: p.update(id="C-WRONG"), "filename"),
        (lambda p: p.update(engines={"diskgraph": "refused"}), "fields"),
        (lambda p: p["tables"].pop("events"), "tables must match"),
        (lambda p: p["tables"]["events"][0].pop("at"), "declared columns"),
        (lambda p: p["tables"]["events"][0].update(at=None), "clock needs integer"),
        (lambda p: p["tables"]["events"].append(p["tables"]["events"][0]), "duplicate primary"),
        (lambda p: p.update(anchors=[]), "1–8 anchors"),
        (lambda p: p["anchors"].append(p["anchors"][0]), "sample IDs"),
        (lambda p: p["anchors"][0].update(as_of_time=True), "integer epoch"),
        (lambda p: p["anchors"][0].update(entity_id=999), "entity must exist"),
        (lambda p: p["expected"].pop("user-1-at-9"), "answer every anchor"),
        (
            lambda p: p["expected"]["user-1-at-9"].update(tensor_values={}),
            "unknown expected-output",
        ),
        (
            lambda p: p["expected"]["user-1-at-9"]["entity_values"].clear(),
            "every declared entity value",
        ),
        (
            lambda p: p["expected"]["user-1-at-9"]["entity_values"].update(
                event_count=float("nan")
            ),
            "finite numbers",
        ),
        (lambda p: p["contract"]["batch"]["batching"].pop("drop_last"), "round-trip"),
    ],
)
def test_invalid_record_fails_before_engine_execution(tmp_path, mutation, reason):
    path = tmp_path / "G-ROWS-001/golden/C-ROWS-001.json"
    path.parent.mkdir(parents=True)
    payload = json.loads(CASE.read_text())
    mutation(payload)
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=reason):
        load_case(path)


def test_duplicate_json_keys_are_rejected(tmp_path):
    path = tmp_path / "G-ROWS-001/golden/C-ROWS-001.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        CASE.read_text().replace('"id": "C-ROWS-001",', '"id": "C-ROWS-001", "id": "x",')
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_case(path)


def test_engine_case_contains_the_actual_json_loaded_contract(monkeypatch):
    from_json = DataLoaderSpec.from_json
    loaded = []

    def record_load(cls, value):
        contract = from_json(value)
        loaded.append(contract)
        return contract

    monkeypatch.setattr(DataLoaderSpec, "from_json", classmethod(record_load))
    case = load_case(CONTRACT_CASE)
    assert len(loaded) == 1
    assert case.contract is loaded[0]


@pytest.mark.parametrize("side", ("writer", "reader"))
@pytest.mark.parametrize("damage", ("sequence_order", "value", "missing_field"))
def test_json_round_trip_rejects_a_broken_writer_or_reader(monkeypatch, side, damage):
    from_json = DataLoaderSpec.from_json

    def damaged(payload):
        values = payload["sample"]["entity_values"]
        fields = payload["batch"]["inputs"]["entity"]["representation"]["fields"]
        if damage == "sequence_order":
            fields.reverse()
        elif damage == "value":
            values["z_first"]["column"] = "second"
        else:
            values.pop("a_second")
            fields[:] = [value for value in fields if value["field"] != "a_second"]
        return payload

    if side == "writer":
        monkeypatch.setattr(
            DataLoaderSpec,
            "to_json",
            lambda self, *, indent=2: json.dumps(damaged(self.to_dict()), indent=indent),
        )
    else:
        monkeypatch.setattr(
            DataLoaderSpec,
            "from_json",
            classmethod(lambda cls, value: cls.from_dict(damaged(from_json(value).to_dict()))),
        )
    with pytest.raises(ValueError, match="preserve values and explicit sequences"):
        load_case(CONTRACT_CASE)


@pytest.mark.parametrize("side", ("writer", "reader"))
def test_json_round_trip_accepts_dictionary_rearrangement(monkeypatch, side):
    from tests.helpers import reorder_dicts

    from_json = DataLoaderSpec.from_json
    if side == "writer":
        monkeypatch.setattr(
            DataLoaderSpec,
            "to_json",
            lambda self, *, indent=2: json.dumps(reorder_dicts(self.to_dict()), indent=indent),
        )
    else:
        monkeypatch.setattr(
            DataLoaderSpec,
            "from_json",
            classmethod(
                lambda cls, value: cls.from_dict(reorder_dicts(from_json(value).to_dict()))
            ),
        )
    assert (
        load_case(CONTRACT_CASE).contract.to_dict()
        == json.loads(CONTRACT_CASE.read_text())["contract"]
    )


def test_engine_assertions_receive_differently_ordered_json_contracts(monkeypatch):
    from tests.conformance import test_guarantees

    contracts = []
    monkeypatch.setattr(
        test_guarantees, "assert_answer", lambda case, _engine: contracts.append(case.contract)
    )
    test_guarantees.test_public_behavior("C-CONTRACT-001", "in_memory")
    assert len(contracts) == 3
    assert {tuple(contract.sample.entity_values) for contract in contracts} == {
        ("a_second", "z_first"),
        ("z_first", "a_second"),
    }
    assert len({contract.fingerprint for contract in contracts}) == 1


@pytest.mark.parametrize("attribute", ("fingerprint", "input_fingerprint"))
def test_json_round_trip_rejects_an_unstable_fingerprint(monkeypatch, attribute):
    monkeypatch.setattr(DataLoaderSpec, attribute, property(lambda self: str(id(self))))
    with pytest.raises(ValueError, match="preserve the .*fingerprint"):
        load_case(CONTRACT_CASE)


@pytest.mark.parametrize("damage", ("document", "link", "empty", "orphan", "duplicate"))
def test_broken_guarantee_layout_is_rejected(tmp_path: Path, damage):
    root = tmp_path / "guarantees"
    shutil.copytree(GUARANTEES, root)
    directory = root / "G-ROWS-001"
    document = directory / "guarantee.md"
    if damage == "document":
        document.unlink()
    elif damage == "link":
        document.write_text(
            document.read_text().replace("golden/C-ROWS-001.json", "golden/C-MISSING.json")
        )
    elif damage == "empty":
        for path in (directory / "golden").glob("*.json"):
            path.unlink()
    elif damage == "orphan":
        (directory / "orphan.json").write_text(CASE.read_text())
    else:
        other = root / "G-ROWS-002"
        shutil.copytree(directory, other)
        note = other / "guarantee.md"
        note.write_text(note.read_text().replace("# G-ROWS-001:", "# G-ROWS-002:"))
    with pytest.raises(ValueError):
        discover_cases(root)
