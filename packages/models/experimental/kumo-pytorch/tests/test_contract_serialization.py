# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Named definitions are unordered; explicit sequences survive saving/loading."""

import hashlib
import itertools
import json
from dataclasses import replace
from unittest.mock import patch

import pytest

from kumo.pytorch import api
from tests.helpers import reorder_dicts


def _contract(scope="entity", values=None):
    if values is None:
        values = {
            "z_base": api.Constant(value=2.0, dtype=api.DType("float64")),
            "a_double": api.Derived("add", ("z_base", "z_base"), dtype=api.DType("float64")),
            "b_total": api.Derived("add", ("z_base", "a_double"), dtype=api.DType("float64")),
        }
    entity = api.EntitySpec("users", entity_type="user")
    if scope == "entity":
        sample = api.SampleSpec(entity, entity_values=values)
        source, layout = (api.EntitySource(), api.Separate())
        requests = (api.SampleRequest("one", 1, 10),)
    else:
        sample = api.SampleSpec(
            entity, candidate_sets={"items": api.CandidateSetSpec(entity, values=values)}
        )
        source, layout = (api.CandidateSource("items"), api.Padded())
        requests = (
            api.SampleRequest("one", 1, 10, candidates={"items": (api.CandidateRequest(2),)}),
        )
    contract = api.DataLoaderSpec(
        api.DataSchema(
            (
                api.TableSpec(
                    "users", "id", {"id": api.DType("int64")}, time_column=api.NON_TEMPORAL
                ),
            ),
            (),
        ),
        sample,
        api.BatchSpec(
            {
                "values": api.Input(
                    source,
                    api.Fields(
                        fields=tuple(
                            api.Encode(field=name, encoding=api.Raw()) for name in sorted(values)
                        ),
                        row=api.Columns(),
                    ),
                    layout,
                )
            }
        ),
    )
    return (contract, requests)


@pytest.mark.parametrize("scope", ("entity", "candidate"))
def test_all_declaration_orders_resolve_dependencies_and_reuse_state(scope):
    original, requests = _contract(scope)
    values = (
        original.sample.entity_values
        if scope == "entity"
        else original.sample.candidate_sets["items"].values
    )
    dataset = api.Dataset({"users": api.InMemoryRows([{"id": 1}, {"id": 2}])})
    fitted = None
    for names in itertools.permutations(values):
        contract, _ = _contract(scope, {name: values[name] for name in names})
        contract.validate()
        assert contract.fingerprint == original.fingerprint
        assert contract.input_fingerprint == original.input_fingerprint
        with api.prepare(contract, dataset, api.InMemoryEngine()) as plan:
            with plan.materialize(requests, fitted=fitted) as task:
                fields = task.preview(requests).batch.inputs["values"]
                for name, expected in {"z_base": 2, "a_double": 4, "b_total": 6}.items():
                    assert fields[name]["values"].flatten().tolist() == [expected]
                    assert fields[name]["valid"].all()
                fitted = task.fitted_state


@pytest.mark.parametrize("scope", ("entity", "candidate"))
@pytest.mark.parametrize("indent", (None, 2, 4))
def test_json_key_order_and_formatting_do_not_change_contract_identity(scope, indent):
    contract, _ = _contract(scope)
    payload = contract.to_dict()
    standardized = json.dumps(payload, sort_keys=True)
    expected_hash = "sha256:" + hashlib.sha256(standardized.encode()).hexdigest()
    assert contract.fingerprint == expected_hash
    for variant in (payload, reorder_dicts(payload), reorder_dicts(payload, shuffled=True)):
        restored = api.DataLoaderSpec.from_json(json.dumps(variant, indent=indent))
        restored.validate()
        assert restored.to_dict() == payload
        assert restored.fingerprint == contract.fingerprint
        assert restored.input_fingerprint == contract.input_fingerprint
        assert api.DataLoaderSpec.from_json(restored.to_json()).fingerprint == expected_hash


@pytest.mark.parametrize("scope", ("entity", "candidate"))
@pytest.mark.parametrize(
    "values,reason",
    [
        ({"a": api.Derived("log1p", ("missing",), dtype=api.DType("float64"))}, "unknown.*missing"),
        ({"a": api.Derived("log1p", ("a",), dtype=api.DType("float64"))}, "cycl"),
        (
            {
                "a": api.Derived("log1p", ("b",), dtype=api.DType("float64")),
                "b": api.Derived("log1p", ("a",), dtype=api.DType("float64")),
            },
            "cycl",
        ),
    ],
)
def test_invalid_dependencies_refuse_before_opening_the_engine(scope, values, reason):
    contract, _ = _contract(scope, values)
    engine = api.InMemoryEngine()
    dataset = api.Dataset({"users": api.InMemoryRows([])})
    issues = api.check(contract, dataset, engine)
    assert issues
    assert all(issue.suggestion for issue in issues)
    with patch.object(engine, "open", side_effect=AssertionError("opened invalid contract")):
        with pytest.raises(api.ValidationError, match=reason):
            api.prepare(contract, dataset, engine)


def test_operand_order_and_repetition_keep_their_meaning():
    values = {
        "result": api.Derived("subtract", ("twice", "base"), dtype=api.DType("float64")),
        "twice": api.Derived("add", ("base", "base"), dtype=api.DType("float64")),
        "base": api.Constant(value=2, dtype=api.DType("float64")),
    }
    original, requests = _contract(values=values)
    swapped, _ = _contract(
        values={
            **values,
            "result": api.Derived("subtract", ("base", "twice"), dtype=api.DType("float64")),
        }
    )
    assert original.fingerprint != swapped.fingerprint
    assert original.input_fingerprint != swapped.input_fingerprint
    dataset = api.Dataset({"users": api.InMemoryRows([{"id": 1}])})
    for contract, expected in ((original, 2), (swapped, -2)):
        result = api.preview(contract, dataset, api.InMemoryEngine(), requests).batch
        assert result.inputs["values"]["result"]["values"].tolist() == [expected]


def test_missing_data_and_undefined_arithmetic_remain_missing():
    contract, requests = _contract(
        values={
            "missing": api.Derived("add", ("base", "base"), dtype=api.DType("float64")),
            "zero_division": api.Derived("divide", ("zero", "zero"), dtype=api.DType("float64")),
            "bad_log": api.Derived("log1p", ("negative",), dtype=api.DType("float64")),
            "base": api.Field(column="value"),
            "zero": api.Constant(value=0, dtype=api.DType("float64")),
            "negative": api.Constant(value=-2, dtype=api.DType("float64")),
        }
    )
    contract = replace(
        contract,
        schema=api.DataSchema(
            (
                api.TableSpec(
                    "users",
                    "id",
                    {"id": api.DType("int64"), "value": api.DType("float64")},
                    time_column=api.NON_TEMPORAL,
                ),
            ),
            (),
        ),
    )
    dataset = api.Dataset({"users": api.InMemoryRows([{"id": 1, "value": None}])})
    fields = api.preview(contract, dataset, api.InMemoryEngine(), requests).batch.inputs["values"]
    for name in ("missing", "zero_division", "bad_log"):
        assert not fields[name]["valid"].any()


@pytest.mark.parametrize("separate_inputs", (False, True))
def test_shared_fitting_is_independent_of_named_input_and_field_placement(separate_inputs):
    values = {"z_large": 1e16, "a_negative": -1e16, "m_small": 1.0}
    contract, requests = _contract(
        values={
            name: api.Constant(value=value, dtype=api.DType("float64"))
            for name, value in values.items()
        }
    )
    fields = {name: api.UseEncoding("scale") for name in values}
    inputs = (
        {
            name: api.Input(
                api.EntitySource(),
                api.Fields(fields=(api.Encode(field=name, encoding=encoding),), row=api.Columns()),
                api.Separate(),
            )
            for name, encoding in fields.items()
        }
        if separate_inputs
        else {
            "values": api.Input(
                api.EntitySource(),
                api.Fields(
                    fields=tuple(
                        (
                            api.Encode(field=name, encoding=encoding)
                            for name, encoding in fields.items()
                        )
                    ),
                    row=api.Columns(),
                ),
                api.Separate(),
            )
        }
    )
    contract = replace(contract, batch=api.BatchSpec(inputs, encodings={"scale": api.ZScore()}))
    dataset = api.Dataset({"users": api.InMemoryRows([{"id": 1}])})
    fingerprints = []
    for variant in (contract, api.DataLoaderSpec.from_dict(reorder_dicts(contract.to_dict()))):
        with api.prepare(variant, dataset, api.InMemoryEngine()) as plan:
            with plan.materialize(requests) as task:
                fingerprints.append(task.bound_contract().fingerprint)
    assert fingerprints[0] == fingerprints[1]


def test_artifact_contract_preserves_executable_meaning(tmp_path):
    contract, requests = _contract()
    dataset = api.Dataset({"users": api.InMemoryRows([{"id": 1}])})
    with api.prepare(contract, dataset, api.InMemoryEngine()) as plan:
        with plan.materialize(requests) as task:
            task.to_artifact().save(tmp_path)
            fitted = task.fitted_state
    restored = api.DataLoaderSpec.from_json((tmp_path / "contract.json").read_text())
    restored.validate()
    assert restored.fingerprint == contract.fingerprint
    with api.prepare(restored, dataset, api.InMemoryEngine()) as plan:
        with plan.materialize(requests, fitted=fitted) as task:
            assert task.preview(requests).batch.inputs["values"]["b_total"]["values"].tolist() == [
                6
            ]
