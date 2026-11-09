# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every golden case has an explicit, reviewed obligation for each engine.

This is ordinary test data: it does not drive kumo-pytorch's runtime support decisions.
The golden answer stays independent of whether an engine currently implements it.
"""

import json
from dataclasses import replace

import pytest

from conformance.assertions import Refusal, assert_answer, assert_refusal, require_engine
from conformance.cases import ENGINES, discover_cases
from conformance.expectations import load_expectations
from kumo.pytorch.api import DataLoaderSpec
from tests.helpers import reorder_dicts

# Updating these assertions never changes the engines' runtime support.
EXPECTATIONS = load_expectations()


def test_catalog():
    cases = discover_cases()
    assert cases.keys() == EXPECTATIONS.keys(), "every case must have executable engine assertions"
    for case_id, decisions in EXPECTATIONS.items():
        assert set(decisions) == set(ENGINES), f"{case_id}: name every required engine"


def pytest_generate_tests(metafunc):
    if "case_id" not in metafunc.fixturenames:
        return
    selected = metafunc.config.getoption("--conformance-engine")
    engines = ENGINES if selected == "all" else (selected,)
    parameters = []
    for case_id, decisions in EXPECTATIONS.items():
        for engine in engines:
            decision = decisions[engine]
            outcome = "answer" if decision == "answer" else "refusal"
            parameters.append(pytest.param(case_id, engine, id=f"{case_id}-{engine}-{outcome}"))
    metafunc.parametrize("case_id,engine_name", parameters)


def test_public_behavior(case_id, engine_name):
    require_engine(engine_name)
    case = discover_cases()[case_id]
    expected = EXPECTATIONS[case_id][engine_name]
    payload = case.contract.to_dict()
    for variant in (payload, reorder_dicts(payload), reorder_dicts(payload, shuffled=True)):
        contract = DataLoaderSpec.from_json(json.dumps(variant))
        contract.validate()
        restored = DataLoaderSpec.from_json(contract.to_json())
        restored.validate()
        assert restored.to_dict() == payload
        assert restored.fingerprint == contract.fingerprint
        assert restored.input_fingerprint == contract.input_fingerprint
        assert contract.to_dict() == payload
        assert contract.fingerprint == case.contract.fingerprint
        assert contract.input_fingerprint == case.contract.input_fingerprint
        loaded_case = replace(case, contract=contract)
        if expected == "answer":
            assert_answer(loaded_case, engine_name)
        else:
            assert_refusal(loaded_case, engine_name, Refusal(**expected))
