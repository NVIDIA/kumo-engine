# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prove the runner rejects wrong answers, wrong refusals, and late reads."""

from contextlib import contextmanager
from dataclasses import replace
from unittest.mock import patch

import pytest

from conformance.assertions import (
    Refusal,
    assert_answer,
    assert_refusal,
    bound_case,
    require_engine,
)
from conformance.cases import discover_cases
from kumo.pytorch.api import ValidationIssue


def test_wrong_expected_answer_does_not_pass():
    case = discover_cases()["C-ROWS-001"]
    wrong = {sample_id: {"entity_values": {"event_count": 99}} for sample_id in case.expected}
    with pytest.raises(AssertionError, match="expected.*got"):
        assert_answer(replace(case, expected=wrong), "in_memory")


@pytest.mark.parametrize(
    "damage", ("accepted", "wrong_reason", "no_suggestion", "extra_issue", "read")
)
def test_refusal_proof_is_specific_and_pre_read(damage):
    case = discover_cases()["C-ROWS-001"]
    expected = Refusal("sample", "not implemented")

    @contextmanager
    def binding(*args, **kwargs):
        with bound_case(case, "in_memory", refusal=True) as (dataset, engine):

            def preflight(contract):
                if damage == "read":
                    list(dataset.tables["events"].rows)
                if damage == "accepted":
                    return ()
                message = "unrelated error" if damage == "wrong_reason" else "not implemented"
                suggestion = None if damage == "no_suggestion" else "use another engine"
                issues = (ValidationIssue("sample", message, suggestion),)
                if damage == "extra_issue":
                    issues += (ValidationIssue("dataset", "missing", "bind data"),)
                return issues

            engine.check_contract = preflight
            yield dataset, engine

    with patch("conformance.assertions.bound_case", binding), pytest.raises(AssertionError):
        assert_refusal(case, "in_memory", expected)


def test_missing_native_dependency_is_a_failure():
    with patch.dict("sys.modules", {"diskgraph": None}), pytest.raises(ModuleNotFoundError):
        require_engine("diskgraph")


@pytest.mark.parametrize("path,message", (("", "reason"), ("sample", " ")))
def test_refusal_expectation_cannot_match_every_error(path, message):
    with pytest.raises(ValueError, match="specific reason"):
        Refusal(path, message)
