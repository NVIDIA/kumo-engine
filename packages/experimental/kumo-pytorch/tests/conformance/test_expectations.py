# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Support assertions must stay explicit and independent of runtime output."""

import json
from copy import deepcopy

import pytest

from conformance.expectations import load_expectations, parse_expectations


@pytest.mark.parametrize(
    "damage",
    ("empty", "unknown_engine", "missing_engine", "wrong_state", "blank_reason", "extra_field"),
)
def test_invalid_expectations_are_rejected(damage):
    payload = deepcopy(load_expectations())
    engines = payload["C-ROWS-002"]
    if damage == "empty":
        payload = {}
    elif damage == "unknown_engine":
        engines["other"] = "answer"
    elif damage == "missing_engine":
        engines.pop("diskgraph")
    elif damage == "wrong_state":
        engines["diskgraph"] = "skip"
    elif damage == "blank_reason":
        engines["diskgraph"]["message_contains"] = " "
    else:
        engines["diskgraph"]["regression"] = "#123"
    with pytest.raises(ValueError):
        parse_expectations(json.dumps(payload))


def test_duplicate_expectation_keys_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        parse_expectations('{"C-ROWS-001": {}, "C-ROWS-001": {}}')
