# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read independent engine assertions as data, never as runtime support declarations."""

import json
import re
from pathlib import Path

ENGINES = ("in_memory", "diskgraph")
EXPECTATIONS = Path(__file__).with_name("expectations.json")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate expectation key {key!r}")
        result[key] = value
    return result


def parse_expectations(text: str) -> dict:
    expectations = json.loads(text, object_pairs_hook=_unique_object)
    if not isinstance(expectations, dict) or not expectations:
        raise ValueError("expectations must be a nonempty object")
    for case_id, decisions in expectations.items():
        if not re.fullmatch(r"C-[A-Z0-9-]+", case_id):
            raise ValueError(f"invalid expectation case ID {case_id!r}")
        if not isinstance(decisions, dict) or set(decisions) != set(ENGINES):
            raise ValueError(f"{case_id}: name every engine exactly once")
        for decision in decisions.values():
            if decision == "answer":
                continue
            if (
                not isinstance(decision, dict)
                or set(decision) != {"path", "message_contains"}
                or not all(isinstance(value, str) and value.strip() for value in decision.values())
            ):
                raise ValueError(f"{case_id}: use 'answer' or a specific refusal path/reason")
    return expectations


def load_expectations() -> dict:
    return parse_expectations(EXPECTATIONS.read_text())
