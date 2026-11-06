# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load small, engine-independent examples beside their English guarantee."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from conformance.expectations import ENGINES as ENGINES
from conformance.fixtures import scalar
from kumo.pytorch.api import DataLoaderSpec, SampleRequest
from kumo.pytorch.contract.schema import table_clock

REPOSITORY = Path(__file__).resolve().parents[1]
GUARANTEES = REPOSITORY / "conformance/spec/guarantees"
SECTIONS = ("Rule", "Boundaries", "Scope", "Golden examples")


@dataclass(frozen=True)
class GoldenCase:
    id: str
    guarantee_id: str
    path: Path
    contract: DataLoaderSpec
    tables: dict[str, list[dict[str, Any]]]
    requests: tuple[SampleRequest, ...]
    expected: dict[str, dict[str, Any]]
    reason: str
    batch: dict[str, Any] | None = None
    state: dict[str, Any] | None = None
    physical_types: dict[str, dict[str, str]] | None = None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _finite(value: Any) -> None:
    if isinstance(value, float):
        _require(math.isfinite(value), "golden data and answers must contain finite numbers")
    elif isinstance(value, dict):
        for child in value.values():
            _finite(child)
    elif isinstance(value, list):
        for child in value:
            _finite(child)


def load_case(path: Path) -> GoldenCase:
    """Validate inputs and output assertions before any engine is opened.

    Raw sample answers and explicitly selected encoded buffers are independent
    assertions. Unknown output projections fail before an engine is opened.
    """
    _require(path.parent.name == "golden", f"{path}: case must live in a golden directory")
    guarantee_id = path.parent.parent.name
    _require(bool(re.fullmatch(r"G-[A-Z0-9-]+", guarantee_id)), "invalid guarantee directory ID")
    _require(bool(re.fullmatch(r"C-[A-Z0-9-]+", path.stem)), "invalid golden filename ID")
    payload = json.loads(path.read_text(), object_pairs_hook=_unique_object)
    _require(isinstance(payload, dict), "golden case must be a JSON object")
    required = {"id", "contract", "tables", "anchors", "expected", "reason"}
    optional = {"batch", "state", "physical_types"}
    _require(
        required <= set(payload) <= required | optional,
        f"{path}: fields must include {sorted(required)}; optional {sorted(optional)}",
    )
    _require(payload["id"] == path.stem, f"{path}: id must match the filename")
    _require(
        isinstance(payload["reason"], str) and bool(payload["reason"].strip()), "missing reason"
    )
    _finite(payload)

    original = DataLoaderSpec.from_dict(payload["contract"])
    original.validate()  # Structural validity, without pretending an engine supports it.
    _require(original.to_dict() == payload["contract"], "contract must round-trip unchanged")
    contract = DataLoaderSpec.from_json(original.to_json())
    contract.validate()
    # Compare independently of the writer, ignoring object placement but not arrays.
    _require(
        json.dumps(contract.to_dict(), sort_keys=True)
        == json.dumps(original.to_dict(), sort_keys=True),
        "contract JSON round-trip must preserve values and explicit sequences",
    )
    _require(
        contract.fingerprint == original.fingerprint,
        "contract JSON round-trip must preserve the fingerprint",
    )
    _require(
        contract.input_fingerprint == original.input_fingerprint,
        "contract JSON round-trip must preserve the input fingerprint",
    )
    sample = contract.sample
    batch = payload.get("batch")
    state = payload.get("state")
    _require(bool(sample.entity_values) or batch is not None, "case must assert a value or batch")
    if batch is not None:
        _require(isinstance(batch, dict) and bool(batch), "batch must be a nonempty object")
        _require(set(batch) <= {"inputs", "targets"}, "unknown batch projection")
        for group, declared_outputs in (
            ("inputs", contract.batch.inputs),
            ("targets", contract.batch.targets),
        ):
            if group in batch:
                _require(
                    isinstance(batch[group], dict) and bool(batch[group]),
                    f"{group}: assert named outputs",
                )
                _require(set(batch[group]) <= set(declared_outputs), f"{group}: unknown output")
        _validate_projection(batch, arrays=True)
    if state is not None:
        _require(isinstance(state, dict) and bool(state), "state must be a nonempty object")
        _validate_projection(state)

    tables = payload["tables"]
    declared = {table.name: table for table in contract.schema.tables}
    _require(isinstance(tables, dict) and set(tables) == set(declared), "tables must match schema")
    for name, rows in tables.items():
        _require(isinstance(rows, list), f"{name}: rows must be a list")
        keys = []
        for row in rows:
            _require(
                isinstance(row, dict) and set(row) == set(declared[name].columns),
                f"{name}: each row must supply exactly its declared columns",
            )
            key = row[declared[name].primary_key]
            _require(type(key) in (int, str), f"{name}: primary keys must be integers or strings")
            keys.append(key)
            clock = table_clock(declared[name])
            if clock is not None:
                _require(
                    type(row[clock]) is int, f"{name}: declared clock needs integer epoch seconds"
                )
            for column, value in tuple(row.items()):
                row[column] = scalar(value)
        _require(len(keys) == len(set(keys)), f"{name}: duplicate primary keys")

    physical_types = payload.get("physical_types")
    if physical_types is not None:
        _require(
            isinstance(physical_types, dict) and set(physical_types) <= set(declared),
            "unknown physical table",
        )
        for name, columns in physical_types.items():
            _require(
                isinstance(columns, dict) and set(columns) <= set(declared[name].columns),
                "unknown physical column",
            )
            _require(
                all(isinstance(value, str) and value for value in columns.values()),
                "physical types must be named Arrow types",
            )

    anchors = payload["anchors"]
    _require(isinstance(anchors, list) and 1 <= len(anchors) <= 8, "supply 1–8 anchors for preview")
    requests = []
    primary_key = declared[sample.entity.table].primary_key
    root_keys = {row[primary_key] for row in tables[sample.entity.table]}
    for anchor in anchors:
        _require(
            isinstance(anchor, dict) and set(anchor) == {"sample_id", "entity_id", "as_of_time"},
            "anchors must contain sample_id, entity_id, and as_of_time only",
        )
        _require(
            isinstance(anchor["sample_id"], str) and bool(anchor["sample_id"]),
            "sample_id must be a nonempty string",
        )
        _require(type(anchor["as_of_time"]) is int, "as_of_time must be integer epoch seconds")
        _require(type(anchor["entity_id"]) in (int, str), "entity_id must be an integer or string")
        _require(anchor["entity_id"] in root_keys, "anchor entity must exist in the root table")
        requests.append(SampleRequest(**anchor))
    ids = [request.sample_id for request in requests]
    _require(len(set(ids)) == len(ids), "sample IDs must be unique")

    expected = payload["expected"]
    _require(
        isinstance(expected, dict) and set(expected) == set(ids), "answer every anchor exactly"
    )
    for answer in expected.values():
        _require(
            isinstance(answer, dict) and set(answer) == {"entity_values"},
            "unknown expected-output projection; only entity_values is currently supported",
        )
        _require(
            isinstance(answer["entity_values"], dict)
            and set(answer["entity_values"]) == set(sample.entity_values),
            "expected entity_values must assert every declared entity value",
        )
    return GoldenCase(
        payload["id"],
        guarantee_id,
        path,
        contract,
        tables,
        tuple(requests),
        expected,
        payload["reason"],
        batch,
        state,
        physical_types,
    )


def _validate_projection(value: Any, *, arrays: bool = False) -> None:
    """Array leaves state their full shape and dtype; mappings select named fields."""
    if isinstance(value, dict):
        if {"dtype", "shape", "values"} <= set(value):
            _require(
                set(value) <= {"dtype", "shape", "values", "atol"}, "unknown array assertion option"
            )
            _require(
                isinstance(value["dtype"], str) and bool(value["dtype"]),
                "array dtype must be named",
            )
            _require(
                isinstance(value["shape"], list)
                and all(type(size) is int and size >= 0 for size in value["shape"]),
                "array shape must contain nonnegative integers",
            )
            _require(
                "atol" not in value or type(value["atol"]) in (float, int) and value["atol"] >= 0,
                "array tolerance must be nonnegative",
            )
            _validate_shape(value["values"], value["shape"])
            return
        if arrays:
            _require(
                not {"dtype", "shape"} & set(value), "array assertion needs dtype, shape and values"
            )
        for child in value.values():
            _validate_projection(child, arrays=arrays)
    elif isinstance(value, list):
        _require(not arrays, "batch arrays need explicit dtype, shape and values")
        for child in value:
            _validate_projection(child)


def _validate_shape(value: Any, shape: list[int]) -> None:
    if not shape:
        _require(not isinstance(value, list), "array values do not match declared shape")
        return
    _require(
        isinstance(value, list) and len(value) == shape[0],
        "array values do not match declared shape",
    )
    for child in value:
        _validate_shape(child, shape[1:])


def discover_case_paths(root: Path) -> list[Path]:
    """Check the rule/example file structure without decoding a contract."""
    directories = sorted(path for path in root.iterdir() if path.is_dir())
    _require(bool(directories), "no guarantees found")
    consumed = set()
    for directory in directories:
        document = directory / "guarantee.md"
        _require(document.is_file(), f"{directory}: missing guarantee.md")
        text = document.read_text()
        _require(
            text.startswith(f"# {directory.name}: "), f"{document}: heading must name directory ID"
        )
        for section in SECTIONS:
            _require(
                bool(re.search(rf"^## {section}\n\n\S", text, re.MULTILINE)),
                f"{document}: missing or empty {section} section",
            )
        paths = sorted((directory / "golden").glob("*.json"))
        _require(bool(paths), f"{directory}: guarantee needs a golden example")
        links = set(re.findall(r"\]\((golden/[^)]+\.json)\)", text))
        _require(
            links == {f"golden/{path.name}" for path in paths},
            f"{document}: golden links must name every case and no missing case",
        )
        for path in paths:
            consumed.add(path)
    _require(
        consumed == set(root.rglob("*.json")), "JSON case outside a guarantee's golden directory"
    )
    return sorted(consumed)


def discover_cases(root: Path = GUARANTEES) -> dict[str, GoldenCase]:
    """Load active examples and reject duplicate IDs after checking their structure."""
    cases = {}
    for path in discover_case_paths(root):
        case = load_case(path)
        _require(case.id not in cases, f"duplicate golden ID {case.id}")
        cases[case.id] = case
    return cases
