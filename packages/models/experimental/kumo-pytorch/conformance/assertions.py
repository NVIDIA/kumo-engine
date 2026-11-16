# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Assert a named answer or refusal through the actual public kumo-pytorch API."""

from __future__ import annotations

import math
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

from conformance.cases import GoldenCase
from conformance.fixtures import values as fixture_values
from conformance.registry import DimensionValue
from kumo.pytorch.api import (
    DataLoaderSpec,
    Dataset,
    DiskGraphEngine,
    InMemoryEngine,
    InMemoryRows,
    Parquet,
    TableSpec,
    ValidationError,
    check,
    prepare,
    preview,
)

Row = Mapping[str, object]
#: Dimensions of the generated case being checked; engine settings come from here.
_CASE: ContextVar[Mapping[str, DimensionValue]] = ContextVar("conformance_case", default={})


@contextmanager
def case_settings(dimensions: Mapping[str, DimensionValue]) -> Iterator[None]:
    """Bind engines to the case's settings; a case without them gets one shard, one thread."""

    token = _CASE.set(dict(dimensions))
    try:
        yield
    finally:
        _CASE.reset(token)


@dataclass(frozen=True)
class Refusal:
    """An independent test assertion, not a runtime support-policy decision."""

    path: str
    message_contains: str

    def __post_init__(self) -> None:
        if not self.path.strip() or not self.message_contains.strip():
            raise ValueError("a refusal assertion needs a path and a specific reason")


class UnreadableRows(Sequence):
    """A real InMemoryRows binding whose contents must not be consumed."""

    def __len__(self) -> int:
        raise AssertionError("refusal read rows while checking the request")

    def __getitem__(self, index: Any) -> Any:
        raise AssertionError("refusal read rows while checking the request")


def require_engine(engine_name: str) -> None:
    """An explicitly requested native engine cannot quietly become a skip."""
    if engine_name == "diskgraph":
        import diskgraph  # noqa: F401
        import pyarrow  # noqa: F401


@contextmanager
def bound_case(
    case: GoldenCase, engine_name: str, *, refusal: bool = False
) -> Iterator[tuple[Dataset, InMemoryEngine | DiskGraphEngine]]:
    with bound_tables(
        case.contract, case.tables, engine_name, refusal=refusal, types=case.physical_types
    ) as bound:
        yield bound


@contextmanager
def bound_tables(
    contract: DataLoaderSpec,
    tables: Mapping[str, Sequence[Row]],
    engine_name: str,
    *,
    refusal: bool = False,
    parts: int = 1,
    types: Mapping[str, Mapping[str, str]] | None = None,
) -> Iterator[tuple[Dataset, InMemoryEngine | DiskGraphEngine]]:
    """Bind rows to an engine; ``parts`` splits each Parquet table into that many files."""
    if engine_name == "in_memory":
        dataset = Dataset(
            {
                name: InMemoryRows(UnreadableRows() if refusal else list(rows))
                for name, rows in tables.items()
            }
        )
        yield dataset, InMemoryEngine()
    elif engine_name == "diskgraph":
        require_engine(engine_name)
        with tempfile.TemporaryDirectory(prefix="conformance-") as temporary:
            directory = Path(temporary)
            bindings = {}
            for name, rows in tables.items():
                path = directory / (f"{name}.parquet" if parts == 1 else name)
                if not refusal:
                    table = next(table for table in contract.schema.tables if table.name == name)
                    write_parquet(table, rows, path, parts=parts, types=(types or {}).get(name))
                # Refusal paths deliberately do not exist. Reading one is a bug.
                bindings[name] = Parquet(str(path))
            settings = _CASE.get()
            yield (
                Dataset(bindings),
                DiskGraphEngine(
                    index=directory / "index",
                    num_threads=int(settings.get("execution.num_threads", 1)),
                    num_shards=int(settings.get("execution.num_shards", 1)),
                ),
            )
    else:
        raise ValueError(f"unknown conformance engine {engine_name!r}")


def write_parquet(
    table: TableSpec,
    rows: Sequence[Row],
    path: Path,
    *,
    parts: int = 1,
    types: Mapping[str, str] | None = None,
) -> None:
    """Write explicit schemas, including empty/all-NULL tables and physical overrides."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    types = types or {}

    def arrow(chunk: Sequence[Row]) -> pa.Table:
        return pa.table(
            {
                column: pa.array(
                    [row[column] for row in chunk],
                    type=(
                        pa.type_for_alias(types[column])
                        if column in types
                        else table.columns[column].to_arrow()
                    ),
                )
                for column in table.columns
            }
        )

    if parts == 1:
        pq.write_table(arrow(rows), path)
        return
    path.mkdir()
    count = min(parts, len(rows)) or 1
    bounds = [len(rows) * index // count for index in range(count + 1)]
    for index in range(count):
        pq.write_table(
            arrow(rows[bounds[index] : bounds[index + 1]]), path / f"part-{index}.parquet"
        )


def _same_value(actual: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and actual.keys() == expected.keys()
            and all(_same_value(actual[key], value) for key, value in expected.items())
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(_same_value(a, e) for a, e in zip(actual, expected, strict=True))
        )
    if type(expected) in (int, float):
        return type(actual) in (int, float) and actual == expected
    return type(actual) is type(expected) and actual == expected


def assert_answer(case: GoldenCase, engine_name: str) -> None:
    with bound_case(case, engine_name) as (dataset, engine):
        issues = check(case.contract, dataset, engine)
        assert not issues, f"{case.id}/{engine_name}: expected support, got {issues}"
        with prepare(case.contract, dataset, engine) as plan:
            with plan.materialize(case.requests) as task:
                result = preview(task, case.requests).to_dict()
                if case.batch is not None:
                    batches = list(task.batches(case.requests, batch_size=len(case.requests)))
                    assert len(batches) == 1, f"{case.id}: expected exactly one batch"
                    batch = batches[0]
                    assert_projection(
                        {"inputs": batch.inputs, "targets": batch.targets},
                        case.batch,
                        path=f"{case.id}/{engine_name}.batch",
                    )
                if case.state is not None:
                    assert_projection(
                        task.fitted_state.to_dict(),
                        case.state,
                        path=f"{case.id}/{engine_name}.state",
                    )
        samples = result["samples"]
        ids = [sample["sample_id"] for sample in samples]
        assert len(ids) == len(set(ids)), f"{case.id}: duplicate output sample IDs"
        actual = {}
        for sample in samples:
            values = sample["entity"]["values"]
            assert set(values) <= set(case.contract.sample.entity_values), "undeclared raw output"
            actual[sample["sample_id"]] = {
                "entity_values": {
                    name: values.get(name) for name in case.contract.sample.entity_values
                }
            }
        assert _same_value(fixture_values(actual), fixture_values(case.expected)), (
            f"{case.id}/{engine_name}: expected {case.expected}, got {actual}"
        )


def assert_projection(actual: Any, expected: Any, *, path: str = "output") -> None:
    """Compare requested leaves completely, including exact array types and shape.

    Mapping projections deliberately select named buffers. A requested missing
    leaf fails. Arrays are never prefix-compared or converted to unordered sets.
    """
    if isinstance(expected, dict) and {"dtype", "shape", "values"} <= set(expected):
        import numpy as np
        import pyarrow as pa
        import torch

        if isinstance(actual, torch.Tensor):
            dtype = str(actual.dtype).removeprefix("torch.")
            shape, emitted = list(actual.shape), actual.detach().cpu().tolist()
        elif isinstance(actual, (pa.Array, pa.ChunkedArray)):
            dtype, shape, emitted = str(actual.type), [len(actual)], actual.to_pylist()
        elif isinstance(actual, np.ndarray):
            dtype, shape, emitted = str(actual.dtype), list(actual.shape), actual.tolist()
        else:
            raise AssertionError(f"{path}: expected typed array, got {type(actual).__name__}")
        assert dtype == expected["dtype"], (
            f"{path}: expected dtype {expected['dtype']}, got {dtype}"
        )
        assert shape == expected["shape"], (
            f"{path}: expected shape {expected['shape']}, got {shape}"
        )
        _assert_exact(emitted, fixture_values(expected["values"]), path, expected.get("atol"))
    elif isinstance(expected, dict):
        assert isinstance(actual, Mapping), f"{path}: expected mapping, got {type(actual).__name__}"
        for key, child in expected.items():
            assert key in actual, f"{path}: missing requested field {key!r}"
            assert_projection(actual[key], child, path=f"{path}.{key}")
    else:
        _assert_exact(fixture_values(actual), fixture_values(expected), path)


def _assert_exact(actual: Any, expected: Any, path: str, atol: float | None = None) -> None:
    if isinstance(expected, dict):
        assert isinstance(actual, Mapping) and actual.keys() == expected.keys(), (
            f"{path}: expected record fields {set(expected)}, got {actual!r}"
        )
        for key, value in expected.items():
            _assert_exact(actual[key], value, f"{path}.{key}", atol)
        return
    if isinstance(expected, list):
        assert isinstance(actual, (tuple, list)), f"{path}: expected sequence, got {actual!r}"
        assert len(actual) == len(expected), (
            f"{path}: expected {len(expected)} values, got {len(actual)}"
        )
        for index, (value, answer) in enumerate(zip(actual, expected, strict=True)):
            _assert_exact(value, answer, f"{path}[{index}]", atol)
        return
    assert type(actual) is type(expected), (
        f"{path}: expected type {type(expected).__name__}, got {type(actual).__name__}"
    )
    if atol is not None:
        assert type(expected) is float, f"{path}: tolerance is only meaningful for floating answers"
        assert math.isclose(actual, expected, abs_tol=atol, rel_tol=0), (
            f"{path}: expected {expected}, got {actual}"
        )
    else:
        assert actual == expected, f"{path}: expected {expected!r}, got {actual!r}"


def assert_refusal(case: GoldenCase, engine_name: str, expected: Refusal) -> None:
    with bound_case(case, engine_name, refusal=True) as (dataset, engine):
        # Patch only the opening operation. The real public check(), real engine
        # preflight, and prepare() remain in use. A late refusal must fail.
        with patch.object(engine, "open", side_effect=AssertionError("opened before refusal")):
            issues = check(case.contract, dataset, engine)
            _assert_issue(case, engine_name, issues, expected)
            try:
                prepare(case.contract, dataset, engine)
            except ValidationError as error:
                _assert_issue(case, engine_name, error.issues, expected)
            else:
                raise AssertionError(
                    f"{case.id}/{engine_name}: prepare() accepted a refused request"
                )


def assert_execution_error(
    case: GoldenCase,
    engine_name: str,
    *,
    stage: str,
    message_contains: str,
    output_format: str = "inspectable",
) -> None:
    """Require a data-dependent error at binding, fitting, or batch construction.

    A failure at an earlier stage propagates rather than satisfying the requested
    assertion. Declaration refusals use assert_refusal with unreadable sources.
    """
    if stage not in {"bind", "fit", "build"} or not message_contains.strip():
        raise ValueError("execution error needs bind/fit/build and a specific reason")

    def expect(action):
        try:
            result = action()
        except (ValueError, RuntimeError) as error:
            assert message_contains in str(error), f"{stage}: wrong error {error!r}"
        else:
            if hasattr(result, "close"):
                result.close()
            raise AssertionError(f"{case.id}/{engine_name}: expected {stage} failure")

    with bound_case(case, engine_name) as (dataset, engine):
        issues = check(case.contract, dataset, engine)
        assert not issues, f"{case.id}: refused before expected {stage} failure: {issues}"
        if stage == "bind":
            expect(lambda: prepare(case.contract, dataset, engine))
            return
        with prepare(case.contract, dataset, engine) as plan:
            if stage == "fit":
                expect(lambda: plan.materialize(case.requests))
                return
            with plan.materialize(case.requests) as task:
                expect(
                    lambda: list(
                        task.batches(
                            case.requests,
                            batch_size=len(case.requests),
                            output_format=output_format,
                        )
                    )
                )


def _assert_issue(case: GoldenCase, engine_name: str, issues: Sequence, expected: Refusal) -> None:
    assert len(issues) == 1, f"{case.id}/{engine_name}: expected one specific refusal, got {issues}"
    issue = issues[0]
    assert issue.path == expected.path, f"wrong refusal path: {issue}"
    assert expected.message_contains in issue.message, f"wrong refusal reason: {issue}"
    assert issue.suggestion and issue.suggestion.strip(), f"refusal needs a suggestion: {issue}"


class PropertyFailure(AssertionError):
    """A property did not hold for one case."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PropertyFailure(message)
