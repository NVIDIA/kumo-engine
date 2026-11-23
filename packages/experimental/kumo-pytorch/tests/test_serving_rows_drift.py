# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`ServingRowsSpec` is a static model of what inference reads.

If it were ever narrower than inference, serving code that fetches what it lists
would silently build other batches. For every golden case and each generated
family (tabular, graph, sequence), this fetches per anchor exactly the rows the
spec selects (`tests.serving_fetch`), assembles them with `ServingAssembler`, and
compares the result with offline batch inference over the whole dataset.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from functools import cache

import pytest

from conformance.cases import discover_cases
from conformance.generate import PHYSICAL_TYPES, GeneratedCase, generate_batch
from conformance.graph_cases import graph_batch
from conformance.properties import SEEDS
from conformance.registry import KNOWN_GAPS
from conformance.sequence_cases import sequence_batch
from kumo.pytorch import api
from tests.serving_fetch import Tables, assert_offline_batch, fetch, offline_batches
from tests.test_inference_lifecycle import assert_same_batches

SEED = 7
# Keep the original generated inputs, including seeds 0, 1 and 2.
GENERATION_PROFILE = "in_memory"
#: Properties whose generated batch is the shared covering array of its family:
#: no known gap removes rows from it.
GRAPH_PROPERTY, SEQUENCE_PROPERTY = "graph_structure", "sequence_point_in_time"


@dataclass(frozen=True)
class DriftCase:
    id: str
    family: str
    contract: api.DataLoaderSpec
    tables: Tables
    requests: tuple[api.SampleRequest, ...]
    dimensions: Mapping[str, object]
    physical_types: Mapping[str, Mapping[str, str]] | None
    generated: GeneratedCase | None = None


def _generated(family: str, case: GeneratedCase, types: Mapping | None = None) -> DriftCase:
    return DriftCase(
        case.id,
        family,
        case.contract,
        case.tables,
        case.requests,
        case.dimensions,
        types,
        case,
    )


def drift_cases() -> Iterator[DriftCase]:
    """Every golden case, then each generated family's seed batches."""

    for case_id, case in discover_cases().items():
        yield DriftCase(
            case_id,
            "golden",
            case.contract,
            case.tables,
            case.requests,
            {},
            case.physical_types,
        )
    for seed in SEEDS[GENERATION_PROFILE]:
        for case in generate_batch(seed, GENERATION_PROFILE):
            yield _generated("tabular", case, PHYSICAL_TYPES)
        for case in graph_batch(seed, GENERATION_PROFILE, GRAPH_PROPERTY):
            yield _generated("graph", case)
        for case in sequence_batch(seed, GENERATION_PROFILE, SEQUENCE_PROPERTY):
            yield _generated("sequence", case)


CASES = {case.id: case for case in drift_cases()}


def test_the_generated_batches_are_the_shared_covering_arrays() -> None:
    assert not any(gap.property in (GRAPH_PROPERTY, SEQUENCE_PROPERTY) for gap in KNOWN_GAPS)


@cache
def _run(case_id: str):
    """The parity and row-deletion checks share the same immutable offline result."""
    case = CASES[case_id]
    return offline_batches(
        case.contract,
        case.tables,
        case.requests,
        case_id=case.id,
        dimensions=case.dimensions,
        types=case.physical_types,
    )


def _served(server: api.ServingAssembler, case: DriftCase, request: api.SampleRequest) -> api.Batch:
    rows = fetch(server.rows_spec, server.contract.schema, case.tables, request)
    (batch,) = server.assemble(rows, [request], seed=SEED)
    return batch


@pytest.mark.parametrize("case_id", sorted(CASES))
def test_serving_what_the_spec_lists_equals_reference_batch_inference(case_id: str) -> None:
    case = CASES[case_id]
    artifact, expected = _run(case.id)
    server = api.ServingAssembler(artifact)
    for request in case.requests:
        assert_offline_batch(
            _served(server, case, request),
            expected[request.sample_id],
            artifact,
            case.tables,
            request,
        )


def test_dropping_a_row_the_spec_lists_changes_some_batches() -> None:
    """The fetch is not vacuous: in every family, some listed row is read."""

    changed: dict[str, int] = {}
    for case in CASES.values():
        changed.setdefault(case.family, 0)
        artifact, _ = _run(case.id)
        server = api.ServingAssembler(artifact)
        changed[case.family] += _some_row_is_read(server, case)
    assert all(changed.values()), changed


def _some_row_is_read(server: api.ServingAssembler, case: DriftCase) -> bool:
    entity = server.rows_spec.entity_table
    for request in case.requests:
        rows = fetch(server.rows_spec, server.contract.schema, case.tables, request)
        (full,) = server.assemble(rows, [request], seed=SEED)
        for name, table in rows.items():
            if name == entity:
                continue
            for index in range(len(table)):
                dropped = {**rows, name: table[:index] + table[index + 1 :]}
                (partial,) = server.assemble(dropped, [request], seed=SEED)
                try:
                    assert_same_batches([partial], [full])
                except AssertionError:
                    return True
    return False


def test_every_family_is_covered() -> None:
    assert {case.family for case in CASES.values()} == {"golden", "tabular", "graph", "sequence"}
