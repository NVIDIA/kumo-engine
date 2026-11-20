# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serving equivalence with explicitly scoped numerical and graph semantics."""

from collections.abc import Mapping
from dataclasses import replace

import torch

from conformance.properties import MOMENT_TOLERANCE, SUMMATION_DEPENDENT
from conformance.sequence_cases import TABLES, _events, assert_sequence_answers
from kumo.pytorch import api
from tests.graph_semantics import SourceRows, assert_graph


def assert_serving_batches(
    actual, expected, artifact, tables, requests, *, entity_answers=None, sequence_case=None
):
    from tests.test_inference_lifecycle import assert_same

    contract = artifact.contract
    state = artifact.fitted_state.to_dict()
    source = tables if isinstance(tables, SourceRows) else SourceRows(contract.schema, tables)
    by_id = {request.sample_id: request for request in requests}
    assert len(actual) == len(expected)
    for left, right in zip(actual, expected, strict=True):
        assert_same(left.targets, right.targets, "targets")
        metadata = []
        for batch in (left, right):
            # Native anchors explicitly report the absent split as null.
            value = dict(batch.metadata)
            value["request_metadata"] = tuple(
                {key: item for key, item in row.items() if key != "split" or item is not None}
                for row in value["request_metadata"]
            )
            metadata.append(value)
        assert_same(*metadata, "metadata")
        assert set(left.inputs) == set(right.inputs) == set(contract.batch.inputs)
        selected = [by_id[sample] for sample in left.metadata["sample_ids"]]
        if sequence_case is not None:
            clocks = {
                (table, row[f"{table[:-1]}_id"]): row["at"]
                for table in TABLES
                for row in sequence_case.tables[table]
            }
            for label, batch in (("served", left), ("native", right)):
                answer = {
                    str(request.sample_id): {
                        name: _events(value, position, clocks, len(selected))
                        for name, value in batch.inputs.items()
                    }
                    for position, request in enumerate(selected)
                }
                assert_sequence_answers(
                    replace(sequence_case, requests=tuple(selected)), answer, label
                )
        for name, spec in contract.batch.inputs.items():
            if sequence_case is not None:
                assert isinstance(spec.source, api.EventSource), "Unvalidated sequence input"
                assert left.inputs[name]["valid"].shape == right.inputs[name]["valid"].shape, (
                    f"{name}: Padded layout differs"
                )
                continue
            if isinstance(spec.representation, api.Graph):
                for batch in (left, right):
                    assert_graph(batch.inputs[name], contract, name, selected, source, state)
                continue
            approximate = set()
            if isinstance(spec.source, api.EntitySource) and isinstance(
                spec.representation, api.Fields
            ):
                for field in spec.representation.fields:
                    value = contract.sample.entity_values[field.field]
                    if (
                        isinstance(value, api.Aggregate)
                        and value.operation in SUMMATION_DEPENDENT
                        and value.dtype.name == "float64"
                        and isinstance(field.encoding, api.Raw)
                    ):
                        approximate.add(f"{name}.{field.field}.values")
            _compare(left.inputs[name], right.inputs[name], name, approximate)
            if entity_answers is not None and isinstance(spec.source, api.EntitySource):
                for field in spec.representation.fields:
                    assert isinstance(field.encoding, api.Raw), (
                        "Generated source oracle expects raw columns"
                    )
                    expected_values = [
                        entity_answers[str(r.sample_id)][field.field] for r in selected
                    ]
                    for batch in (left, right):
                        value = batch.inputs[name][field.field]
                        assert value["valid"].tolist() == [v is not None for v in expected_values]
                        reference = torch.tensor(
                            [0 if v is None else v for v in expected_values],
                            dtype=value["values"].dtype,
                        )
                        mask = value["valid"]
                        _compare(
                            value["values"][mask],
                            reference[mask],
                            f"{name}.{field.field}.values",
                            approximate,
                        )


def _compare(actual, expected, path, approximate):
    from tests.test_inference_lifecycle import assert_same

    if path in approximate:
        # G-CALC-001/G-SERVING-001 permit approximate float64 moments. Reuse the
        # conformance fixture family's 1e-12 bound; masks/counts/layout stay exact.
        assert isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor)
        assert actual.dtype == expected.dtype == torch.float64
        torch.testing.assert_close(
            actual,
            expected,
            rtol=MOMENT_TOLERANCE,
            atol=MOMENT_TOLERANCE,
            equal_nan=False,
            msg=path,
        )
    elif isinstance(expected, Mapping):
        assert isinstance(actual, Mapping) and actual.keys() == expected.keys(), path
        for key in expected:
            _compare(actual[key], expected[key], f"{path}.{key}", approximate)
    else:
        assert_same(actual, expected, path)
