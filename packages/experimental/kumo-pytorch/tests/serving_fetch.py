# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A reference row lookup that follows a `ServingRowsSpec` and nothing else.

It fetches, for one request, exactly what the spec's public data describes: each
route's join, time window and limit (every row tied at a limit's boundary kept),
hop by hop, then only the required tables and columns. Tests use it to check
that what the spec lists is enough to rebuild batch inference.
"""

from __future__ import annotations

import json
import lzma
from collections.abc import Mapping, Sequence
from functools import cache
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from conformance.assertions import bound_tables, case_settings
from conformance.properties import MOMENT_TOLERANCE, SUMMATION_DEPENDENT
from kumo.pytorch import api
from kumo.pytorch.api import DataSchema, SampleRequest
from kumo.pytorch.assembly import _fingerprint, _jsonable
from kumo.pytorch.data.records import cutoff_to_epoch, timestamp_to_epoch
from kumo.pytorch.serving import RowLimit, RowsJoin, ServingRowsSpec, TableRoute
from tests.graph_semantics import SourceRows, assert_graph
from tests.native_fixtures import request_anchor_table
from tests.test_inference_lifecycle import assert_same

__all__ = ["Row", "Tables", "fetch"]

Row = Mapping[str, object]
Tables = Mapping[str, Sequence[Row]]


def fetch(
    spec: ServingRowsSpec, schema: DataSchema, tables: Tables, request: SampleRequest
) -> dict[str, list[dict[str, object]]]:
    """What a caller fetches for ``request`` by following ``spec`` over ``tables``."""

    keys = {table.name: table.primary_key for table in schema.tables}
    # Separate source occurrences can even be aliases of the same input mapping.
    sources = {
        name: [dict(row) for row in rows] if keys[name] is None else rows
        for name, rows in tables.items()
    }
    cutoff = cutoff_to_epoch(request.as_of_time)
    fetched: dict[str, dict[object, Row]] = {}
    routes = sorted(
        ((table.name, route) for table in spec.tables.values() for route in table.routes),
        key=lambda item: item[1].hop,
    )
    for name, route in routes:
        rows = _joined(sources[name], route.join, request, fetched)
        rows = _windowed(rows, route, cutoff)
        if route.limit is not None:
            rows = _limited(rows, route.limit, route.join)
        bucket = fetched.setdefault(name, {})
        for row in rows:
            identity = row[keys[name]] if keys[name] is not None else id(row)
            bucket.setdefault(identity, row)
    return {
        name: [
            {column: row[column] for column, rows in table.columns.items() if rows.required}
            for row in fetched.get(name, {}).values()
        ]
        for name, table in spec.tables.items()
        if table.required
    }


def _joined(
    rows: Sequence[Row], join: RowsJoin, request: SampleRequest, fetched: dict[str, dict]
) -> list[Row]:
    if join.request_field == "entity_id":
        values = {request.entity_id}
    elif join.request_field is not None:
        set_name = join.request_field.removeprefix("candidates[").removesuffix("]").strip("'\"")
        values = {candidate.entity_id for candidate in request.candidates.get(set_name, ())}
    else:
        assert join.from_table is not None and join.from_column is not None
        values = {row[join.from_column] for row in fetched.get(join.from_table, {}).values()}
    values.discard(None)
    return [row for row in rows if row[join.column] in values]


def _seconds(value: object) -> int:
    if value is None:
        raise ValueError("a time column the spec filters on has a NULL")
    return timestamp_to_epoch(value)


def _windowed(rows: list[Row], route: TableRoute, cutoff: int) -> list[Row]:
    if route.time_column is None:
        return rows
    start, stop = (None, 0) if route.window is None else (route.window.start, route.window.stop)
    return [
        row
        for row in rows
        if (start is None or _seconds(row[route.time_column]) > cutoff + start)
        and (stop is None or _seconds(row[route.time_column]) <= cutoff + stop)
    ]


def _limited(rows: list[Row], limit: RowLimit, join: RowsJoin) -> list[Row]:
    groups: dict[object, list[Row]] = {}
    for row in rows:
        groups.setdefault(None if limit.per == "request" else row[join.column], []).append(row)
    kept: list[Row] = []
    for members in groups.values():
        ordered = sorted(
            members,
            key=lambda row: _seconds(row[limit.time_column]),
            reverse=limit.order == "newest",
        )
        if len(ordered) > limit.count:
            boundary = _seconds(ordered[limit.count - 1][limit.time_column])
            ordered = ordered[: limit.count] + [
                row
                for row in ordered[limit.count :]
                if _seconds(row[limit.time_column]) == boundary
            ]
        kept.extend(ordered)
    return kept


@cache
def _recorded():
    # #482 (5ab2640d), test_serving_rows_drift._run and test_serving._batch_inference.
    # The archive contains Artifact.save files and inspectable batches from offline inference.
    with lzma.open(Path(__file__).with_name("data") / "serving-reference.json.xz", "rt") as handle:
        return json.load(handle)


def offline_batches(
    contract,
    tables,
    requests,
    *,
    case_id=None,
    dimensions=None,
    types=None,
    fit_split=None,
    artifact=None,
):
    dimensions = dimensions or {}
    archive = _recorded()
    record = archive["offline"].get(case_id) or archive["native"].get(case_id)
    if record is not None:
        source = [contract.fingerprint, tables, requests, dimensions, types]
        assert _fingerprint(source) == record["source"], f"{case_id}: offline source changed"
    if case_id in ("C-FIELDS-002", "C-ROWS-002", "event_reference_lookup") or (
        dimensions.get("schema_shape.parallel_relation") == "present"
    ):
        # Native storage cannot ingest repeated physical FKs or evaluate these joined fields.
        assert case_id in archive["offline"]
        with TemporaryDirectory() as directory:
            for name in ("contract.json", "fitted-state.json", "manifest.json", "task.json"):
                (Path(directory) / name).write_text(record["artifact"][name])
            return api.Artifact.load(directory), {
                k: (None, v) for k, v in record["batches"].items()
            }
    with (
        case_settings(dimensions),
        bound_tables(contract, tables, "diskgraph", types=types) as (dataset, engine),
        api.prepare(contract, dataset, engine) as plan,
    ):
        anchors = request_anchor_table(requests, schema=plan.contract.schema)
        if artifact is None:
            with plan.materialize(anchors, fit_split=fit_split) as task:
                artifact = task.to_artifact()
        # Explicit scoring requests retain their caller-supplied split in the native
        # graph path. The full conformance corpus uses Parquet scoring anchors.
        scoring = requests if fit_split is not None else anchors
        with plan.materialize(scoring, fitted=artifact.fitted_state, inference=True) as task:
            batches = {
                batch.metadata["sample_ids"][0]: batch
                for batch in task.batches(batch_size=1, seed=7)
            }
    assert set(batches) == {request.sample_id for request in requests}
    originals = record["batches"] if record is not None else {}
    return artifact, {k: (batch, originals.get(k, {})) for k, batch in batches.items()}


def assert_offline_batch(served, offline, artifact, tables, request):
    expected, original = offline
    if expected is None:
        assert_same(
            _jsonable([served.inputs, served.targets, served.metadata]),
            original,
            str(request.sample_id),
        )
        return
    contract = artifact.contract
    assert_same(served.targets, expected.targets, "targets")
    left, right = dict(served.metadata), dict(expected.metadata)
    assert_same(
        left["request_metadata"], ({"split": request.split, **request.metadata},), "served metadata"
    )
    # An absent optional split can be omitted or explicitly null in request metadata.
    for metadata in (left, right):
        metadata["request_metadata"] = tuple(
            {key: value for key, value in row.items() if key != "split" or value is not None}
            for row in metadata["request_metadata"]
        )
    if not contract.requires_anchor_time:
        assert_same(
            left.pop("as_of_time"),
            torch.tensor([cutoff_to_epoch(request.as_of_time)]),
            "served cutoff",
        )
        assert_same(right.pop("as_of_time"), torch.tensor([0]), "native cutoff")
    assert_same(left, right, "metadata")
    assert served.inputs.keys() == expected.inputs.keys() == contract.batch.inputs.keys()
    for name, spec in contract.batch.inputs.items():
        actual, reference = served.inputs[name], expected.inputs[name]
        if isinstance(spec.representation, api.Graph):
            # The served sampler must retain its original seeded ordering as well.
            assert_same(_jsonable(actual), original[name], f"{name} seeded serving graph")
            source = SourceRows(contract.schema, tables)
            for graph in (actual, reference):
                assert_graph(
                    graph, contract, name, [request], source, artifact.fitted_state.to_dict()
                )
        elif isinstance(spec.source, api.EntitySource) and isinstance(
            spec.representation, api.Fields
        ):
            approximate = {
                field.field
                for field in spec.representation.fields
                if isinstance(
                    value := contract.sample.entity_values.get(field.field), api.Aggregate
                )
                and value.operation in SUMMATION_DEPENDENT
                and value.dtype.name == "float64"
                and isinstance(field.encoding, api.Raw)
            }
            assert actual.keys() == reference.keys()
            for key in actual:
                if key not in approximate:
                    assert_same(actual[key], reference[key], f"{name}.{key}")
                    continue
                assert_same(
                    _jsonable(actual[key]["values"]), original[name][key], f"{name}.{key}.served"
                )
                assert actual[key].keys() == reference[key].keys()
                for part in actual[key]:
                    tolerance = (MOMENT_TOLERANCE, MOMENT_TOLERANCE) if part == "values" else None
                    assert_same(
                        actual[key][part],
                        reference[key][part],
                        f"{name}.{key}.{part}",
                        tolerance=tolerance,
                    )
        else:
            assert_same(actual, reference, name)
