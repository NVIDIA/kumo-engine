# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write tiny test rows as Parquet and exercise the real DiskGraph binding."""

import hashlib
import json
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from conformance.assertions import write_parquet
from kumo.pytorch import api

_RESOURCES = ContextVar("native_test_resources")
_ROOT = ContextVar("native_test_directory")


@contextmanager
def native_resources():
    with ExitStack() as stack:
        token = _RESOURCES.set(stack)
        root_token = _ROOT.set(None)
        try:
            yield
        finally:
            _ROOT.reset(root_token)
            _RESOURCES.reset(token)


def directory():
    return Path(_RESOURCES.get().enter_context(TemporaryDirectory(prefix="pytorch-test-")))


def _root():
    root = _ROOT.get()
    if root is None:
        root = directory()
        _ROOT.set(root)
    return root


def parquet_dataset(schema, tables):
    """Fixture conversion only; callers still use the public native validation path."""
    if isinstance(tables, api.Dataset):
        tables = tables.tables
    rows = {
        name: source.rows if isinstance(source, api.InMemoryRows) else source
        for name, source in tables.items()
    }
    identity = hashlib.sha256(
        json.dumps([repr(schema), rows], sort_keys=True, default=repr).encode()
    ).hexdigest()
    root = _root() / f"tables-{identity}"
    root.mkdir(parents=True, exist_ok=True)
    bindings = {}
    for name, rows in tables.items():
        if isinstance(rows, api.Parquet):
            bindings[name] = rows
            continue
        if isinstance(rows, api.InMemoryRows):
            rows = rows.rows
        table = next((table for table in schema.tables if table.name == name), None)
        path = root / f"{name}.parquet"
        if table is not None and not path.exists():
            write_parquet(table, rows, path)
        bindings[name] = api.Parquet(str(path))
    return api.Dataset(bindings)


def native_engine():
    return api.DiskGraphEngine(index=directory() / "index", num_threads=1, num_shards=1)


def prepare_rows(contract, dataset, *, inference=False):
    dataset = parquet_dataset(contract.schema, dataset)
    plan = api.prepare(contract, dataset, native_engine(), inference=inference)
    _RESOURCES.get().callback(plan.close)
    return plan


def check_rows(contract, dataset, **kwargs):
    return api.check(contract, parquet_dataset(contract.schema, dataset), native_engine(), **kwargs)


class NativeSamples:
    """Raw native samples, validated with the same contract as public preparation."""

    def __init__(self, schema, tables):
        self.schema, self.tables = schema, tables

    def build(self, sample, requests):
        batch = api.BatchSpec(
            inputs={
                "entity": api.Input(
                    source=api.EntitySource(),
                    representation=api.Fields((api.Encode("fixture", api.Raw()),)),
                    layout=api.Separate(),
                )
            }
        )
        # The batch is only needed to prepare the source; build the original raw
        # sample below, without adding this preparation-only constant to its output.
        contract = api.DataLoaderSpec(
            self.schema,
            replace(
                sample,
                entity_values={
                    **sample.entity_values,
                    "fixture": api.Constant(0, api.DType("int64")),
                },
            ),
            batch,
        )
        plan = prepare_rows(contract, self.tables)
        return plan._source.build(sample, requests)


def preview_rows(contract, dataset, requests, **kwargs):
    return api.preview(
        contract, parquet_dataset(contract.schema, dataset), native_engine(), requests, **kwargs
    )


def write_anchor_table(requests, path, *, schema=None):
    """Project bounded expected records to native anchors, retaining labels and metadata."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from kumo.pytorch.api import NON_TEMPORAL, AnchorTable, CandidateColumn, Parquet

    requests = list(requests)
    targets = sorted({target.name for row in requests for target in row.targets})
    metadata = sorted({name for row in requests for name in row.metadata})
    candidates = {
        name: sorted(
            {
                target.name
                for row in requests
                for item in row.candidates.get(name, ())
                for target in item.targets
            }
        )
        for name in sorted({name for row in requests for name in row.candidates})
    }
    rows = [
        {
            "sample_id": row.sample_id,
            "entity_id": row.entity_id,
            "as_of_time": row.as_of_time,
            "split": row.split,
            **{
                f"target_{name}": next((t.value for t in row.targets if t.name == name), None)
                for name in targets
            },
            **{name: row.metadata.get(name) for name in metadata},
            **{
                f"candidates_{name}": [
                    {
                        "entity_id": item.entity_id,
                        **{f"target_{target.name}": target.value for target in item.targets},
                    }
                    for item in row.candidates.get(name, ())
                ]
                for name in candidates
            },
        }
        for row in requests
    ]
    data = (
        pa.table(
            {
                name: pa.array([row[name] for row in rows], type=pa.uint64())
                if any(type(row[name]) is int and row[name] > 2**63 - 1 for row in rows)
                else [row[name] for row in rows]
                for name in rows[0]
            }
        )
        if rows
        else pa.table(
            {
                "sample_id": pa.array([], type=pa.string()),
                "entity_id": pa.array([], type=pa.int64()),
                "as_of_time": pa.array([], type=pa.timestamp("s", tz="UTC")),
                "split": pa.array([], type=pa.string()),
            }
        )
    )
    clock = data.schema.field("as_of_time")
    if pa.types.is_integer(clock.type):
        position = data.schema.get_field_index("as_of_time")
        data = data.set_column(
            position, "as_of_time", data["as_of_time"].cast(pa.timestamp("s", tz="UTC"))
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        pq.write_table(data, path)
    else:
        existing = pq.read_table(path)
        assert existing.equals(data.cast(existing.schema)), (
            "anchor projection path was reused for different rows"
        )
    return AnchorTable(
        Parquet(str(path.resolve())),
        "entity_id",
        "as_of_time"
        if schema is None
        or any(table.time_column not in (None, NON_TEMPORAL) for table in schema.tables)
        else None,
        sample_id="sample_id",
        split="split" if any(row.split is not None for row in requests) else None,
        targets={name: f"target_{name}" for name in targets},
        candidates={
            name: CandidateColumn(
                f"candidates_{name}",
                "entity_id",
                targets={target: f"target_{target}" for target in labels},
            )
            for name, labels in candidates.items()
        },
        metadata=metadata,
    )


def request_anchor_table(requests, *, schema=None):
    """Expected records describe fixtures; the task consumes their Parquet projection."""
    if isinstance(requests, api.AnchorTable):
        return requests
    requests = tuple(requests)
    records = [
        [
            row.sample_id,
            row.entity_id,
            row.as_of_time,
            row.split,
            [(target.name, target.value) for target in row.targets],
            dict(row.metadata),
            dict(row.candidates),
        ]
        for row in requests
    ]
    identity = hashlib.sha256(
        json.dumps(records, sort_keys=True, default=repr).encode()
    ).hexdigest()
    return write_anchor_table(requests, _root() / f"anchors-{identity}.parquet", schema=schema)


def native_anchor_table(*args, **kwargs):
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = (
        args[0]
        if len(args) == 1 and isinstance(args[0], api.AnchorTable)
        else api.AnchorTable(*args, **kwargs)
    )
    if not isinstance(table.source, api.InMemoryRows):
        return table
    rows = list(table.source.rows)
    data = pa.Table.from_pylist(rows)
    clock = table.as_of_time
    if clock in data.column_names and pa.types.is_integer(data[clock].type):
        position = data.column_names.index(clock)
        data = data.set_column(position, clock, data[clock].cast(pa.timestamp("s", tz="UTC")))
    identity = hashlib.sha256(
        json.dumps([str(data.schema), rows], sort_keys=True, default=repr).encode()
    ).hexdigest()
    path = _root() / f"anchors-{identity}.parquet"
    if not path.exists():
        pq.write_table(data, path)
    return replace(table, source=api.Parquet(str(path)))
