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

from conformance.assertions import write_anchor_table, write_parquet
from kumo.pytorch import api

_RESOURCES = ContextVar("native_test_resources")
_ROOT = ContextVar("native_test_directory")


@contextmanager
def native_resources():
    with ExitStack() as stack:
        token = _RESOURCES.set(stack)
        root_token = _ROOT.set(
            Path(stack.enter_context(TemporaryDirectory(prefix="pytorch-test-")))
        )
        try:
            yield
        finally:
            _ROOT.reset(root_token)
            _RESOURCES.reset(token)


def directory():
    return Path(_RESOURCES.get().enter_context(TemporaryDirectory(prefix="pytorch-test-")))


def parquet_dataset(schema, tables, root=None):
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
    root = _ROOT.get() / f"tables-{identity}" if root is None else Path(root)
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


def check_rows_or_refuse(contract, dataset):
    from conformance.refusals import check_or_refuse

    check_or_refuse(contract, parquet_dataset(contract.schema, dataset), native_engine())


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


def request_anchor_table(requests, *, root=None, schema=None):
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
    root = _ROOT.get() if root is None else Path(root)
    return write_anchor_table(requests, root / f"anchors-{identity}.parquet", schema=schema)


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
    path = _ROOT.get() / f"anchors-{identity}.parquet"
    if not path.exists():
        pq.write_table(data, path)
    return replace(table, source=api.Parquet(str(path)))


@contextmanager
def forbid_deprecated_engine(
    *, message="InMemory execution is forbidden; assert DiskGraph against fixture answers"
):
    """Forbid test backends/oracles; allow ServingAssembler's package-owned dependency."""
    import sys
    from functools import wraps
    from inspect import signature
    from unittest.mock import patch

    from kumo.pytorch.engines.in_memory.anchors import InMemoryAnchors
    from kumo.pytorch.engines.in_memory.graph import InMemoryGraphSampler
    from kumo.pytorch.engines.in_memory.samples import InMemorySampleBuilder
    from kumo.pytorch.engines.in_memory.source import InMemorySource

    serving_calls = {api.ServingAssembler.__init__.__code__, api.ServingAssembler.assemble.__code__}

    def serving_is_executing():
        frame = sys._getframe(1)
        try:
            while frame is not None:
                if frame.f_code in serving_calls:
                    return True
                frame = frame.f_back
            return False
        finally:
            del frame

    def forbid(original, *, serving_dependency):
        @wraps(original)
        def forbidden(*args, **kwargs):
            if serving_dependency and serving_is_executing():
                return original(*args, **kwargs)
            raise AssertionError(message)

        return forbidden

    with ExitStack() as stack:
        # Preserve public signatures for the API-reference checks, including
        # classes whose original constructor is object.__init__.
        stack.enter_context(
            patch.object(
                api.InMemoryEngine, "__signature__", signature(api.InMemoryEngine), create=True
            )
        )
        methods = [
            (implementation, "__init__")
            for implementation in (
                api.InMemoryEngine,
                InMemoryAnchors,
                InMemoryGraphSampler,
                InMemorySampleBuilder,
                InMemorySource,
            )
        ] + [(api.InMemoryEngine, method) for method in ("open", "materialize_anchors")]
        for implementation, name in methods:
            stack.enter_context(
                patch.object(
                    implementation,
                    name,
                    forbid(
                        getattr(implementation, name),
                        serving_dependency=name == "__init__"
                        and implementation is not InMemoryAnchors,
                    ),
                )
            )
        yield
