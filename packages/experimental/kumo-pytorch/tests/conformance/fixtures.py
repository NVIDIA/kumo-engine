# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small reproducible Parquet fixtures; all production delivery uses AnchorTable."""

import json
import random
from contextlib import contextmanager

import pyarrow as pa
import pyarrow.parquet as pq
import torch

import kumo.pytorch.api as r
from conformance.assertions import write_parquet
from conformance.refusals import check_or_refuse, executing_case

INT = r.DType("int64")
FLOAT = r.DType("float64")
TEXT = r.DType("string")
TIME = r.DType("timestamp", unit="s", timezone="UTC")


def sources(contract, rows, directory, *, seed=0, parts=3):
    directory.mkdir(parents=True, exist_ok=True)
    bindings = {}
    for table in contract.schema.tables:
        values = list(rows[table.name])
        random.Random(seed).shuffle(values)
        path = directory / table.name
        write_parquet(table, values, path, parts=parts)
        bindings[table.name] = r.Parquet(str(path))
    return r.Dataset(bindings)


def anchors(directory, records, *, targets=None, candidates=None, entity_dtype=None):
    # Anchor identities use their declared type, including uint64 above INT64_MAX.
    columns = {
        key: [row.get(key) for row in records]
        for key in dict.fromkeys(key for row in records for key in row)
    }
    if entity_dtype is not None:
        columns["entity"] = pa.array(columns["entity"], type=entity_dtype.to_arrow())
    data = pa.table(columns)
    path = directory / "anchors.parquet"
    pq.write_table(data, path)
    return r.AnchorTable(
        r.Parquet(str(path)),
        "entity",
        "at",
        sample_id="sample",
        split="split",
        targets=targets or {},
        candidates=candidates or {},
    )


@contextmanager
def materialized(contract, rows, records, directory, *, seed=0, targets=None, candidates=None):
    """Preflight refusals are distinct; any refusal after preparation remains a failure."""
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    dataset = sources(contract, rows, directory / "source", seed=seed)
    entity = next(
        table for table in contract.schema.tables if table.name == contract.sample.entity.table
    )
    table = anchors(
        directory,
        records,
        targets=targets,
        candidates=candidates,
        entity_dtype=entity.columns.get(entity.primary_key),
    )
    (directory / "case.json").write_text(
        json.dumps(
            {
                "contract": contract.to_dict(),
                "rows": rows,
                "anchors": records,
                "seed": seed,
            },
            indent=2,
            default=str,
        )
        + "\n"
    )
    engine = r.DiskGraphEngine(index=directory / "index", num_threads=1, num_shards=3)
    check_or_refuse(contract, dataset, engine)
    with r.prepare(contract, dataset, engine) as plan:
        with (
            executing_case(),
            plan.materialize(table, fit_split="train", encoder_cutoff=100) as task,
        ):
            yield task, plan, table


def batches(task, *, split="train", epoch=0, **kwargs):
    assert callable(getattr(task._anchors, "batches", None)), "Native anchors are required"
    if task.contract.batch.batching.max_samples is None:
        kwargs.setdefault("batch_size", 4)
    return task.batches(split=split, epoch=epoch, shuffle=False, output_format="model", **kwargs)


def record(sample, entity, *, at=100, split="train", **extra):
    return {"sample": sample, "entity": entity, "at": at, "split": split, **extra}


def fields(*names):
    return r.Fields(tuple(r.Encode(name, r.Raw()) for name in names))


def assert_tree(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor) and left.dtype == right.dtype
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tree(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_tree(a, b)
    else:
        assert left == right
