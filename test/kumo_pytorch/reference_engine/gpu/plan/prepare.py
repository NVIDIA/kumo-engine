# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`prepare`: the entry point, in place of upstream Kumo PyTorch's `prepare(contract, dataset, engine)`.

`prepare`, `GpuPlan` and `GpuTask`: upstream Kumo PyTorch's preparation shape, on the GPU.
``test.kumo_pytorch.reference_engine.gpu.prepare(contract, dataset)`` checks the contract with upstream Kumo PyTorch's
own validation and this backend's checks of what it has built so far
(`_Engine`: its sources, and `_unbuilt` for the parts of a contract), so a
contract that needs anything else is refused by name before a row is read. It then loads,
onto the GPU, only the tables and columns the contract reads (as upstream's
own reading of the contract lists them, targets included), and returns a
`GpuPlan`.

`GpuPlan.materialize` binds anchors (a Parquet `AnchorTable`, or
`SampleRequest` values built in code) with upstream's checks and fit-split
rules, fits the inputs' encoders on the fit split, and returns a `GpuTask`,
whose `batches` follow upstream's `Task.batches` (`test.kumo_pytorch.reference_engine.gpu.batching`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from kumo.pytorch.dataset import Dataset
    from kumo.pytorch.spec import DataLoaderSpec

from test.kumo_pytorch.reference_engine.gpu.plan.check import _Engine
from test.kumo_pytorch.reference_engine.gpu.plan.columns import _narrowed
from test.kumo_pytorch.reference_engine.gpu.plan.gpu_plan import GpuPlan
from test.kumo_pytorch.reference_engine.gpu.plan.groups import _build_bytes

__all__ = ['prepare']


def prepare(contract: DataLoaderSpec, dataset: Dataset, *, inference: bool = False,
            split_dir: str | Path | None = None, split: str = "if_missing", num_partitions: int | None = None,
            model_bytes: int | None = None) -> GpuPlan:
    """Check `contract` against what this backend builds, then load what it reads onto the GPU.

    Raises upstream's ``ValidationError`` listing every problem, before any row
    is read. With ``inference=True`` the targets are checked as upstream's
    ``check(..., inference=True)`` checks them, and only the columns the inputs
    read are loaded.

    When the columns the contract reads do not fit on the GPU
    (`test.kumo_pytorch.reference_engine.gpu.split.routing.choose_route`), the dataset is split by its entity and
    streamed through the GPU a partition at a time (`test.kumo_pytorch.reference_engine.gpu.split.plan`). The
    split is kept in `split_dir`, as upstream's DiskGraph engine keeps its index
    in a folder: a split there made from the same files, holding what this
    contract reads, is reused; otherwise one is written there, or, with
    ``split="never"``, the dataset is refused. A split that does not match is
    never overwritten. A dataset that fits is loaded directly, and `split_dir`
    is not used.

    `model_bytes` is the model's measured peak GPU memory while training
    (`test.kumo_pytorch.reference_engine.gpu.measure_model_bytes`): the data then gets what the model and a
    reserve leave, rather than half the GPU, both to decide whether it fits and
    to size the split's partitions. `num_partitions` instead splits the dataset
    into exactly that many partitions, whatever its size (a densely linked
    dataset the sizing refuses, or a chosen granularity); it needs `split_dir`.
    A split found there is reused only with that many partitions; without
    `num_partitions`, only when its largest partition fits the room left now.
    """

    from test.kumo_pytorch.reference_engine.gpu.data.tables import load_tables
    from test.kumo_pytorch.reference_engine.gpu.split.plan import (
        SPLIT_POLICIES,
    )
    from test.kumo_pytorch.reference_engine.gpu.split.routing import (
        choose_route,
    )

    from kumo.pytorch.preparation import _check, _inference_view
    from kumo.pytorch.spec import ValidationError

    if split not in SPLIT_POLICIES:
        raise ValueError(f"split must be one of {', '.join(map(repr, SPLIT_POLICIES))}, not {split!r}")
    if num_partitions is not None:
        if type(num_partitions) is not int or num_partitions < 1:
            raise ValueError(f"num_partitions must be a positive number of partitions, not {num_partitions!r}")
        if model_bytes is not None:
            raise ValueError("pass num_partitions or model_bytes, not both: num_partitions fixes the split, "
                             "model_bytes sizes it")
        if split_dir is None:
            raise ValueError("num_partitions splits the dataset: pass split_dir, a folder for the split")
    if model_bytes is not None and (type(model_bytes) is not int or model_bytes < 0):
        raise ValueError(f"model_bytes must be a non-negative number of bytes, not {model_bytes!r}")
    engine = _Engine()
    contract, issues = _check(contract, dataset, engine, inference=inference)
    if issues:
        raise ValidationError(issues)
    viewed = _inference_view(contract) if inference else contract  # inference: no target's columns
    schema = _narrowed(viewed)
    route = None if num_partitions is not None else \
        choose_route(schema, dataset, contract.sample.entity.table, sample=viewed.sample, model_bytes=model_bytes)
    if route is None or route.route != "direct":
        if split_dir is None:
            raise ValueError(
                f"the columns this contract reads, and the indexes built on them, take about "
                f"{route.dataset_bytes / 1e9:.1f} GB ({route.peak_bytes / 1e9:.1f} GB at their peak), more than this "
                f"{route.gpu_bytes / 1e9:.0f} GB GPU holds directly: pass split_dir, a folder for the dataset split "
                "by entity (written once, reused while the files are unchanged)"
            )
        from test.kumo_pytorch.reference_engine.gpu.split.plan import (
            prepare_split,
        )

        plan = prepare_split(contract, dataset, engine, schema, viewed.sample, split_dir,
                             if_missing=split == "if_missing", inference=inference, num_partitions=num_partitions,
                             model_bytes=model_bytes, room=None if route is None else route.partition_bytes,
                             peak_room=None if route is None else route.partition_peak_bytes)
        held = None
        if route is not None and model_bytes is not None:  # the shared tables and two partitions, at most
            from test.kumo_pytorch.reference_engine.gpu.data.sources import (
                resolve_sources,
            )
            from test.kumo_pytorch.reference_engine.gpu.split.routing import (
                _held,
                _loads,
            )

            shared = [table.name for table in schema.tables if plan.layout.kind(table.name) == "shared"]
            loads = _loads(schema, viewed.sample, resolve_sources(schema, dataset))
            held = int(_held(loads, shared)) + 2 * route.partition_bytes
        plan.build_bytes = _build_bytes(model_bytes, held)
        plan.model_bytes = model_bytes
        return plan
    from test.kumo_pytorch.reference_engine.gpu.sampling.indexes import (
        EntityIndex,
    )
    from test.kumo_pytorch.reference_engine.gpu.sampling.sampler import Sampler

    tables = load_tables(schema, dataset, only=[table.name for table in schema.tables])
    entity = contract.sample.entity.table
    index = EntityIndex(tables[entity], next(t.primary_key for t in contract.schema.tables if t.name == entity))
    plan = GpuPlan(contract, dataset, engine, tables, index, Sampler(contract, tables), inference=inference)
    plan.build_bytes = _build_bytes(model_bytes, route.dataset_bytes)
    plan.model_bytes = model_bytes
    return plan
