# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Building a call's batches a group at a time, within a GPU memory budget."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:

    from kumo.pytorch.spec import DataLoaderSpec


#: Without the model's measured peak (`prepare(model_bytes=)`), the share of the GPU that building one call's
#: batches may hold at once: the data may take half of the GPU, and the model the quarter left.
BUILD_FRACTION = 0.25

#: A call of more anchors than twice this is built a group of whole batches at a time (`_in_groups`): this many
#: first, then as many as half of `GpuPlan.build_bytes` holds at the most bytes an anchor has taken so far.
PROBE_ANCHORS = 1 << 14

#: Headroom on the bytes an anchor was measured to take, for anchors with longer histories than those measured.
BUILD_HEADROOM = 1.25


def _build_bytes(model_bytes: int | None, held: int | None) -> int:
    """The GPU memory building one call's batches may hold at once (`_in_groups`): with the model's measured peak,
    what it and the data on the GPU (`held` bytes, decoded) leave, which the route kept at least a tenth of the
    GPU; without it, `BUILD_FRACTION` of the GPU."""

    from test.kumo_pytorch.reference_engine.gpu.split.routing import (
        _gpu_memory,
    )

    gpu = _gpu_memory()
    if model_bytes is None or held is None:
        return int(gpu * BUILD_FRACTION)
    return int(gpu - model_bytes - held)


def _groups(size: int, options: dict[str, Any], from_requests: bool) -> list[tuple[int, int]]:
    """The groups a token budget or length buckets compose batches within: materialized anchors a chunk
    (``chunk_batches`` batches' samples) at a time, requests all at once, as upstream batches them."""

    step = size if from_requests else options["maximum"] * options["chunk"]
    return [(start, min(start + step, size)) for start in range(0, size, max(step, 1))]


def _measured(build: Any, contract: DataLoaderSpec, rows: Any, padding: Any, budget: int) -> dict[str, Any]:
    """Each sample's length in ``Batching.length_input`` (`test.kumo_pytorch.reference_engine.gpu.batching.compose._lengths`), measured on that input
    alone (and the input it reads events of): no target changes it, as upstream measures it. Many samples are
    measured a group at a time, as `_in_groups` builds them, within `budget` bytes."""

    import numpy as np

    from kumo.pytorch.contract.batch import EventsOf, Graph

    name = contract.batch.batching.length_input
    if name not in contract.batch.inputs or isinstance(contract.batch.inputs[name].representation, Graph):
        raise ValueError(f"Batching.length_input {name!r} names no input with fields")
    if not rows.size:
        return {"total": [], "events": [], "longest": []}
    kept = {name, *([contract.batch.inputs[name].source.input]
                    if isinstance(contract.batch.inputs[name].source, EventsOf) else [])}
    alone = replace(contract, batch=replace(contract.batch, targets={}, inputs={
        key: value for key, value in contract.batch.inputs.items() if key in kept}))
    count = int(rows.size)
    if count <= 2 * PROBE_ANCHORS:
        return next(build(alone, rows=rows, padding=padding, targets=False, measure=name))
    sizer, parts, start = _Sizer(budget), [], 0
    while start < count:
        stop = min(count, start + sizer.size())
        with sizer.measured(stop - start):
            parts.append(next(build(alone, rows=rows[start:stop], padding=padding[start:stop], targets=False,
                                    measure=name)))
        start = stop
    return {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}


class _Sizer:
    """How many anchors the next group of a call takes, within `budget` bytes of GPU memory: `PROBE_ANCHORS` first,
    then as many as half of `budget` holds (the consumer may still hold the group before's last batch, and with
    it that group's arrays) at the most bytes an anchor has taken, measured with RMM's allocation statistics over
    each group's build before its first batch is handed out (no consumer's work inside), with `BUILD_HEADROOM`."""

    def __init__(self, budget: int) -> None:
        self.budget, self.rate = budget, None  # rate: the most bytes an anchor has taken

    def size(self) -> int:
        if self.rate is None:
            return PROBE_ANCHORS
        return max(1, int(self.budget / 2 / (self.rate * BUILD_HEADROOM)))

    @contextmanager
    def measured(self, anchors: int) -> Iterator[None]:
        import rmm.statistics

        rmm.statistics.enable_statistics()  # once; keeps the pool beneath it (`test.kumo_pytorch.reference_engine.gpu.data.memory.ensure_pool`)
        rmm.statistics.push_statistics()
        try:
            yield
        finally:
            peak = rmm.statistics.pop_statistics().peak_bytes
        rate = peak / max(anchors, 1)
        self.rate = rate if self.rate is None else max(self.rate, rate)


def _in_groups(build: Any, rows: Any, padding: Any, plan: Any, budget: int, first: int = 0) -> Iterator[Any]:
    """`build`'s batches of the anchors at `rows` as `plan` (`test.kumo_pytorch.reference_engine.gpu.batching.compose.BatchPlan`) cuts them, the same as one
    call gives them, built a group of whole batches at a time so that a call of many anchors holds no more than
    `budget` bytes of GPU memory at once (`_Sizer`). A call of at most twice `PROBE_ANCHORS` is built at once.
    `first` is the place of `rows`' first anchor in the caller's call (padding rows' sample IDs number from it)."""

    from test.kumo_pytorch.reference_engine.gpu.batching.compose import (
        BatchPlan,
    )
    from test.kumo_pytorch.reference_engine.gpu.data.memory import (
        release_unused,
    )

    import numpy as np

    bounds = plan.bounds
    if not bounds or bounds[-1][1] <= 2 * PROBE_ANCHORS:
        yield from build(rows=rows, padding=padding, plan=plan, first=first)
        return
    cuts, sizer, done = plan.cuts, _Sizer(budget), 0
    while done < len(bounds):
        end = int(np.searchsorted(cuts, cuts[done] + sizer.size(), side="right")) - 1
        end = min(max(end, done + 1), len(bounds))  # whole batches, at least one
        start, stop = int(cuts[done]), int(cuts[end])
        failing = done <= plan.failure_at < end
        part = BatchPlan([(left - start, right - start) for left, right in bounds[done:end]],
                         max(0, min(plan.emit, end) - done), plan.failure if failing else None,
                         plan.failure_at - done if failing else -1)
        batches = build(rows=rows[start:stop], padding=padding[start:stop], plan=part, first=first + start)
        with sizer.measured(stop - start):  # the group's build: every anchor's values, before its first batch
            head = next(batches, None)
        if head is not None:
            yield head
            yield from batches
        del batches, head
        release_unused()  # the group's arrays, for PyTorch's allocations
        done = end
