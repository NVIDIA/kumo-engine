# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Composing batches: their sizes under token budgets and length buckets, and the length summary."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:

    pass
from test.kumo_pytorch.reference_engine.gpu.batching.layout import _Nested


@dataclass
class BatchPlan:
    """How a call's samples are cut into batches: `bounds` (start, stop) of each, in order, covering every
    sample; the first `emit` are yielded (`drop_last` drops one); at batch `failure_at`, `failure` is raised
    instead (upstream's refusal of a sample over the token budget, met when its chunk is reached)."""

    bounds: list[tuple[int, int]]
    emit: int
    failure: str | None = None
    failure_at: int = -1

    @property
    def cuts(self) -> Any:
        """Where each batch starts, then the end (host int64)."""

        import numpy as np

        return np.asarray([start for start, _ in self.bounds] + [self.bounds[-1][1] if self.bounds else 0],
                          dtype=np.int64)

    @classmethod
    def fixed(cls, count: int, maximum: int | None, drop_last: bool) -> BatchPlan:
        """`maximum` samples a batch, the last one short (dropped if `drop_last`); every sample in one batch when
        `maximum` is None (a token budget's lengths being measured)."""

        if maximum is None:
            return cls([(0, count)] if count else [], int(bool(count)))
        bounds = [(start, min(start + maximum, count)) for start in range(0, count, maximum)]
        short = bool(bounds) and bounds[-1][1] - bounds[-1][0] < maximum
        return cls(bounds, len(bounds) - int(drop_last and short))


def _lengths(name: str, sequences: Mapping[str, Any], bounds: Mapping[str, Any], count: int) -> dict[str, Any]:
    """Each sample's length in input `name`, as upstream measures it for batching (``_measure_values``, before
    any target changes the input): its tokens (a token input, flattened or grouped), else its rows (events; 1 for
    an entity's columns); for a grouped token input also its events and its longest event (host int64)."""

    import cupy
    import numpy as np

    if name in sequences:
        built = sequences[name][0]
        if isinstance(built, _Nested):
            row_bounds = cupy.asarray(bounds[name])
            per_event = cupy.diff(built.tokens.bounds)
            owner = cupy.searchsorted(row_bounds[1:], cupy.arange(per_event.size, dtype=cupy.int64), side="right")
            longest = cupy.zeros(count, dtype=cupy.int64)
            if per_event.size:
                cupy.maximum.at(longest, owner, per_event)
            return {"total": cupy.diff(built.tokens.bounds[row_bounds]).get(), "events": np.diff(bounds[name]),
                    "longest": longest.get()}
        return {"total": np.diff(built.bounds)}
    if name in bounds:
        return {"total": np.diff(bounds[name])}
    return {"total": np.ones(count, dtype=np.int64)}


def plan_batches(lengths: dict[str, Any], layout: Any, groups: list[tuple[int, int]], maximum: int | None,
                 max_tokens: int | None, bucket: bool, drop_last: bool) -> tuple[Any, BatchPlan]:
    """Upstream's batch composition (``_compose_batches``) over each group of samples (a chunk of materialized
    anchors, or every request): ordered by length when `bucket` (stable), then each batch takes samples while it
    holds fewer than `maximum` and its cost stays within `max_tokens`: the length input's layout prices it,
    packed as its total length, nested with packed fields as its tokens, nested with padded fields as its
    events times its longest event, else as its samples times its longest. Returns the samples' new order and
    the plan; `drop_last` drops a short last batch of the last group."""

    import numpy as np

    total = np.asarray(lengths["total"], dtype=np.int64)
    costs = _pricer(lengths, layout)
    order, bounds, failure, failure_at = [], [], None, -1
    for number, (low, high) in enumerate(groups):
        members = np.arange(low, high, dtype=np.int64)
        if bucket:
            members = members[np.argsort(total[members], kind="stable")]
        start = len(np.concatenate(order)) if order else 0
        order.append(members)
        position = 0
        while position < members.size:
            reach = members.size - position if maximum is None else min(maximum, members.size - position)
            take = reach
            if max_tokens is not None:
                window = 1024 if maximum is None else reach
                while True:
                    span = min(window, members.size - position)
                    priced = costs(members[position:position + span])
                    over = np.flatnonzero(priced > max_tokens)
                    if over.size or span == members.size - position or maximum is not None:
                        take = min(int(over[0]) if over.size else span, reach)
                        break
                    window *= 2
                if take == 0:  # upstream refuses a sample the budget cannot hold, when its chunk is batched
                    if failure is None:
                        failure, failure_at = "one sample exceeds Batching.max_tokens", len(bounds)
                    take = 1
            bounds.append((start + position, start + position + take))
            position += take
    emit = len(bounds)
    if drop_last and maximum is not None and bounds and bounds[-1][1] - bounds[-1][0] < maximum:
        emit -= 1
    if failure is not None:  # upstream composes a whole chunk before yielding any of it
        failure_at = next(index for index, (begin, _) in enumerate(bounds)
                          if begin >= _group_start(groups, bounds[failure_at][0], order))
    return (np.concatenate(order) if order else np.zeros(0, dtype=np.int64)), BatchPlan(bounds, emit, failure,
                                                                                        failure_at)


def _pricer(lengths: dict[str, Any], layout: Any) -> Any:
    """What a batch of samples costs under the length input's `layout` (upstream's ``_batch_cost``): for
    samples ``chosen``, the cost of a batch of ``chosen[:k + 1]``, for each k."""

    import numpy as np

    from kumo.pytorch.contract.batch.layouts import Nested, Packed

    total = np.asarray(lengths["total"], dtype=np.int64)
    if isinstance(layout, Nested) and not isinstance(layout.fields, Packed):  # its events times its longest
        width, height = np.asarray(lengths["events"], dtype=np.int64), np.asarray(lengths["longest"], dtype=np.int64)
        return lambda chosen: np.cumsum(width[chosen]) * np.maximum.accumulate(height[chosen])
    if isinstance(layout, (Packed, Nested)):  # its tokens
        return lambda chosen: np.cumsum(total[chosen])
    return lambda chosen: np.arange(1, chosen.size + 1) * np.maximum.accumulate(total[chosen])  # samples x longest


def length_summary(lengths: dict[str, Any], layout: Any, policy: Any) -> dict[str, Any] | None:
    """Upstream's summary of the fitted samples' lengths in ``Batching.length_input`` (``_length_summary``,
    the manifest's ``lengths``): how many, their mean and percentiles (nearest), and the share of padding when
    they are batched by length under the contract's ``max_samples`` and ``max_tokens``. None without a length
    input or a sample."""

    import numpy as np

    total = np.asarray(lengths["total"], dtype=np.int64)
    if policy.length_input is None or not total.size:
        return None
    order, plan = plan_batches(lengths, layout, [(0, int(total.size))], policy.max_samples, policy.max_tokens, True,
                               False)
    costs = _pricer(lengths, layout)
    capacity = sum(int(costs(order[begin:end])[-1]) for begin, end in plan.bounds)
    actual = int(total.sum())
    return {
        "input": policy.length_input,
        "count": int(total.size),
        "mean": float(np.mean(total)),
        **{f"p{percentile}": int(np.percentile(total, percentile, method="nearest"))
           for percentile in (0, 25, 50, 75, 90, 95, 100)},
        "expected_padded_fraction": float(capacity - actual) / capacity if capacity else 0.0,
    }


def _group_start(groups: list[tuple[int, int]], position: int, order: list[Any]) -> int:
    """Where (in the new order) the group holding new position `position` starts."""

    start = 0
    for members in order:
        if position < start + members.size:
            return start
        start += members.size
    return start
