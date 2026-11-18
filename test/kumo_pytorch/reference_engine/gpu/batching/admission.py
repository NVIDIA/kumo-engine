# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Refusing a batch call whose widest padded batch cannot fit on the GPU, before batches are handed out.

A padded layout lays each sample's events out as wide as the longest history in
its batch, as upstream pads them: ``[samples, longest]`` per padded array. With
no history limit, one long history makes one batch need more GPU memory than
the data leaves; MBD ``transaction-forecast``'s widest batch of 128 holds
46.9 GiB. Rather than run out of memory part way through delivery, a call
first counts each anchor's events (`event_counts`: the sampler's row selection,
reading no fields), works out every batch's width, and refuses the call when
the widest batch's padded arrays alone take more than the GPU has free beside
the data and the model (`admitted`). An out-of-memory error while a batch is
built is reported the same way, with the batch, its widths and the memory.

Counted: event inputs (``EventSource``, ``EventsOf``) whose events are padded
(``Padded``, or ``Nested`` with padded events) and whose events no field value
drops. Flattened histories, value-dependent event selection and graphs are not
counted before the first batch; a batch of theirs that does not fit fails
while it is built, with `BatchMemoryError`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from test.kumo_pytorch.reference_engine.gpu.sampling.sampler import Sampler

    from kumo.pytorch.spec import DataLoaderSpec

__all__ = ["Admission", "BatchMemoryError", "admitted", "event_counts", "padded_event_inputs"]

#: Bytes a padded slot takes at least, before the first batch shows what it takes: its validity and one byte.
MIN_SLOT_BYTES = 2

REMEDY = ("Bound each history (EventSource.max_events, or a limit on its event rows), use smaller batches, or "
          "lay the events out Packed; a padded batch is as wide as its longest history, as upstream pads it")


class BatchMemoryError(MemoryError):
    """A batch the GPU cannot hold: refused before delivery, or out of memory while it was built."""


def padded_event_inputs(contract: DataLoaderSpec) -> dict[str, str]:
    """The inputs laid out ``[samples, events]``, each mapped to the ``EventSource`` input whose events it lays
    out, when the row selection gives each sample's event count exactly (no field value drops events)."""

    from kumo.pytorch.contract.batch import (
        Composite,
        EventsOf,
        EventSource,
        Flatten,
    )
    from kumo.pytorch.contract.batch.layouts import Nested, Padded

    inputs = contract.batch.inputs
    result = {}
    for name, value in inputs.items():
        parent = value.source.input if isinstance(value.source, EventsOf) else name
        if parent not in inputs or not isinstance(inputs[parent].source, EventSource):
            continue
        if isinstance(getattr(value.representation, "history", None), Flatten):
            continue
        layout = value.layout
        if not (isinstance(layout, Padded) or (isinstance(layout, Nested) and isinstance(layout.events, Padded))):
            continue
        row = getattr(inputs[parent].representation, "row", None)
        if (isinstance(row, Composite) and row.required_fields) or getattr(row, "missing", None) == "skip_event":
            continue
        result[name] = parent
    return result


def event_counts(contract: DataLoaderSpec, sampler: Sampler, entity_rows: Any, cutoffs: Any,
                 parents: Sequence[str]) -> dict[str, Any]:
    """Each anchor's events in each ``EventSource`` input of `parents` (CuPy int64), as
    `test.kumo_pytorch.reference_engine.gpu.sampling.events.event_inputs` selects them: every component's selected rows, each
    component's ``limit`` kept, summed, then the input's ``max_events`` kept."""

    from test.kumo_pytorch.reference_engine.gpu.sampling.reductions import (
        _counts,
    )

    import cupy

    count = int(entity_rows.size)
    selected: dict[tuple[str, str], Any] = {}
    result = {}
    for parent in parents:
        source = contract.batch.inputs[parent].source
        collection = "events" if source.collection == "history" else "current_events"
        total = cupy.zeros(count, dtype=cupy.int64)
        for component in source.events:
            key = (collection, component)
            if key not in selected:
                spec = getattr(contract.sample, collection)[component]
                rows = sampler.select_rows(spec.rows, entity_rows, cutoffs, [])
                counts = _counts(rows.owner, count)
                selected[key] = counts if spec.limit is None else cupy.minimum(counts, spec.limit)
            total += selected[key]
        result[parent] = total if source.max_events is None else cupy.minimum(total, source.max_events)
    return result


@dataclass
class Admission:
    """A call's batches as the check sees them: each one's samples and each counted input's width (host
    int64 arrays, one entry a batch), the GPU memory free for a batch, and what the data keeps beside it."""

    sizes: Any
    widths: dict[str, Any]
    available: int
    resident: int = 0

    @classmethod
    def of(cls, inputs: Mapping[str, str], counts: Mapping[str, Any], bounds: Sequence[tuple[int, int]],
           available: int, resident: int = 0) -> Admission | None:
        """From each counted parent's per-sample `counts` in the call's order and the batches' `bounds`."""

        import numpy as np

        if not bounds:
            return None
        starts = np.asarray([start for start, _ in bounds], dtype=np.int64)
        stops = np.asarray([stop for _, stop in bounds], dtype=np.int64)
        widths = {}
        for name, parent in inputs.items():
            values = counts[parent].get() if hasattr(counts[parent], "get") else np.asarray(counts[parent])
            ends = np.concatenate([values, [0]])  # reduceat needs a start inside the array
            widths[name] = np.maximum.reduceat(ends, starts)[: len(bounds)] if values.size else \
                np.zeros(len(bounds), dtype=np.int64)
            widths[name][stops == starts] = 0
        return cls(stops - starts, widths, int(available), int(resident))

    def bytes(self, slots: Mapping[str, float]) -> Any:
        """Each batch's padded bytes at `slots` bytes a slot per input."""

        import numpy as np

        total = np.zeros(self.sizes.size, dtype=np.float64)
        for name, widths in self.widths.items():
            total += self.sizes * widths * slots.get(name, MIN_SLOT_BYTES)
        return total

    def check(self, slots: Mapping[str, float]) -> None:
        """Refuse when the widest batch's padded arrays take more than the memory free for a batch."""

        import numpy as np

        needed = self.bytes(slots)
        worst = int(np.argmax(needed))
        if needed[worst] > self.available:
            widths = ", ".join(f"{name} {int(widths[worst]):,}" for name, widths in self.widths.items())
            raise BatchMemoryError(
                f"batch {worst + 1} of {self.sizes.size} ({int(self.sizes[worst])} samples) is padded to its longest "
                f"history ({widths} events wide), so its padded arrays alone take {needed[worst] / 2**30:.1f} GiB, "
                f"more than the {self.available / 2**30:.1f} GiB the GPU has free beside the data"
                + (f" ({self.resident / 2**30:.1f} GiB of partitions)" if self.resident else "")
                + f" and the model. {REMEDY}.")

    def describe(self, number: int) -> str:
        if number >= self.sizes.size:
            return ""
        widths = ", ".join(f"{name} {int(widths[number]):,}" for name, widths in self.widths.items())
        return f"; it is {widths} events wide"


def measured_slots(batch: Any, contract: DataLoaderSpec, admission: Admission) -> dict[str, float]:
    """Bytes a padded slot of each counted input takes in `batch`, the call's first one: its ``[samples,
    width]`` tensors (the ``sequence`` of a ``Nested`` layout) and those of the targets laid out beside it."""

    import torch

    from kumo.pytorch.contract.batch import AlignedEventTarget
    from kumo.pytorch.contract.batch.layouts import Nested

    count = int(admission.sizes[0])
    slots = {}
    for name, widths in admission.widths.items():
        width = int(widths[0])
        if not count or not width:
            continue
        value = batch.inputs.get(name, {})
        if isinstance(contract.batch.inputs[name].layout, Nested):
            value = value.get("sequence", {}) if isinstance(value, dict) else {}
        found = [value]
        found += [batch.targets.get(target) for target, spec in contract.batch.targets.items()
                  if isinstance(spec, AlignedEventTarget) and spec.input == name]
        total = 0
        stack = list(found)
        while stack:
            item = stack.pop()
            if isinstance(item, dict):
                stack.extend(item.values())
            elif isinstance(item, torch.Tensor) and tuple(item.shape) == (count, width):
                total += item.numel() * item.element_size()
        if total:
            slots[name] = max(total / (count * width), MIN_SLOT_BYTES)
    return slots


def admitted(contract: DataLoaderSpec, batches: Callable[[], Iterator[Any]],
             admission: Callable[[], Admission | None]) -> Iterator[Any]:
    """`batches`' batches, refused before the first is handed out when `admission` (worked out first) says
    the widest cannot fit; an out-of-memory error while one is built becomes a `BatchMemoryError`."""

    from test.kumo_pytorch.reference_engine.gpu.data.memory import free_bytes

    import torch

    checked = admission()
    if checked is not None:
        checked.check({})  # the least each slot takes: refused before building anything
    number = 0
    iterator = batches()
    while True:
        try:
            batch = next(iterator)
        except StopIteration:
            return
        except BatchMemoryError:
            raise
        except (MemoryError, torch.OutOfMemoryError) as error:
            raise BatchMemoryError(
                f"the GPU ran out of memory building batch {number + 1}"
                + (f" of {checked.sizes.size} ({int(checked.sizes[number])} samples){checked.describe(number)}"
                   if checked is not None and number < checked.sizes.size else "")
                + f"; {free_bytes() / 2**30:.1f} GiB is free now, {torch.cuda.memory_allocated() / 2**30:.1f} GiB "
                f"held by PyTorch. {REMEDY}.") from error
        if number == 0 and checked is not None:
            checked.check(measured_slots(batch, contract, checked))
        yield batch
        number += 1
