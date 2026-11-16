"""Adapters supply work and semantic counts; the runner never inspects payloads."""

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BatchCounts:
    """Actual occurrences, not budgets. None means not measured, never zero.

    Tokens include input values/markers/BPE pieces, excluding tensor padding,
    targets and parallel field IDs. Events/nodes/edges count emitted occurrences.
    Output elements/bytes describe the adapter's documented emitted payload.
    """

    samples: int
    padding_samples: int
    tokens: int | None = None
    nodes: int | None = None
    edges: int | None = None
    events: int | None = None
    output_elements: int | None = None
    output_bytes: int | None = None

    def __post_init__(self):
        for name, value in vars(self).items():
            if value is None and name not in {"samples", "padding_samples"}:
                continue
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.padding_samples > self.samples:
            raise ValueError("padding_samples exceeds emitted samples")


def _nothing():
    pass


def _before_epoch(epoch, recorder):
    pass


@dataclass
class Workload:
    """One configured experiment; no dataset, backend or model name dispatch.

    ``prepare(recorder)`` owns source/index/fitting instrumentation. ``batches``
    returns the complete selected epoch. ``count`` must be read-only and exact.
    Optional ``step(batch, recorder)`` returns whether an optimizer update ran.
    A device adapter must supply ``synchronize`` for completed-window timing.
    ``close`` must tolerate partially failed preparation and release resources.

    Factories create fresh objects per repetition; experiments requiring fresh
    OS processes or controlled caches must arrange those conditions explicitly.
    Inapplicability requires a per-metric semantic reason, not a missing probe.
    """

    identity: Mapping[str, Any]
    prepare: Callable[[Any], None]
    batches: Callable[[int], Iterable[Any]]
    count: Callable[[Any], BatchCounts]
    step: Callable[[Any, Any], bool] | None = None
    synchronize: Callable[[], None] = _nothing
    close: Callable[[], None] = _nothing
    inapplicable: Mapping[str, str] = field(default_factory=dict)
    device: str = "cpu"
    before_epoch: Callable[[int, Any], None] = _before_epoch

    def validate(self):
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("device must identify the execution device")
        if self.device != "cpu" and self.synchronize is _nothing:
            raise ValueError(
                "Device execution requires an explicit completion callback"
            )
