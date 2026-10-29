# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Progress events reported by `Trainer.fit` and `Graph.snapshot`."""

import hashlib
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Literal,
    Mapping,
    Optional,
    Sequence,
    cast,
)

from kumo_core.instrumentation import NoOpTrainingInstrumentation

logger = logging.getLogger(__name__)

EventName = Literal['fit_started', 'trial_started', 'epoch_ended',
                    'trial_ended', 'fit_ended', 'snapshot_started',
                    'table_staged', 'snapshot_ended']


@dataclass(frozen=True)
class Event:
    r"""One progress event. ``trial`` and ``epoch`` count from 1."""

    name: EventName
    time: float
    trial: Optional[int] = None
    epoch: Optional[int] = None
    metrics: Mapping[str, float] = field(default_factory=dict)
    params: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def query_sha256(query: str) -> str:
    return hashlib.sha256(query.encode()).hexdigest()


def _finite_floats(metrics: Mapping[str, Any]) -> Dict[str, float]:
    result = {}
    for key, value in metrics.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            result[key] = number
    return result


class EventEmitter(NoOpTrainingInstrumentation):
    def __init__(self, callbacks: Sequence[Callable[[Event], None]]) -> None:
        self._callbacks = list(callbacks)

    def emit(self, name: EventName, *, trial: Optional[int] = None,
             epoch: Optional[int] = None,
             metrics: Optional[Mapping[str, float]] = None,
             params: Optional[Mapping[str, Any]] = None) -> None:
        event = Event(name=name, time=time.time(), trial=trial, epoch=epoch,
                      metrics=_finite_floats(metrics or {}),
                      params=dict(params or {}))
        for callback in list(self._callbacks):
            try:
                callback(event)
            except Exception:
                logger.warning(
                    'Progress callback %r failed; it is disabled for the '
                    'rest of this call', callback, exc_info=True)
                self._callbacks.remove(callback)

    def event(self, name: str, **fields: Any) -> None:
        self.emit(cast(EventName, name), **fields)
