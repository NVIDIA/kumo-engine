# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Progress events reported by `Trainer.fit`."""

import hashlib
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    TypeVar,
    cast,
)

from kumo_core.instrumentation import NoOpTrainingInstrumentation

logger = logging.getLogger(__name__)
T = TypeVar('T')

EventName = Literal['fit_started', 'trial_started', 'epoch_ended',
                    'trial_ended', 'fit_ended']


@dataclass(frozen=True)
class Event:
    r"""One progress event. ``trial`` and ``epoch`` count from 1."""

    name: EventName
    run_id: str
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


def call_each(targets: List[T], call: Callable[[T], Any], what: str,
              dropped: Optional[Callable[[T], None]] = None) -> List[Any]:
    values = []
    for target in list(targets):
        try:
            values.append(call(target))
        except Exception:
            logger.warning('%s %r failed; it is not called again in this run',
                           what, target, exc_info=True)
            targets.remove(target)
            if dropped is not None:
                dropped(target)
    return values


class EventEmitter(NoOpTrainingInstrumentation):
    def __init__(self, run_id: str,
                 callbacks: Sequence[Callable[[Event], None]]) -> None:
        self.run_id = run_id
        self._callbacks = list(callbacks)

    def emit(self, name: EventName, *, trial: Optional[int] = None,
             epoch: Optional[int] = None,
             metrics: Optional[Mapping[str, float]] = None,
             params: Optional[Mapping[str, Any]] = None) -> None:
        event = Event(name=name, run_id=self.run_id, time=time.time(),
                      trial=trial, epoch=epoch,
                      metrics=_finite_floats(metrics or {}),
                      params=dict(params or {}))
        call_each(self._callbacks, lambda callback: callback(event),
                  'Progress callback')

    def event(self, name: str, **fields: Any) -> None:
        self.emit(cast(EventName, name), **fields)
