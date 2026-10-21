# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Progress events reported by `Trainer.fit`."""

import logging
import math
import time
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
)

logger = logging.getLogger(__name__)

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
        return {
            'name': self.name,
            'run_id': self.run_id,
            'time': self.time,
            'trial': self.trial,
            'epoch': self.epoch,
            'metrics': dict(self.metrics),
            'params': dict(self.params),
        }


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


class EventEmitter:
    def __init__(self, run_id: str,
                 callbacks: Sequence[Callable[[Event], None]]) -> None:
        self.run_id = run_id
        self._callbacks: List[Callable[[Event], None]] = list(callbacks)

    def emit(self, name: EventName, *, trial: Optional[int] = None,
             epoch: Optional[int] = None,
             metrics: Optional[Mapping[str, float]] = None,
             params: Optional[Mapping[str, Any]] = None) -> None:
        if not self._callbacks:
            return
        try:
            event = Event(name=name, run_id=self.run_id, time=time.time(),
                          trial=trial, epoch=epoch,
                          metrics=_finite_floats(metrics or {}),
                          params=dict(params or {}))
        except Exception:
            logger.warning('Could not build progress event %r', name,
                           exc_info=True)
            return
        for callback in list(self._callbacks):
            try:
                callback(event)
            except Exception:
                logger.warning(
                    'Progress callback %r failed; it is disabled for the '
                    'rest of run %s', callback, self.run_id, exc_info=True)
                self._callbacks.remove(callback)

    @property
    def listening(self) -> bool:
        return bool(self._callbacks)

    @property
    def core_sink(self) -> Optional[Callable[[Dict[str, Any]], None]]:
        if not self._callbacks:
            return None
        return self._emit_core

    def _emit_core(self, payload: Dict[str, Any]) -> None:
        try:
            self.emit(**payload)
        except Exception:
            logger.warning('Could not report progress event %r', payload,
                           exc_info=True)
