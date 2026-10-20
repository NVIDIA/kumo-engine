# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Experiment tracking hooks for `Trainer.fit`."""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional, Protocol

from kumo.engine._events import Event

if TYPE_CHECKING:
    from kumo.engine.trainer.trainer import TrainingResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FitContext:
    r"""What a fit trains, known once its model plan is resolved."""

    run_id: str
    task_type: str
    snapshot_id: str
    training_table_id: str
    query: str
    model_plan: Mapping[str, Any]


class Tracking(Protocol):
    r"""Receives one `Trainer.fit`'s lifecycle. Errors are logged, never
    raised into the fit."""
    def start(self, context: FitContext) -> None:
        ...

    def trial_logger(self, trial: int, attempt: int) -> Optional[Any]:
        r"""A Lightning logger that also receives one search-trial attempt's
        metrics, or None."""
        ...

    def on_event(self, event: Event) -> None:
        ...

    def finish(self, result: Optional['TrainingResult'],
               error: Optional[BaseException]) -> None:
        ...


class TrackingSession:
    r"""Calls one fit's `Tracking` hooks, best-effort."""
    def __init__(self, tracking: Optional[Tracking]) -> None:
        self._tracking = tracking
        self._started = False

    @property
    def enabled(self) -> bool:
        return self._tracking is not None

    def start(self, context: Callable[[], FitContext]) -> None:
        if self._tracking is None:
            return
        self._started = True
        self._call(lambda: self._tracking.start(context()))

    def trial_logger(self, trial: int, attempt: int) -> Optional[Any]:
        return self._call(lambda: self._tracking.trial_logger(trial, attempt))

    def on_event(self, event: Event) -> None:
        if self._started:
            self._call(lambda: self._tracking.on_event(event))

    def finish(self, result: Optional['TrainingResult'],
               error: Optional[BaseException]) -> None:
        if self._started:
            self._call(lambda: self._tracking.finish(result, error))

    @staticmethod
    def _call(hook: Callable[[], Any]) -> Any:
        try:
            return hook()
        except Exception:
            logger.warning('Experiment tracking failed; the fit continues',
                           exc_info=True)
            return None
