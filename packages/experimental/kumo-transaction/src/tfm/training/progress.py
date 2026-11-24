# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Training progress: one formatted line every `log_every` steps, and the per-step curve.

Every backend feeds the same object, so a run reads the same on one GPU or
several. Only the first rank prints; every rank keeps its own curve. Losses
are the rank's own minibatch losses, so on several GPUs the printed loss is
the first GPU's.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from tfm.training.distributed import distributed_info


def _duration(seconds: float) -> str:
    seconds = max(0, round(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m{secs:02d}s"


def _print(line: str) -> None:
    print(line, flush=True)


@dataclass
class TrainingProgress:
    """Receives every step from a trainer; prints and records it.

    `log_every=None` prints nothing but still records the curve. `rank`
    defaults to the process's rank in `torch.distributed` at `start`.
    """

    log_every: int | None = None
    write: Callable[[str], None] = _print
    rank: int | None = None
    curve: list[dict[str, float]] = field(default_factory=list)
    _max_steps: int = 0
    _batches_per_epoch: int = 1
    _started: float = 0.0
    _interval_tokens: float = 0.0
    _interval_started: float = 0.0

    def __post_init__(self) -> None:
        if self.log_every is not None and self.log_every < 1:
            raise ValueError(f"log_every must be >= 1, got {self.log_every}.")

    @property
    def printing(self) -> bool:
        return self.log_every is not None and self.rank == 0

    def start(self, *, max_steps: int, batches_per_epoch: int, world_size: int = 1) -> None:
        if self.rank is None:
            self.rank = distributed_info()[1]
        self._max_steps = max_steps
        self._batches_per_epoch = max(1, batches_per_epoch)
        self._started = self._interval_started = time.perf_counter()
        self._interval_tokens = 0.0
        if self.printing:
            gpus = f" on {world_size} processes" if world_size > 1 else ""
            self.write(
                f"training {max_steps:,} steps{gpus}, {batches_per_epoch:,} minibatches per epoch"
                + (" per process" if world_size > 1 else "")
            )

    def step(self, step: int, *, loss: float, lr: float, grad_norm: float, tokens: float) -> None:
        """Record `step` (1-based); print it when due."""
        now = time.perf_counter()
        self._interval_tokens += tokens
        self.curve.append(
            {
                "step": float(step),
                "epoch": step / self._batches_per_epoch,
                "loss": loss,
                "lr": lr,
                "grad_norm": grad_norm,
                "tokens": tokens,
                "seconds": now - self._started,
            }
        )
        due = self.log_every is not None and (step % self.log_every == 0 or step == self._max_steps)
        if not (self.printing and due):
            return
        elapsed = now - self._started
        rate = self._interval_tokens / max(now - self._interval_started, 1e-9)
        left = elapsed / step * (self._max_steps - step)
        width = len(f"{self._max_steps:,}")
        grad = f" | grad norm {grad_norm:.3f}" if math.isfinite(grad_norm) else ""
        self.write(
            f"step {step:>{width},}/{self._max_steps:,} | epoch {step / self._batches_per_epoch:.2f}"
            f" | loss {loss:.4f} | lr {lr:.2e}{grad}"
            f" | {rate / 1e3:,.0f}k tok/s | {_duration(elapsed)} elapsed | ~{_duration(left)} left"
        )
        self._interval_tokens = 0.0
        self._interval_started = now


__all__ = ["TrainingProgress"]
