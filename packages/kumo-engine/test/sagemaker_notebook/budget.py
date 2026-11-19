# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Leave ten minutes of the installed one-hour job limit for evidence."""
import json
import math
import signal
import time
from contextlib import contextmanager
from pathlib import Path

WORK_SECONDS = 3000


def remaining(root, maximum=None):
    deadline = json.loads((Path(root) / 'execution-deadline.json').read_text())
    if not isinstance(deadline, (int, float)) or not math.isfinite(deadline):
        raise ValueError('Invalid notebook execution deadline')
    seconds = deadline - time.monotonic()
    if seconds <= 0:
        raise TimeoutError(
            'Notebook work exhausted its evidence-reserving budget')
    return min(seconds, maximum) if maximum is not None else seconds


@contextmanager
def limit(root):
    seconds = remaining(root)
    if signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
        raise RuntimeError('Notebook kernel already has an execution timer')

    def expired(*_):
        raise TimeoutError(
            'Notebook work exhausted its evidence-reserving budget')

    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
