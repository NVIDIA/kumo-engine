# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep the last output of a failed ``kumo-auto`` or ``kumo-pred`` child.

Kubernetes pool placements run the child in an activity pod, and releasing
the placement deletes that pod together with its log. The activity therefore
saves a bounded tail of the child's output next to the job config. The ML
workflow worker reads it back, logs it, and uses it for the job's failure
message.

This module has no I/O so that workflow code can import it.
"""
from __future__ import annotations

import collections
import re
from typing import Optional

# Workflow patch that reads the saved output on the failure path.
POOL_FAILURE_OUTPUT_PATCH = 'pool-failure-output-v1'

MAX_LINES = 50
MAX_LINE_CHARS = 2000
MAX_CHARS = 16 << 10
MAX_SUMMARY_CHARS = 500

_EXCEPTION_LINE = re.compile(r'^(?:[A-Za-z_]\w*\.)*[A-Za-z_]\w*'
                             r'(?:Error|Exception|Exit|Interrupt)\b')


class OutputTail:
    """Bounded record of the last non-empty lines that a child printed."""
    def __init__(self, max_lines: int = MAX_LINES) -> None:
        self._lines: collections.deque[str] = collections.deque(
            maxlen=max_lines)

    def add(self, record: bytes) -> None:
        text = record.decode('utf-8', errors='replace').rstrip()
        if text.strip():
            self._lines.append(text[:MAX_LINE_CHARS])

    def text(self) -> str:
        return bound_text('\n'.join(self._lines))


def bound_text(text: str) -> str:
    """Keep the end of ``text``, where a child prints its error."""
    return text if len(text) <= MAX_CHARS else text[-MAX_CHARS:]


def failure_output_path(job_config_path: str, binary_name: str,
                        shard_index: Optional[int] = None) -> str:
    """Return where the pool saves the output of a failed child.

    The path sits next to the shared job config, so both the activity pod
    and the ML workflow worker can reach it.
    """
    shard = '' if shard_index is None else f'.shard-{shard_index:05d}'
    return f'{job_config_path}.{binary_name}{shard}.failure.txt'


def failure_summary(output: str) -> Optional[str]:
    """Return the output line that names the failure, bounded in length.

    A Python traceback ends with the exception line, so the last such line
    wins. Otherwise the last non-empty line is the best available summary.
    """
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        return None
    for line in reversed(lines):
        if _EXCEPTION_LINE.match(line):
            return line[:MAX_SUMMARY_CHARS]
    return lines[-1][:MAX_SUMMARY_CHARS]
