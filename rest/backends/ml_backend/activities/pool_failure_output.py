# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read the output that a failed pool child saved.

A Kubernetes pool placement runs ``kumo-auto`` or ``kumo-pred`` in an
activity pod, and releasing the placement deletes that pod with its log.
This activity runs on the long-lived ML workflow worker, so the output it
logs stays available after the activity pod is gone.
"""
import os.path as osp
from typing import Optional

from temporalio import activity

from io_utils import fsutil, logging
from pool.failure_output import bound_text

log = logging.getLogger(__name__)


@activity.defn
def read_pool_failure_output(job_id: str, paths: list[str]) -> Optional[str]:
    """Return and log the saved output of failed pool children.

    Missing or unreadable files are skipped because the output is diagnostic
    only. Returns ``None`` when no path holds any output.
    """
    sections = []
    for path in paths:
        try:
            if not fsutil.exists(path):
                continue
            text = bound_text(fsutil.read_text(path).strip())
        except Exception:
            log.warning(
                'Could not read the saved pool failure output %s for job %s',
                path, job_id, exc_info=True)
            continue
        if text:
            sections.append(text if len(paths) ==
                            1 else f'[{osp.basename(path)}]\n{text}')
    if not sections:
        return None
    output = bound_text('\n'.join(sections))
    log.error('Job `%s` failed in a pool subprocess. Last output:\n%s', job_id,
              output)
    return output
