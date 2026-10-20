# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime versions and scheduler job identity recorded in manifests."""

import json
import os
import platform
import re
import subprocess
from functools import cache
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Optional

PROCESSING_JOB_CONFIG = Path('/opt/ml/config/processingjobconfig.json')


def _version(distribution: str) -> Optional[str]:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


@cache
def _build_commit(version: str) -> Optional[str]:
    if '+' in version:
        return version.split('+', 1)[1]
    root = Path(__file__).resolve().parents[2]
    if root.name in ('site-packages', 'dist-packages'):
        return None
    command = ['git', '-C', str(root), 'rev-parse', '--short', 'HEAD']
    try:
        return subprocess.run(command, capture_output=True, text=True,
                              check=True, timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def environment() -> Dict[str, Optional[str]]:
    from kumo.engine._version import __version__

    return {
        'kumo_engine': __version__,
        'build_commit': _build_commit(__version__),
        'kumo_core': _version('kumo-core'),
        'torch': _version('torch'),
        'python': platform.python_version(),
    }


def _context(platform_name: str, job_id: Optional[str],
             job_run_id: Optional[str], source: str) -> Dict[str, Any]:
    return {
        'platform': platform_name,
        'job_id': job_id,
        'job_run_id': job_run_id,
        'source': source,
    }


def _processing_job() -> Optional[Dict[str, Any]]:
    try:
        config = json.loads(PROCESSING_JOB_CONFIG.read_text())
    except (OSError, ValueError):
        return None
    return config if isinstance(config, dict) else None


def _detect() -> Optional[Dict[str, Any]]:
    env = os.environ
    if env.get('TRAINING_JOB_NAME'):
        return _context('sagemaker', env['TRAINING_JOB_NAME'],
                        env.get('TRAINING_JOB_ARN'), 'auto')
    processing = _processing_job()
    if processing is not None and processing.get('ProcessingJobName'):
        return _context('sagemaker', processing['ProcessingJobName'],
                        processing.get('ProcessingJobArn'), 'auto')
    service = env.get('SNOWFLAKE_SERVICE_NAME')
    if service:
        return _context('snowflake', re.sub(r'_[A-Za-z0-9]{8}$', '', service),
                        service, 'auto')
    if env.get('DATABRICKS_RUNTIME_VERSION'):
        return _context('databricks', None, None, 'auto')
    return None


def scheduler_context() -> Optional[Dict[str, Any]]:
    env = os.environ
    scheduler = env.get('KUMO_SCHEDULER')
    job_id = env.get('KUMO_JOB_ID')
    job_run_id = env.get('KUMO_JOB_RUN_ID')
    detected = _detect()
    if scheduler or job_id or job_run_id:
        if not scheduler:
            scheduler = detected['platform'] if detected else 'other'
        return _context(scheduler, job_id, job_run_id, 'env')
    return detected
