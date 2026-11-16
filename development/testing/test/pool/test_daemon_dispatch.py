# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for controller-side dispatch to pool worker queues."""
from types import SimpleNamespace
from typing import cast

from pool.daemon_dispatch import _task_queue_for_config
from service_framework.deploy_env_config import DeployEnvConfig


def test_singlehost_snapshot_deleter_uses_instance_scoped_queue():
    config = SimpleNamespace(
        data_compute=SimpleNamespace(pool_strategy='singlehost'),
        task_queue=lambda queue: f'{queue}-devserver-train-a',
    )

    queue_for_worker = _task_queue_for_config(cast(DeployEnvConfig, config))

    assert queue_for_worker('singlehost') == (
        'diskgraph-daemon-tq-devserver-train-a')
