# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Boundary between Kumo-Python and the DiskGraph runtime.

Usage::

    from kumo.diskgraph import get_service

    service = get_service()
    local_path = service.local_snapshot_path(handle)

See :mod:`kumo.diskgraph.service` for the implementation.
"""

import torch

import service_framework.deploy_env_config as deploy_env_config
from kumo.diskgraph.service import (
    DiskgraphHandle,
    DiskgraphService,
    IngestResult,
    MaterializeResult,
    PredTableResult,
    TrainTableResult,
)
from service_framework._raw_deploy_config import DiskgraphConfig
from service_framework.env_var import EnvVar

# PyTorch patch to reduce dataloader wait time when killing hanging threads:
torch.utils.data._utils.MP_STATUS_CHECK_INTERVAL = 0.1

# Loading-time budget for graph engine startup.
_MAX_LOADING_TIME = EnvVar(
    'GRAPH_ENGINE_MAX_LOADING_TIME_SECONDS', type=int, required=False,
    default_value=2 * 60 * 60, description='Maximum loading time budget '
    '(seconds) propagated to the k8s job manifest.')

# Per-edge-type cap enforced at ingest-time downsampling and at job
# pre-flight sizing.
MAX_EDGES_PER_EDGE_TYPE = 2 * 10**9

__all__ = [
    'DiskgraphHandle',
    'DiskgraphService',
    'IngestResult',
    'MaterializeResult',
    'MAX_EDGES_PER_EDGE_TYPE',
    'PredTableResult',
    'TrainTableResult',
    'get_service',
]


def get_service() -> DiskgraphService:
    """Return a :class:`DiskgraphService` bound to the active deploy config.

    Plain factory — no process-wide caching. Construction is trivial
    (stores a config reference), so factory-level caching is redundant
    and would only make test reset harder.
    """
    data_compute = deploy_env_config.deploy_env().data_compute
    if not isinstance(data_compute, DiskgraphConfig):
        raise RuntimeError(
            'get_service() called in non-DiskGraph mode; '
            f'data_compute is {type(data_compute).__name__}', )
    return DiskgraphService(data_compute)
