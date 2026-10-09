# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""The execution backend for the GNN path.

:func:`cloud_backend` is the single lookup used by
:class:`~kumoai_studio.graph.Table`, :class:`~kumoai_studio.graph.Graph`,
:class:`~kumoai_studio.pquery.PredictiveQuery`, and
:class:`~kumoai_studio.trainer.Trainer`. It always returns the Kumo cloud
platform backend.
"""

import threading
from typing import List, Optional

from kumoai_studio.backend.base import (
    ArtifactExportBackend,
    Backend,
    ConnectorBackend,
    GraphBackend,
    IdentityBackend,
    JobBackend,
    PQueryBackend,
    PredictionBackend,
    PredictionTableBackend,
    TableBackend,
    TrainingBackend,
    TrainTableBackend,
)
from kumoai_studio.backend.cloud import CloudBackend, RestSourceBackend

#: Snowflake, Databricks, BigQuery, S3 and GCS are absent because none of
#: them live here any more: ``kumo_snowflake``, ``kumo_databricks``,
#: ``kumo_bigquery``, ``kumo_s3`` and ``kumo_gcs`` each ship their own reader
#: and sink and register them by entry point. A caller naming one imports it
#: from there.
__all__ = [
    'ArtifactExportBackend',
    'Backend',
    'CloudBackend',
    'ConnectorBackend',
    'GraphBackend',
    'IdentityBackend',
    'JobBackend',
    'PQueryBackend',
    'PredictionBackend',
    'PredictionTableBackend',
    'RestSourceBackend',
    'TableBackend',
    'TrainTableBackend',
    'TrainingBackend',
    'cloud_backend',
]


def __dir__() -> List[str]:
    return sorted(__all__)


_lock = threading.Lock()
_backend: Optional[CloudBackend] = None


def cloud_backend() -> CloudBackend:
    r"""Returns the process-wide :class:`~kumoai_studio.backend.cloud.CloudBackend`.

    Constructed lazily and holds no connection state of its own, so calling
    this before :func:`kumoai_studio.init` is safe; the error surfaces on
    first use.
    """
    global _backend
    if _backend is None:
        with _lock:
            if _backend is None:
                _backend = CloudBackend()
    return _backend
