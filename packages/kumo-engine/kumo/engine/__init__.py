# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Local, in-process relational deep learning with Kumo.

No `Backend` protocol, no `set_backend`, no remote/hosted mode, no
hosted project.
There is nothing to call analogous to the cloud SDK's
`kumoai_studio.init(url=..., api_key=...)`: connectors and tables are constructed
and threaded through directly (`Table`, `Graph`, `Connector` subclasses), and
`Graph`/`Trainer` objects carry their own scratch-directory identity, so no
process-wide setup call is *required* before use. `configure(...)` is the one
*optional* one -- see `kumo.engine.config` -- for pointing the shared scratch-
directory root (DiskGraph ingest, GNN training) at, e.g., a local NVMe SSD
mount.
"""

from kumo_core.embedding_models import (
    EmbeddingModelStore,
    GloVeVectors,
    TokenEmbeddingModel,
)
from kumo_core.utils.nofile import ensure_nofile
from kumoapi.graph import Edge
from kumoapi.plugin import ModelRegistryLocation
from kumoapi.table import Column

from kumo.engine._version import __version__
from kumo.engine.config import configure
from kumo.engine.connector import (
    BigQueryConnector,
    BigQueryIdentity,
    Connector,
    DatabricksConnector,
    DatabricksIdentity,
    FileConnector,
    GCSConnector,
    GCSIdentity,
    Identity,
    S3Connector,
    S3Identity,
    SnowflakeConnector,
    SnowflakeIdentity,
    SourceTable,
    SQLSourceTable,
)
from kumo.engine.graph import Graph, GraphSnapshot
from kumo.engine.pquery import PredictionTable, PredictiveQuery, TrainingTable
from kumo.engine.table import Table
from kumo.engine.trainer import (
    ModelOutputConfig,
    PredictionOutputConfig,
    PredictionResult,
    Trainer,
    TrainingResult,
    TrainingTableOutputConfig,
)

__all__ = [
    "__version__",
    "configure",
    "EmbeddingModelStore",
    "GloVeVectors",
    "TokenEmbeddingModel",
    "ModelRegistryLocation",
    "Connector",
    "FileConnector",
    "SnowflakeConnector",
    "S3Connector",
    "GCSConnector",
    "BigQueryConnector",
    "DatabricksConnector",
    "SourceTable",
    "SQLSourceTable",
    "Identity",
    "SnowflakeIdentity",
    "S3Identity",
    "GCSIdentity",
    "BigQueryIdentity",
    "DatabricksIdentity",
    "Column",
    "Table",
    "Edge",
    "Graph",
    "GraphSnapshot",
    "PredictiveQuery",
    "TrainingTable",
    "PredictionTable",
    "Trainer",
    "TrainingResult",
    "PredictionResult",
    "ModelOutputConfig",
    "PredictionOutputConfig",
    "TrainingTableOutputConfig",
]

ensure_nofile()
