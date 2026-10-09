# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from service_framework.env_var import EnvVar

# ML job environment variables ###############################################
ML_PRED_MAX_NUM_ENTITIES_PER_PARTITION = 'ML_PRED_MAX_NUM_ENTITIES_PER_PARTITION'  # noqa: E501
ML_PRED_MAX_NUM_ROWS_PER_PARTITION = 'ML_PRED_MAX_NUM_ROWS_PER_PARTITION'
# The max number of processes for the data backend worker. One process consumes
# about 10G of memory maximal.
MAX_DATA_BACKEND_WORKER_PROCESSES = 'MAX_DATA_BACKEND_WORKER_PROCESSES'

# Data Platform Env variables #################################################
# refresh timeout used for whole graph and individual table refreshes
DATA_BACKEND_REFRESH_TIMEOUT_HOURS = EnvVar(
    name="DATA_BACKEND_REFRESH_TIMEOUT",
    type=int,
    default_value=7,
    description="Data backend refresh timeout in hours",
).parse_value()
DATA_BACKEND_INGEST_TIMEOUT_HOURS = EnvVar(
    name="DATA_BACKEND_INGEST_TIMEOUT",
    type=int,
    default_value=7,
    description="Data backend ingest timeout in hours",
).parse_value()
DATA_BACKEND_IO_TIMEOUT_HOURS = 1
# Native DiskGraph ingestion previously used a hard-coded four-hour timeout.
# Keep its shorter activity budget independently configurable while the
# enclosing workflow retains the existing ingest timeout.
DATA_BACKEND_DISKGRAPH_WORKFLOW_TIMEOUT_HOURS = EnvVar(
    name="DATA_BACKEND_DISKGRAPH_WORKFLOW_TIMEOUT",
    type=int,
    default_value=6,
    description="DiskGraph native ingestion activity timeout in hours",
).parse_value()
if (DATA_BACKEND_DISKGRAPH_WORKFLOW_TIMEOUT_HOURS >=
        DATA_BACKEND_INGEST_TIMEOUT_HOURS):
    raise ValueError(
        'DATA_BACKEND_DISKGRAPH_WORKFLOW_TIMEOUT must be less than '
        'DATA_BACKEND_INGEST_TIMEOUT')
DATA_BACKEND_TEXT_EMBEDDING_TIMEOUT_HOURS = 2
# materialize timeout used for generating train/pred tables and materialize
DATA_BACKEND_MATERIALIZE_TIMEOUT_HOURS = 7
# bp stats task timeout
DATA_BACKEND_BP_STATS_TIMEOUT_HOURS = 7

# Per-dispatch pooled train/predict activity cap (21 days). The coordinator can
# dispatch up to three times (63 days of activity execution); heartbeats
# enforce liveness.
TRAINING_AND_PREDICTION_TIMEOUT_HOURS = 24 * 21

# Custom SQL Tables: aligned max-staleness floor (in hours). A Custom SQL
# table's data_version is a content hash over the SQL + its referenced-table
# versions, but that parsed referenced set is an incomplete/unreliable proxy for
# what the query actually reads: SQL functions/constructs whose data we cannot
# recover as table references (TVFs, read_files(), generate_series, no FROM) can
# sit alongside ordinary tables, and warehouse views change underneath a
# reference whose version we cannot see move — so the extracted versions can
# stay static forever while the real data drifts. This bucket folds an aligned
# wall-clock window into data_version so re-materialization fires at least this
# often regardless of detected change. Default 24h; env-overridable. Clamped to
# >= 1h: this is a mandatory safety floor (never disabled), and a 0/negative
# override would make the aligned bucket ill-defined (division by zero).
CUSTOM_SQL_MAX_STALENESS_HOURS = max(
    1,
    EnvVar(
        name="CUSTOM_SQL_MAX_STALENESS",
        type=int,
        default_value=24,
        description="Max hours a Custom SQL table may go without re-ingest, "
        "regardless of detected referenced-table change",
    ).parse_value(),
)
