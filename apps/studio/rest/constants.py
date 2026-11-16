# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import timedelta

from temporalio.common import RetryPolicy

DATA_BACKEND_WORKER_NAME = 'data-backend-worker'

DATA_PERSIST_DIR_NAME = 'data-persist-dir'
# Non-compute, lightweight IO activties on data backend are meant to be
# executed via data-backend task queue, running in either threadpool or
# event loop.
DATA_BACKEND_TASK_QUEUE = 'data-backend-task-queue'

# Heartbeat timeout for long-running async activities, e.g., waiting/polling
# for result of another workflow.
HEARTBEAT_TIMEOUT_ASYNC_ACTIVITY = timedelta(minutes=2)

# This retry policy helps make child workflow status polling more resilient to
# transient errors/timeouts.
CHILD_WF_WAITING_RETRY_POLICY = RetryPolicy(
    maximum_attempts=20,
    initial_interval=timedelta(seconds=5),
    maximum_interval=timedelta(minutes=2),
    non_retryable_error_types=['WorkflowFailureError'],
)

# In-process data compute activities only retry once: they share the worker's
# memory so OOM is the dominant failure mode and is not retryable, and there
# is no checkpointing — a retry would restart from scratch.
_LOCAL_DATA_COMPUTE_ACTIVITY_RETRY_POLICY = RetryPolicy(maximum_attempts=2)

COMPUTE_ACTIVITY_RETRY_POLICY = _LOCAL_DATA_COMPUTE_ACTIVITY_RETRY_POLICY

VALIDATE_STATS_RETRY_POLICY = RetryPolicy(
    maximum_attempts=10,
    initial_interval=timedelta(minutes=1),
    maximum_interval=timedelta(minutes=10),
    non_retryable_error_types=['ValueError'],
)

# Heartbeat timeout for long-running data compute activities. Set high to
# tolerate delayed heartbeats from in-process compute (context switching
# between the heartbeat reporter thread and the activity execution thread).
HEARTBEAT_TIMEOUT_COMPUTE_ACTIVITY = timedelta(minutes=15)
HEARTBEAT_TIMEOUT_TRAIN_TABLE_SPLIT = timedelta(minutes=15)

# Signal name for registering a running workflow as watcher to receive status
# update signals from a table/view refresh workflow.
REGISTER_WATCHER_SIGNAL = 'register_watcher'
# Signal name for receiving and processing table/view refresh status updates.
RECEIVE_REFRESH_STATUS_UPDATE_SIGNAL = 'receive_refresh_status_update'

# Env vars for data backend worker settings
DISABLE_REMOTE_INGESTION = 'DISABLE_REMOTE_INGESTION'

ML_WORKFLOW_TASK_QUEUE = 'ml-workflow-task-queue'
ML_WORKFLOW_WORKER = 'ml-workflow-worker'
# Activity heartbeats may be throttled by worker so setting a longer
# timeout to prevent false timeout.
# TODO(siyang): Reduce the heartbeat timeout to detect unresponsive/dead worker
# faster.
DEFAULT_HEARTBEAT_TIMEOUT = timedelta(minutes=5)
# Retry is disabled for training and batch prediction.
DISABLE_RETRY_POLICY = RetryPolicy(maximum_attempts=1)
RETRY_ONCE_POLICY = RetryPolicy(maximum_attempts=2)
