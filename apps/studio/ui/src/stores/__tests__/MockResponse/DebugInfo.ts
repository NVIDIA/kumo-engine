// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export const debug_info = {
    "db_path": ".",
    "release_version": "vitest",
    "temporal_url": "localhost:7233",
    "feature_flags": {
        "ENABLE_COLUMN_ANALYSIS": true,
        "MAX_TABLES_IN_A_GRAPH": 12,
        "IS_DATABRICKS_DEPLOY": false,
        "GCP_WORKLOAD_EXCHANGE_ENABLED": false
    },
    "job_queue_status":
        "{ 'gpu_slots_occupied': 0,\n  'job_slots_occupied': 0,\n  'queued_jobs': [],\n  'running_jobs': [],\n  'settings': { 'max_concurrent_gpu_worker_slots': 10,\n                'max_concurrent_jobs_total': 10,\n                'max_job_queue_length': 10,\n                'use_job_queue': True}}"
};
