// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
    GeneratePredictionTableResponse,
    KumoapiCommonJobStatus
} from "gen/v0/types";

export const doneResponse: GeneratePredictionTableResponse = {
    "resource": {
        "job_id": "gen-predtable-job-00041000000000000000000000000000",
        "project_id": "project_00000000000000000000000000000001",
        "job_status_report": {
            "status": "DONE",
            "tracking_url": "n/a (under development)",
            "start_time": "2025-07-07T09:45:26.791707+00:00",
            "end_time": "2025-07-07T09:47:00.024795+00:00",
            "event_log": [
                {
                    "stage_name": "Generating Prediction Data",
                    "last_updated_at": "2025-07-07T09:46:59.997610+00:00",
                    "detail":
                        "Successfully completed generating prediction table."
                }
            ],
            "validation_response": undefined
        },
        "created_at": "2025-07-07T09:45:26.791707+00:00",
        "tags": {
            "test_new_pred_page": "test_new_pred_page",
            "pquery_id": "pquery-00066000000000000000000000000000"
        },
        "config": {
            "job_tags": {},
            "pquery_id": "pquery-00066000000000000000000000000000",
            "plan": {
                "anchor_time": "inferred",
                "lag_timesteps": "???",
                "forecast_length": undefined,
                "year_over_year": false
            },
            "graph_snapshot_id":
                "graph-00030000000000000000000000000000@00035000000000000000000000000000"
        },
        "source": "UI"
    }
};

export const failedResponse = {
    "status": "FAILED" as KumoapiCommonJobStatus,
    "tracking_url": "n/a (under development)",
    "start_time": "2025-04-09T05:18:35.943169+00:00",
    "end_time": "2025-04-09T05:20:35.834570+00:00",
    "event_log": [
        {
            "stage_name": "Failed",
            "last_updated_at": "2025-04-09T05:20:35.682830+00:00",
            "detail": "Training workflow has expired and timed out."
        }
    ],
    "validation_response": undefined
};

export const cancelledResponse = {
    "status": "CANCELLED" as KumoapiCommonJobStatus,
    "tracking_url": "n/a (under development)",
    "start_time": "2025-04-09T05:18:35.943169+00:00",
    "end_time": "2025-04-09T05:20:35.834570+00:00",
    "event_log": [
        {
            "stage_name": "Canceled",
            "last_updated_at": "2025-04-09T05:20:35.682830+00:00",
            "detail":
                "BATCH_PREDICTION_JOB is cancelled: Job has been terminated: by parent close policy"
        }
    ],
    "validation_response": undefined
};

export const runningResponse = {
    "status": "RUNNING" as KumoapiCommonJobStatus,
    "tracking_url": "n/a (under development)",
    "start_time": "2025-04-09T05:18:35.943169+00:00",
    "end_time": "2025-04-09T05:20:35.834570+00:00",
    "event_log": [
        {
            "stage_name": "Generating prediction table...",
            "last_updated_at": "2025-04-09T05:20:35.682830+00:00",
            "detail": "This job is processing data from your warehouse."
        }
    ],
    "validation_response": undefined
};

export const queuedResponse = {
    "status": "QUEUED" as KumoapiCommonJobStatus,
    "tracking_url": "n/a (under development)",
    "start_time": "2025-04-09T05:18:35.943169+00:00",
    "end_time": "2025-04-09T05:20:35.834570+00:00",
    "event_log": [
        {
            "stage_name": "Queued",
            "last_updated_at": "2025-04-09T05:20:35.682830+00:00",
            "detail": "This job is queued"
        }
    ],
    "validation_response": undefined
};
