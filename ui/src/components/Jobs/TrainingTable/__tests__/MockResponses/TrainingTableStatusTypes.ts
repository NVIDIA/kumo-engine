// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
    GenerateTrainTableResponse,
    KumoapiCommonJobStatus
} from "gen/v0/types";

export const doneResponse: GenerateTrainTableResponse = {
    "resource": {
        "job_id": "gen-traintable-job-9f8b259b4f764c8f932ad5912a926ec2",
        "project_id": "project_00000000000000000000000000000001",
        "job_status_report": {
            "status": "DONE" as KumoapiCommonJobStatus,
            "tracking_url": "n/a (under development)",
            "start_time": "2025-03-12T14:55:39.208629+00:00",
            "end_time": "2025-03-12T14:59:22.825522+00:00",
            "event_log": [
                {
                    "stage_name": "Generating Training Data",
                    "last_updated_at": "2025-03-12T14:59:22.669626+00:00",
                    "detail": "Successfully completed generating train table."
                }
            ],
            "validation_response": undefined
        },
        "created_at": "2025-03-12T14:55:39.208629+00:00",
        "tags": {
            "new_train_tag": "new_train_tag",
            "pquery_id": "pquery-629e15c329bb119cbae6de2fc2021165",
            "parent_job": "trainingjob-9f8b259b4f764c8f932ad5912a926ec2"
        },
        "config": {
            "job_tags": {},
            "pquery_id": "pquery-629e15c329bb119cbae6de2fc2021165",
            "plan": {
                "split": "ApproxDateOffsetSplit(ratio=[0.8, 0.1, 0.1])",
                "train_start_offset": "inferred",
                "train_end_offset": "inferred",
                "start_time": "???",
                "end_time": "???",
                "timeframe_step": "inferred",
                "forecast_length": "???",
                "lag_timesteps": "???",
                "year_over_year": "???",
                "entity_candidate": undefined,
                "entity_candidate_aggregation": undefined,
                "task_path_override": undefined,
                "train_table_path_override": undefined
            },
            "graph_snapshot_id":
                "graph-d445eeb8b2aa1b79a142406239fa74e8@6f0ae33ac31edb0c321fbe182e39e460"
        },
        "source": "UI",
        "user": "test-user@example.com"
    }
};

export const doneResponseWithoutParentJob: GenerateTrainTableResponse = {
    "resource": {
        "job_id": "gen-traintable-job-9f8b259b4f764c8f932ad5912a926ec2",
        "project_id": "project_00000000000000000000000000000001",
        "job_status_report": {
            "status": "DONE" as KumoapiCommonJobStatus,
            "tracking_url": "n/a (under development)",
            "start_time": "2025-03-12T14:55:39.208629+00:00",
            "end_time": "2025-03-12T14:59:22.825522+00:00",
            "event_log": [
                {
                    "stage_name": "Generating Training Data",
                    "last_updated_at": "2025-03-12T14:59:22.669626+00:00",
                    "detail": "Successfully completed generating train table."
                }
            ],
            "validation_response": undefined
        },
        "created_at": "2025-03-12T14:55:39.208629+00:00",
        "tags": {
            "new_train_tag": "new_train_tag",
            "pquery_id": "pquery-629e15c329bb119cbae6de2fc2021165"
        },
        "config": {
            "job_tags": {},
            "pquery_id": "pquery-629e15c329bb119cbae6de2fc2021165",
            "plan": {
                "split": "ApproxDateOffsetSplit(ratio=[0.8, 0.1, 0.1])",
                "train_start_offset": "inferred",
                "train_end_offset": "inferred",
                "start_time": "???",
                "end_time": "???",
                "timeframe_step": "inferred",
                "forecast_length": "???",
                "lag_timesteps": "???",
                "year_over_year": "???",
                "entity_candidate": undefined,
                "entity_candidate_aggregation": undefined,
                "task_path_override": undefined,
                "train_table_path_override": undefined
            },
            "graph_snapshot_id":
                "graph-d445eeb8b2aa1b79a142406239fa74e8@6f0ae33ac31edb0c321fbe182e39e460"
        },
        "source": "UI"
    }
};

export const failedResponse = {
    "status": "FAILED" as KumoapiCommonJobStatus,
    "tracking_url": "n/a (under development)",
    "start_time": "2025-04-03T15:42:19.976993+00:00",
    "end_time": "2025-04-03T16:22:01.373419+00:00",
    "event_log": [
        {
            "stage_name": "Failed",
            "last_updated_at": "2025-04-03T16:22:01.372668+00:00",
            "detail":
                "GENERATE_TRAIN_TABLE_JOB workflow has expired and not found."
        }
    ],
    "validation_response": undefined
};

export const cancelledResponse = {
    "status": "CANCELLED" as KumoapiCommonJobStatus,
    "tracking_url": "n/a (under development)",
    "start_time": "2025-03-20T15:38:17.246243+00:00",
    "end_time": "2025-03-21T01:52:34.119752+00:00",
    "event_log": [
        {
            "stage_name": "Canceled",
            "last_updated_at": "2025-03-21T01:52:34.117533+00:00",
            "detail":
                "BATCH_PREDICTION_JOB is cancelled: Job has been terminated: by parent close policy"
        }
    ],
    "validation_response": undefined
};

export const runningResponse = {
    "status": "RUNNING" as KumoapiCommonJobStatus,
    "tracking_url": "n/a (under development)",
    "start_time": "2025-03-20T15:38:17.246243+00:00",
    "end_time": "2025-03-21T01:52:34.119752+00:00",
    "event_log": [
        {
            "stage_name": "Running",
            "last_updated_at": "2025-03-21T01:52:34.117533+00:00",
            "detail": "Generating Prediction Table.."
        }
    ],
    "validation_response": undefined
};

export const queuedResponse = {
    "status": "QUEUED" as KumoapiCommonJobStatus,
    "tracking_url": "n/a (under development)",
    "start_time": "2025-03-20T15:38:17.246243+00:00",
    "end_time": "2025-03-21T01:52:34.119752+00:00",
    "event_log": [
        {
            "stage_name": "Queued",
            "last_updated_at": "2025-03-21T01:52:34.117533+00:00",
            "detail": "This training job is queued"
        }
    ],
    "validation_response": undefined
};
