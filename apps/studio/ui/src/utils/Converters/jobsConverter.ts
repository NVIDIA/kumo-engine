// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
    GeneratePredictionTableResponse,
    GenerateTrainTableResponse
} from "gen/v0/types";

import { PredictionTableGeneration } from "types/batchPredictions";
import { TrainTableGeneration } from "types/training_job";

import { TAG_KEY_ARCHIVED } from "utils/jobs";
import {
    formatToNumericLocalDateString,
    formatToShortLocalDateString,
    jobRunTimeFromJobResponse
} from "utils/timeConverter";

/**
 * Helper function to convert a Pydantic TrainTableGenResponse dataclass returned
 * from the REST API into a TrainTableGeneration that can be used in Javascript
 * @param trainTableGenResponse API response of get API
 */
/** Run time when both start and end timestamps are present, else "". */
function genRunTime(
    report:
        | GenerateTrainTableResponse["resource"]["job_status_report"]
        | GeneratePredictionTableResponse["resource"]["job_status_report"]
        | undefined
): string {
    if (report?.start_time && report?.end_time) {
        return jobRunTimeFromJobResponse(report.start_time, report.end_time);
    }
    return "";
}

export function parseTrainTableGenResponse(
    trainTableGenResponse: GenerateTrainTableResponse
) {
    const resource = trainTableGenResponse?.resource;
    const report = resource?.job_status_report;
    const config = resource?.config;
    const tags = resource?.tags;
    const generationDetails: TrainTableGeneration = {
        jobId: resource?.job_id,
        config: config,
        created: resource?.created_at,
        createdAt: resource?.created_at,
        dateCreated: formatToShortLocalDateString(resource?.created_at),
        source: resource?.source,
        tags: tags,
        jobStatusReport: report,
        runTime: genRunTime(report),
        graphSnapshotId: config?.graph_snapshot_id,
        status: report?.status,
        archived: tags?.archived === TAG_KEY_ARCHIVED,
        user: resource?.user
    };
    return generationDetails;
}

export function parsePredictionTableGenResponse(
    predictionTableGenResponse: GeneratePredictionTableResponse
) {
    const resource = predictionTableGenResponse?.resource;
    const report = resource?.job_status_report;
    const config = resource?.config;
    const generationDetails: PredictionTableGeneration = {
        jobId: resource?.job_id,
        config: config,
        created: formatToNumericLocalDateString(resource?.created_at),
        createdAt: resource?.created_at,
        dateCreated: formatToShortLocalDateString(resource?.created_at),
        source: resource?.source,
        tags: resource?.tags,
        jobStatusReport: report,
        runTime: genRunTime(report),
        graphSnapshotId: config?.graph_snapshot_id,
        status: report?.status,
        user: resource?.user
    };
    return generationDetails;
}
