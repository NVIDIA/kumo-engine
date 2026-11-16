// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AxiosResponse } from "axios";
import { BatchPredictionJobResponse } from "gen/v1/types";
import { testResponseType } from "tests/testResponseType";
import { createMockResponse } from "tests/testUtil";

import { jobStatus } from "utils/Constants";

import mockBatchPredictionCancelledResponse from "../MockResponses/BatchPredictionCancelledResponse.json";
import mockBatchPredictionFailedResponse from "../MockResponses/BatchPredictionFailedResponse.json";
import mockBatchPredictionQueuesResponse from "../MockResponses/BatchPredictionQueuedResponse.json";
import mockBatchPredictionJob from "../MockResponses/BatchPredictionResponse.json";
import mockBatchPredictionRunningResponse from "../MockResponses/BatchPredictionRunningResponse.json";
let currentTestResponseType = "";
let currentStatus = jobStatus.Done;
let customResponseOverride: BatchPredictionJobResponse | null = null;

export const mockBatchPredictionJobResponse = (): Promise<
    AxiosResponse<BatchPredictionJobResponse>
> => {
    if (customResponseOverride !== null) {
        return Promise.resolve(
            createMockResponse(
                customResponseOverride
            ) as unknown as AxiosResponse<BatchPredictionJobResponse>
        );
    }
    if (currentTestResponseType === testResponseType.ERROR) {
        return mockErrorBatchPredictionJobResponse();
    } else if (currentTestResponseType === testResponseType.EMPTY) {
        return mockEmptyBatchPredictionJobResponse();
    } else if (currentTestResponseType === testResponseType.INVALID) {
        return mockInvalidBatchPredictionJobResponse();
    } else if (currentTestResponseType === testResponseType.ARCHIVED) {
        return mockArchivedBatchPredictionResponse();
    }

    return mockDefaultBatchPredictionJobResponse();
};

export const setBatchPredictionResponseType = (
    responseType: testResponseType
) => {
    currentTestResponseType = responseType;
};

export const setCustomBatchPredictionResponse = (
    response: BatchPredictionJobResponse | null
) => {
    customResponseOverride = response;
};

/**
 * Minimal BP job response factory for fast component tests. Omits nested job
 * IDs/links so child components (TrainingJobOverview, PredictionTableOverview,
 * GraphOverview) do not mount and fire extra queries, keeping tests fast.
 * Pair with {@link setCustomBatchPredictionResponse} to override per test.
 *
 * `trainingJobId` keeps the related-training section mounting (so its
 * `warnings` call still fires) while every other nested cascade stays
 * suppressed — pair it with a minimal training-workflow override.
 */
export const minimalBpJob = (
    id: string,
    status: string,
    tagsOverride: Record<string, string> | null = null,
    trainingJobId: string | null = null
): BatchPredictionJobResponse =>
    ({
        id,
        config: {
            predictive_query_id: "NA-minimal",
            training_job_id: trainingJobId,
            generate_pred_table_job_id: null,
            graph_id: null,
            graph_data_version: null,
            outputs: [],
            explanations: false,
            task_type: null,
            num_classes_to_return: null,
            binary_classification_threshold: null,
            prediction_time: null,
            lag_timesteps: null,
            year_over_year: false,
            forecast_length: null,
            filter_override_args: null,
            transformation: null,
            output_table_name: null,
            output_directory: null,
            output_types: null,
            num_parallel_workers_limit: null,
            custom_pred_table_source: null,
            pquery_def_fpid: null
        },
        timestamps: null,
        predictive_query_link: null,
        execution: {
            status,
            log: null,
            start_time: null,
            last_updated_at: null,
            workflow_end_state: null,
            failure_info: null,
            progress: null,
            results: null
        },
        graph_link: null,
        description: "",
        batch_prediction_workflow_id: null,
        tags: tagsOverride,
        job_source: null
    }) as unknown as BatchPredictionJobResponse;

export const setBatchPredictionResponseStatus = (responseStatus: jobStatus) => {
    currentStatus = responseStatus;
};

const mockDefaultBatchPredictionJobResponse = (): Promise<
    AxiosResponse<BatchPredictionJobResponse>
> => {
    const resp =
        currentStatus === jobStatus.Failed
            ? { ...mockBatchPredictionFailedResponse }
            : currentStatus === jobStatus.Cancelled
              ? { ...mockBatchPredictionCancelledResponse }
              : currentStatus === jobStatus.Queued
                ? { ...mockBatchPredictionQueuesResponse }
                : currentStatus === jobStatus.Running
                  ? { ...mockBatchPredictionRunningResponse }
                  : { ...mockBatchPredictionJob };
    return Promise.resolve(
        createMockResponse(
            resp
        ) as unknown as AxiosResponse<BatchPredictionJobResponse>
    );
};

export const mockArchivedBatchPredictionResponse = (): Promise<
    AxiosResponse<BatchPredictionJobResponse>
> => {
    return Promise.resolve(
        createMockResponse({
            ...mockBatchPredictionJob,
            tags: {
                ...mockBatchPredictionJob.tags,
                archived: "archived"
            }
        }) as unknown as AxiosResponse<BatchPredictionJobResponse>
    );
};

export const mockEmptyBatchPredictionJobResponse = (): Promise<
    AxiosResponse<BatchPredictionJobResponse>
> => {
    return Promise.resolve(
        createMockResponse(
            null
        ) as unknown as AxiosResponse<BatchPredictionJobResponse>
    );
};

export const mockInvalidBatchPredictionJobResponse = (): Promise<
    AxiosResponse<BatchPredictionJobResponse>
> => {
    return Promise.reject({
        response: {
            status: 404,
            statusText: "Unprocessable Entity",
            data: "Error in Batch Prediction Job"
        }
    });
};

export const expectedBatchPredictionJobResponse = () => {
    return mockBatchPredictionJob;
};

export const mockErrorBatchPredictionJobResponse = () => {
    return Promise.reject(
        createMockResponse(new Error("Internal Server Error"), 500)
    );
};
