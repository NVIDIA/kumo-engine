// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AxiosResponse } from "axios";
import { testResponseType } from "tests/testResponseType";
import { createMockResponse } from "tests/testUtil";

import { jobStatus } from "utils/Constants";

import mockWorkflowResponse from "../MockResponses/TrainingWorkflow.test.json";
import mockWorkflowResponseCancelledJob from "../MockResponses/TrainingWorkflowCancelled.test.json";
import mockWorkflowResponseFailedJob from "../MockResponses/TrainingWorkflowFailed.test.json";
import mockWorkflowResponseQueuedJob from "../MockResponses/TrainingWorkflowQueued.test.json";
import mockWorkflowResponseRunningJob from "../MockResponses/TrainingWorkflowRunning.test.json";
let currentTestResponseType = "";
let currentStatus = jobStatus.Done;
let customWorkflowOverride: unknown = null;
export const expectedNewPredictionJobTableHeader = "Job IDUserStartedTags";
export const mockTrainingWorkflowResponse = (
    id: string
): Promise<AxiosResponse<unknown>> => {
    if (customWorkflowOverride !== null) {
        return Promise.resolve(
            createMockResponse(
                customWorkflowOverride
            ) as unknown as AxiosResponse<unknown>
        );
    }
    if (currentTestResponseType === testResponseType.ERROR) {
        return mockErrorResponse();
    } else if (currentTestResponseType === testResponseType.EMPTY) {
        return mockEmptyResponse();
    } else if (currentTestResponseType === testResponseType.INVALID) {
        return mockInvalidResponse();
    } else if (currentTestResponseType === testResponseType.ARCHIVED) {
        return mockArchivedResponse();
    }
    return mockDefaultResponse(id);
};

export const setTrainingWorkflowTestResponseType = (
    responseType: testResponseType
) => {
    currentTestResponseType = responseType;
};
export const setTrainingWorkflowResponseStatus = (
    responseStatus: jobStatus
) => {
    currentStatus = responseStatus;
};

export const setCustomTrainingWorkflowResponse = (response: unknown) => {
    customWorkflowOverride = response;
};

/**
 * Minimal training workflow response for fast tests. Keeps `training_job`
 * (so the header, status chip, and archive/cancel controls render) but nulls
 * the fields that drive nested children and the evaluation/explanation tab
 * cascade (task_type, graph snapshot, train-table job, generation job). Use
 * for tests that exercise header actions (archive, cancel, status) and do not
 * assert on related jobs or eval/explanation content.
 */
export const minimalTrainingWorkflow = (
    id: string,
    status: jobStatus | string = jobStatus.Done,
    tagsOverride: Record<string, string> | null = null
) => ({
    pquery_definition: null,
    training_job: {
        id,
        execution: {
            status,
            log: { detail: "" },
            start_time: null,
            failure_info: null
        },
        advanced_options: null,
        pquery_id: null,
        pquery_def_fpid: null,
        graph_data_version: null,
        generate_train_table_job_id: null,
        job_source: null,
        tags: tagsOverride,
        enable_baselines: false
    },
    task_type: null,
    query_type: null,
    generation_job: null,
    legacy_graph_id: null,
    job_source: null,
    graph_snapshot_id: null
});

const mockEncodersMap = {
    "customerltv_mini_integ_test.transaction.Description":
        "Hash(num_components=100, na_strategy='separate')",
    "customerltv_mini_integ_test.transaction.Quantity":
        "Index(min_occ=17, na_strategy='separate')",
    "customerltv_mini_integ_test.transaction.InvoiceDate":
        "Datetime(include_minute=True, include_hour=True, include_day_of_week=True, include_day_of_month=True, include_day_of_year=True, include_year=True, num_year_periods=None)",
    "customerltv_mini_integ_test.transaction.UnitPrice":
        "Numerical(scaler='robust', na_strategy='mean')",
    "customerltv_mini_integ_test.transaction.Country":
        "Index(min_occ=5, na_strategy='separate')"
};

export const mockEncodersDefaultResponse = (): Promise<
    AxiosResponse<unknown>
> => {
    return Promise.resolve(
        createMockResponse(mockEncodersMap) as unknown as AxiosResponse<unknown>
    );
};

// The v1 predictive query encoders endpoint wraps the encoders map in a
// PQueryEncodersResponse object: { encoders: { ... } }.
export const mockPredictiveQueryEncodersResponse = (): Promise<
    AxiosResponse<unknown>
> => {
    return Promise.resolve(
        createMockResponse({
            encoders: mockEncodersMap
        }) as unknown as AxiosResponse<unknown>
    );
};

const mockDefaultResponse = (id: string): Promise<AxiosResponse<unknown>> => {
    const resp =
        currentStatus === jobStatus.Failed
            ? { ...mockWorkflowResponseFailedJob }
            : currentStatus === jobStatus.Cancelled
              ? { ...mockWorkflowResponseCancelledJob }
              : currentStatus === jobStatus.Queued
                ? { ...mockWorkflowResponseQueuedJob }
                : currentStatus === jobStatus.Running
                  ? { ...mockWorkflowResponseRunningJob }
                  : { ...mockWorkflowResponse };
    resp.training_job.id = id;
    return Promise.resolve(
        createMockResponse(resp) as unknown as AxiosResponse<unknown>
    );
};

export const mockArchivedResponse = (): Promise<AxiosResponse<unknown>> => {
    return Promise.resolve(
        createMockResponse({
            ...mockWorkflowResponse,
            training_job: {
                ...mockWorkflowResponse.training_job,
                tags: {
                    ...mockWorkflowResponse.training_job.tags,
                    archived: "archived"
                }
            }
        }) as unknown as AxiosResponse<unknown>
    );
};

export const mockEmptyResponse = (): Promise<AxiosResponse<unknown>> => {
    return Promise.resolve(
        createMockResponse({}) as unknown as AxiosResponse<unknown>
    );
};

export const mockInvalidResponse = (): Promise<AxiosResponse<unknown>> => {
    return Promise.reject({
        response: {
            status: 422,
            statusText: "Unprocessable Entity",
            data: "Error in Training Job"
        }
    });
};

export const mockErrorResponse = () => {
    return Promise.reject(
        createMockResponse(new Error("Internal Server Error"), 500)
    );
};

export const expectedTrainingJobWorkflow = (id: string) => {
    const resp = { ...mockWorkflowResponse };
    resp.training_job.id = id;
};
