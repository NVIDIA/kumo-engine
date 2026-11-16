// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AxiosResponse } from "axios";
import { GeneratePredictionTableResponse } from "gen/v0/types";
import { testResponseType } from "tests/testResponseType";
import { createMockResponse } from "tests/testUtil";

import { jobStatus } from "utils/Constants";

import {
    cancelledResponse,
    doneResponse,
    failedResponse,
    queuedResponse,
    runningResponse
} from "../MockResponses/PredictionTableGeneration";

let currentTestResponseType = "";
let currentStatus = jobStatus.Done;
let customResponseOverride: GeneratePredictionTableResponse | null = null;

export const mockGenPredictionTableJobResponse = (
    id: string
): Promise<AxiosResponse<GeneratePredictionTableResponse>> => {
    if (customResponseOverride !== null) {
        return Promise.resolve(
            createMockResponse(
                customResponseOverride
            ) as unknown as AxiosResponse<GeneratePredictionTableResponse>
        );
    }
    if (currentTestResponseType === testResponseType.ERROR) {
        return mockErrorGenPredictionTableJobResponse();
    } else if (currentTestResponseType === testResponseType.EMPTY) {
        return mockEmptyGenPredictionTableJobResponse();
    } else if (currentTestResponseType === testResponseType.INVALID) {
        return mockInvalidGenPredictionTableJobResponse();
    } else if (currentTestResponseType === testResponseType.ARCHIVED) {
        return mockArchivedGenPredictionTableJobResponse();
    }

    return mockDefaultGenPredictionTableJobResponse(id);
};

export const setGenPredictionTableJobTestResponseType = (
    responseType: testResponseType
) => {
    currentTestResponseType = responseType;
};
export const setGenPredictionTableJobResponseStatus = (
    responseStatus: jobStatus
) => {
    currentStatus = responseStatus;
};

export const setCustomGenPredictionTableResponse = (
    response: GeneratePredictionTableResponse | null
) => {
    customResponseOverride = response;
};

/**
 * Minimal gen-pred-table response for fast dispatcher/overview tests. Nulls the
 * nested graph_snapshot_id and pquery_id so the embedded GraphSnapshotOverview
 * and predictive-query panels never mount (suppressing their graph-viz / metadata
 * fetch cascade), while keeping job_id + status so the header and display id
 * still render.
 */
export const minimalGenPredTableJob = (
    id: string,
    status: string = jobStatus.Done
): GeneratePredictionTableResponse =>
    ({
        resource: {
            job_id: id,
            job_status_report: {
                status,
                tracking_url: "",
                start_time: null,
                end_time: null,
                event_log: [],
                validation_response: undefined
            },
            created_at: null,
            tags: {},
            config: {
                job_tags: {},
                pquery_id: null,
                plan: {
                    anchor_time: "inferred",
                    lag_timesteps: "",
                    forecast_length: undefined,
                    year_over_year: false
                },
                graph_snapshot_id: null
            },
            source: "UI"
        }
    }) as unknown as GeneratePredictionTableResponse;

const mockDefaultGenPredictionTableJobResponse = (
    id: string
): Promise<AxiosResponse<GeneratePredictionTableResponse>> => {
    return Promise.resolve(
        createMockResponse(
            expectedGenPredictionTableJobResponse(id)
        ) as unknown as AxiosResponse<GeneratePredictionTableResponse>
    );
};

export const mockEmptyGenPredictionTableJobResponse = (): Promise<
    AxiosResponse<GeneratePredictionTableResponse>
> => {
    return Promise.resolve(
        createMockResponse(
            []
        ) as unknown as AxiosResponse<GeneratePredictionTableResponse>
    );
};

export const mockInvalidGenPredictionTableJobResponse = (): Promise<
    AxiosResponse<GeneratePredictionTableResponse>
> => {
    return Promise.resolve(
        createMockResponse({
            "resource": false
        }) as unknown as AxiosResponse<GeneratePredictionTableResponse>
    );
};

export const expectedGenPredictionTableJobResponse = (id: string) => {
    const resp = doneResponse;
    switch (currentStatus) {
        case jobStatus.Failed:
            resp.resource.job_status_report = failedResponse;
            break;
        case jobStatus.Cancelled:
            resp.resource.job_status_report = cancelledResponse;
            break;
        case jobStatus.Running:
            resp.resource.job_status_report = runningResponse;
            break;
        case jobStatus.Queued:
            resp.resource.job_status_report = queuedResponse;
            break;
        default:
            break;
    }
    resp.resource.job_id = id;
    return resp;
};

export const mockArchivedGenPredictionTableJobResponse = (): Promise<
    AxiosResponse<GeneratePredictionTableResponse>
> => {
    return Promise.resolve(
        createMockResponse({
            ...doneResponse,
            resource: {
                ...doneResponse.resource,
                tags: {
                    ...doneResponse.resource.tags,
                    archived: "archived"
                }
            }
        }) as unknown as AxiosResponse<GeneratePredictionTableResponse>
    );
};

export const mockErrorGenPredictionTableJobResponse = () => {
    return Promise.reject(
        createMockResponse(new Error("Internal Server Error"), 500)
    );
};

export const mockPredictionTotalEntities = (): Promise<
    AxiosResponse<unknown>
> => {
    return Promise.resolve(
        createMockResponse("2") as unknown as AxiosResponse<unknown>
    );
};
