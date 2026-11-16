// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AxiosResponse } from "axios";
import { testResponseType } from "tests/testResponseType";
import { createMockResponse } from "tests/testUtil";

import { jobStatus } from "utils/Constants";

import {
    cancelledResponse,
    doneResponse,
    doneResponseWithoutParentJob,
    failedResponse,
    queuedResponse,
    runningResponse
} from "../MockResponses/TrainingTableStatusTypes";
let currentTestResponseType = "";
let currentStatus = jobStatus.Done;

export const mockTrainingTableGenResponse = (
    id: string
): Promise<AxiosResponse<unknown>> => {
    if (currentTestResponseType === testResponseType.ERROR) {
        return mockErrorResponse();
    } else if (currentTestResponseType === testResponseType.EMPTY) {
        return mockEmptyResponse();
    } else if (currentTestResponseType === testResponseType.INVALID) {
        return mockInvalidResponse();
    } else if (currentTestResponseType === testResponseType.ARCHIVED) {
        return mockArchivedTrainingGenResponse();
    } else if (currentTestResponseType == testResponseType.FILTERED_OUT) {
        return mockResponseWithoutParentJob();
    }
    return mockDefaultResponse(id);
};

export const setTrainingGenTestResponseType = (
    responseType: testResponseType
) => {
    currentTestResponseType = responseType;
};
export const setTrainingGenResponseStatus = (responseStatus: jobStatus) => {
    currentStatus = responseStatus;
};

export const mockEncodersDefaultResponse = (): Promise<
    AxiosResponse<unknown>
> => {
    const resp = {
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

    return Promise.resolve(createMockResponse(resp) as AxiosResponse<unknown>);
};

const mockDefaultResponse = (id: string): Promise<AxiosResponse<unknown>> => {
    return Promise.resolve(
        createMockResponse(
            expectedTrainingTableGenResponse(id)
        ) as AxiosResponse<unknown>
    );
};

export const mockArchivedTrainingGenResponse = (): Promise<
    AxiosResponse<unknown>
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
        }) as AxiosResponse<unknown>
    );
};

export const mockResponseWithoutParentJob = (): Promise<
    AxiosResponse<unknown>
> => {
    return Promise.resolve(
        createMockResponse(
            doneResponseWithoutParentJob
        ) as AxiosResponse<unknown>
    );
};

export const mockEmptyResponse = (): Promise<AxiosResponse<unknown>> => {
    return Promise.resolve(createMockResponse({}) as AxiosResponse<unknown>);
};

export const mockInvalidResponse = (): Promise<AxiosResponse<unknown>> => {
    return Promise.reject({
        response: {
            status: 422,
            statusText: "Unprocessable Entity",
            data: "Error in Training Table Job"
        }
    });
};

export const mockErrorResponse = () => {
    return Promise.reject(
        createMockResponse(new Error("Internal Server Error"), 500)
    );
};

export const expectedTrainingTableGenResponse = (id: string) => {
    const resp = JSON.parse(JSON.stringify(doneResponse));
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
