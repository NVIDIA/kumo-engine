// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AxiosResponse } from "axios";
import { testResponseType } from "tests/testResponseType";
import { createMockResponse } from "tests/testUtil";

import { CreatePredictionJobArgs } from "types/predictionJob";

import mockGetNewBpJobResponse from "../MockResponse/NewBpJobResponse.json";

let currentTestResponseType = "";

export const mockBatchPredictionJobsResponse = (): Promise<
    AxiosResponse<unknown>
> => {
    switch (currentTestResponseType) {
        case testResponseType.ERROR:
            return mockErrorResponse();
        case testResponseType.INVALID:
            return mockInvalidResponse();
        default:
            return mockValidResponse();
    }
};

export const setBatchPredictionJobsResponseType = (
    responseType: string = ""
) => {
    currentTestResponseType = responseType;
    return currentTestResponseType;
};

const mockValidResponse = (): Promise<AxiosResponse<unknown>> => {
    return Promise.resolve(
        createMockResponse({
            id: "bp-job-349fee63191149bf85731f7bb7ffcf45"
        }) as unknown as AxiosResponse<unknown>
    );
};

const mockInvalidResponse = (): Promise<AxiosResponse<unknown>> => {
    return Promise.reject({
        response: {
            status: 422,
            statusText: "Unprocessable Entity",
            data: "Error in Creating Batch Prediction"
        }
    });
};

const mockErrorResponse = () => {
    return Promise.reject(
        createMockResponse(new Error("Internal Server Error"), 500)
    );
};

export const expectedNewBpJobResponse = (): CreatePredictionJobArgs => {
    return { ...mockGetNewBpJobResponse } as unknown as CreatePredictionJobArgs;
};
