// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AxiosResponse } from "axios";
import { testResponseType } from "tests/testResponseType";
import { createMockResponse } from "tests/testUtil";

import mockSDKPredictionCodeModalResponse from "../MockResponses/SDKPredictionCodeModal.json";

let currentTestResponseType = "";

export const mockSDKPredictionCodeModalResponses = (): Promise<
    AxiosResponse<unknown>
> => {
    switch (currentTestResponseType) {
        case testResponseType.ERROR:
            return mockErrorResponse();
        case testResponseType.INVALID:
            return mockInvalidResponse();
        case testResponseType.EMPTY:
            return mockEmptyResponse();
        default:
            return mockValidResponse();
    }
};

export const setSDKPredictionResponseTestType = (responseType: string = "") => {
    currentTestResponseType = responseType;
    return currentTestResponseType;
};

const mockValidResponse = (): Promise<AxiosResponse<unknown>> => {
    return Promise.resolve(
        createMockResponse(
            mockSDKPredictionCodeModalResponse
        ) as unknown as AxiosResponse<unknown>
    );
};

const mockInvalidResponse = (): Promise<AxiosResponse<unknown>> => {
    return Promise.reject({
        response: {
            status: 422,
            statusText: "Unprocessable Entity",
            data: "Error in Prediction SDK Response"
        }
    });
};

const mockEmptyResponse = (): Promise<AxiosResponse<unknown>> => {
    return Promise.resolve(
        createMockResponse(undefined) as unknown as AxiosResponse<unknown>
    );
};

const mockErrorResponse = () => {
    return Promise.reject(
        createMockResponse(new Error("Internal Server Error"), 500)
    );
};
