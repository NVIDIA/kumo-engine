// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AxiosResponse } from "axios";
import { testResponseType } from "tests/testResponseType";
import { createMockResponse } from "tests/testUtil";

import mockPredictionTableResponse from "pages/NewJob/Prediction/__tests__/MockResponse/OutputSchema.json";

let currentTestResponseType = "";

export const mockPredictionTableSchemaResponses = (): Promise<
    AxiosResponse<unknown>
> => {
    switch (currentTestResponseType) {
        case testResponseType.ERROR:
            return mockErrorPredictionTableSchemaResponse();
        case testResponseType.INVALID:
            return mockInvalidPredictionTableSchemaResponse();
        default:
            return mockValidPredictionTableSchemaResponse();
    }
};

export const setPredictionTableSchemaTestResponseType = (
    responseType: string = ""
) => {
    currentTestResponseType = responseType;
    return currentTestResponseType;
};

const mockValidPredictionTableSchemaResponse = (): Promise<
    AxiosResponse<unknown>
> => {
    return Promise.resolve(
        createMockResponse(
            mockPredictionTableResponse
        ) as unknown as AxiosResponse<unknown>
    );
};

const mockInvalidPredictionTableSchemaResponse = (): Promise<
    AxiosResponse<unknown>
> => {
    return Promise.reject({
        response: {
            status: 422,
            statusText: "Unprocessable Entity",
            data: "Error in Creating Table"
        }
    });
};

const mockErrorPredictionTableSchemaResponse = () => {
    return Promise.reject(
        createMockResponse(new Error("Internal Server Error"), 500)
    );
};
