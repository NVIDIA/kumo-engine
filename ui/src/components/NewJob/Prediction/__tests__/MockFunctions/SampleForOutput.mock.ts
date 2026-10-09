// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AxiosResponse } from "axios";
import { testResponseType } from "tests/testResponseType";
import { createMockResponse } from "tests/testUtil";

import mockSampleForOutputResponse from "../MockResponses/SampleOutputSchema.json";

let currentTestResponseType = "";

export const mockSampleForOutputResponses = (): Promise<
    AxiosResponse<unknown>
> => {
    switch (currentTestResponseType) {
        case testResponseType.ERROR:
            return mockErrorSampleForOutputResponse();
        case testResponseType.INVALID:
            return mockInvalidSampleForOutputResponse();
        default:
            return mockValidSampleForOutputResponse();
    }
};

export const setSamplesForOutputTestResponseType = (
    responseType: string = ""
) => {
    currentTestResponseType = responseType;
    return currentTestResponseType;
};

const mockValidSampleForOutputResponse = (): Promise<
    AxiosResponse<unknown>
> => {
    return Promise.resolve(
        createMockResponse(
            mockSampleForOutputResponse
        ) as unknown as AxiosResponse<unknown>
    );
};

const mockInvalidSampleForOutputResponse = (): Promise<
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

const mockErrorSampleForOutputResponse = () => {
    return Promise.reject(
        createMockResponse(new Error("Internal Server Error"), 500)
    );
};
