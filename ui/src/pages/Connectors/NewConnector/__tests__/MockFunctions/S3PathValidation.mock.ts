// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AxiosResponse } from "axios";
import { testResponseType } from "tests/testResponseType";
import { createMockResponse } from "tests/testUtil";

import {
    invalidS3PathResponse,
    validResponse
} from "../MockResponses/S3PathValidation";

let currentTestResponseType = "";

export const mockS3PathValidationResponse = (): Promise<
    AxiosResponse<unknown>
> => {
    switch (currentTestResponseType) {
        case testResponseType.ERROR:
            return mockErrorResponse();
        case testResponseType.INVALID:
            return mockInvalidS3PathResponse();
        case testResponseType.EMPTY:
            return mockEmptyS3PathResponse();
        default:
            return mockValidS3PathResponse();
    }
};

const mockEmptyS3PathResponse = (): Promise<AxiosResponse<unknown>> => {
    return Promise.resolve(
        createMockResponse(null) as unknown as AxiosResponse<unknown>
    );
};

export const setS3PathResponseType = (responseType: string = "") => {
    currentTestResponseType = responseType;
    return currentTestResponseType;
};

const mockValidS3PathResponse = (): Promise<AxiosResponse<unknown>> => {
    return Promise.resolve(
        createMockResponse(validResponse) as unknown as AxiosResponse<unknown>
    );
};

const mockInvalidS3PathResponse = (): Promise<AxiosResponse<unknown>> => {
    return Promise.reject({
        response: {
            status: 422,
            statusText: "Unprocessable Entity",
            data: invalidS3PathResponse
        }
    });
};

const mockErrorResponse = () => {
    return Promise.reject(
        createMockResponse(new Error("Internal Server Error"), 500)
    );
};
