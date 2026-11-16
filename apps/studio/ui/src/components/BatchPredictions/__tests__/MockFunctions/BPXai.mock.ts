// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AxiosResponse } from "axios";
import { testResponseType } from "tests/testResponseType";
import { createMockResponse } from "tests/testUtil";

import mockBPXai from "../MockResponses/BPXai.json";

let currentTestResponseType = "";

export const mockBPXaiResponse = (): Promise<AxiosResponse<unknown>> => {
    switch (currentTestResponseType) {
        case testResponseType.ERROR:
            return mockErrorResponse();
        case testResponseType.INVALID:
            return mockInvalidResponse();
        case testResponseType.EMPTY:
            return mockEmptyResponse();
        default:
            return mockDefaultResponse();
    }
};

export const setBPXaiResponseType = (responseType: string = "") => {
    currentTestResponseType = responseType;
    return currentTestResponseType;
};

const mockDefaultResponse = (): Promise<AxiosResponse<unknown>> => {
    return Promise.resolve(
        createMockResponse(mockBPXai) as unknown as AxiosResponse<unknown>
    );
};

const mockInvalidResponse = (): Promise<AxiosResponse<unknown>> => {
    return Promise.resolve(
        createMockResponse({
            mappings: false
        }) as unknown as AxiosResponse<unknown>
    );
};

const mockErrorResponse = () => {
    return Promise.reject(
        createMockResponse(new Error("Internal Server Error"), 500)
    );
};

export const mockEmptyResponse = (): Promise<AxiosResponse<unknown>> => {
    return Promise.resolve(
        createMockResponse({}) as unknown as AxiosResponse<unknown>
    );
};
