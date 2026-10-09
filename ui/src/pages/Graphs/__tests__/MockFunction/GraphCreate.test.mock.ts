// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AxiosResponse } from "axios";
import { testResponseType } from "tests/testResponseType";
import { createMockResponse } from "tests/testUtil";

let currentTestResponseType = "";

export const mockCreateGraphResponse = (): Promise<AxiosResponse<unknown>> => {
    switch (currentTestResponseType) {
        case testResponseType.EMPTY:
            return mockCreateGraphEmptyResponse();
        case testResponseType.ERROR:
            return mockErrorGraphCreateResponse();
        case testResponseType.INVALID:
            return mockInvalidGraphCreateResponse();
        default:
            return mockCreateGraphDefaultResponse();
    }
};

export const setGraphCreateResponseType = (responseType: string = "") => {
    currentTestResponseType = responseType;
    return currentTestResponseType;
};

const mockCreateGraphDefaultResponse = (): Promise<AxiosResponse<unknown>> => {
    return Promise.resolve(
        createMockResponse({
            id: "graph-4973156df602a3f4346186713bd6782d"
        }) as unknown as AxiosResponse<unknown>
    );
};

const mockInvalidGraphCreateResponse = (): Promise<AxiosResponse<unknown>> => {
    return Promise.reject({
        response: {
            status: 422,
            statusText: "Unprocessable Entity",
            data: "Error in Creating Graph"
        }
    });
};

const mockCreateGraphEmptyResponse = (): Promise<AxiosResponse<unknown>> => {
    return Promise.resolve(
        createMockResponse([]) as unknown as AxiosResponse<unknown>
    );
};

export const mockErrorGraphCreateResponse = () => {
    return Promise.reject(
        createMockResponse(new Error("Internal Server Error"), 500)
    );
};
