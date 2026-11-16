// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AxiosResponse } from "axios";
import { testResponseType } from "tests/testResponseType";
import { createMockResponse } from "tests/testUtil";

import mockGetManyResponse from "../MockResponse/GetManyResponse.json";

let currentTestResponseType = "";

export const mockModelGetManyResponse = (): Promise<AxiosResponse<unknown>> => {
    if (currentTestResponseType === testResponseType.ERROR) {
        return mockErrorGetManyResponse();
    } else if (currentTestResponseType === testResponseType.EMPTY) {
        return mockEmptyGetManyResponse();
    } else if (currentTestResponseType === testResponseType.INVALID) {
        return mockInvalidGetManyResponse();
    }
    return mockDefaultGetManyResponse();
};

export const setTestGetManyResponseType = (responseType: string = "") => {
    currentTestResponseType = responseType;
};

const mockDefaultGetManyResponse = (): Promise<AxiosResponse<unknown>> => {
    return Promise.resolve(
        createMockResponse(
            mockGetManyResponse
        ) as unknown as AxiosResponse<unknown>
    );
};

export const mockEmptyGetManyResponse = (): Promise<AxiosResponse<unknown>> => {
    return Promise.resolve(
        createMockResponse([]) as unknown as AxiosResponse<unknown>
    );
};

export const mockInvalidGetManyResponse = (): Promise<
    AxiosResponse<unknown>
> => {
    return Promise.resolve(
        createMockResponse({}) as unknown as AxiosResponse<unknown>
    );
};

export const mockErrorGetManyResponse = () => {
    return Promise.reject(
        createMockResponse(new Error("Internal Server Error"), 500)
    );
};

export const expectedGetManyResponse = () => {
    return Promise.resolve(
        createMockResponse(
            mockGetManyResponse
        ) as unknown as AxiosResponse<unknown>
    );
};
