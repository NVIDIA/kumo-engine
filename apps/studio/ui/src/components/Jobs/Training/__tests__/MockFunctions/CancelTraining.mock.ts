// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AxiosResponse } from "axios";
import { testResponseType } from "tests/testResponseType";
import { createMockResponse } from "tests/testUtil";

let currentTestResponseType = "";

export const mockCancelTraining = (): Promise<AxiosResponse<unknown>> => {
    if (currentTestResponseType === testResponseType.ERROR) {
        return mockErrorResponse();
    } else if (currentTestResponseType === testResponseType.EMPTY) {
        return mockEmptyResponse();
    }
    return mockDefaultResponse();
};

export const setCancelTrainingResponseType = (
    responseType: testResponseType
) => {
    currentTestResponseType = responseType;
};

const mockDefaultResponse = (): Promise<AxiosResponse<unknown>> => {
    return Promise.resolve(
        createMockResponse(true) as unknown as AxiosResponse<unknown>
    );
};

export const mockEmptyResponse = (): Promise<AxiosResponse<unknown>> => {
    return Promise.resolve(
        createMockResponse({}) as unknown as AxiosResponse<unknown>
    );
};

export const mockErrorResponse = () => {
    return Promise.reject(
        createMockResponse(new Error("Internal Server Error"), 500)
    );
};
