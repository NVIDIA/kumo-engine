// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AxiosResponse } from "axios";
import { testResponseType } from "tests/testResponseType";
import { createMockResponse } from "tests/testUtil";

let currentTestResponseType = "";

export const mockActiveJobsResponse = (): Promise<AxiosResponse<unknown>> => {
    if (currentTestResponseType === testResponseType.ERROR) {
        return mockErrorActiveJobsResponse();
    } else if (currentTestResponseType === testResponseType.INVALID) {
        return mockInvalidActiveJobsResponse();
    }
    return mockDefaultActiveJobsResponse();
};

export const setTestActiveJobsResponseType = (responseType: string = "") => {
    currentTestResponseType = responseType;
};

const mockDefaultActiveJobsResponse = (): Promise<AxiosResponse<unknown>> => {
    return Promise.resolve(
        createMockResponse({}) as unknown as AxiosResponse<unknown>
    );
};

export const mockInvalidActiveJobsResponse = (): Promise<
    AxiosResponse<unknown>
> => {
    return Promise.resolve(
        createMockResponse(undefined) as unknown as AxiosResponse<unknown>
    );
};

export const mockErrorActiveJobsResponse = () => {
    return Promise.reject(
        createMockResponse(new Error("Internal Server Error"), 500)
    );
};
