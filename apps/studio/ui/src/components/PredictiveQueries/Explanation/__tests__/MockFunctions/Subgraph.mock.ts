// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AxiosResponse } from "axios";
import { testResponseType } from "tests/testResponseType";
import { createMockResponse } from "tests/testUtil";

import mockSungraphResponse from "../MockResponses/SubgraphResponse.json";
let currentTestResponseType = "";

export const mockSubgraphResponse = (): Promise<AxiosResponse<unknown>> => {
    if (currentTestResponseType === testResponseType.ERROR) {
        return mockErrorResponse();
    } else if (currentTestResponseType === testResponseType.EMPTY) {
        return mockEmptyResponse();
    } else if (currentTestResponseType === testResponseType.INVALID) {
        return mockInvalidResponse();
    }
    return mockDefaultResponse();
};

export const setSubgraphResponseType = (responseType: testResponseType) => {
    currentTestResponseType = responseType;
};

const mockDefaultResponse = (): Promise<AxiosResponse<unknown>> => {
    return Promise.resolve(
        createMockResponse(
            expectedSubgraphResponse()
        ) as unknown as AxiosResponse<unknown>
    );
};

export const mockEmptyResponse = (): Promise<AxiosResponse<unknown>> => {
    return Promise.resolve(
        createMockResponse({}) as unknown as AxiosResponse<unknown>
    );
};

export const mockInvalidResponse = (): Promise<AxiosResponse<unknown>> => {
    return Promise.resolve(
        createMockResponse({
            feature_store: {},
            subgraph_store: {}
        }) as unknown as AxiosResponse<unknown>
    );
};

export const mockErrorResponse = () => {
    return Promise.reject(
        createMockResponse(new Error("Internal Server Error"), 500)
    );
};

export const expectedSubgraphResponse = () => {
    const resp = { ...mockSungraphResponse };
    return resp;
};
