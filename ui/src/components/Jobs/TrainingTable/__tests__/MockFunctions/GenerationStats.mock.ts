// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AxiosResponse } from "axios";
import { testResponseType } from "tests/testResponseType";
import { createMockResponse } from "tests/testUtil";

import mockGenStatsResource from "../MockResponses/GenerationStatsResponse.test.json";
let currentTestResponseType = testResponseType.EMPTY;

export const mockGenerationStatsResponse = (): Promise<
    AxiosResponse<unknown>
> => {
    if (currentTestResponseType === testResponseType.ERROR) {
        return mockErrorResponse();
    } else if (currentTestResponseType === testResponseType.EMPTY) {
        return mockEmptyResponse();
    } else if (currentTestResponseType === testResponseType.INVALID) {
        return mockInvalidResponse();
    } else if (currentTestResponseType === testResponseType.FILTERED_OUT) {
        return mockFilteredOutResponse();
    }
    return mockDefaultResponse();
};

export const setGenStatsResponseType = (responseType: testResponseType) => {
    currentTestResponseType = responseType;
};

const mockDefaultResponse = (): Promise<AxiosResponse<unknown>> => {
    return Promise.resolve(
        createMockResponse(
            mockGenStatsResource
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
            test: {}
        }) as unknown as AxiosResponse<unknown>
    );
};

export const mockErrorResponse = () => {
    return Promise.reject(
        createMockResponse(new Error("Internal Server Error"), 500)
    );
};

export const mockFilteredOutResponse = (): Promise<AxiosResponse<unknown>> => {
    return Promise.resolve(
        createMockResponse({
            num_expected_iterations: 10,
            num_finished_iterations: 7,
            iteration_data: [
                {
                    min_date: "1998-12-01T00:00:00",
                    max_date: "1998-12-31T00:00:00",
                    anchor_date: "1998-12-01T00:00:00",
                    num_train_rows: 1000,
                    num_entities: 1000,
                    num_targets: 2,
                    index: 0
                }
            ]
        }) as unknown as AxiosResponse<unknown>
    );
};
