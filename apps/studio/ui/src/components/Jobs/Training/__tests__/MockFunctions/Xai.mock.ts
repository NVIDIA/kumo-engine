// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AxiosResponse } from "axios";
import { testResponseType } from "tests/testResponseType";
import { createMockResponse } from "tests/testUtil";

import mockXaiResponse from "../MockResponses/Xai.test.json";
let currentTestResponseType = "";

export const mockTrainingXaiResponse = (): Promise<AxiosResponse<unknown>> => {
    if (currentTestResponseType === testResponseType.ERROR) {
        return mockErrorResponse();
    } else if (currentTestResponseType === testResponseType.EMPTY) {
        return mockEmptyResponse();
    } else if (currentTestResponseType === testResponseType.INVALID) {
        return mockInvalidResponse();
    } else if (currentTestResponseType === testResponseType.FILTERED_OUT) {
        return mockNaResponse();
    } else if (currentTestResponseType === testResponseType.VALID) {
        return mockDifColTypeResponse();
    }
    return mockDefaultResponse();
};

export const setXaiTestResponseType = (responseType: testResponseType) => {
    currentTestResponseType = responseType;
};

const mockDefaultResponse = (): Promise<AxiosResponse<unknown>> => {
    return Promise.resolve(
        createMockResponse(mockXaiResponse) as unknown as AxiosResponse<unknown>
    );
};

export const mockEmptyResponse = (): Promise<AxiosResponse<unknown>> => {
    return Promise.resolve(
        createMockResponse({
            tables: {},
            navigation: {}
        }) as unknown as AxiosResponse<unknown>
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

export const mockNaResponse = () => {
    return Promise.resolve(
        createMockResponse({
            ...mockXaiResponse,
            tables: {
                ...mockXaiResponse.tables,
                "FINANCE_TRANS": {
                    ...mockXaiResponse.tables.FINANCE_TRANS,
                    "NA_COL": {
                        stype: "categorical",
                        variance_score: "N/A",
                        cohort: mockXaiResponse.tables.FINANCE_TRANS[
                            "ACCOUNT (1 hop)"
                        ].cohort
                    }
                }
            }
        }) as unknown as AxiosResponse<unknown>
    );
};

export const mockDifColTypeResponse = () => {
    return Promise.resolve(
        createMockResponse({
            ...mockXaiResponse,
            tables: {
                ...mockXaiResponse.tables,
                "FINANCE_TRANS": {
                    ...mockXaiResponse.tables.FINANCE_TRANS,
                    "TIMESTAMP_COL": {
                        stype: "timestamp",
                        variance_score: 0.5,
                        cohort: mockXaiResponse.tables.FINANCE_TRANS[
                            "ACCOUNT (1 hop)"
                        ].cohort
                    }
                }
            }
        }) as unknown as AxiosResponse<unknown>
    );
};
