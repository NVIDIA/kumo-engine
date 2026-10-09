// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AxiosResponse } from "axios";
import { testResponseType } from "tests/testResponseType";
import { createMockResponse } from "tests/testUtil";

import { GraphDefinition, ValidateGraphResponse } from "types/graphs";

import { mockNewGraphResponse } from "../MockResponse/NewGraphResponse";

let currentTestResponseType = "";

export const mockValidateGraphResponse = (): Promise<
    AxiosResponse<unknown>
> => {
    switch (currentTestResponseType) {
        case testResponseType.EMPTY:
            return mockGraphValidateEmptyResponse();
        case testResponseType.ERROR:
            return mockErrorResponse();
        case testResponseType.INVALID:
            return mockInvalidResponse();
        case testResponseType.INVALID_RESPONSE:
            return mockInvalidGraphResponse();
        case testResponseType.UNREACHABLE_TABLES:
            return mockUnreachableTablesResponse();
        case testResponseType.RUNNING:
            return mockPendingValidateResponse();
        default:
            return mockDefaultValidateResponse();
    }
};

/**
 * A validate response that never resolves, simulating a round-trip
 * that is still in flight.
 */
const mockPendingValidateResponse = (): Promise<AxiosResponse<unknown>> => {
    return new Promise<AxiosResponse<unknown>>(() => {});
};

export const setGraphValidateResponseType = (responseType: string = "") => {
    currentTestResponseType = responseType;
    return currentTestResponseType;
};

const mockDefaultValidateResponse = (): Promise<
    AxiosResponse<Array<ValidateGraphResponse>>
> => {
    return Promise.resolve(
        createMockResponse({
            "warnings": [],
            "errors": [],
            "info_items": []
        }) as unknown as AxiosResponse<Array<ValidateGraphResponse>>
    );
};

const mockGraphValidateEmptyResponse = (): Promise<
    AxiosResponse<Array<ValidateGraphResponse>>
> => {
    return Promise.resolve(
        createMockResponse([]) as unknown as AxiosResponse<
            Array<ValidateGraphResponse>
        >
    );
};

const mockInvalidGraphResponse = (): Promise<AxiosResponse> => {
    return Promise.resolve({
        data: {
            errors: [{ message: "Error in Creating Graph" }]
        },
        status: 422,
        statusText: "Unprocessable Entity",
        headers: {},
        config: {}
    } as AxiosResponse);
};

const mockInvalidResponse = (): Promise<AxiosResponse<unknown>> => {
    return Promise.reject({
        response: {
            status: 422,
            statusText: "Unprocessable Entity",
            data: "Error in Creating Graph"
        }
    });
};

const mockErrorResponse = () => {
    return Promise.reject(
        createMockResponse(new Error("Internal Server Error"), 500)
    );
};

const mockUnreachableTablesResponse = (): Promise<AxiosResponse> => {
    return Promise.resolve({
        data: {
            warnings: [],
            errors: [
                {
                    title: "Graph Validation Failure",
                    message:
                        "Invalid Graph Parameters: Tables {'ltv_test.transactions_no_time'} are not reachable from any table in the graph. Please make sure that all tables are fully linked into a single graph via column groups, or remove the tables that cannot be linked."
                }
            ],
            info_items: []
        },
        status: 200,
        statusText: "OK",
        headers: {},
        config: {}
    } as AxiosResponse);
};

export const expectedNewGraphResponse = (): GraphDefinition => {
    return { ...mockNewGraphResponse } as unknown as GraphDefinition;
};
