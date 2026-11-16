// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AxiosResponse } from "axios";
import { SourceTableListResponse } from "gen/v0/types";
import { testResponseType } from "tests/testResponseType";
import { createMockResponse } from "tests/testUtil";

import mockGetSourceTables from "../MockResponse/SourceTables.json";

let currentResponseType = "";

export const mockSourceTablesResponse = (): Promise<
    AxiosResponse<SourceTableListResponse>
> => {
    switch (currentResponseType) {
        case testResponseType.EMPTY:
            return mockEmptySourceTablesResponse();
        case testResponseType.INVALID:
        case testResponseType.ERROR:
            return mockErrorSourceTablesResponse();
        default:
            return mockSourceTablesResponses();
    }
};

export const setSourceTablesResponsesType = (responseType: string = "") => {
    currentResponseType = responseType;
    return currentResponseType;
};

const mockSourceTablesResponses = (): Promise<
    AxiosResponse<SourceTableListResponse>
> => {
    return Promise.resolve(
        createMockResponse(
            structuredClone(mockGetSourceTables)
        ) as unknown as AxiosResponse<SourceTableListResponse>
    );
};

const mockEmptySourceTablesResponse = (): Promise<
    AxiosResponse<SourceTableListResponse>
> => {
    return Promise.resolve(
        createMockResponse({
            table_names: []
        }) as unknown as AxiosResponse<SourceTableListResponse>
    );
};

const mockErrorSourceTablesResponse = () => {
    return Promise.reject(
        createMockResponse(new Error("Internal Server Error"), 500)
    );
};

export const expectedSourceTablesResponse = () => {
    return mockGetSourceTables as unknown as SourceTableListResponse;
};
