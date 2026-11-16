// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AxiosResponse } from "axios";
import { testResponseType } from "tests/testResponseType";
import { createMockResponse } from "tests/testUtil";

import { TableListResponse } from "types/tables";

import mockGetTables from "../MockResponse/TableList.test.response.json";

let currentTestResponseType = "";

// Opt-in override for tests that need a specific list payload (e.g. the "Kind"
// filter partitioning). Defaults to null so existing tests are unaffected.
let customTableListRows: TableListResponse[] | null = null;

export const setCustomTableListRows = (
    rows: TableListResponse[] | null
): void => {
    customTableListRows = rows;
};

export const mockTableListResponse = (): Promise<
    AxiosResponse<TableListResponse[]>
> => {
    if (customTableListRows !== null) {
        return Promise.resolve(
            createMockResponse(customTableListRows) as unknown as AxiosResponse<
                TableListResponse[]
            >
        );
    }
    if (currentTestResponseType === testResponseType.ERROR) {
        return mockErrorTableListResponse();
    } else if (currentTestResponseType === testResponseType.EMPTY) {
        return mockEmptyTableListResponse();
    } else if (currentTestResponseType === testResponseType.INVALID) {
        return mockInvalidTableListResponse();
    }
    return mockTablesListResponse();
};

export const setTableListResponseType = (responseType: string = "") => {
    currentTestResponseType = responseType;
};

const mockTablesListResponse = (): Promise<
    AxiosResponse<TableListResponse[]>
> => {
    return Promise.resolve(
        createMockResponse(mockGetTables) as unknown as AxiosResponse<
            TableListResponse[]
        >
    );
};

export const mockEmptyTableListResponse = (): Promise<
    AxiosResponse<TableListResponse[]>
> => {
    return Promise.resolve(
        createMockResponse([]) as unknown as AxiosResponse<TableListResponse[]>
    );
};

export const mockInvalidTableListResponse = (): Promise<
    AxiosResponse<TableListResponse[]>
> => {
    return Promise.resolve(
        createMockResponse([
            {
                id: "",
                name: 5498,
                timestamps: {
                    created_at: "",
                    updated_at: ""
                }
            }
        ]) as unknown as AxiosResponse<TableListResponse[]>
    );
};

export const expectedTableListResponse = () => {
    return [...mockGetTables];
};

export const mockErrorTableListResponse = () => {
    return Promise.reject(
        createMockResponse(new Error("Internal Server Error"), 500)
    );
};
