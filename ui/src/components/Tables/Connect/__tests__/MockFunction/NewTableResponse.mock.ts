// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AxiosResponse } from "axios";
import { testResponseType } from "tests/testResponseType";
import { createMockResponse } from "tests/testUtil";

import { TableDetails } from "types/tables";

import {
    mockNewCustomerTableResponse,
    mockNewTable
} from "../MockResponse/NewTableResponse";

let currentTestResponseType = "";

export const mockSampleRowsResponse = (
    source_table?: string
): Promise<AxiosResponse<TableDetails>> => {
    if (currentTestResponseType === testResponseType.ERROR) {
        return mockErrorNewTableResponse();
    } else if (currentTestResponseType === testResponseType.EMPTY) {
        return mockEmptyNewTableResponse();
    } else if (currentTestResponseType === testResponseType.INVALID) {
        return mockInvalidNewTableResponse();
    }

    return mockDefaultNewTableResponse(source_table);
};

export const setNewTableResponseType = (responseType: string = "") => {
    currentTestResponseType = responseType;
};

const mockDefaultNewTableResponse = (
    source_table?: string
): Promise<AxiosResponse<TableDetails>> => {
    const responseData =
        source_table === "customer"
            ? mockNewCustomerTableResponse
            : mockNewTable;

    return Promise.resolve(
        createMockResponse(
            responseData
        ) as unknown as AxiosResponse<TableDetails>
    );
};

export const mockEmptyNewTableResponse = (): Promise<
    AxiosResponse<TableDetails>
> => {
    return Promise.resolve(
        createMockResponse([]) as unknown as AxiosResponse<TableDetails>
    );
};

export const mockInvalidNewTableResponse = (): Promise<
    AxiosResponse<TableDetails>
> => {
    return Promise.resolve(
        createMockResponse({
            cols: [],
            end_time_col: false,
            pkey: "",
            source_table: {},
            time_col: ""
        }) as unknown as AxiosResponse<TableDetails>
    );
};

export const expectedNewTableResponse = (
    source_table?: string
): TableDetails => {
    const responseData =
        source_table === "customer"
            ? mockNewCustomerTableResponse
            : mockNewTable;

    return { ...responseData } as unknown as TableDetails;
};

export const mockErrorNewTableResponse = () => {
    return Promise.reject(
        createMockResponse(new Error("Internal Server Error"), 500)
    );
};
