// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AxiosResponse } from "axios";
import { SourceTableDataResponse } from "gen/v0/types";
import { testResponseType } from "tests/testResponseType";
import { createMockResponse } from "tests/testUtil";

import { SourceColumn } from "types/tables";

import {
    mockGetColumnResponse,
    mockGetCustomerColumnResponse
} from "../MockResponse/ColumnResponse";

let currentTestResponseType = "";

export const mockColumnResponse = (
    source_table?: string
): Promise<AxiosResponse<SourceTableDataResponse>> => {
    if (currentTestResponseType === testResponseType.ERROR) {
        return mockErrorColumnResponse();
    } else if (currentTestResponseType === testResponseType.EMPTY) {
        return mockEmptyColumnResponse();
    } else if (currentTestResponseType === testResponseType.INVALID) {
        return mockInvalidColumnResponse();
    }
    return mockDefaultColumnResponse(source_table);
};

export const setColumnResponseType = (responseType: string = "") => {
    currentTestResponseType = responseType;
};

const mockDefaultColumnResponse = (
    source_table?: string
): Promise<AxiosResponse<SourceTableDataResponse>> => {
    const responseData =
        source_table === "customer"
            ? mockGetCustomerColumnResponse
            : mockGetColumnResponse;

    return Promise.resolve(
        createMockResponse({
            table_name: source_table ?? "",
            cols: structuredClone(responseData.source_cols)
        }) as unknown as AxiosResponse<SourceTableDataResponse>
    );
};

export const mockEmptyColumnResponse = (): Promise<
    AxiosResponse<SourceTableDataResponse>
> => {
    return Promise.resolve(
        createMockResponse({
            table_name: "",
            cols: []
        }) as unknown as AxiosResponse<SourceTableDataResponse>
    );
};

export const mockInvalidColumnResponse = (): Promise<
    AxiosResponse<SourceTableDataResponse>
> => {
    return Promise.reject({
        response: {
            status: 422,
            statusText: "Unprocessable Entity",
            data: "Failed to load columns"
        }
    });
};

export const mockErrorColumnResponse = () => {
    return Promise.reject(
        createMockResponse(new Error("Internal Server Error"), 500)
    );
};

export const expectedColumnResponse = (): SourceColumn[] => {
    return [...mockGetColumnResponse.source_cols] as unknown as SourceColumn[];
};
