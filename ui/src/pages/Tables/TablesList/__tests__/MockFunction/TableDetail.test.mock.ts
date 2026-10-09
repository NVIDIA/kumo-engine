// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AxiosResponse } from "axios";
import { testResponseType } from "tests/testResponseType";
import { createMockResponse } from "tests/testUtil";

import { TableResponse } from "types/tables";

import mockGetInvalidTablesDetail from "../MockResponse/InvalidTableDetail.test.response.json";
import {
    mockGetTablesDetail,
    mockGetTablesDetailStock,
    mockGetTablesDetailWithAllFields
} from "../MockResponse/TableDetail.test.response";

let currentTestResponseType = "";
let tableUsesProjectUploadConnector = false;

export const setTableUsesProjectUploadConnector = (enabled = false) => {
    tableUsesProjectUploadConnector = enabled;
};

export const mockTableDetailResponse = (
    tableId: string
): Promise<AxiosResponse<TableResponse>> => {
    if (currentTestResponseType === testResponseType.ERROR) {
        return mockErrorTableDetailResponse();
    } else if (currentTestResponseType === testResponseType.EMPTY) {
        return mockEmptyTableDetailResponse();
    } else if (currentTestResponseType === testResponseType.INVALID) {
        return mockInvalidTableDetailResponse();
    }
    return mockDefaultTableResponse(tableId);
};

export const setTestResponseType = (responseType: string = "") => {
    currentTestResponseType = responseType;
};

const mockDefaultTableResponse = (
    tableId: string
): Promise<AxiosResponse<TableResponse>> => {
    let tableDetail;
    if (tableId === "NA.NA-b656de36982dfe8ea68ab7ab1ebac6a1") {
        tableDetail = mockGetTablesDetailStock;
    } else if (tableId === "NA.NA-d5828d994e8679119b05ec8f07de920a") {
        tableDetail = mockGetTablesDetailWithAllFields;
    } else {
        tableDetail = mockGetTablesDetail;
    }
    const response = structuredClone(tableDetail);
    if (tableUsesProjectUploadConnector) {
        response.table.source_table.connector_id = "project_upload_connector";
        response.connector = {
            id: "project_upload_connector",
            name: "Local Upload"
        };
    }
    return Promise.resolve(
        createMockResponse(response) as unknown as AxiosResponse<TableResponse>
    );
};

export const mockEmptyTableDetailResponse = (): Promise<
    AxiosResponse<TableResponse>
> => {
    return Promise.resolve(
        createMockResponse([]) as unknown as AxiosResponse<TableResponse>
    );
};

export const mockInvalidTableDetailResponse = (): Promise<
    AxiosResponse<TableResponse>
> => {
    return Promise.resolve(
        createMockResponse(
            mockGetInvalidTablesDetail
        ) as unknown as AxiosResponse<TableResponse>
    );
};

export const expectedTableDetailsResponse = (name?: string): TableResponse => {
    return (name === "stock"
        ? mockGetTablesDetailStock
        : mockGetTablesDetail) as unknown as TableResponse;
};

export const mockErrorTableDetailResponse = () => {
    return Promise.reject(
        createMockResponse(new Error("Internal Server Error"), 500)
    );
};
