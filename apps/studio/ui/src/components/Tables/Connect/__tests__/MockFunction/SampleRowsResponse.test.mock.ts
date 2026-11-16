// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AxiosResponse } from "axios";
import { SourceTableDataResponse } from "gen/v0/types";
import { testResponseType } from "tests/testResponseType";
import { createMockResponse } from "tests/testUtil";

import {
    mockGetCustomerSampleRow,
    mockGetSampleRows
} from "../MockResponse/SampleRowsResponse";

let currentTestResponseType = "";
// Per-column stype overrides for the sample response, letting a test make
// the sample-based recommendations diverge from the stats-free column
// preview (the fixtures agree by default, so no recommendation is emitted).
let sampleColumnStypeOverrides: Record<string, string> | null = null;

export const setSampleRowsStypeOverrides = (
    overrides: Record<string, string> | null = null
) => {
    sampleColumnStypeOverrides = overrides;
};

export const mockSampleRowsResponse = (
    source_table?: string
): Promise<AxiosResponse<SourceTableDataResponse>> => {
    if (currentTestResponseType === testResponseType.ERROR) {
        return mockErrorSampleRows();
    } else if (currentTestResponseType === testResponseType.EMPTY) {
        return mockEmptySampleRows();
    } else if (currentTestResponseType === testResponseType.INVALID) {
        return mockInvalidSampleRows();
    }

    return mockDefaultSampleRows(source_table);
};

export const setSampleRowsResponseType = (responseType: string = "") => {
    currentTestResponseType = responseType;
};

const mockDefaultSampleRows = (
    source_table?: string
): Promise<AxiosResponse<SourceTableDataResponse>> => {
    const responseData =
        source_table === "customer"
            ? mockGetCustomerSampleRow
            : mockGetSampleRows;
    const data = responseData.rows.map((row) =>
        Object.fromEntries(
            responseData.columns.map((column, index) => [
                column.name,
                row[index]
            ])
        )
    );

    const cols = structuredClone(responseData.columns).map((column) =>
        sampleColumnStypeOverrides?.[column.name]
            ? { ...column, stype: sampleColumnStypeOverrides[column.name] }
            : column
    );

    return Promise.resolve(
        createMockResponse({
            table_name: source_table ?? "",
            cols,
            sample_rows: JSON.stringify({ data })
        }) as unknown as AxiosResponse<SourceTableDataResponse>
    );
};

export const mockEmptySampleRows = (): Promise<
    AxiosResponse<SourceTableDataResponse>
> => {
    return Promise.resolve(
        createMockResponse({
            table_name: "",
            cols: [],
            sample_rows: null
        }) as unknown as AxiosResponse<SourceTableDataResponse>
    );
};

export const mockInvalidSampleRows = (): Promise<
    AxiosResponse<SourceTableDataResponse>
> => {
    return Promise.resolve(
        createMockResponse({
            table_name: "invalid",
            cols: [],
            sample_rows: "not-json"
        }) as unknown as AxiosResponse<SourceTableDataResponse>
    );
};

export const expectedSampleRows = () => {
    return Promise.resolve(
        mockGetSampleRows as unknown as SourceTableDataResponse
    );
};

export const mockErrorSampleRows = () => {
    return Promise.reject(
        createMockResponse(new Error("Internal Server Error"), 500)
    );
};
