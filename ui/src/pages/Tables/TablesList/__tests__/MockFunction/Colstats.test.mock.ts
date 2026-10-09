// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AxiosResponse } from "axios";
import { ColStatsResponse } from "gen/v0/types";
import { testResponseType } from "tests/testResponseType";
import { createMockResponse } from "tests/testUtil";

// import { ColStatsResponse } from "types/tables";
import {
    mockGetColStats,
    mockGetColStatsFinanceOrder,
    mockGetColStatsFinanceTrans,
    mockGetColStatsWithAllFields
} from "../MockResponse/Colstats";

let currentTestResponseType = "";

export const mockColStatsResponse = (
    tableId: string
): Promise<AxiosResponse<ColStatsResponse>> => {
    if (currentTestResponseType === testResponseType.ERROR) {
        return mockErrorColStatsResponse();
    } else if (currentTestResponseType === testResponseType.EMPTY) {
        return mockEmptyColStatsResponse();
    } else if (currentTestResponseType === testResponseType.INVALID) {
        return mockInvalidColStatsResponse();
    }

    return mockDefaultColStatsResponse(tableId);
};

export const setColStatsResponseType = (responseType: string = "") => {
    currentTestResponseType = responseType;
};

const mockDefaultColStatsResponse = (
    tableId: string
): Promise<AxiosResponse<ColStatsResponse>> => {
    let colStats = mockGetColStats;
    if (
        [
            "NA.NA-bc150313117da0324bfe31ca4ff499e5",
            "table-bc150313117da0324bfe31ca4ff499e5@1664307208"
        ].includes(tableId)
    ) {
        colStats = mockGetColStatsFinanceTrans;
    } else if (
        [
            "NA.NA-ccaf9a4a904985c04cd0be0c96d6857c",
            "table-ccaf9a4a904985c04cd0be0c96d6857c@1659426523"
        ].includes(tableId)
    ) {
        colStats = mockGetColStatsFinanceOrder;
    } else if (tableId === "NA.NA-d5828d994e8679119b05ec8f07de920a") {
        colStats = mockGetColStatsWithAllFields;
    }
    return Promise.resolve(
        createMockResponse(
            colStats
        ) as unknown as AxiosResponse<ColStatsResponse>
    );
};

export const mockEmptyColStatsResponse = (): Promise<
    AxiosResponse<ColStatsResponse>
> => {
    return Promise.resolve(
        createMockResponse([]) as unknown as AxiosResponse<ColStatsResponse>
    );
};

export const mockInvalidColStatsResponse = (): Promise<
    AxiosResponse<ColStatsResponse>
> => {
    return Promise.resolve(
        createMockResponse({
            approx: true,
            columns: [],
            rows_used: 0,
            statistics_available: false
        }) as unknown as AxiosResponse<ColStatsResponse>
    );
};

export const expectedColStatsResponse = (name?: string): ColStatsResponse => {
    if (name === "FINANCE_TRANS") {
        return {
            ...mockGetColStatsFinanceTrans
        } as unknown as ColStatsResponse;
    } else if (name === "FINANCE_ORDERS") {
        return {
            ...mockGetColStatsFinanceOrder
        } as unknown as ColStatsResponse;
    } else {
        return { ...mockGetColStats } as unknown as ColStatsResponse;
    }
};

export const mockErrorColStatsResponse = () => {
    return Promise.reject(
        createMockResponse(new Error("Internal Server Error"), 500)
    );
};
