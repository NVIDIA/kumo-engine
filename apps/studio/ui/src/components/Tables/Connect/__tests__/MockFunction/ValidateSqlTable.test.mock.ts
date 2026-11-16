// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AxiosResponse } from "axios";
import { SqlTableResolveResponse } from "gen/v1/types";
import { testResponseType } from "tests/testResponseType";
import { createMockResponse } from "tests/testUtil";

import {
    mockCostGatedValidateResponse,
    mockEmptyColumnsValidateResponse,
    mockValidateSqlTableResponse,
    scopedConnectorErrorMessage
} from "../MockResponse/ValidateSqlTableResponse";

let currentTestResponseType = "";

export const mockValidateSqlTableResponseFn = (): Promise<
    AxiosResponse<SqlTableResolveResponse>
> => {
    if (currentTestResponseType === testResponseType.ERROR) {
        return mockScopedConnectorError();
    }
    if (currentTestResponseType === testResponseType.EMPTY) {
        return mockEmptyColumns();
    }
    // PARTIAL == schema confirmed but the preview was cost-gated (no rows).
    if (currentTestResponseType === testResponseType.PARTIAL) {
        return mockCostGated();
    }
    return mockValidResponse();
};

export const setValidateSqlTableResponseType = (responseType: string = "") => {
    currentTestResponseType = responseType;
};

const mockValidResponse = (): Promise<AxiosResponse<SqlTableResolveResponse>> =>
    Promise.resolve(
        createMockResponse(
            structuredClone(mockValidateSqlTableResponse)
        ) as unknown as AxiosResponse<SqlTableResolveResponse>
    );

const mockEmptyColumns = (): Promise<AxiosResponse<SqlTableResolveResponse>> =>
    Promise.resolve(
        createMockResponse(
            structuredClone(mockEmptyColumnsValidateResponse)
        ) as unknown as AxiosResponse<SqlTableResolveResponse>
    );

const mockCostGated = (): Promise<AxiosResponse<SqlTableResolveResponse>> =>
    Promise.resolve(
        createMockResponse(
            structuredClone(mockCostGatedValidateResponse)
        ) as unknown as AxiosResponse<SqlTableResolveResponse>
    );

// The scoped (RBAC group) connector rejection arrives as a `ConnectorError`.
const mockScopedConnectorError = () =>
    Promise.reject({
        response: {
            status: 400,
            statusText: "Bad Request",
            data: scopedConnectorErrorMessage
        }
    });
