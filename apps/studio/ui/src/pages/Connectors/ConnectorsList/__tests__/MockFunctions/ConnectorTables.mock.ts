// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AxiosResponse } from "axios";
import { testResponseType } from "tests/testResponseType";
import { createMockResponse } from "tests/testUtil";

import { TableResponse } from "types/tables";

import mockGetConnectorsDetail from "../MockResponses/ConnectorTables.json";

let currentResponseType = "";

export const mockConnectorTablesResponse = (): Promise<
    AxiosResponse<TableResponse>
> => {
    switch (currentResponseType) {
        case testResponseType.EMPTY:
            return mockEmptyConnectorTablesResponse();
        case testResponseType.INVALID:
        case testResponseType.ERROR:
            return mockErrorConnectorTablesResponse();
        default:
            return mockConnectorTablesResponses();
    }
};

export const setConnectorTablesResponseType = (responseType: string = "") => {
    currentResponseType = responseType;
    return currentResponseType;
};

const mockConnectorTablesResponses = (): Promise<
    AxiosResponse<TableResponse>
> => {
    return Promise.resolve(
        createMockResponse(
            mockGetConnectorsDetail
        ) as unknown as AxiosResponse<TableResponse>
    );
};

const mockEmptyConnectorTablesResponse = (): Promise<
    AxiosResponse<TableResponse>
> => {
    return Promise.resolve(
        createMockResponse([]) as unknown as AxiosResponse<TableResponse>
    );
};

const mockErrorConnectorTablesResponse = () => {
    return Promise.reject(
        createMockResponse(new Error("Internal Server Error"), 500)
    );
};

export const expectedConnectorDetailsResponse = () => {
    return mockGetConnectorsDetail as unknown as TableResponse[];
};
