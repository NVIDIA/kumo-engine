// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AxiosResponse } from "axios";
import { testResponseType } from "tests/testResponseType";
import { createMockResponse } from "tests/testUtil";

import { TableDetails } from "types/tables";

import { tableWithNoConnector } from "../MockResponse/TableDetailWithNoConnector";

let currentTestResponseType = "";
export const mockTableDetailNoConnectorResponse = (): Promise<
    AxiosResponse<TableDetails>
> => {
    if (currentTestResponseType === testResponseType.ERROR) {
        return mockErrorTableDetailNoConnectorResponse();
    } else if (currentTestResponseType === testResponseType.EMPTY) {
        return mockEmptyTableDetailNoConnectorResponse();
    } else if (currentTestResponseType === testResponseType.INVALID) {
        return mockInvalidTableDetailNoConnectorResponse();
    }
    return mockDefaultTableNoConnectorResponse();
};

export const setTestTableDetailNoConnectorResponseType = (
    responseType: string = ""
) => {
    currentTestResponseType = responseType;
};

const mockDefaultTableNoConnectorResponse = (): Promise<
    AxiosResponse<TableDetails>
> => {
    return Promise.resolve(
        createMockResponse(
            tableWithNoConnector
        ) as unknown as AxiosResponse<TableDetails>
    );
};

export const mockEmptyTableDetailNoConnectorResponse = (): Promise<
    AxiosResponse<TableDetails>
> => {
    return Promise.resolve(
        createMockResponse([]) as unknown as AxiosResponse<TableDetails>
    );
};

export const mockInvalidTableDetailNoConnectorResponse = (): Promise<
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

export const expectedTableDetailsNoConnectorResponse = (): TableDetails => {
    return { ...tableWithNoConnector } as unknown as TableDetails;
};

export const mockErrorTableDetailNoConnectorResponse = () => {
    return Promise.reject(
        createMockResponse(new Error("Internal Server Error"), 500)
    );
};
