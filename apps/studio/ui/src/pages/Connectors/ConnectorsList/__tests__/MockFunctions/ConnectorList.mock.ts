// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AxiosResponse } from "axios";
import { ConnectorResponse } from "gen/v0/types";
import { testResponseType } from "tests/testResponseType";
import { createMockResponse } from "tests/testUtil";

import mockGetConnectors from "../MockResponses/ConnectorList.json";
import invalidConnectors from "../MockResponses/ConnectorListInvalid.json";

let currentTestResponseType = "";
let customConnectorListResponse: ConnectorResponse[] | null = null;

const currentConnector = (value: (typeof mockGetConnectors)[number]) => ({
    ...value,
    project_id: "project-tests",
    identity_id: value.identity_id
});

const currentConnectors = mockGetConnectors.map(currentConnector);

export const mockConnectorListResponse = (): Promise<
    AxiosResponse<ConnectorResponse[]>
> => {
    if (customConnectorListResponse) {
        return Promise.resolve(
            createMockResponse(
                customConnectorListResponse
            ) as unknown as AxiosResponse<ConnectorResponse[]>
        );
    }
    switch (currentTestResponseType) {
        case testResponseType.ERROR:
            return mockErrorConnectorListResponse();
        case testResponseType.EMPTY:
            return mockEmptyConnectorListResponse();
        case testResponseType.INVALID:
            return mockInvalidConnectorListResponse();
        case testResponseType.ALL:
            return mockAllConnectorsListResponse();
        default:
            // No customer connector is provisioned implicitly.
            return mockDefaultConnectorResponse();
    }
};

export const setConnectorListResponseType = (responseType: string = "") => {
    currentTestResponseType = responseType;
    return currentTestResponseType;
};

export const setCustomConnectorListResponse = (
    response: ConnectorResponse[] | null
) => {
    customConnectorListResponse = response;
};

export const getTestResponseType = () => {
    return currentTestResponseType;
};

const mockAllConnectorsListResponse = (): Promise<
    AxiosResponse<ConnectorResponse[]>
> => {
    const mockresp = currentConnectors;
    return Promise.resolve(
        createMockResponse(mockresp) as unknown as AxiosResponse<
            ConnectorResponse[]
        >
    );
};

const mockDefaultConnectorResponse = (): Promise<
    AxiosResponse<ConnectorResponse[]>
> => {
    return Promise.resolve(
        createMockResponse([]) as unknown as AxiosResponse<ConnectorResponse[]>
    );
};

const mockEmptyConnectorListResponse = (): Promise<
    AxiosResponse<ConnectorResponse[]>
> => {
    return Promise.resolve(
        createMockResponse([]) as unknown as AxiosResponse<ConnectorResponse[]>
    );
};

const mockInvalidConnectorListResponse = (): Promise<
    AxiosResponse<ConnectorResponse[]>
> => {
    return Promise.resolve(
        createMockResponse(invalidConnectors) as unknown as AxiosResponse<
            ConnectorResponse[]
        >
    );
};

const mockErrorConnectorListResponse = () => {
    return Promise.reject(
        createMockResponse(new Error("Internal Server Error"), 500)
    );
};

export const expectedConnectorListResponse = () => {
    return currentConnectors as unknown as ConnectorResponse[];
};

export const connectorListWithDistinctNamesResponse = () => {
    const [warehouseConnector, lakehouseConnector] =
        expectedConnectorListResponse();
    return [
        {
            ...warehouseConnector,
            id: "warehouse-connector-id",
            name: "Customer Warehouse"
        },
        {
            ...lakehouseConnector,
            id: "lakehouse-connector-id",
            name: "Analytics Lakehouse"
        }
    ];
};

export const getConnectorResponseById = (id: string) => {
    const mockresp = currentConnectors.find((d) => d.id === id);
    return mockresp as unknown as ConnectorResponse;
};

export const getInvalidConnectorResponseById = (id: string) => {
    const mockresp = [...invalidConnectors].find((d) => d.id === id);
    return mockresp as unknown as ConnectorResponse;
};
