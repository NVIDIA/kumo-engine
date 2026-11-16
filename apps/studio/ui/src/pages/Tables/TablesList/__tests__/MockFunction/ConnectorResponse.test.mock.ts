// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AxiosResponse } from "axios";
import { ConnectorResponse } from "gen/v0/types";
import { testResponseType } from "tests/testResponseType";
import { createMockResponse } from "tests/testUtil";

import mockGetConnectors from "../../../../Connectors/ConnectorsList/__tests__/MockResponses/ConnectorList.json";

let currentTestResponseType = "";

export const mockConnectorResponse = (): Promise<
    AxiosResponse<ConnectorResponse>
> => {
    if (currentTestResponseType === testResponseType.ERROR) {
        return mockErrorConnectorResponse();
    } else if (currentTestResponseType === testResponseType.EMPTY) {
        return mockEmptyConnectorResponse();
    } else if (currentTestResponseType === testResponseType.INVALID) {
        return mockInvalidConnectorResponse();
    }
    return mockDefaultConnectorResponse();
};

export const setConTestResponseType = (responseType: string = "") => {
    currentTestResponseType = responseType;
};

const mockDefaultConnectorResponse = (): Promise<
    AxiosResponse<ConnectorResponse>
> => {
    const connector = mockGetConnectors[0];
    return Promise.resolve(
        createMockResponse({
            ...connector,
            project_id: "project-tests",
            identity_id: connector.identity_id,
            name: connector.name
        }) as unknown as AxiosResponse<ConnectorResponse>
    );
};

export const mockEmptyConnectorResponse = (): Promise<
    AxiosResponse<ConnectorResponse>
> => {
    return Promise.resolve(
        createMockResponse({}) as unknown as AxiosResponse<ConnectorResponse>
    );
};

export const mockInvalidConnectorResponse = (): Promise<
    AxiosResponse<ConnectorResponse>
> => {
    return Promise.resolve(
        createMockResponse({
            config: {
                name: "",
                root_dir: "",
                type: ""
            },
            name: "",
            root_dir: "",
            type: "",
            description: "",
            id: "",
            timestamps: {
                created_at: "",
                updated_at: ""
            }
        }) as unknown as AxiosResponse<ConnectorResponse>
    );
};

export const mockErrorConnectorResponse = () => {
    return Promise.reject(
        createMockResponse(new Error("Internal Server Error"), 500)
    );
};

export const expectedConnectorResponse = () => {
    return Promise.resolve(mockGetConnectors as unknown as ConnectorResponse[]);
};
