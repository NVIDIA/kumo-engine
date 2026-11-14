// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ConnectorResponse } from "gen/v0/types";
import { testResponseType } from "tests/testResponseType";
import { createMockResponse } from "tests/testUtil";

export const mockCopiedConnector: ConnectorResponse = {
    id: "connector_00000000000000000000000000000c09",
    project_id: "project-target",
    identity_id: "identity_00000000000000000000000000000001",
    name: "Sales analytics",
    description: "",
    timestamps: {
        created_at: "2026-01-01T00:00:00Z",
        updated_at: "2026-01-01T00:00:00Z"
    },
    config: {
        type: "s3",
        target: { mode: "rooted", bucket: "sales", prefix: "", region: null }
    }
} as ConnectorResponse;

let copyConnectorResponseType = "";

export const mockCopyConnectorResponse = () => {
    if (copyConnectorResponseType === testResponseType.ERROR) {
        return Promise.reject({
            response: {
                status: 409,
                data: {
                    detail: "Connector named 'Sales analytics' already exists in this project"
                }
            }
        });
    }
    return Promise.resolve(createMockResponse(mockCopiedConnector));
};

export const setCopyConnectorResponseType = (type: string = "") => {
    copyConnectorResponseType = type;
};
