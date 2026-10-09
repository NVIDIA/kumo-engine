// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ConnectorResponse } from "gen/v0/types";

import {
    formatToLongLocalDateString,
    formatToShortLocalDateString
} from "utils/timeConverter";

import {
    connectorFromConnectorResp,
    connectorListFromConnectorResp
} from "../connectorsConverter";

const connector = (
    id: string,
    createdAt: string,
    updatedAt: string
): ConnectorResponse => ({
    id,
    project_id: "project-sales",
    identity_id: "identity-sales",
    name: id,
    description: "",
    timestamps: { created_at: createdAt, updated_at: updatedAt },
    config: {
        type: "databricks",
        catalog: "sales",
        schema_name: "analytics",
        warehouse_id: "warehouse-1"
    }
});

describe("connectorListFromConnectorResp", () => {
    it("sorts project connectors by updated time descending", () => {
        const first = connector(
            "first",
            "2025-01-01T00:00:00Z",
            "2025-01-02T00:00:00Z"
        );
        const second = connector(
            "second",
            "2025-01-02T00:00:00Z",
            "2025-01-03T00:00:00Z"
        );

        expect(connectorListFromConnectorResp([first, second])).toEqual([
            expect.objectContaining({ id: "second" }),
            expect.objectContaining({ id: "first" })
        ]);
    });
});

describe("connectorFromConnectorResp", () => {
    it("preserves project and identity coordinates", () => {
        const response = connector(
            "sales",
            "2025-01-01T00:00:00Z",
            "2025-01-02T00:00:00Z"
        );

        expect(connectorFromConnectorResp(response)).toEqual({
            id: "sales",
            projectId: "project-sales",
            identityId: "identity-sales",
            name: "sales",
            description: "",
            name_link: "/project/project-sales/connectors/sales",
            created: formatToLongLocalDateString(
                response.timestamps.created_at
            ),
            dateCreated: formatToShortLocalDateString(
                response.timestamps.created_at
            ),
            sortCreated: response.timestamps.created_at,
            updated: formatToLongLocalDateString(
                response.timestamps.updated_at
            ),
            dateUpdated: formatToLongLocalDateString(
                response.timestamps.updated_at
            ),
            sortUpdated: response.timestamps.updated_at,
            timestamps: response.timestamps,
            source: "databricks",
            config: response.config,
            showDelete: true
        });
    });
});
