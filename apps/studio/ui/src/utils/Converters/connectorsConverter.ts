// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ConnectorResponse } from "gen/v0/types";

import { JsConnector } from "types/connectors";

import { getProjectPath } from "utils/Constants";
import {
    formatToLongLocalDateString,
    formatToShortLocalDateString
} from "utils/timeConverter";

export function connectorListFromConnectorResp(
    connectors: ConnectorResponse[],
    sortData = true
): JsConnector[] {
    const values = connectors.map(connectorFromConnectorResp);
    if (sortData) {
        values.sort((a, b) => (a.sortCreated > b.sortCreated ? -1 : 1));
        values.sort((a, b) => (a.sortUpdated > b.sortUpdated ? -1 : 1));
    }
    return values;
}

export function connectorFromConnectorResp(
    connector: ConnectorResponse
): JsConnector {
    const dateCreated = formatToShortLocalDateString(
        connector.timestamps.created_at
    );
    const dateUpdated = formatToLongLocalDateString(
        connector.timestamps.updated_at
    );
    return {
        id: connector.id,
        projectId: connector.project_id,
        identityId: connector.identity_id,
        name: connector.name,
        description: connector.description ?? "",
        name_link:
            getProjectPath(connector.project_id).Connectors + connector.id,
        created: formatToLongLocalDateString(connector.timestamps.created_at),
        dateCreated,
        // Sort keys stay raw ISO so ordering never depends on the display
        // format or the viewer's timezone.
        sortCreated: connector.timestamps.created_at,
        updated: formatToLongLocalDateString(connector.timestamps.updated_at),
        dateUpdated,
        sortUpdated: connector.timestamps.updated_at,
        timestamps: connector.timestamps,
        source: connector.config.type ?? "",
        config: connector.config,
        showDelete: true
    };
}
