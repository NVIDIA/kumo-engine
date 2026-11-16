// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ConnectorResponse, ResourceTimestamps } from "gen/v0/types";

export type ConnectorID = string;

export type ConnectorConfig = ConnectorResponse["config"];

export type JsConnector = {
    id: ConnectorID;
    projectId: string;
    identityId: string;
    name: string;
    description: string;
    name_link: string;
    created: string;
    dateCreated: string;
    /** Raw ISO timestamp used as the sort key, never a display string. */
    sortCreated: Date | string;
    updated: string;
    dateUpdated: string;
    /** Raw ISO timestamp used as the sort key, never a display string. */
    sortUpdated: Date | string;
    source: string;
    config: ConnectorConfig;
    showDelete: boolean;
    /** Raw ISO timestamps from the API; created/updated above are pre-formatted. */
    timestamps: ResourceTimestamps;
};
