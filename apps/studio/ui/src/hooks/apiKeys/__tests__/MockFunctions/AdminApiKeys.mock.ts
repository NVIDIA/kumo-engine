// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { APIKeySummary, APIKeyType, CreateAPIKeyResponse } from "gen/v0/types";
import { testResponseType } from "tests/testResponseType";
import { createMockResponse } from "tests/testUtil";

export const mockAdminApiKeysList: APIKeySummary[] = [
    {
        id: "admin-key-001",
        name: "Admin Key One",
        type: APIKeyType.ADMIN,
        owner_user_id: "user-001",
        owner_display_name: "Alice Admin",
        created_time: "2026-01-10T08:00:00.000Z",
        value_preview: "kmo_adm...001"
    },
    {
        // USER type is intentional — the admin list endpoint returns all key types
        id: "admin-key-002",
        name: "Admin Key Two",
        type: APIKeyType.USER,
        owner_user_id: "user-002",
        owner_display_name: "Bob User",
        created_time: "2026-02-05T12:00:00.000Z",
        value_preview: "kmo_usr...002"
    }
];

export const mockCreateAdminApiKeyResponse: CreateAPIKeyResponse = {
    id: "admin-key-new",
    name: "New Admin Key",
    type: APIKeyType.ADMIN,
    created_time: "2026-04-01T00:00:00.000Z",
    value_preview: "kmo_adm...new",
    value: "kmo_adm_super_secret_value"
};

let adminApiKeysResponseType = "";

export const mockAdminApiKeysResponse = () => {
    switch (adminApiKeysResponseType) {
        case testResponseType.ERROR:
            return mockAdminErrorResponse();
        default:
            return Promise.resolve(createMockResponse(mockAdminApiKeysList));
    }
};

export const setAdminApiKeysResponseType = (type: string = "") => {
    adminApiKeysResponseType = type;
};

let createAdminApiKeyResponseType = "";

export const mockCreateAdminApiKeyApiResponse = () => {
    switch (createAdminApiKeyResponseType) {
        case testResponseType.ERROR:
            return mockAdminErrorResponse();
        default:
            return Promise.resolve(
                createMockResponse(mockCreateAdminApiKeyResponse)
            );
    }
};

export const setCreateAdminApiKeyResponseType = (type: string = "") => {
    createAdminApiKeyResponseType = type;
};

let deleteAdminApiKeyResponseType = "";

export const mockDeleteAdminApiKeyResponse = () => {
    switch (deleteAdminApiKeyResponseType) {
        case testResponseType.ERROR:
            return mockAdminErrorResponse();
        default:
            return Promise.resolve(createMockResponse(null));
    }
};

export const setDeleteAdminApiKeyResponseType = (type: string = "") => {
    deleteAdminApiKeyResponseType = type;
};

const mockAdminErrorResponse = () => {
    return Promise.reject(
        createMockResponse(new Error("Internal Server Error"), 500)
    );
};
