// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { APIKeySummary, APIKeyType, CreateAPIKeyResponse } from "gen/v0/types";
import { testResponseType } from "tests/testResponseType";
import { createMockResponse } from "tests/testUtil";

export const mockUserApiKeysList: APIKeySummary[] = [
    {
        id: "user-key-001",
        name: "My Personal Key",
        type: APIKeyType.USER,
        owner_user_id: "user-100",
        owner_display_name: "Current User",
        created_time: "2026-03-01T09:00:00.000Z",
        value_preview: "kmo_usr...001"
    }
];

export const mockCreateUserApiKeyResponse: CreateAPIKeyResponse = {
    id: "user-key-new",
    name: "New User Key",
    type: APIKeyType.USER,
    created_time: "2026-04-01T00:00:00.000Z",
    value_preview: "kmo_usr...new",
    value: "kmo_usr_super_secret_value"
};

// --- List ---

let currentUserApiKeysResponseType = "";

export const mockUserApiKeysResponse = () => {
    switch (currentUserApiKeysResponseType) {
        case testResponseType.ERROR:
            return mockUserListErrorResponse();
        case testResponseType.INVALID:
            return mockUserListInvalidResponse();
        default:
            return mockUserListValidResponse();
    }
};

export const setUserApiKeysResponseType = (responseType: string = "") => {
    currentUserApiKeysResponseType = responseType;
    return currentUserApiKeysResponseType;
};

const mockUserListValidResponse = () =>
    Promise.resolve(createMockResponse(mockUserApiKeysList));

const mockUserListInvalidResponse = () =>
    Promise.reject({
        response: {
            status: 422,
            statusText: "Unprocessable Entity",
            data: "Invalid query parameters"
        }
    });

const mockUserListErrorResponse = () =>
    Promise.reject({
        response: {
            status: 500,
            statusText: "Internal Server Error",
            data: "Internal Server Error"
        }
    });

// --- Create ---

let currentCreateUserApiKeyResponseType = "";

export const mockCreateUserApiKeyApiResponse = () => {
    switch (currentCreateUserApiKeyResponseType) {
        case testResponseType.ERROR:
            return mockCreateUserErrorResponse();
        case testResponseType.INVALID:
            return mockCreateUserInvalidResponse();
        default:
            return mockCreateUserValidResponse();
    }
};

export const setCreateUserApiKeyResponseType = (responseType: string = "") => {
    currentCreateUserApiKeyResponseType = responseType;
    return currentCreateUserApiKeyResponseType;
};

const mockCreateUserValidResponse = () =>
    Promise.resolve(createMockResponse(mockCreateUserApiKeyResponse));

const mockCreateUserInvalidResponse = () =>
    Promise.reject({
        response: {
            status: 422,
            statusText: "Unprocessable Entity",
            data: "API key name already exists"
        }
    });

const mockCreateUserErrorResponse = () =>
    Promise.reject({
        response: {
            status: 500,
            statusText: "Internal Server Error",
            data: "Internal Server Error"
        }
    });

// --- Delete ---

let currentDeleteUserApiKeyResponseType = "";

export const mockDeleteUserApiKeyResponse = () => {
    switch (currentDeleteUserApiKeyResponseType) {
        case testResponseType.ERROR:
            return mockDeleteUserErrorResponse();
        case testResponseType.INVALID:
            return mockDeleteUserInvalidResponse();
        default:
            return mockDeleteUserValidResponse();
    }
};

export const setDeleteUserApiKeyResponseType = (responseType: string = "") => {
    currentDeleteUserApiKeyResponseType = responseType;
    return currentDeleteUserApiKeyResponseType;
};

const mockDeleteUserValidResponse = () =>
    Promise.resolve(createMockResponse(null));

const mockDeleteUserInvalidResponse = () =>
    Promise.reject({
        response: {
            status: 422,
            statusText: "Unprocessable Entity",
            data: "API key not found"
        }
    });

const mockDeleteUserErrorResponse = () =>
    Promise.reject({
        response: {
            status: 500,
            statusText: "Internal Server Error",
            data: "Internal Server Error"
        }
    });
