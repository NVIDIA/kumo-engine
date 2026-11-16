// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AdminResponse } from "gen/v0/types";
import { testResponseType } from "tests/testResponseType";
import { createMockResponse } from "tests/testUtil";

import { AxiosResponse } from "types/common";

import adminInfoResponse from "../MockResponses/AdminInfo.response.json";
let currentTestResponseType = "";
let mockUserIsAdmin = true;

export const setMockAdminUserIsAdmin = (isAdmin: boolean) => {
    mockUserIsAdmin = isAdmin;
};

export const mockAdminInfoResponse = (): Promise<
    AxiosResponse<AdminResponse>
> => {
    switch (currentTestResponseType) {
        case testResponseType.ERROR:
            return mockErrorResponse();
        case testResponseType.INVALID:
            return mockInvalidResponse();
        case testResponseType.EMPTY:
            return mockEmptyResponse();
        default:
            return mockValidResponse();
    }
};

export const setAdminInfoResponseType = (responseType: string = "") => {
    currentTestResponseType = responseType;
    return currentTestResponseType;
};

const mockValidResponse = (): Promise<AxiosResponse<AdminResponse>> => {
    return Promise.resolve(
        createMockResponse({
            ...adminInfoResponse,
            user_is_customer_admin: mockUserIsAdmin
        }) as unknown as AxiosResponse<AdminResponse>
    );
};

const mockInvalidResponse = (): Promise<AxiosResponse<AdminResponse>> => {
    return Promise.reject({
        response: {
            status: 422,
            statusText: "Unprocessable Entity",
            data: "Failed to get Admin Info"
        }
    });
};

const mockErrorResponse = () => {
    return Promise.reject(
        createMockResponse(new Error("Internal Server Error"), 500)
    );
};

const mockEmptyResponse = (): Promise<AxiosResponse<AdminResponse>> => {
    return Promise.resolve(
        createMockResponse({}) as unknown as AxiosResponse<AdminResponse>
    );
};
