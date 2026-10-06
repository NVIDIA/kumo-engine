// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { OrganizationMember } from "gen/v0/types";
import { testResponseType } from "tests/testResponseType";
import { createMockResponse } from "tests/testUtil";

import { AxiosResponse } from "types/common";

import orgMembers from "../MockResponses/OrgMembers.response.json";
let currentTestResponseType = "";

export const mockOrgMembersInfoResponse = (): Promise<
    AxiosResponse<OrganizationMember>
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

export const setOrgMembersResponseType = (responseType: string = "") => {
    currentTestResponseType = responseType;
    return currentTestResponseType;
};

const mockValidResponse = (): Promise<AxiosResponse<OrganizationMember>> => {
    return Promise.resolve(
        createMockResponse(
            orgMembers
        ) as unknown as AxiosResponse<OrganizationMember>
    );
};

const mockInvalidResponse = (): Promise<AxiosResponse<OrganizationMember>> => {
    return Promise.reject({
        response: {
            status: 422,
            statusText: "Unprocessable Entity",
            data: "Failed to get Members Info"
        }
    });
};

const mockEmptyResponse = (): Promise<AxiosResponse<OrganizationMember>> => {
    return Promise.resolve(
        createMockResponse([]) as unknown as AxiosResponse<OrganizationMember>
    );
};

const mockErrorResponse = () => {
    return Promise.reject(
        createMockResponse(new Error("Internal Server Error"), 500)
    );
};
