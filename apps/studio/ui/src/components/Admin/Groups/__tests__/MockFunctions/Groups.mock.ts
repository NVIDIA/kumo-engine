// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { KumoGroupDBEntry } from "gen/v0/types";
import { testResponseType } from "tests/testResponseType";
import { createMockResponse } from "tests/testUtil";

import { AxiosResponse } from "types/common";

import groupsResponse from "../MockResponses/Groups.response.json";

let currentTestResponseType = "";

export const mockGroupsResponse = (): Promise<
    AxiosResponse<KumoGroupDBEntry[]>
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

export const setGroupsResponseType = (responseType: string = "") => {
    currentTestResponseType = responseType;
    return currentTestResponseType;
};

const mockValidResponse = (): Promise<AxiosResponse<KumoGroupDBEntry[]>> => {
    return Promise.resolve(
        createMockResponse(groupsResponse) as unknown as AxiosResponse<
            KumoGroupDBEntry[]
        >
    );
};

const mockInvalidResponse = (): Promise<AxiosResponse<KumoGroupDBEntry[]>> => {
    return Promise.reject({
        response: {
            status: 422,
            statusText: "Unprocessable Entity",
            data: "Failed to get Groups"
        }
    });
};

const mockEmptyResponse = (): Promise<AxiosResponse<KumoGroupDBEntry[]>> => {
    return Promise.resolve(
        createMockResponse([]) as unknown as AxiosResponse<KumoGroupDBEntry[]>
    );
};

const mockErrorResponse = () => {
    return Promise.reject(
        createMockResponse(new Error("Internal Server Error"), 500)
    );
};
