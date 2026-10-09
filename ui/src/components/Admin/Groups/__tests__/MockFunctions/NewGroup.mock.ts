// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { KumoGroupDBEntry } from "gen/v0/types";
import { testResponseType } from "tests/testResponseType";
import { createMockResponse } from "tests/testUtil";

import { AxiosResponse } from "types/common";

import groupsResponse from "../MockResponses/Groups.response.json";

let currentTestResponseType = "";

export const mockCreateGroupResponse = (): Promise<
    AxiosResponse<KumoGroupDBEntry>
> => {
    switch (currentTestResponseType) {
        case testResponseType.ERROR:
            return mockErrorResponse();
        case testResponseType.INVALID:
            return mockInvalidResponse();
        case "conflict":
            return Promise.reject({
                response: {
                    status: 409,
                    statusText: "Conflict",
                    data: "Group slug 'risk-owners' is already in use."
                }
            });
        default:
            return mockValidResponse();
    }
};

export const setCreateGroupResponseType = (responseType: string = "") => {
    currentTestResponseType = responseType;
    return currentTestResponseType;
};

const mockValidResponse = (): Promise<AxiosResponse<KumoGroupDBEntry>> => {
    return Promise.resolve(
        createMockResponse(
            groupsResponse[0]
        ) as unknown as AxiosResponse<KumoGroupDBEntry>
    );
};

const mockInvalidResponse = (): Promise<AxiosResponse<KumoGroupDBEntry>> => {
    return Promise.reject({
        response: {
            status: 422,
            statusText: "Unprocessable Entity",
            data: "Failed to create Group"
        }
    });
};

const mockErrorResponse = () => {
    return Promise.reject(
        createMockResponse(new Error("Internal Server Error"), 500)
    );
};
