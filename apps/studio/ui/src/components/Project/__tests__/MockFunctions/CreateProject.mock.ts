// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { testResponseType } from "tests/testResponseType";
import { createMockResponse } from "tests/testUtil";

let currentTestResponseType = "";

export const mockCreateProjectResponse = () => {
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
                    data: "Project slug 'fraud-detection' is already in use."
                }
            });
        default:
            return mockValidResponse();
    }
};

export const setCreateProjectResponseType = (responseType: string = "") => {
    currentTestResponseType = responseType;
    return currentTestResponseType;
};

const mockValidResponse = () => {
    return Promise.resolve(createMockResponse({ id: "new-project-123" }));
};

const mockInvalidResponse = () => {
    return Promise.reject({
        response: {
            status: 422,
            statusText: "Unprocessable Entity",
            data: "Failed to create Project"
        }
    });
};

const mockErrorResponse = () => {
    return Promise.reject(
        createMockResponse(new Error("Internal Server Error"), 500)
    );
};
