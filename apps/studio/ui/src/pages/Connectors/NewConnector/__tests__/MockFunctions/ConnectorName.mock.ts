// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AxiosResponse } from "axios";
import { testResponseType } from "tests/testResponseType";
import { createMockResponse } from "tests/testUtil";

import {
    alreadyNameExists,
    invalidName,
    validName
} from "../MockResponses/ConnectorNameValidation";

let currentTestResponseType = "";

export const mockConnectorNameValidationResponse = (
    name: string
): Promise<AxiosResponse<unknown>> => {
    switch (currentTestResponseType) {
        case testResponseType.ERROR:
            return mockErrorResponse();
        case testResponseType.INVALID:
            return mockInvalidNameResponse(name);
        case testResponseType.EXISTS:
            return mockAlreadyExistsNameResponse(name);
        default:
            return mockValidNameResponse();
    }
};

export const setConnectorNameResponseType = (responseType: string = "") => {
    currentTestResponseType = responseType;
    return currentTestResponseType;
};

const mockValidNameResponse = (): Promise<AxiosResponse<unknown>> => {
    return Promise.resolve(
        createMockResponse(validName) as unknown as AxiosResponse<unknown>
    );
};

const mockInvalidNameResponse = (
    name: string
): Promise<AxiosResponse<unknown>> => {
    const resp = { ...invalidName };
    resp.error_msg = resp.error_msg.replace("<name>", name);
    return Promise.resolve(
        createMockResponse(resp) as unknown as AxiosResponse<unknown>
    );
};

const mockAlreadyExistsNameResponse = (
    name: string
): Promise<AxiosResponse<unknown>> => {
    const resp = { ...alreadyNameExists };
    resp.error_msg = resp.error_msg.replace("<name>", name);
    return Promise.resolve(
        createMockResponse(resp) as unknown as AxiosResponse<unknown>
    );
};

const mockErrorResponse = () => {
    return Promise.reject(
        createMockResponse(new Error("Internal Server Error"), 500)
    );
};
