// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AxiosResponse } from "axios";
import { testResponseType } from "tests/testResponseType";
import { createMockResponse } from "tests/testUtil";

import {
    ValidateQueryStringRequest,
    ValidateQueryStringResponse
} from "types/predictiveQueries";

import { validatePayloadDeatails } from "../MockResponses/ValidatePayload";

let currentTestResponseType = "";

export const mockValidatePayloadResponse = (): Promise<
    AxiosResponse<ValidateQueryStringResponse>
> => {
    if (currentTestResponseType === testResponseType.INVALID) {
        return mockInvalidValidatePayloadResponse();
    } else if (currentTestResponseType === testResponseType.ERROR) {
        return mockEmptyValidatePayloadResponse();
    }
    return mockDefaultValidatePayloadResponse();
};

export const setTestValidatePayloadResponseType = (
    responseType: string = ""
) => {
    currentTestResponseType = responseType;
};

const mockDefaultValidatePayloadResponse = (): Promise<
    AxiosResponse<ValidateQueryStringResponse>
> => {
    return Promise.resolve(
        createMockResponse({
            items: []
        }) as unknown as AxiosResponse<ValidateQueryStringResponse>
    );
};

export const mockEmptyValidatePayloadResponse = (): Promise<
    AxiosResponse<ValidateQueryStringResponse>
> => {
    return Promise.resolve(
        createMockResponse(
            []
        ) as unknown as AxiosResponse<ValidateQueryStringResponse>
    );
};

export const mockInvalidValidatePayloadResponse = (): Promise<
    AxiosResponse<ValidateQueryStringResponse>
> => {
    return Promise.resolve(
        createMockResponse({
            items: [
                {
                    type: 0,
                    title: "Invalid Predictive Query",
                    description:
                        "Encountered the following issues during parsing: Errors:\n1. Invalid Syntax. The target (PREDICT) clause in this query is empty or invalid. This clause describes what you want to predict and will be used to generate labels for your prediction problem.\n2. Invalid Syntax. Line 1, col 8; Input Mismatch, got 'customers', but expected one of '(', FQN_COLUMN, WILDCARD_COLUMN, NOT, AGGR.",
                    cta: null
                }
            ]
        }) as unknown as AxiosResponse<ValidateQueryStringResponse>
    );
};

export const expectedValidatePayloadResponse = () => {
    return Promise.resolve(
        validatePayloadDeatails as unknown as ValidateQueryStringRequest
    );
};
