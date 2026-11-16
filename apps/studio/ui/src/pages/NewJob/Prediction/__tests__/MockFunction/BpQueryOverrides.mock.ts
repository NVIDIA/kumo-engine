// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AxiosResponse } from "axios";
import { EditableQuery } from "gen/v1/types";
import { testResponseType } from "tests/testResponseType";
import { createMockResponse } from "tests/testUtil";

import mockGetBpOverrideFilterResponse from "../MockResponse/BpQueryOverrides.json";

let currentTestResponseType = "";

export const mockBpOverrideFilterResponse = (): Promise<
    AxiosResponse<EditableQuery>
> => {
    if (currentTestResponseType === testResponseType.ERROR) {
        return mockErrorBpOverrideFilterResponse();
    } else if (currentTestResponseType === testResponseType.EMPTY) {
        return mockEmptyBpOverrideFilterResponse();
    } else if (currentTestResponseType === testResponseType.INVALID) {
        return mockInvalidBpOverrideFilterResponse();
    }
    return mockDefaultBpOverrideFilterResponse();
};

export const setTestBpOverrideFilterResponseType = (
    responseType: string = ""
) => {
    currentTestResponseType = responseType;
};

const mockDefaultBpOverrideFilterResponse = (): Promise<
    AxiosResponse<EditableQuery>
> => {
    return Promise.resolve(
        createMockResponse(
            mockGetBpOverrideFilterResponse
        ) as unknown as AxiosResponse<EditableQuery>
    );
};

export const mockEmptyBpOverrideFilterResponse = (): Promise<
    AxiosResponse<EditableQuery>
> => {
    return Promise.resolve(
        createMockResponse([]) as unknown as AxiosResponse<EditableQuery>
    );
};

export const mockInvalidBpOverrideFilterResponse = (): Promise<
    AxiosResponse<EditableQuery>
> => {
    return Promise.resolve(
        createMockResponse({
            text: false,
            editable_fields: [],
            validation_response: null
        }) as unknown as AxiosResponse<EditableQuery>
    );
};

export const mockErrorBpOverrideFilterResponse = () => {
    return Promise.reject(
        createMockResponse(new Error("Internal Server Error"), 500)
    );
};

export const expectedBpOverrideFilterResponse = () => {
    return Promise.resolve(mockGetBpOverrideFilterResponse);
};
