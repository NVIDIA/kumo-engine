// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AxiosResponse } from "axios";
import { PQueryResource } from "gen/v0/types";
import { testResponseType } from "tests/testResponseType";
import { createMockResponse } from "tests/testUtil";

let currentTestResponseType = "";

export const mockAddTagResponse = (): Promise<
    AxiosResponse<PQueryResource>
> => {
    if (currentTestResponseType === testResponseType.ERROR) {
        return mockErrorAddTagResponse();
    } else if (currentTestResponseType === testResponseType.EMPTY) {
        return mockEmptyAddTagResponse();
    } else if (currentTestResponseType === testResponseType.INVALID) {
        return mockInvalidAddTagResponse();
    }

    return mockDefaultAddTagResponse();
};

export const setAddTagTestResponseType = (responseType: string = "") => {
    currentTestResponseType = responseType;
};

const mockDefaultAddTagResponse = (): Promise<
    AxiosResponse<PQueryResource>
> => {
    return Promise.resolve(
        createMockResponse(true) as unknown as AxiosResponse<PQueryResource>
    );
};

export const mockEmptyAddTagResponse = (): Promise<
    AxiosResponse<PQueryResource>
> => {
    return Promise.resolve(
        createMockResponse([]) as unknown as AxiosResponse<PQueryResource>
    );
};

export const mockInvalidAddTagResponse = (): Promise<
    AxiosResponse<PQueryResource>
> => {
    return Promise.resolve(
        createMockResponse(0) as unknown as AxiosResponse<PQueryResource>
    );
};

export const mockErrorAddTagResponse = () => {
    return Promise.reject(
        createMockResponse(new Error("Internal Server Error"), 500)
    );
};
