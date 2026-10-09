// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AxiosResponse } from "axios";
import { PQueryResource } from "gen/v0/types";
import { testResponseType } from "tests/testResponseType";
import { createMockResponse } from "tests/testUtil";

let currentTestResponseType = "";

export const mockDeleteTagResponse = (): Promise<
    AxiosResponse<PQueryResource>
> => {
    if (currentTestResponseType === testResponseType.ERROR) {
        return mockErrorDeleteTagResponse();
    } else if (currentTestResponseType === testResponseType.EMPTY) {
        return mockEmptyDeleteTagResponse();
    } else if (currentTestResponseType === testResponseType.INVALID) {
        return mockInvalidDeleteTagResponse();
    }

    return mockDefaultDeleteTagResponse();
};

export const setDeleteTagTestResponseType = (responseType: string = "") => {
    currentTestResponseType = responseType;
};

const mockDefaultDeleteTagResponse = (): Promise<
    AxiosResponse<PQueryResource>
> => {
    return Promise.resolve(
        createMockResponse(true) as unknown as AxiosResponse<PQueryResource>
    );
};

export const mockEmptyDeleteTagResponse = (): Promise<
    AxiosResponse<PQueryResource>
> => {
    return Promise.resolve(
        createMockResponse([]) as unknown as AxiosResponse<PQueryResource>
    );
};

export const mockInvalidDeleteTagResponse = (): Promise<
    AxiosResponse<PQueryResource>
> => {
    return Promise.resolve(
        createMockResponse(0) as unknown as AxiosResponse<PQueryResource>
    );
};

export const mockErrorDeleteTagResponse = () => {
    return Promise.reject(
        createMockResponse(new Error("Internal Server Error"), 500)
    );
};
