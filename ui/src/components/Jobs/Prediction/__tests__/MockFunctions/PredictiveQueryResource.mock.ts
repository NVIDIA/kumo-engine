// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AxiosResponse } from "axios";
import { PQueryResource } from "gen/v0/types";
import { testResponseType } from "tests/testResponseType";
import { createMockResponse } from "tests/testUtil";

import { PQueryResourceResponse } from "../MockResponses/PredictiveQueryResource";

let currentTestResponseType = "";

export const mockPredictiveQueryResoureResponse = (): Promise<
    AxiosResponse<PQueryResource>
> => {
    if (currentTestResponseType === testResponseType.ERROR) {
        return mockErrorPredictiveQueryResoureResponse();
    } else if (currentTestResponseType === testResponseType.EMPTY) {
        return mockEmptyPredictiveQueryResoureResponse();
    } else if (currentTestResponseType === testResponseType.INVALID) {
        return mockInvalidPredictiveQueryResoureResponse();
    }

    return mockDefaultPredictiveQueryResoureResponse();
};

export const setPredictiveQueryResoureTestResponseType = (
    responseType: string = ""
) => {
    currentTestResponseType = responseType;
};

const mockDefaultPredictiveQueryResoureResponse = (): Promise<
    AxiosResponse<PQueryResource>
> => {
    return Promise.resolve(
        createMockResponse(
            PQueryResourceResponse
        ) as unknown as AxiosResponse<PQueryResource>
    );
};

export const mockEmptyPredictiveQueryResoureResponse = (): Promise<
    AxiosResponse<PQueryResource>
> => {
    return Promise.resolve(
        createMockResponse([]) as unknown as AxiosResponse<PQueryResource>
    );
};

export const mockInvalidPredictiveQueryResoureResponse = (): Promise<
    AxiosResponse<PQueryResource>
> => {
    return Promise.resolve(
        createMockResponse({
            query_string: "",
            graph: [],
            name: 0,
            desc: false
        }) as unknown as AxiosResponse<PQueryResource>
    );
};

export const expectedPredictiveQueryResoureResponse = () => {
    return PQueryResourceResponse;
};

export const mockErrorPredictiveQueryResoureResponse = () => {
    return Promise.reject(
        createMockResponse(new Error("Internal Server Error"), 500)
    );
};
