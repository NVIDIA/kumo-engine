// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AxiosResponse } from "axios";
import { ModelPlansResponse } from "gen/v0/types";
import { testResponseType } from "tests/testResponseType";
import { createMockResponse } from "tests/testUtil";

import mockModelPlanResponse from "../MockResponses/ModelPlanResponse.json";

let currentTestResponseType = "";

export const mockModelPlanResponses = (): Promise<
    AxiosResponse<ModelPlansResponse[]>
> => {
    if (currentTestResponseType === testResponseType.ERROR) {
        return mockErrorModelPlanResponse();
    } else if (currentTestResponseType === testResponseType.EMPTY) {
        return mockEmptyModelPlanResponse();
    }
    return mockDefaultModelPlanResponse();
};

export const setTestModelPlanResponseType = (responseType: string = "") => {
    currentTestResponseType = responseType;
};

const mockDefaultModelPlanResponse = (): Promise<
    AxiosResponse<ModelPlansResponse[]>
> => {
    return Promise.resolve(
        createMockResponse(mockModelPlanResponse) as unknown as AxiosResponse<
            ModelPlansResponse[]
        >
    );
};

export const mockEmptyModelPlanResponse = (): Promise<
    AxiosResponse<ModelPlansResponse[]>
> => {
    return Promise.resolve(
        createMockResponse([]) as unknown as AxiosResponse<ModelPlansResponse[]>
    );
};

export const mockErrorModelPlanResponse = () => {
    return Promise.reject(
        createMockResponse(new Error("Internal Server Error"), 500)
    );
};

export const expectedModelTagsResponse = () => {
    return Promise.resolve(
        mockModelPlanResponse as unknown as ModelPlansResponse[]
    );
};
