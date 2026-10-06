// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AxiosResponse } from "axios";
import { testResponseType } from "tests/testResponseType";
import { createMockResponse } from "tests/testUtil";

import {
    NoMatchingValue,
    RegressionHeatMapResponse,
    SmallerActualValue,
    SmallerPredictionValue
} from "../MockResponses/RegressionHeatMap";

let currentTestResponseType = "";

export const mockRegressionHeatMapResponses = (
    trainingJobId: string
): Promise<AxiosResponse<unknown>> => {
    if (currentTestResponseType === testResponseType.ERROR) {
        return mockErrorResponse();
    } else if (currentTestResponseType === testResponseType.EMPTY) {
        return mockEmptyResponse();
    } else if (currentTestResponseType === testResponseType.INVALID) {
        return mockInvalidResponse();
    }
    return mockDefaultResponse(trainingJobId);
};

export const setRegressionHeatMapResponseType = (
    responseType: testResponseType
) => {
    currentTestResponseType = responseType;
};

const mockDefaultResponse = (
    trainingJobId: string
): Promise<AxiosResponse<unknown>> => {
    let regressionHeatMap = RegressionHeatMapResponse;
    if (trainingJobId === "trainingjob-0c4623b61d0643d6a29b32c90c62f39c") {
        regressionHeatMap = SmallerPredictionValue;
    } else if (
        trainingJobId === "trainingjob-6f8321a97ace4897b30eb4a0600e41bd"
    ) {
        regressionHeatMap = SmallerActualValue;
    } else if (
        trainingJobId === "trainingjob-afbc69bc7e60441ab732d6ad95f4c687"
    ) {
        regressionHeatMap = NoMatchingValue;
    }
    return Promise.resolve(
        createMockResponse(
            regressionHeatMap
        ) as unknown as AxiosResponse<unknown>
    );
};

export const mockEmptyResponse = (): Promise<AxiosResponse<unknown>> => {
    return Promise.resolve(
        createMockResponse({}) as unknown as AxiosResponse<unknown>
    );
};

const mockInvalidResponse = (): Promise<AxiosResponse<unknown>> => {
    return Promise.reject({
        response: {
            status: 422,
            statusText: "Unprocessable Entity",
            data: "Failed to get Regression Heatmap Stats!"
        }
    });
};

export const mockErrorResponse = () => {
    return Promise.reject(
        createMockResponse(new Error("Internal Server Error"), 500)
    );
};
