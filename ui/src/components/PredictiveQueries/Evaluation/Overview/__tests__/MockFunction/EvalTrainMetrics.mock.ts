// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AxiosResponse } from "axios";
import { testResponseType } from "tests/testResponseType";
import { createMockResponse } from "tests/testUtil";

import { eval_metrics_res_link_pred } from "components/PredictiveQueries/Evaluation/Metrics/__tests__/MockResponses/EvalMetricsResponse";

let currentTestResponseType = "";

export const mockEvalTrainMetricsResponse = (
    jobId: string
): Promise<AxiosResponse<unknown>> => {
    if (currentTestResponseType === testResponseType.ERROR) {
        return mockErrorResponse();
    } else if (currentTestResponseType === testResponseType.EMPTY) {
        return mockEmptyResponse();
    } else if (currentTestResponseType === testResponseType.INVALID) {
        return mockInvalidResponse();
    }
    return mockDefaultResponse(jobId);
};

export const setmockEvalMetricsResponseType = (
    responseType: testResponseType
) => {
    currentTestResponseType = responseType;
};

const mockDefaultResponse = (
    jobId: string
): Promise<AxiosResponse<unknown>> => {
    let metricsResponse = null;

    if (jobId === "trainingjob-00047000000000000000000000000000") {
        metricsResponse = eval_metrics_res_link_pred;
    }

    if (!metricsResponse) {
        throw new Error(`Unknown jobId: ${jobId}`);
    }

    return Promise.resolve(
        createMockResponse(metricsResponse) as unknown as AxiosResponse<unknown>
    );
};

export const mockEmptyResponse = (): Promise<AxiosResponse<unknown>> => {
    return Promise.resolve(
        createMockResponse({ metrics: {} }) as unknown as AxiosResponse<unknown>
    );
};

export const mockInvalidResponse = (): Promise<AxiosResponse<unknown>> => {
    return Promise.resolve(
        createMockResponse({
            "classes": undefined
        }) as unknown as AxiosResponse<unknown>
    );
};

export const mockErrorResponse = () => {
    return Promise.reject(
        createMockResponse(new Error("Internal Server Error"), 500)
    );
};
