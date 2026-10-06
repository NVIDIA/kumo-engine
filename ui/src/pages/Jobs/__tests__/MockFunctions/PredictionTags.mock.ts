// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AxiosResponse } from "axios";
import { JobTags } from "gen/v0/types";
import { testResponseType } from "tests/testResponseType";
import { createMockResponse } from "tests/testUtil";

import mockGetPredictionTags from "../MockResponses/PredictionTags.test.json";

let currentTestResponseType = "";

export const expectedJobsListTableHeader =
    "Job IDJob TypeStartedStatusSourceDurationTags";

export const mockPredictionTagsResponse = (): Promise<
    AxiosResponse<JobTags[]>
> => {
    if (currentTestResponseType === testResponseType.ERROR) {
        return mockErrorPredictionTagsResponse();
    } else if (currentTestResponseType === testResponseType.EMPTY) {
        return mockEmptyPredictionTagsResponse();
    } else if (currentTestResponseType === testResponseType.INVALID) {
        return mockInvalidPredictionTagsResponse();
    }
    return mockDefaultPredictionTagsResponse();
};

export const setTestPredictionTagsResponseType = (
    responseType: string = ""
) => {
    currentTestResponseType = responseType;
};

const mockDefaultPredictionTagsResponse = (): Promise<
    AxiosResponse<JobTags[]>
> => {
    return Promise.resolve(
        createMockResponse(mockGetPredictionTags) as unknown as AxiosResponse<
            JobTags[]
        >
    );
};

export const mockEmptyPredictionTagsResponse = (): Promise<
    AxiosResponse<JobTags[]>
> => {
    return Promise.resolve(
        createMockResponse([]) as unknown as AxiosResponse<JobTags[]>
    );
};

export const mockInvalidPredictionTagsResponse = (): Promise<
    AxiosResponse<JobTags[]>
> => {
    return Promise.resolve(
        createMockResponse({
            timestamps: {
                created_at: "",
                updated_at: ""
            },
            tags: ""
        }) as unknown as AxiosResponse<JobTags[]>
    );
};

export const mockErrorPredictionTagsResponse = () => {
    return Promise.reject(
        createMockResponse(new Error("Internal Server Error"), 500)
    );
};

export const expectedPredictionTagsResponse = (): JobTags[] => {
    return { ...mockGetPredictionTags } as unknown as JobTags[];
};
