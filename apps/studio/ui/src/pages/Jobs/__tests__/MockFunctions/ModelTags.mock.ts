// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AxiosResponse } from "axios";
import { JobTags } from "gen/v0/types";
import { testResponseType } from "tests/testResponseType";
import { createMockResponse } from "tests/testUtil";

import mockGetModelTags from "../MockResponses/ModelTags.test.json";

let currentTestResponseType = "";

export const expectedJobsListTableHeader =
    "Job IDJob TypeStartedStatusSourceDurationTags";

export const mockModelTagsResponse = (): Promise<AxiosResponse<JobTags[]>> => {
    if (currentTestResponseType === testResponseType.ERROR) {
        return mockErrorModelTagsResponse();
    } else if (currentTestResponseType === testResponseType.EMPTY) {
        return mockEmptyModelTagsResponse();
    } else if (currentTestResponseType === testResponseType.INVALID) {
        return mockInvalidModelTagsResponse();
    }
    return mockDefaultModelTagsResponse();
};

export const setTestModelTagsResponseType = (responseType: string = "") => {
    currentTestResponseType = responseType;
};

const mockDefaultModelTagsResponse = (): Promise<AxiosResponse<JobTags[]>> => {
    return Promise.resolve(
        createMockResponse(mockGetModelTags) as unknown as AxiosResponse<
            JobTags[]
        >
    );
};

export const mockEmptyModelTagsResponse = (): Promise<
    AxiosResponse<JobTags[]>
> => {
    return Promise.resolve(
        createMockResponse([]) as unknown as AxiosResponse<JobTags[]>
    );
};

export const mockInvalidModelTagsResponse = (): Promise<
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

export const mockErrorModelTagsResponse = () => {
    return Promise.reject(
        createMockResponse(new Error("Internal Server Error"), 500)
    );
};

export const expectedModelTagsResponse = (): JobTags[] => {
    return { ...mockGetModelTags } as unknown as JobTags[];
};
