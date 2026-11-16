// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AxiosResponse } from "axios";
import { testResponseType } from "tests/testResponseType";
import { createMockResponse } from "tests/testUtil";

import { GraphResponse } from "types/graphs";

import { graphDetail } from "../MockResponse/GraphDetail.response";

let currentTestGraphDetailResponseType = "";

export const setGraphsDetailResponseType = (responseType: string = "") => {
    currentTestGraphDetailResponseType = responseType;
};

export const mockGraphDetailResponse = (): Promise<
    AxiosResponse<GraphResponse>
> => {
    if (currentTestGraphDetailResponseType === testResponseType.ERROR) {
        return mockErrorGraphsListResponse();
    } else if (
        currentTestGraphDetailResponseType === testResponseType.INVALID
    ) {
        return mockInvalidGraphDetailResponse();
    } else if (currentTestGraphDetailResponseType === testResponseType.EMPTY) {
        return mockEmptyGraphDetailResponse();
    }

    return mockDefaultGraphDetailResponse();
};

const mockDefaultGraphDetailResponse = (): Promise<
    AxiosResponse<GraphResponse>
> => {
    return Promise.resolve(
        createMockResponse(
            graphDetail
        ) as unknown as AxiosResponse<GraphResponse>
    );
};

export const expectedGraphDetailResponse = (): GraphResponse => {
    return { ...graphDetail } as unknown as GraphResponse;
};

export const mockEmptyGraphDetailResponse = (): Promise<
    AxiosResponse<GraphResponse>
> => {
    return Promise.resolve(
        createMockResponse([]) as unknown as AxiosResponse<GraphResponse>
    );
};

export const mockInvalidGraphDetailResponse = (): Promise<
    AxiosResponse<GraphResponse>
> => {
    return Promise.resolve(
        createMockResponse([
            {
                id: "",
                name: 5498,
                timestamps: {
                    created_at: "",
                    updated_at: ""
                }
            }
        ]) as unknown as AxiosResponse<GraphResponse>
    );
};

export const mockErrorGraphsListResponse = () => {
    return Promise.reject(
        createMockResponse(new Error("Internal Server Error"), 500)
    );
};
