// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AxiosResponse } from "axios";
import { GraphSnapshotResponse } from "gen/v0/types";
import mockGetGraphSnapshotResponse from "tests/fixtures/GraphSnapshot.json";
import { testResponseType } from "tests/testResponseType";
import { createMockResponse } from "tests/testUtil";

import {
    cancelled_res,
    failed_res,
    queued_res,
    running_res
} from "../MockResponse/GraphSnapshotResponse";

let currentTestResponseType = "";

export const mockGraphSnapshotResponse = (): Promise<
    AxiosResponse<GraphSnapshotResponse>
> => {
    if (currentTestResponseType === testResponseType.ERROR) {
        return mockErrorGraphSnapshotResponse();
    } else if (currentTestResponseType === testResponseType.EMPTY) {
        return mockEmptyGraphSnapshotResponse();
    } else if (currentTestResponseType === testResponseType.INVALID) {
        return mockInvalidGraphSnapshotResponse();
    } else if (currentTestResponseType === testResponseType.FAILED) {
        return mockFailedGraphSnapshotResponse();
    } else if (currentTestResponseType === testResponseType.QUEUED) {
        return mockQueuedGraphSnapshotResponse();
    } else if (currentTestResponseType === testResponseType.CANCELLED) {
        return mockCancelledGraphSnapshotResponse();
    } else if (currentTestResponseType === testResponseType.RUNNING) {
        return mockRunningGraphSnapshotResponse();
    }
    return mockDefaultGraphSnapshotResponse();
};

export const setTestGraphSnapshotResponseType = (responseType: string = "") => {
    currentTestResponseType = responseType;
};

const mockDefaultGraphSnapshotResponse = (): Promise<
    AxiosResponse<GraphSnapshotResponse>
> => {
    return Promise.resolve(
        createMockResponse(
            mockGetGraphSnapshotResponse
        ) as unknown as AxiosResponse<GraphSnapshotResponse>
    );
};

export const mockEmptyGraphSnapshotResponse = (): Promise<
    AxiosResponse<GraphSnapshotResponse>
> => {
    return Promise.resolve(
        createMockResponse(
            []
        ) as unknown as AxiosResponse<GraphSnapshotResponse>
    );
};

const mockInvalidGraphSnapshotResponse = (): Promise<
    AxiosResponse<GraphSnapshotResponse>
> => {
    return Promise.reject({
        response: {
            status: 422,
            statusText: "Unprocessable Entity",
            data: "Please provide a valid graph snapshot ID or a valid graph definition and graph version to fetch a graph snapshot."
        }
    });
};

export const mockErrorGraphSnapshotResponse = () => {
    return Promise.reject(
        createMockResponse(new Error("Internal Server Error"), 500)
    );
};

const mockFailedGraphSnapshotResponse = (): Promise<
    AxiosResponse<GraphSnapshotResponse>
> => {
    return Promise.resolve(
        createMockResponse(
            failed_res
        ) as unknown as AxiosResponse<GraphSnapshotResponse>
    );
};

const mockQueuedGraphSnapshotResponse = (): Promise<
    AxiosResponse<GraphSnapshotResponse>
> => {
    return Promise.resolve(
        createMockResponse(
            queued_res
        ) as unknown as AxiosResponse<GraphSnapshotResponse>
    );
};

const mockCancelledGraphSnapshotResponse = (): Promise<
    AxiosResponse<GraphSnapshotResponse>
> => {
    return Promise.resolve(
        createMockResponse(
            cancelled_res
        ) as unknown as AxiosResponse<GraphSnapshotResponse>
    );
};

const mockRunningGraphSnapshotResponse = (): Promise<
    AxiosResponse<GraphSnapshotResponse>
> => {
    return Promise.resolve(
        createMockResponse(
            running_res
        ) as unknown as AxiosResponse<GraphSnapshotResponse>
    );
};

export const expectedGraphSnapshotResponse = (): GraphSnapshotResponse => {
    return {
        ...mockGetGraphSnapshotResponse
    } as unknown as GraphSnapshotResponse;
};
