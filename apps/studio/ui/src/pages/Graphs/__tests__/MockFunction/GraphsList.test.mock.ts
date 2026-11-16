// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AxiosResponse } from "axios";
import { IDResponse } from "gen/v1/types";
import { testResponseType } from "tests/testResponseType";
import { createMockResponse } from "tests/testUtil";

import { Graph, GraphListResponse } from "types/graphs";

import { graphsList } from "../MockResponse/GraphsList.response";

let currentTestGraphListResponseType = "";
export let invalidField = "";

export const setGraphsListInvalidField = (field: string = "") => {
    invalidField = field;
};

export const setGraphsListResponseType = (responseType: string = "") => {
    currentTestGraphListResponseType = responseType;
};

export const mockGraphsListResponse = (): Promise<
    AxiosResponse<Array<GraphListResponse>>
> => {
    if (currentTestGraphListResponseType === testResponseType.ERROR) {
        return mockErrorGraphsListResponse();
    } else if (currentTestGraphListResponseType === testResponseType.EMPTY) {
        return mockEmptyGraphsListResponse();
    } else if (currentTestGraphListResponseType === testResponseType.INVALID) {
        return mockInvalidGraphsListResponse();
    }

    return mockDefaultGraphsListResponse();
};

const mockDefaultGraphsListResponse = (): Promise<
    AxiosResponse<Array<GraphListResponse>>
> => {
    return Promise.resolve(
        createMockResponse(graphsList) as unknown as AxiosResponse<
            Array<GraphListResponse>
        >
    );
};

export const expectedGraphsListResponse = (): GraphListResponse[] => {
    return [...graphsList] as unknown as GraphListResponse[];
};

export const mockEmptyGraphsListResponse = (): Promise<
    AxiosResponse<Array<GraphListResponse>>
> => {
    return Promise.resolve(
        createMockResponse([]) as unknown as AxiosResponse<
            Array<GraphListResponse>
        >
    );
};

export const mockInvalidGraphsListResponse = (): Promise<
    AxiosResponse<GraphListResponse[]>
> => {
    const resp = [];
    if (invalidField === "id") {
        resp.push({
            id: 22,
            name: "",
            timestamps: {
                created_at: "",
                updated_at: ""
            }
        });
    } else if (invalidField === "timestamps") {
        resp.push({
            id: "",
            name: "",
            timestamps: {
                created_at: 2.2,
                updated_at: 3.3
            }
        });
    } else {
        resp.push({
            id: "",
            name: 5498,
            timestamps: {
                created_at: "",
                updated_at: ""
            }
        });
    }
    return Promise.resolve(
        createMockResponse(resp) as unknown as AxiosResponse<
            GraphListResponse[]
        >
    );
};

export const mockErrorGraphsListResponse = () => {
    return Promise.reject(
        createMockResponse(new Error("Internal Server Error"), 500)
    );
};

export const mockDeleteGraphResponse = (
    payload: Graph
): Promise<AxiosResponse<IDResponse>> => {
    const resp: IDResponse = { id: payload ? "graph-mock-id" : "" };

    return Promise.resolve(
        createMockResponse(resp) as unknown as AxiosResponse<IDResponse>
    );
};

export const mockEditGraphResponse = (
    payload: Graph
): Promise<AxiosResponse<IDResponse>> => {
    const resp: IDResponse = { id: payload ? "graph-mock-id" : "" };

    return Promise.resolve(
        createMockResponse(resp) as unknown as AxiosResponse<IDResponse>
    );
};
