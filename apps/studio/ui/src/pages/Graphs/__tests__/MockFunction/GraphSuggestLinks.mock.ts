// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AxiosResponse } from "axios";
import { testResponseType } from "tests/testResponseType";
import { createMockResponse } from "tests/testUtil";

import { ColumnGroup } from "types/graphs";

import { stockTransLink } from "../MockResponse/GraphSuggestLinks.response";

let currentTestResponseType = "";

export const mockGraphSuggestLinksResponse = (): Promise<
    AxiosResponse<Array<ColumnGroup>>
> => {
    switch (currentTestResponseType) {
        case testResponseType.EMPTY:
            return mockSuggestLinksEmptyResponse();
        default:
            return mockSuggestLinksResponse();
    }
};

export const setGraphSuggestLinksResponseType = (responseType: string = "") => {
    currentTestResponseType = responseType;
    return currentTestResponseType;
};

const mockSuggestLinksResponse = (): Promise<
    AxiosResponse<Array<ColumnGroup>>
> => {
    return Promise.resolve(
        createMockResponse(stockTransLink) as unknown as AxiosResponse<
            Array<ColumnGroup>
        >
    );
};

const mockSuggestLinksEmptyResponse = (): Promise<
    AxiosResponse<Array<ColumnGroup>>
> => {
    return Promise.resolve(
        createMockResponse([]) as unknown as AxiosResponse<Array<ColumnGroup>>
    );
};
