// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AxiosResponse } from "axios";
import { testResponseType } from "tests/testResponseType";
import { createMockResponse } from "tests/testUtil";

let currentTestResponseType: testResponseType = testResponseType.EMPTY;

export const mockGetBpJobWarningResponse = (): Promise<
    AxiosResponse<unknown>
> => {
    if (currentTestResponseType === testResponseType.ERROR) {
        return mockErrorResponse();
    } else if (currentTestResponseType === testResponseType.EMPTY) {
        return mockEmptyResponse();
    } else if (currentTestResponseType === testResponseType.INVALID) {
        return mockInvalidResponse();
    }
    return mockDefaultResponse();
};

export const setGetBpJobWarningTestResponseType = (
    responseType: testResponseType
) => {
    currentTestResponseType = responseType;
};

const mockDefaultResponse = (): Promise<AxiosResponse<unknown>> => {
    const resp = {
        "bp-job-3f5f52a53f01473d8c5123ff7ba4fd15": {
            "errors": [],
            "warnings": [],
            "info_items": [
                {
                    "is_error": false,
                    "title": "Possible Data Missing",
                    "description":
                        "FINANCE_TRANS includes data from a time range that may not be covered by FINANCE_ACCOUNT. The latest fact in FINANCE_TRANS (1998-12-31 UTC) occurs 12 months after the latest fact in FINANCE_ACCOUNT (1997-12-29 UTC), so data between 1997-12-29 UTC and 1998-12-31 UTC may be dropped. If this is not expected, please update your data source to include data from this time range.",
                    "is_dismissible": false,
                    "cta": null,
                    "id": "possible_data_missing"
                }
            ]
        }
    };

    return Promise.resolve(
        createMockResponse(resp) as unknown as AxiosResponse<unknown>
    );
};

export const mockEmptyResponse = (): Promise<AxiosResponse<unknown>> => {
    return Promise.resolve(
        createMockResponse({}) as unknown as AxiosResponse<unknown>
    );
};

export const mockInvalidResponse = (): Promise<AxiosResponse<unknown>> => {
    return Promise.resolve(
        createMockResponse({
            jobs: [[]],
            total_num_items: 0
        }) as unknown as AxiosResponse<unknown>
    );
};

export const mockErrorResponse = () => {
    return Promise.reject(
        createMockResponse(new Error("Internal Server Error"), 500)
    );
};
