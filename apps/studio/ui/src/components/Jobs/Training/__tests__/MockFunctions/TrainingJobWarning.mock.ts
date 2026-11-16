// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AxiosResponse } from "axios";
import { testResponseType } from "tests/testResponseType";
import { createMockResponse } from "tests/testUtil";

let currentTestResponseType: testResponseType = testResponseType.EMPTY;

export const mockTrainingWarningResponse = (
    id: string
): Promise<AxiosResponse<unknown>> => {
    if (currentTestResponseType === testResponseType.ERROR) {
        return mockErrorResponse();
    } else if (currentTestResponseType === testResponseType.EMPTY) {
        return mockEmptyResponse();
    } else if (currentTestResponseType === testResponseType.INVALID) {
        return mockInvalidResponse();
    }
    return mockDefaultResponse(id);
};

export const setTrainingWarningTestResponseType = (
    responseType: testResponseType
) => {
    currentTestResponseType = responseType;
};

const mockDefaultResponse = (id: string): Promise<AxiosResponse<unknown>> => {
    const resp = {
        [`${id}`]: {
            "errors": [],
            "warnings": [],
            "info_items": [
                {
                    "is_error": false,
                    "title": "Duplicate primary keys",
                    "description":
                        "102310 (90.3%) of the rows in the 'transaction' table's primary key column (InvoiceNo) are duplicated. Consider cleaning your data by removing rows with duplicate primary keys for better model performance, or do not specify primary key if rows in the table are not used as entities. Features on different rows with the same primary key will be aggregated. NOTE: Column statistics are calculated after dropping invalid rows.",
                    "is_dismissible": false,
                    "cta": null,
                    "id": "duplicate_primary_keys"
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
