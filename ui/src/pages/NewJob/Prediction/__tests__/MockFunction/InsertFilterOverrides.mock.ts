// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AxiosResponse } from "axios";
import { EditableQuery, InsertFilterOverridesRequest } from "gen/v1/types";
import { testResponseType } from "tests/testResponseType";
import { createMockResponse } from "tests/testUtil";

import mockGetInsertFilterOverridesResponse from "../MockResponse/InsertFilterOverrides.json";

let currentTestResponseType = "";

export const mockInsertFilterOverridesResponse = (
    payload: InsertFilterOverridesRequest
): Promise<AxiosResponse<EditableQuery>> => {
    if (currentTestResponseType === testResponseType.ERROR) {
        return mockErrorInsertFilterOverridesResponse();
    } else if (currentTestResponseType === testResponseType.EMPTY) {
        return mockEmptyInsertFilterOverridesResponse();
    } else if (currentTestResponseType === testResponseType.INVALID) {
        return mockInvalidInsertFilterOverridesResponse();
    } else if (currentTestResponseType === testResponseType.WITH_WARNINGS) {
        return mockWarningInsertFilterOverridesResponse();
    }
    return mockDefaultInsertFilterOverridesResponse(payload);
};

export const setTestInsertFilterOverridesResponseType = (
    responseType: string = ""
) => {
    currentTestResponseType = responseType;
};

const mockDefaultInsertFilterOverridesResponse = (
    payload: InsertFilterOverridesRequest
): Promise<AxiosResponse<EditableQuery>> => {
    const response = { ...mockGetInsertFilterOverridesResponse };
    if (payload.entity_filter) {
        response.text += " WHERE " + payload.entity_filter;
    }
    return Promise.resolve(
        createMockResponse(response) as unknown as AxiosResponse<EditableQuery>
    );
};

export const mockEmptyInsertFilterOverridesResponse = (): Promise<
    AxiosResponse<EditableQuery>
> => {
    return Promise.resolve(
        createMockResponse([]) as unknown as AxiosResponse<EditableQuery>
    );
};

export const mockInvalidInsertFilterOverridesResponse = (): Promise<
    AxiosResponse<EditableQuery>
> => {
    return Promise.resolve(
        createMockResponse({
            "text": "PREDICT COUNT(FINANCE_TRANS.*, 0, 30, days) = 0 FOR EACH FINANCE_ACCOUNT.ACCOUNT_ID WHERE diomdm#5359",
            "editable_fields": [
                {
                    "field_name": "Entity filter",
                    "text": "diomdm#5359",
                    "position": [83, 101],
                    "full_filter_format_string": " WHERE {0}",
                    "empty_filter_default": "",
                    "bold_text_position": [90, 101]
                }
            ],
            "validation_response": {
                "warnings": [],
                "errors": [
                    {
                        "title": "Invalid query syntax.",
                        "message":
                            "Encountered the following issues during parsing: Errors:\n1. Invalid Syntax. The entity filter (FOR EACH ... WHERE) clause in this query is empty or invalid. This clause allows you to filter the entities that the model will generate training examples for.\n2. Invalid Syntax. Line 1, col 90; Input Mismatch, got 'diomdm', but expected one of '(', FQN_COLUMN, WILDCARD_COLUMN, NOT, AGGR."
                    }
                ],
                "info_items": []
            }
        }) as unknown as AxiosResponse<EditableQuery>
    );
};

export const mockErrorInsertFilterOverridesResponse = () => {
    return Promise.reject(
        createMockResponse(new Error("Internal Server Error"), 500)
    );
};

export const mockWarningInsertFilterOverridesResponse = (): Promise<
    AxiosResponse<EditableQuery>
> => {
    return Promise.resolve(
        createMockResponse({
            "text": "PREDICT MAX(transaction_1.Quantity, 0, 30, days) FOR EACH customer_2.CustomerID",
            "editable_fields": [
                {
                    "field_name": "Entity filter",
                    "text": "",
                    "position": [79, 79],
                    "full_filter_format_string": " WHERE {0}",
                    "empty_filter_default": "",
                    "bold_text_position": [79, 79]
                }
            ],
            "validation_response": {
                "warnings": [
                    {
                        "title": "Semantic Type Mismatch",
                        "message":
                            "Aggregation MAX is intended to operate on numerical, text, and timestamp columns, but transaction_1.Quantity has semantic type categorical. Consider changing the underlying semantic type to fit the task better."
                    }
                ],
                "errors": [],
                "info_items": []
            }
        }) as unknown as AxiosResponse<EditableQuery>
    );
};

export const expectedInsertFilterOverridesResponse = () => {
    return Promise.resolve(mockGetInsertFilterOverridesResponse);
};
