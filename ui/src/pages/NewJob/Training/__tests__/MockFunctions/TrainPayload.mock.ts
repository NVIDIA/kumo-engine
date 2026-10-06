// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AxiosResponse } from "axios";
import { testResponseType } from "tests/testResponseType";
import { createMockResponse } from "tests/testUtil";

import { trainPayloadDeatails } from "../MockResponses/TrainPayload";

let currentTestResponseType = "";

export const mockTrainResponse = (): Promise<AxiosResponse<unknown>> => {
    if (currentTestResponseType === testResponseType.ERROR) {
        return mockErrorTrainResponse();
    } else if (currentTestResponseType === testResponseType.INVALID) {
        return mockInvalidTrainResponse();
    }
    return mockDefaultTrainResponse();
};

export const setTestTrainResponseType = (responseType: string = "") => {
    currentTestResponseType = responseType;
};

const mockDefaultTrainResponse = (): Promise<AxiosResponse<unknown>> => {
    return Promise.resolve(
        createMockResponse({
            id: "trainingjob-bed43f8f2fc54ea985439b026dbe0865"
        }) as unknown as AxiosResponse<unknown>
    );
};

const mockInvalidTrainResponse = (): Promise<AxiosResponse<unknown>> => {
    return Promise.reject({
        response: {
            status: 422,
            statusText: "Unprocessable Entity",
            data: ""
        }
    });
};

const mockErrorTrainResponse = () => {
    return Promise.reject(
        createMockResponse(new Error("Internal Server Error"), 500)
    );
};

export const expectedTrainResponse = () => {
    return Promise.resolve(trainPayloadDeatails);
};
