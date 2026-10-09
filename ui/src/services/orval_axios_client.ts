// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import axios, { AxiosRequestConfig, AxiosResponse } from "axios";
import { ZodError, ZodSafeParseResult, ZodSchema } from "zod";

import eventBus from "utils/eventBus";
import { isUiSchemaValidationEnabled } from "utils/uiSchemaValidation";
import {
    VALIDATION_SCHEMA_ERROR_EVENT,
    ValidationSchemaErrorEvent
} from "utils/validationSchemaEvent";

import axiosClient from "./axios_client";

interface OrvalRequestParams extends AxiosRequestConfig {
    url: string;
    method: AxiosRequestConfig["method"];
}

export const customClient = async <T>(
    { url, method, data, params, headers, signal }: OrvalRequestParams, // Type annotation only
    schema: unknown
): Promise<AxiosResponse<T>> => {
    try {
        const response = await axiosClient({
            url,
            method,
            data,
            params,
            signal,
            headers
        });

        let parsed = {
            success: true,
            error: null
        } as unknown as ZodSafeParseResult<unknown>;

        // Optional: parse or validate the response
        if (
            response.status >= 200 &&
            response.status < 300 &&
            schema &&
            isUiSchemaValidationEnabled()
        ) {
            parsed = (schema as ZodSchema).safeParse(response.data);
            const methodToUse = method?.toUpperCase() ?? "GET";
            const apiResponse = JSON.stringify(response.data, null, 2);
            const validationErrors = JSON.stringify(
                parsed.error?.format(),
                null,
                2
            );
            const apiPayload = data ? JSON.stringify(data) : data;

            if (!parsed.success && validationErrors) {
                eventBus.dispatch<ValidationSchemaErrorEvent>(
                    VALIDATION_SCHEMA_ERROR_EVENT,
                    {
                        method: methodToUse,
                        url,
                        apiResponse,
                        validationErrors,
                        apiPayload
                    }
                );
            }
        }
        return response;
    } catch (error) {
        if (axios.isCancel(error)) {
            throw error;
        } else if (error instanceof ZodError) {
            console.error("Zod validation failed:", error);
        } else {
            console.error("API call failed:", error);
        }
        throw error;
    }
};
