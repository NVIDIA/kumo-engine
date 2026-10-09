// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React from "react";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook } from "@testing-library/react";
import { AxiosRequestHeaders, AxiosResponse } from "axios";

export function createMockResponse<T>(
    data: T,
    status = 200,
    statusText = "OK"
): Partial<AxiosResponse<T>> {
    return {
        data,
        status,
        statusText,
        headers: {
            "content-type": "application/json"
        },
        config: {
            url: "",
            method: "",
            headers: {
                "Accept": "application/json"
            } as AxiosRequestHeaders
        }
    };
}

export const normalizeString = (str: string) => str.replace(/\s+/g, " ").trim();

export const createQueryClient = (retry = false) => {
    return new QueryClient({
        defaultOptions: {
            queries: {
                retry,
                retryDelay: 0
            }
        }
    });
};

export const createWrapper = (queryClient: QueryClient) => {
    return ({ children }: { children: React.ReactNode }) =>
        React.createElement(
            QueryClientProvider,
            { client: queryClient },
            children
        );
};

export const renderWithQueryClient = <T>(hook: () => T) =>
    renderHook(hook, { wrapper: createWrapper(createQueryClient()) });
