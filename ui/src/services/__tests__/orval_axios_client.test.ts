// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import axios, { AxiosResponse, InternalAxiosRequestConfig } from "axios";
import { useDebugStore } from "stores/debugStore";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { z, ZodError } from "zod";

import eventBus from "utils/eventBus";
import { VALIDATION_SCHEMA_ERROR_EVENT } from "utils/validationSchemaEvent";

import axiosClient from "../axios_client";
import { customClient } from "../orval_axios_client";

vi.unmock("services/orval_axios_client");

vi.mock("services/axios_client", () => ({
    default: vi.fn()
}));

// The orval client emits an eventBus event instead of importing the toast
// component (services/ must not import components/). Spy on the dispatch.
let dispatchSpy: ReturnType<typeof vi.spyOn>;

const createMockResponse = (
    overrides?: Partial<AxiosResponse>
): AxiosResponse => ({
    data: { id: 1, name: "Test" },
    status: 200,
    statusText: "OK",
    headers: {},
    config: {} as InternalAxiosRequestConfig,
    ...overrides
});

const enableSchemaValidation = () => {
    useDebugStore.setState({
        data: {
            ...useDebugStore.getState().data,
            feature_flags: {
                ...useDebugStore.getState().data.feature_flags,
                ENABLE_UI_SCHEMA_VALIDATION: true
            }
        }
    });
};

describe("orval_axios_client", () => {
    beforeEach(() => {
        vi.clearAllMocks();
        useDebugStore.getState().reset();
        dispatchSpy = vi.spyOn(eventBus, "dispatch");
    });

    afterEach(() => {
        dispatchSpy.mockRestore();
        vi.clearAllMocks();
    });

    describe("customClient", () => {
        it("Should make successful API request without schema", async () => {
            const mockResponse = createMockResponse();
            vi.mocked(axiosClient).mockResolvedValue(mockResponse);

            const result = await customClient(
                { url: "/api/test", method: "GET" },
                null
            );

            expect(result).toEqual(mockResponse);
            expect(axiosClient).toHaveBeenCalledWith({
                url: "/api/test",
                method: "GET",
                data: undefined,
                params: undefined,
                signal: undefined,
                headers: undefined
            });
        });

        it("Should pass all request parameters to axiosClient", async () => {
            const requestParams = {
                url: "/api/users",
                method: "POST" as const,
                data: { name: "John", email: "john@example.com" },
                params: { page: 1, limit: 10 },
                headers: { "X-Custom-Header": "value" },
                signal: new AbortController().signal
            };
            const mockResponse = createMockResponse({
                data: { success: true },
                status: 201,
                statusText: "Created"
            });
            vi.mocked(axiosClient).mockResolvedValue(mockResponse);

            const result = await customClient(requestParams, null);

            expect(result).toEqual(mockResponse);
            expect(axiosClient).toHaveBeenCalledWith(requestParams);
        });

        it.each([
            {
                scenario: "schema is null",
                schema: null,
                enableValidation: false
            },
            {
                scenario: "ENABLE_UI_SCHEMA_VALIDATION is false",
                schema: z.object({ id: z.number() }),
                enableValidation: false
            }
        ])(
            "Should not validate when $scenario",
            async ({ schema, enableValidation }) => {
                if (!enableValidation) {
                    useDebugStore.setState({
                        data: {
                            ...useDebugStore.getState().data,
                            feature_flags: {
                                ...useDebugStore.getState().data.feature_flags,
                                ENABLE_UI_SCHEMA_VALIDATION: false
                            }
                        }
                    });
                }

                const mockResponse = createMockResponse();
                vi.mocked(axiosClient).mockResolvedValue(mockResponse);

                await customClient({ url: "/api/test", method: "GET" }, schema);

                expect(dispatchSpy).not.toHaveBeenCalled();
            }
        );

        it.each([
            {
                method: "GET" as const
            },
            {
                method: "POST" as const
            },
            {
                method: "PUT" as const
            },
            {
                method: "DELETE" as const
            },
            {
                method: "PATCH" as const
            }
        ])("Should handle $method method correctly", async ({ method }) => {
            const mockResponse = createMockResponse({
                data: { success: true }
            });
            vi.mocked(axiosClient).mockResolvedValue(mockResponse);

            await customClient(
                { url: `/api/${method.toLowerCase()}`, method },
                null
            );

            expect(axiosClient).toHaveBeenCalledWith(
                expect.objectContaining({ method })
            );
        });

        it("Should validate response when schema is provided and flag is enabled", async () => {
            const schema = z.object({ id: z.number(), name: z.string() });
            const mockResponse = createMockResponse();
            enableSchemaValidation();
            vi.mocked(axiosClient).mockResolvedValue(mockResponse);

            const result = await customClient(
                { url: "/api/test", method: "GET" },
                schema
            );

            expect(result).toEqual(mockResponse);
            expect(dispatchSpy).not.toHaveBeenCalled();
        });

        it("Should trigger validation toast when schema validation fails", async () => {
            const schema = z.object({ id: z.number(), name: z.string() });
            const mockResponse = createMockResponse({
                data: { id: "invalid", name: 123 }
            });
            enableSchemaValidation();
            vi.mocked(axiosClient).mockResolvedValue(mockResponse);

            const result = await customClient(
                { url: "/api/test", method: "POST", data: { test: "payload" } },
                schema
            );

            expect(result).toEqual(mockResponse);
            expect(dispatchSpy).toHaveBeenCalledWith(
                VALIDATION_SCHEMA_ERROR_EVENT,
                {
                    method: "POST",
                    url: "/api/test",
                    apiResponse: JSON.stringify(mockResponse.data, null, 2),
                    validationErrors: expect.any(String),
                    apiPayload: JSON.stringify({ test: "payload" })
                }
            );
        });

        it("Should not validate when response status is not 2xx", async () => {
            const schema = z.object({ id: z.number() });
            const mockResponse = createMockResponse({
                data: { error: "Not found" },
                status: 404,
                statusText: "Not Found"
            });
            enableSchemaValidation();
            vi.mocked(axiosClient).mockResolvedValue(mockResponse);

            await customClient({ url: "/api/test", method: "GET" }, schema);

            expect(dispatchSpy).not.toHaveBeenCalled();
        });

        it("Should default to GET method in validation when method is undefined", async () => {
            const schema = z.object({ id: z.number() });
            const mockResponse = createMockResponse({
                data: { id: "invalid" }
            });
            enableSchemaValidation();
            vi.mocked(axiosClient).mockResolvedValue(mockResponse);

            await customClient({ url: "/api/test", method: undefined }, schema);

            expect(dispatchSpy).toHaveBeenCalledWith(
                VALIDATION_SCHEMA_ERROR_EVENT,
                expect.objectContaining({ method: "GET" })
            );
        });

        it("Should not include apiPayload in toast when data is undefined", async () => {
            const schema = z.object({ id: z.number() });
            const mockResponse = createMockResponse({
                data: { id: "invalid" }
            });
            enableSchemaValidation();
            vi.mocked(axiosClient).mockResolvedValue(mockResponse);

            await customClient({ url: "/api/test", method: "GET" }, schema);

            expect(dispatchSpy).toHaveBeenCalledWith(
                VALIDATION_SCHEMA_ERROR_EVENT,
                expect.objectContaining({ apiPayload: undefined })
            );
        });

        it("Should return response even when validation fails", async () => {
            const schema = z.object({ id: z.number() });
            const mockResponse = createMockResponse({
                data: { id: "invalid" }
            });
            enableSchemaValidation();
            vi.mocked(axiosClient).mockResolvedValue(mockResponse);

            const result = await customClient(
                { url: "/api/test", method: "GET" },
                schema
            );

            expect(result).toEqual(mockResponse);
            expect(result.data).toEqual({ id: "invalid" });
        });
    });

    describe("error handling", () => {
        it("Should rethrow canceled requests without logging an API failure", async () => {
            const canceled = new axios.CanceledError("canceled");
            const consoleErrorSpy = vi
                .spyOn(console, "error")
                .mockImplementation(() => {});
            vi.mocked(axiosClient).mockRejectedValue(canceled);

            await expect(
                customClient({ url: "/api/test", method: "GET" }, null)
            ).rejects.toBe(canceled);

            expect(consoleErrorSpy).not.toHaveBeenCalled();
            consoleErrorSpy.mockRestore();
        });

        it("Should throw error when axiosClient fails", async () => {
            const mockError = new Error("Network error");
            vi.mocked(axiosClient).mockRejectedValue(mockError);

            await expect(
                customClient({ url: "/api/test", method: "GET" }, null)
            ).rejects.toThrow("Network error");
        });

        it.each([
            {
                errorType: "ZodError",
                error: new ZodError([
                    {
                        code: "invalid_type",
                        expected: "string",
                        path: ["name"],
                        message: "Expected string, received number"
                    }
                ]),
                expectedLog: "Zod validation failed:"
            },
            {
                errorType: "generic error",
                error: new Error("API call failed"),
                expectedLog: "API call failed:"
            }
        ])(
            "Should log and rethrow $errorType",
            async ({ error, expectedLog }) => {
                const consoleErrorSpy = vi
                    .spyOn(console, "error")
                    .mockImplementation(() => {});
                vi.mocked(axiosClient).mockRejectedValue(error);

                await expect(
                    customClient({ url: "/api/test", method: "GET" }, null)
                ).rejects.toThrow();

                expect(consoleErrorSpy).toHaveBeenCalledWith(
                    expectedLog,
                    error
                );
                consoleErrorSpy.mockRestore();
            }
        );
    });

    describe("validation flow", () => {
        it.each([
            {
                scenario: "valid data",
                data: { id: 1, name: "John Doe", email: "john@example.com" },
                shouldTriggerToast: false
            },
            {
                scenario: "invalid data",
                data: { id: "not-a-number", name: 123, email: "invalid-email" },
                shouldTriggerToast: true
            }
        ])(
            "Should handle validation flow with $scenario",
            async ({ data, shouldTriggerToast }) => {
                const schema = z.object({
                    id: z.number(),
                    name: z.string(),
                    email: z.string().email()
                });
                const mockResponse = createMockResponse({ data });
                enableSchemaValidation();
                vi.mocked(axiosClient).mockResolvedValue(mockResponse);

                const result = await customClient(
                    {
                        url: "/api/users",
                        method: "POST",
                        data: { name: "John", email: "john@example.com" }
                    },
                    schema
                );

                expect(result.data).toEqual(data);

                if (shouldTriggerToast) {
                    expect(dispatchSpy).toHaveBeenCalledWith(
                        VALIDATION_SCHEMA_ERROR_EVENT,
                        {
                            method: "POST",
                            url: "/api/users",
                            apiResponse: JSON.stringify(data, null, 2),
                            validationErrors: expect.stringContaining("id"),
                            apiPayload: JSON.stringify({
                                name: "John",
                                email: "john@example.com"
                            })
                        }
                    );
                } else {
                    expect(dispatchSpy).not.toHaveBeenCalled();
                }
            }
        );

        it("Should not validate when feature flag is disabled", async () => {
            const schema = z.object({ id: z.number() });
            const mockResponse = createMockResponse();

            useDebugStore.setState({
                data: {
                    ...useDebugStore.getState().data,
                    feature_flags: {
                        ...useDebugStore.getState().data.feature_flags,
                        ENABLE_UI_SCHEMA_VALIDATION: false
                    }
                }
            });
            vi.mocked(axiosClient).mockResolvedValue(mockResponse);
            await customClient({ url: "/api/test", method: "GET" }, schema);

            expect(dispatchSpy).not.toHaveBeenCalled();
        });
    });
});
