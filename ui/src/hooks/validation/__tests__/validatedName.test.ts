// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { QueryClient } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import { testResponseType } from "tests/testResponseType";
import { createQueryClient, createWrapper } from "tests/testUtil";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { setGraphNameResponseType } from "pages/Graphs/__tests__/MockFunction/GraphName.mock";

import { kumoResource } from "utils/Constants";

import { useValidatedName } from "../validatedName";

import "tests/mockAxios";

let queryClient: QueryClient;

const onValidationStart = vi.fn();
const onValidationEnd = vi.fn();
const onShouldValidateChange = vi.fn();

beforeEach(() => {
    queryClient = createQueryClient();
    vi.clearAllMocks();
    setGraphNameResponseType();
});

afterEach(() => {
    queryClient.clear();
});

const renderValidatedName = (
    args: {
        initialValue?: string;
        shouldValidate?: boolean;
        validateOnMount?: boolean;
        resource?: kumoResource;
        entityId?: string;
    } = {}
) => {
    const {
        initialValue = "",
        shouldValidate = true,
        validateOnMount = true,
        resource = kumoResource.Graph,
        entityId = ""
    } = args;

    return renderHook(
        (props) =>
            useValidatedName(
                props.initialValue,
                onValidationStart,
                onValidationEnd,
                onShouldValidateChange,
                props.resource,
                props.shouldValidate,
                props.validateOnMount,
                props.entityId
            ),
        {
            wrapper: createWrapper(queryClient),
            initialProps: {
                initialValue,
                shouldValidate,
                validateOnMount,
                resource,
                entityId
            }
        }
    );
};

const waitForValidationEnd = (expectedResponse: {
    isValid: boolean;
    errMsg?: string;
}) =>
    waitFor(
        () => {
            expect(onValidationEnd).toHaveBeenCalledWith(
                expect.objectContaining({
                    validationResponse:
                        expect.objectContaining(expectedResponse)
                })
            );
        },
        { timeout: 3000 }
    );

describe("useValidatedName", () => {
    describe("return value", () => {
        it("Should return initialValue and a setter function", () => {
            const { result } = renderValidatedName({
                initialValue: "my_graph"
            });
            const [input, setInput] = result.current;

            expect(input).toBe("my_graph");
            expect(typeof setInput).toBe("function");
        });

        it("Should update input when setter is called", () => {
            const { result } = renderValidatedName();

            act(() => {
                result.current[1]("new_name");
            });

            expect(result.current[0]).toBe("new_name");
        });
    });

    describe("onShouldValidateChange callback", () => {
        it("Should call onShouldValidateChange on mount", () => {
            renderValidatedName();

            expect(onShouldValidateChange).toHaveBeenCalledTimes(1);
        });

        it("Should call onShouldValidateChange when shouldValidate changes", () => {
            const { rerender } = renderValidatedName({ shouldValidate: true });
            vi.clearAllMocks();

            rerender({
                initialValue: "",
                shouldValidate: false,
                validateOnMount: true,
                resource: kumoResource.Graph,
                entityId: ""
            });

            expect(onShouldValidateChange).toHaveBeenCalledTimes(1);
        });
    });

    describe("onValidationStart behavior", () => {
        it.each([
            {
                shouldValidate: true,
                expectedCalls: 1,
                description: "shouldValidate is true"
            },
            {
                shouldValidate: false,
                expectedCalls: 0,
                description: "shouldValidate is false"
            }
        ])(
            "Should call onValidationStart $expectedCalls time(s) on mount when $description",
            ({ shouldValidate, expectedCalls }) => {
                renderValidatedName({ shouldValidate });

                expect(onValidationStart).toHaveBeenCalledTimes(expectedCalls);
            }
        );

        it.each([
            {
                shouldValidate: true,
                expectedCalls: 1,
                description: "shouldValidate is true"
            },
            {
                shouldValidate: false,
                expectedCalls: 0,
                description: "shouldValidate is false"
            }
        ])(
            "Should call onValidationStart $expectedCalls time(s) on input change when $description",
            ({ shouldValidate, expectedCalls }) => {
                const { result } = renderValidatedName({ shouldValidate });
                vi.clearAllMocks();

                act(() => {
                    result.current[1]("new_name");
                });

                expect(onValidationStart).toHaveBeenCalledTimes(expectedCalls);
            }
        );
    });

    describe("debounce behavior", () => {
        beforeEach(() => {
            vi.useFakeTimers();
        });

        afterEach(() => {
            vi.useRealTimers();
        });

        it("Should not trigger validation before 1 second debounce elapses", () => {
            renderValidatedName({ initialValue: "valid_graph" });

            act(() => {
                vi.advanceTimersByTime(999);
            });

            expect(onValidationEnd).not.toHaveBeenCalled();
        });

        it("Should reset debounce timer when input changes before timer elapses", () => {
            const { result } = renderValidatedName({ initialValue: "graph" });

            act(() => {
                vi.advanceTimersByTime(500);
            });

            act(() => {
                result.current[1]("graph_v");
            });

            act(() => {
                vi.advanceTimersByTime(500);
            });

            expect(onValidationEnd).not.toHaveBeenCalled();
        });
    });

    describe("onValidationEnd callback", () => {
        it.each([
            {
                resource: kumoResource.Graph,
                initialValue: "valid_graph",
                description: "valid Graph name",
                setup: () => setGraphNameResponseType(),
                expectedResponse: { isValid: true }
            },
            {
                resource: kumoResource.Graph,
                initialValue: "invalid-graph",
                description: "invalid name from API",
                setup: () => setGraphNameResponseType(testResponseType.INVALID),
                expectedResponse: { isValid: false }
            },
            {
                resource: kumoResource.Graph,
                initialValue: "",
                description: "empty name (no API call)",
                setup: () => setGraphNameResponseType(),
                expectedResponse: {
                    isValid: false,
                    errMsg: "Please enter Graph Name"
                }
            }
        ])(
            "Should call onValidationEnd with correct response for $description",
            async ({ resource, initialValue, setup, expectedResponse }) => {
                setup();

                renderValidatedName({ initialValue, resource });

                await waitForValidationEnd(expectedResponse);
            }
        );

        it("Should not call onValidationEnd when shouldValidate is false", () => {
            renderValidatedName({
                initialValue: "valid_graph",
                shouldValidate: false
            });

            expect(onValidationEnd).not.toHaveBeenCalled();
        });
    });
});
