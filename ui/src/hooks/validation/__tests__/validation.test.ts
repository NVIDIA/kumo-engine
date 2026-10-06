// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MutableRefObject } from "react";

import { act, renderHook } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { CustomError } from "types/common";

import { errorStatusCode } from "utils/Constants";

import { useValidationState } from "../validation";

const setPageLoading = vi.fn();

beforeEach(() => {
    vi.clearAllMocks();
});

const renderValidationState = () =>
    renderHook(() => useValidationState(setPageLoading));

const createError = (status: errorStatusCode, data: string): CustomError => ({
    name: "Error",
    message: data,
    response: { status, data }
});

const callHandleError = (error: CustomError, title?: string) => {
    const { result } = renderValidationState();
    act(() => {
        result.current[2](error, title);
    });
    return result;
};

describe("useValidationState", () => {
    it("Should initialise banner and alert as null", () => {
        const { result } = renderValidationState();
        const [banner, alert] = result.current;

        expect(banner).toBeNull();
        expect(alert).toBeNull();
    });

    describe("handleError", () => {
        it("Should set banner for 500 Internal Server Error", () => {
            const result = callHandleError(
                createError(errorStatusCode.InternalServerError, "crash")
            );

            expect(result.current[0]).toMatchObject({
                title: "Internal error",
                warnings: [
                    "We could not process your request due to an internal error, please try again!"
                ]
            });
            expect(result.current[1]).toBeNull();
            expect(setPageLoading).toHaveBeenCalledWith(false);
        });

        it.each([
            {
                title: undefined,
                expectedTitle:
                    "Sorry, there is an error with one of the inputs",
                description: "no custom title"
            },
            {
                title: "Custom Error Title",
                expectedTitle: "Custom Error Title",
                description: "custom title"
            }
        ])(
            "Should set banner with correct title for 422 error when $description",
            ({ title, expectedTitle }) => {
                const result = callHandleError(
                    createError(
                        errorStatusCode.UnProcessableEntityError,
                        "Invalid input"
                    ),
                    title
                );

                expect(result.current[0]).toMatchObject({
                    title: expectedTitle,
                    warnings: ["Invalid input"]
                });
                expect(result.current[1]).toBeNull();
                expect(setPageLoading).toHaveBeenCalledWith(false);
            }
        );

        it("Should set alert for generic errors", () => {
            const result = callHandleError({
                name: "Error",
                message: "Not found"
            } as CustomError);

            expect(result.current[1]).toMatchObject({
                title: "Something went wrong!",
                description: "Not found"
            });
            expect(result.current[0]).toBeNull();
            expect(setPageLoading).toHaveBeenCalledWith(false);
        });

        it("Should not set alert for Request aborted error", () => {
            const result = callHandleError({
                name: "Error",
                message: "Request aborted"
            } as CustomError);

            expect(result.current[1]).toBeNull();
            expect(setPageLoading).toHaveBeenCalledWith(false);
        });
    });

    describe("clear", () => {
        it("Should reset banner and alert to null", () => {
            const { result } = renderValidationState();

            act(() => {
                result.current[2](
                    createError(errorStatusCode.InternalServerError, "crash")
                );
                result.current[2]({
                    name: "Error",
                    message: "some error"
                } as CustomError);
            });

            expect(result.current[0]).not.toBeNull();
            expect(result.current[1]).not.toBeNull();

            act(() => {
                result.current[3]();
            });

            expect(result.current[0]).toBeNull();
            expect(result.current[1]).toBeNull();
        });
    });

    describe("openDialog", () => {
        it("Should call toggleDialog on dialogRef with title and description", () => {
            const { result } = renderValidationState();
            const toggleDialog = vi.fn();

            act(() => {
                (
                    result.current[4] as MutableRefObject<{
                        toggleDialog: typeof toggleDialog;
                    }>
                ).current = { toggleDialog };
                result.current[5]("Dialog Title", "Dialog Description");
            });

            expect(toggleDialog).toHaveBeenCalledWith(
                "Dialog Title",
                "Dialog Description"
            );
        });

        it("Should not throw when dialogRef.current is null", () => {
            const { result } = renderValidationState();

            expect(() => {
                act(() => {
                    result.current[5]("Title", "Description");
                });
            }).not.toThrow();
        });
    });
});
