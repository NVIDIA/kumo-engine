// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { waitFor } from "@testing-library/dom";
import { mockAxiosInstance } from "tests/mockAxios";
import type { Mock } from "vitest";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { CustomError } from "types/common";
import { AlertData, BannerData } from "types/validation";

import { errorStatusCode } from "utils/Constants";
import {
    adjustError,
    handleErrorFunctional,
    logStackTrace,
    parseErrorMessage
} from "utils/errorHandler";

const createError = (overrides: Partial<CustomError>): CustomError => ({
    name: "Error",
    message: "Default error message",
    ...overrides
});

describe("parseErrorMessage", () => {
    it.each([
        {
            description: "Should parse simple string error response",
            errorOverrides: {
                response: { data: "Invalid input" }
            },
            expected: "Invalid input"
        },
        {
            description: "Should parse array of error messages",
            errorOverrides: {
                response: {
                    data: [
                        { msg: "Field A required" },
                        { msg: "Invalid value" }
                    ]
                }
            },
            expected: "Field A required, Invalid value"
        },
        {
            description: "Should parse nested detail array",
            errorOverrides: {
                response: {
                    data: { detail: [{ msg: "Nested A" }, { msg: "Nested B" }] }
                }
            } as unknown as Partial<CustomError>,
            expected: "Nested A, Nested B"
        },
        {
            description: "Should parse detail string",
            errorOverrides: {
                response: { data: { detail: "Detail string" } }
            } as unknown as Partial<CustomError>,
            expected: "Detail string"
        },
        {
            description: "Should parse fallback error string",
            errorOverrides: {
                response: { data: { error: "Fatal error" } }
            } as unknown as Partial<CustomError>,
            expected: "Fatal error"
        },
        {
            description: "Should parse generic error object to string",
            errorOverrides: {
                response: { data: { unexpected: true } }
            } as unknown as Partial<CustomError>,
            expected: "[object Object]"
        },
        {
            description: "Should parse message when no response exists",
            errorOverrides: { message: "Fallback message" },
            expected: "Fallback message"
        },
        {
            description: "Should parse request fallback",
            errorOverrides: { message: "", request: "request error" },
            expected: "request error"
        }
    ])("$description", ({ errorOverrides, expected }) => {
        const error = createError(errorOverrides);
        expect(parseErrorMessage(error)).toBe(expected);
    });

    it.each([
        {
            description: "raw HTML string body (nginx 504 page)",
            errorOverrides: {
                response: {
                    status: errorStatusCode.GatewayTimeout,
                    data: "<html><body><h1>504 Gateway Time-out</h1></body></html>"
                }
            }
        },
        {
            description: "leading-whitespace HTML body",
            errorOverrides: {
                response: { data: "  \n<!DOCTYPE html><html></html>" }
            }
        },
        {
            description: "text/html content type with non-markup body",
            errorOverrides: {
                response: {
                    data: "gateway error",
                    headers: { "content-type": "text/html; charset=utf-8" }
                }
            }
        }
    ])(
        "Should never surface raw markup: $description",
        ({ errorOverrides }) => {
            const error = createError(
                errorOverrides as unknown as Partial<CustomError>
            );
            const message = parseErrorMessage(error);
            expect(message).not.toContain("<");
            expect(message).toContain("unexpected response");
        }
    );
});

describe("handleErrorFunctional", () => {
    let showBanner: Mock<(bannerData: BannerData) => void>;
    let alertError: Mock<(alertData: AlertData) => void>;

    beforeEach(() => {
        showBanner = vi.fn<(bannerData: BannerData) => void>();
        alertError = vi.fn<(alertData: AlertData) => void>();
    });

    it("Should show banner for InternalServerError with isV2=true", () => {
        const error = createError({
            response: { status: errorStatusCode.InternalServerError }
        });

        handleErrorFunctional(
            error,
            showBanner,
            alertError,
            "Test Title",
            false
        );

        expect(showBanner).toHaveBeenCalledWith({
            title: "Internal error",
            subheader: "",
            warnings: [
                "We could not process your request due to an internal error, please try again!"
            ],
            isWarning: false
        });
        expect(alertError).not.toHaveBeenCalled();
    });

    it.each([
        {
            description:
                "Should show banner for UnProcessableEntityError with custom title",
            title: "Input error",
            expectedTitle: "Input error"
        },
        {
            description:
                "Should show default banner for UnProcessableEntityError without title",
            title: undefined,
            expectedTitle: "Sorry, there is an error with one of the inputs"
        }
    ])("$description", ({ title, expectedTitle }) => {
        const error = createError({
            response: {
                status: errorStatusCode.UnProcessableEntityError,
                data: "Invalid"
            }
        });

        handleErrorFunctional(error, showBanner, alertError, title);

        expect(showBanner).toHaveBeenCalledWith({
            title: expectedTitle,
            subheader: "",
            warnings: ["Invalid"],
            isWarning: false
        });
    });

    it("Should alert error for other error types", () => {
        const error = createError({
            response: { status: 400, data: "General error" }
        } as unknown as CustomError);

        handleErrorFunctional(error, showBanner, alertError, "Oops");

        expect(alertError).toHaveBeenCalledWith({
            title: "Oops",
            description: "General error"
        });
        expect(showBanner).not.toHaveBeenCalled();
    });

    it("Should not alert when error message is 'Request aborted'", () => {
        const error = createError({ message: "Request aborted" });

        handleErrorFunctional(error, showBanner, alertError, "Abort test");

        expect(alertError).not.toHaveBeenCalled();
        expect(showBanner).not.toHaveBeenCalled();
    });

    it.each([
        errorStatusCode.BadGateway,
        errorStatusCode.ServiceUnavailable,
        errorStatusCode.GatewayTimeout
    ])(
        "Should show an actionable banner (not raw HTML) for gateway status %i",
        (status) => {
            const error = createError({
                response: {
                    status,
                    data: "<html>\n<head><title>504 Gateway Time-out</title></head>\n<body><center><h1>504 Gateway Time-out</h1></center></body>\n</html>"
                }
            } as unknown as CustomError);

            handleErrorFunctional(
                error,
                showBanner,
                alertError,
                "Failed to load columns"
            );

            expect(alertError).not.toHaveBeenCalled();
            expect(showBanner).toHaveBeenCalledTimes(1);
            const bannerArg = showBanner.mock.calls[0][0];
            // Caller-supplied title is preserved for context.
            expect(bannerArg.title).toBe("Failed to load columns");
            // Message must be actionable, generic (no file-size copy), and
            // never contain raw markup.
            expect(bannerArg.warnings[0]).toContain("Please retry");
            expect(bannerArg.warnings[0]).not.toContain("file size");
            bannerArg.warnings.forEach((w) => expect(w).not.toContain("<"));
        }
    );

    it.each([
        {
            status: errorStatusCode.BadGateway,
            expectedTitle: "Server error"
        },
        {
            status: errorStatusCode.ServiceUnavailable,
            expectedTitle: "Server unavailable"
        },
        {
            status: errorStatusCode.GatewayTimeout,
            expectedTitle: "Server timeout"
        }
    ])(
        "Should use a status-neutral title when no caller title is supplied ($status)",
        ({ status, expectedTitle }) => {
            const error = createError({
                response: { status, data: "<html></html>" }
            } as unknown as CustomError);

            // No title argument -> falls back to the default sentinel.
            handleErrorFunctional(error, showBanner, alertError);

            expect(showBanner).toHaveBeenCalledTimes(1);
            expect(showBanner.mock.calls[0][0].title).toBe(expectedTitle);
        }
    );
});

describe("adjustError", () => {
    const originalHref = window.location.href;

    beforeEach(() => {
        // Set a known test URL
        window.location.href = "https://test.kumo.ai/test-page";
    });

    afterEach(() => {
        // Restore original location
        window.location.href = originalHref;
    });

    it.each([
        {
            description: "returns custom message for empty string",
            input: "",
            expectedContains: [
                "We're sorry, an unexpected error occurred",
                "Page URL: https://test.kumo.ai/test-page",
                "please contact Kumo Support"
            ],
            expectedExactMatch: null
        },
        {
            description: "returns original message for 'uncaught' error",
            input: "Uncaught error occurred",
            expectedContains: [],
            expectedExactMatch: "Uncaught error occurred"
        },
        {
            description: "returns original message for 'activity' error",
            input: "Activity failed",
            expectedContains: [],
            expectedExactMatch: "Activity failed"
        },
        {
            description: "returns original message for 'workflow' error",
            input: "Workflow terminated",
            expectedContains: [],
            expectedExactMatch: "Workflow terminated"
        },
        {
            description: "returns original message for validation errors",
            input: "Validation failed for field X",
            expectedContains: [],
            expectedExactMatch: "Validation failed for field X"
        },
        {
            description: "returns original message for any error with text",
            input: "An UnCaUgHt error",
            expectedContains: [],
            expectedExactMatch: "An UnCaUgHt error"
        }
    ])("$description", ({ input, expectedContains, expectedExactMatch }) => {
        const result = adjustError(input);

        if (expectedExactMatch) {
            expect(result).toBe(expectedExactMatch);
        }

        expectedContains.forEach((substring) => {
            expect(result).toContain(substring);
        });
    });
});

describe("logStackTrace", () => {
    const originalEnv = process.env;
    const message = "Something went wrong";
    const stackTrace = { stack: "Error: stack trace" };

    beforeEach(() => {
        vi.clearAllMocks();
        process.env = { ...originalEnv };
    });

    afterEach(() => {
        process.env = originalEnv;
    });

    it("calls logError in production with valid stack", async () => {
        process.env.NODE_ENV = "production";
        await logStackTrace(message, stackTrace);
        await waitFor(() => {
            expect(mockAxiosInstance.post).toHaveBeenCalledWith(
                `v0/log/log_ui_error`,
                {
                    "level": "info",
                    "message": "Something went wrong",
                    "stacktrace": "Error: stack trace"
                },
                {
                    "headers": {
                        "Content-Type": "application/json"
                    }
                }
            );
        });
    });

    it("calls console.trace in development", async () => {
        process.env.NODE_ENV = "development";
        const traceSpy = vi
            .spyOn(console, "trace")
            .mockImplementation(() => {});

        await logStackTrace(message, stackTrace);

        expect(traceSpy).toHaveBeenCalledWith(stackTrace);
        traceSpy.mockRestore();
    });
});
