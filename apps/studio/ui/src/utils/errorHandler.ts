// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { createLogLogUiError } from "gen/v0/api/log/log";

import { CustomError, ErrorMode } from "types/common";
import { AlertData, BannerData } from "types/validation";

import { errorStatusCode, UNEXPECTED_RESPONSE_MESSAGE } from "./Constants";

/**
 * Log stack trace to backend for production error tracking.
 * In non-production, logs to console for debugging.
 */
export async function logStackTrace(
    message: string,
    stackTrace: { stack?: string }
) {
    if (process.env.NODE_ENV === "production") {
        try {
            if (stackTrace.stack)
                await createLogLogUiError({
                    message: message,
                    level: "info",
                    stacktrace: stackTrace.stack
                });
        } catch (error) {
            console.trace(error);
        }
    } else if (process.env.NODE_ENV !== "test") {
        console.trace(stackTrace);
    }
}

/** Status codes emitted by an upstream proxy/gateway (nginx, load balancer). */
const GATEWAY_ERROR_STATUSES: ReadonlyArray<number> = [
    errorStatusCode.BadGateway,
    errorStatusCode.ServiceUnavailable,
    errorStatusCode.GatewayTimeout
];

/**
 * Generic, actionable message shown when an upstream gateway/proxy returns a
 * 502/503/504. handleErrorFunctional is shared app-wide, so this stays generic
 * (no column/file-size specifics) - see #30031.
 */
const GATEWAY_ERROR_MESSAGE =
    "The server is temporarily unavailable or took too long to respond. " +
    "Please retry. If the problem persists, contact Kumo Support.";

/** Status-neutral banner titles per gateway status (502/503/504 are not all timeouts). */
const GATEWAY_ERROR_TITLES: Readonly<Record<number, string>> = {
    [errorStatusCode.BadGateway]: "Server error",
    [errorStatusCode.ServiceUnavailable]: "Server unavailable",
    [errorStatusCode.GatewayTimeout]: "Server timeout"
};

function isGatewayError(status?: number): boolean {
    return status !== undefined && GATEWAY_ERROR_STATUSES.includes(status);
}

/**
 * Prefer the caller-supplied title for context (e.g. "Failed to load columns")
 * and otherwise fall back to a status-neutral title per gateway status.
 */
function gatewayBannerTitle(title: string, status?: number): string {
    if (title !== "Something went wrong!") {
        return title;
    }
    return (
        (status !== undefined && GATEWAY_ERROR_TITLES[status]) ||
        "Server unavailable"
    );
}

/** True when a body string looks like raw HTML/XML markup rather than text/JSON. */
function isMarkupBody(value: string): boolean {
    return value.trimStart().startsWith("<");
}

/** True when the response advertises an HTML content type. */
function isHtmlContentType(error: CustomError): boolean {
    const headers = error?.response?.headers;
    if (!headers) return false;
    const contentType = headers["content-type"] ?? headers["Content-Type"];
    return (
        typeof contentType === "string" &&
        contentType.toLowerCase().includes("text/html")
    );
}

/**
 * Method to parse the error and then call method to show alert dialog or
 * error/warning banner based on the error status code.
 *
 * @param title Title for alert dialog or error/warning component
 * @param error error object
 */
export function handleErrorFunctional(
    error: CustomError,
    showBanner: (bannerData: BannerData) => void,
    alertError: (alertData: AlertData) => void,
    title: string = "Something went wrong!",
    isWarning: boolean = false
) {
    const description = parseErrorMessage(error);
    const status = error?.response?.status;
    if (
        status === errorStatusCode.InternalServerError &&
        error?.mode !== ErrorMode.ALERT
    ) {
        return showBanner({
            title: "Internal error",
            subheader: "",
            warnings: [
                "We could not process your request due to an internal error, please try again!"
            ],
            isWarning: false
        });
    } else if (isGatewayError(status) && error?.mode !== ErrorMode.ALERT) {
        // Gateway/proxy errors (502/503/504) arrive from nginx, often as raw
        // HTML; show a clear, actionable message instead of the raw body,
        // preserving the caller-supplied title for context (#30031).
        return showBanner({
            title: gatewayBannerTitle(title, status),
            subheader: "",
            warnings: [GATEWAY_ERROR_MESSAGE],
            isWarning: false
        });
    } else if (
        status === errorStatusCode.UnProcessableEntityError &&
        error?.mode !== ErrorMode.ALERT
    ) {
        if (title !== "Something went wrong!") {
            return showBanner({
                title: title,
                subheader: "",
                warnings: [description],
                isWarning: isWarning
            });
        } else {
            return showBanner({
                title: "Sorry, there is an error with one of the inputs",
                subheader: "",
                warnings: [description],
                isWarning: isWarning
            });
        }
    } else {
        // Skipping `Request aborted` is because, this error comes only when API
        // request is in progress and user reloads page. So no need to alert
        if (description !== "Request aborted")
            return alertError({
                "title": title,
                "description": description
            });
    }
}

/**
 * Parsing error data to get the proper error message
 * @param error tryCatch or Promise errors
 * @returns ```message``` string message
 */
type ResponseData = NonNullable<NonNullable<CustomError["response"]>["data"]>;

/** Join the `msg` of each item in a FastAPI-style error array. */
function joinErrorMsgs(items: Array<{ msg?: string }>): string {
    return items.map((val) => val?.msg).join(", ");
}

/** Extract a message string from the various shapes `response.data` can take. */
function parseResponseData(data: ResponseData): string {
    if (typeof data === "string") {
        return data;
    }
    if (Array.isArray(data)) {
        return joinErrorMsgs(data);
    }
    if (typeof data === "object") {
        if (typeof data?.detail === "string") {
            return data.detail;
        }
        if (Array.isArray(data?.detail)) {
            return joinErrorMsgs(data.detail);
        }
        if (data?.error) {
            return data.error.toString();
        }
        return data?.toString();
    }
    return "";
}

export function parseErrorMessage(error: CustomError) {
    let retValue = "";
    if (error?.response) {
        if (error.response.data) {
            retValue = parseResponseData(error.response.data);
        }
    } else if (error?.message) {
        retValue = error?.message?.toString();
    } else if (error?.request) {
        retValue = error?.request?.toString();
    } else {
        retValue = error?.toString();
    }

    // Log the raw value for observability before any sanitization.
    logStackTrace(retValue, error);

    // Never surface a raw markup body (e.g. a proxy-generated nginx HTML error
    // page) to the user; return a safe, generic message instead (#30031).
    if (isMarkupBody(retValue) || isHtmlContentType(error)) {
        return UNEXPECTED_RESPONSE_MESSAGE;
    }

    return retValue;
}

export function prepareCustomError(
    error: string,
    statusCode = errorStatusCode.Success,
    errorMode?: ErrorMode
) {
    const customError: CustomError = {
        response: {
            status: statusCode,
            data: error
        },
        name: "",
        message: error,
        mode: errorMode
    };
    return customError;
}

/**
 * Adjusts error description to provide helpful messaging when empty.
 * Returns the original description if provided, otherwise returns
 * a user-friendly message with the current page URL for support.
 */
export const adjustError = (errorDescription: string) => {
    const currentPageUrl = window.location.href;

    if (!errorDescription) {
        return (
            "We're sorry, an unexpected error occurred while " +
            "processing your request. \n\n" +
            "To help us resolve this quickly, please contact " +
            "Kumo Support with the following information: \n\n" +
            `Page URL: ${currentPageUrl} \n\n` +
            "Thank you for your patience while we work to resolve the issue."
        );
    }

    return errorDescription;
};
