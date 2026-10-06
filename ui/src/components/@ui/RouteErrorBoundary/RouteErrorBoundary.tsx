// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { isRouteErrorResponse, useRouteError } from "react-router";

import { ErrorFallback } from "@kumo-ai/kumo-ui-catalog";

const RouteErrorBoundary = () => {
    const error = useRouteError();

    let errorMessage = "An unexpected error occurred";
    let errorStatus: number | undefined;
    let errorStack: string | undefined;

    if (isRouteErrorResponse(error)) {
        errorStatus = error.status;
        errorMessage =
            error.status === 404
                ? "Page not found"
                : error.statusText || error.data?.message || errorMessage;
    } else if (error instanceof Error) {
        errorMessage = error.message;
        errorStack = error.stack;
    } else if (typeof error === "string") {
        errorMessage = error;
    }

    return (
        <ErrorFallback
            message={errorMessage}
            statusCode={errorStatus}
            stack={errorStack}
            showGoBack
            centered
        />
    );
};

export default RouteErrorBoundary;
