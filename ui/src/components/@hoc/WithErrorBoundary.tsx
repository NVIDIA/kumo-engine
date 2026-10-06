"use client";
// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React, { ComponentType, forwardRef } from "react";
import { ErrorBoundary, FallbackProps } from "react-error-boundary";

import { ErrorFallback } from "@kumo-ai/kumo-ui-catalog";

// Extract and format the component name from the stack trace
export const getComponentNameFromStack = (stack?: string): string => {
    if (!stack) return "Unknown Component";
    const match = stack.match(/at\s+([A-Za-z0-9_$]+)\s/);
    return match && match?.length > 0 ? match[1] : "Unknown Component";
};

// Error fallback component to display when an error is caught
const ComponentErrorFallback = ({ error }: FallbackProps) => {
    const componentName = getComponentNameFromStack(error?.stack);
    return (
        <ErrorFallback
            message="An error occurred while rendering this component."
            componentName={componentName}
            stack={error?.stack}
        />
    );
};

export const WithErrorBoundaryForwardRef = <P extends object>(
    WrappedComponent: ComponentType<P>
) => {
    return forwardRef<unknown, P>((props, ref: React.ForwardedRef<unknown>) => {
        return (
            <ErrorBoundary FallbackComponent={ComponentErrorFallback}>
                <WrappedComponent {...(props as P)} ref={ref} />
            </ErrorBoundary>
        );
    });
};

const WithErrorBoundary = <P extends object>(
    WrappedComponent: ComponentType<P>
) => {
    const ComponentWithErrorBoundary = (props: P) => {
        return (
            <ErrorBoundary FallbackComponent={ComponentErrorFallback}>
                <WrappedComponent {...props} />
            </ErrorBoundary>
        );
    };
    return ComponentWithErrorBoundary;
};

export default WithErrorBoundary;
