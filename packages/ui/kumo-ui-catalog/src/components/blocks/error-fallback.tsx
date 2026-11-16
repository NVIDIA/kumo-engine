// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export interface ErrorFallbackProps {
    message: string;
    componentName?: string;
    statusCode?: number;
    stack?: string;
    showGoBack?: boolean;
    centered?: boolean;
    showStackTrace?: boolean;
}

function ErrorFallback({
    message,
    componentName,
    statusCode,
    stack,
    showGoBack = false,
    centered = false,
    showStackTrace = false
}: ErrorFallbackProps) {
    const content = (
        <Alert
            variant="destructive"
            enableClose={false}
            className={cn(centered && "max-w-lg")}>
            <AlertTitle>
                {statusCode && `Error ${statusCode}: `}
                Something went wrong
                {componentName && (
                    <>
                        {" "}
                        in{" "}
                        <strong className="text-destructive">
                            {componentName}
                        </strong>
                    </>
                )}
            </AlertTitle>
            <AlertDescription>
                <p className="mb-3">
                    {message} Please contact Kumo support or try refreshing the
                    page if the issue persists.
                </p>
                {showStackTrace && stack && (
                    <details className="mb-3">
                        <summary className="cursor-pointer text-sm">
                            Stack Trace
                        </summary>
                        <pre className="text-title-secondary mt-2 max-h-40 overflow-auto whitespace-pre-wrap text-xs">
                            {stack}
                        </pre>
                    </details>
                )}
                <div className="mt-3 flex gap-2 pb-2">
                    <Button
                        onClick={() => window.location.reload()}
                        variant="light-outline"
                        size="sm">
                        Refresh Page
                    </Button>
                    {showGoBack && (
                        <Button
                            onClick={() => window.history.back()}
                            variant="light-outline"
                            size="sm">
                            Go Back
                        </Button>
                    )}
                </div>
            </AlertDescription>
        </Alert>
    );

    if (centered) {
        return (
            <div className="min-h-100 flex items-center justify-center p-4">
                {content}
            </div>
        );
    }

    return content;
}

export { ErrorFallback };
