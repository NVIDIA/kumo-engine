// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React, { useEffect, useState } from "react";

import {
    Button,
    ChevronDown,
    ChevronUp,
    CloseSmall,
    Collapsible,
    CollapsibleContent,
    CollapsibleTrigger,
    CopyToClipBoard,
    Error,
    TextScrollArea,
    toast,
    Tooltip,
    TooltipContent,
    TooltipTrigger
} from "@kumo-ai/kumo-ui-catalog";

import {
    VALIDATION_SCHEMA_ERROR_EVENT,
    ValidationSchemaErrorEvent
} from "utils/validationSchemaEvent";

export interface ValidationSchemaToastProps {
    method: string;
    url: string;
    apiResponse: string;
    validationErrors: string;
    apiPayload?: string;
    onClose?: () => void;
}

export function triggerValidationSchemaToast({
    method,
    url,
    apiResponse,
    validationErrors,
    apiPayload
}: ValidationSchemaToastProps) {
    const toastId = toast(
        <ValidationSchemaToast
            method={method}
            url={url}
            apiResponse={apiResponse}
            validationErrors={validationErrors}
            apiPayload={apiPayload}
            onClose={() => toast.dismiss(toastId)}
        />,
        {
            duration: Infinity
        }
    );
}

/**
 * Bridges the orval client's schema-validation-error event (services/ cannot
 * import this component) to the toast. Mount once at the app root.
 *
 * Listens on `document` directly (the event is dispatched as a CustomEvent via
 * eventBus.dispatch) so the same handler reference is used for add/remove —
 * cleanup is correct even under StrictMode double-mount.
 */
export function useValidationSchemaToastListener(): void {
    useEffect(() => {
        const handler = (event: Event) => {
            const detail = (event as CustomEvent<ValidationSchemaErrorEvent>)
                .detail;
            triggerValidationSchemaToast(detail);
        };
        document.addEventListener(VALIDATION_SCHEMA_ERROR_EVENT, handler);
        return () =>
            document.removeEventListener(
                VALIDATION_SCHEMA_ERROR_EVENT,
                handler
            );
    }, []);
}

export const ValidationSchemaToast: React.FC<ValidationSchemaToastProps> = ({
    method,
    url,
    apiResponse,
    validationErrors,
    apiPayload,
    onClose
}) => {
    const isUrlTruncated = url.length > 32;

    return (
        <div
            className="pointer-events-auto"
            onPointerDownCapture={(e) => {
                e.stopPropagation();
            }}>
            <div className="flex justify-between flex items-center items-space-between w-full h-10">
                <Tooltip>
                    <TooltipTrigger>
                        <div className="text-rose-700 flex items-center font-semibold text-base">
                            <Error className="fill-rose-700 mr-2" />
                            API response validation failed
                        </div>
                    </TooltipTrigger>
                    <TooltipContent
                        side="left"
                        sideOffset={20}
                        className="text-xs max-w-100">
                        This notification appears in select environments when
                        the backend API response is not in the format that the
                        UI expects. Please inform the experience pod. If you see
                        this error during development, you can update the
                        expected types using `make generate-ui-client-code`.
                    </TooltipContent>
                </Tooltip>
                <Button
                    variant="outline"
                    size="xs"
                    aria-label="Close"
                    onClick={onClose}>
                    <CloseSmall />
                </Button>
            </div>

            <div className="flex justify-between w-full overflow-hidden font-semibold">
                {url && (
                    <Tooltip id={"tooltip-apiurl"}>
                        <TooltipTrigger>
                            <span className="text-sm inline-block max-w-full overflow-hidden truncate w-62.5">
                                {isUrlTruncated
                                    ? `${method}: ${url.slice(0, 31)}...`
                                    : `${method}: ${url}`}
                            </span>
                        </TooltipTrigger>

                        {isUrlTruncated && (
                            <TooltipContent
                                arrow={false}
                                side="left"
                                sideOffset={30}>
                                {url}
                            </TooltipContent>
                        )}
                    </Tooltip>
                )}
            </div>

            <div className=" h-40 ps-2 pt-2 rounded-lg overflow-auto min-w-80">
                <CollapsibleItem
                    title="Show API Response"
                    content={apiResponse}
                />
                <CollapsibleItem
                    title="Show Validation Errors"
                    content={validationErrors}
                />

                {apiPayload && (
                    <CollapsibleItem
                        title="Show API Payload"
                        content={apiPayload}
                    />
                )}
            </div>
        </div>
    );
};

const CollapsibleItem = ({
    title,
    content
}: {
    title: string;
    content: string;
}) => {
    const [toggleItems, setToggleItems] = useState(false);

    return (
        <Collapsible
            className="rounded-lg flex flex-col gap-2 mb-2"
            open={toggleItems}
            onOpenChange={setToggleItems}>
            <div className="flex items-center gap-2">
                <CollapsibleTrigger asChild>
                    <div className="min-w-65">
                        <Button
                            variant="light-outline"
                            size="sm"
                            className="justify-between w-full items-center">
                            {title}
                            {toggleItems ? <ChevronUp /> : <ChevronDown />}
                        </Button>
                    </div>
                </CollapsibleTrigger>
                <div onClick={() => navigator.clipboard.writeText(content)}>
                    <CopyToClipBoard copiedText={content} size="lg" />
                </div>
            </div>
            <CollapsibleContent className=" mt-2">
                <TextScrollArea
                    className="max-h-32 overflow-auto max-w-72.5"
                    text={content}
                />
            </CollapsibleContent>
        </Collapsible>
    );
};
