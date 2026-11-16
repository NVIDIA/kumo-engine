// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { type ReactNode, useState } from "react";

import { Button, cn, Input, Label } from "@kumo-ai/kumo-ui-catalog";

import { deriveResourceSlug, resourceSlugError } from "utils/resourceSlug";

interface ResourceSlugFieldProps {
    displayName: string;
    id: string;
    value: string;
    onChange: (value: string) => void;
    className?: string;
    // When true, render as a label-left / input-right row (to line up with
    // sibling fields in a two-column grid) instead of the stacked default.
    horizontal?: boolean;
    // Optional className applied to the Label element (e.g. to match sibling
    // field label colors in a form).
    labelClassName?: string;
    // When true, hide the example ("fraud-detection") placeholder and the
    // "Preview: <slug>" hint line. The derived ID still shows as the input's
    // placeholder. Scoped to the identity form; other consumers keep the hints.
    hidePreviewHint?: boolean;
}

export const ResourceSlugField = ({
    displayName,
    id,
    value,
    onChange,
    className = "",
    horizontal = false,
    labelClassName,
    hidePreviewHint = false
}: ResourceSlugFieldProps) => {
    const preview = deriveResourceSlug(displayName);
    const error = resourceSlugError(value, displayName);
    const descriptionId = `${id}-description`;
    const errorId = `${id}-error`;

    const label = (
        <Label
            htmlFor={id}
            className={cn(
                "text-sm font-medium",
                labelClassName,
                // In the horizontal layout, align the label to the top of the
                // right column so it sits beside the input, not the hint text.
                horizontal && "self-start pt-2"
            )}>
            ID <span className="font-normal">(optional)</span>
        </Label>
    );

    const dynamicHint = value
        ? `Selected ID: ${value}`
        : !hidePreviewHint && preview
          ? `Preview: ${preview}`
          : null;

    let hintNode: ReactNode;
    if (dynamicHint) {
        hintNode = (
            <>
                <p aria-live="polite">{dynamicHint}</p>
                <p>This ID cannot be changed later. The name can be changed.</p>
            </>
        );
    } else if (hidePreviewHint) {
        hintNode = (
            <p aria-live="polite">
                Enter a name to preview the automatically generated ID. This ID
                cannot be changed later. The name can be changed.
            </p>
        );
    } else {
        hintNode = (
            <>
                <p aria-live="polite">
                    Enter a name to preview the automatically generated ID.
                </p>
                <p>This ID cannot be changed later. The name can be changed.</p>
            </>
        );
    }

    const body = (
        <>
            <Input
                id={id}
                name="slug"
                value={value}
                placeholder={
                    hidePreviewHint ? preview : preview || "fraud-detection"
                }
                aria-describedby={`${descriptionId}${error ? ` ${errorId}` : ""}`}
                aria-invalid={Boolean(error)}
                onChange={(event) => onChange(event.target.value)}
            />
            <div id={descriptionId} className="text-sm text-title-secondary">
                {hintNode}
            </div>
            {error && (
                <p
                    id={errorId}
                    role="alert"
                    className="text-sm text-destructive">
                    {error}
                </p>
            )}
        </>
    );

    return horizontal ? (
        <div
            className={cn(
                "grid grid-cols-[var(--form-label-col)_1fr] items-start gap-2",
                className
            )}>
            {label}
            <div className="flex flex-col gap-2">{body}</div>
        </div>
    ) : (
        <div className={cn("flex flex-col gap-2", className)}>
            {label}
            {body}
        </div>
    );
};

interface ImmutableResourceSlugProps {
    id: string;
    slug: string;
    nameEditable?: boolean;
    className?: string;
}

export const ImmutableResourceSlug = ({
    id,
    slug,
    nameEditable = true,
    className = ""
}: ImmutableResourceSlugProps) => {
    const [copyStatus, setCopyStatus] = useState("");
    const descriptionId = `${id}-description`;

    const copySlug = async () => {
        if (!navigator.clipboard) {
            setCopyStatus("Copy is not available in this browser.");
            return;
        }
        try {
            await navigator.clipboard.writeText(slug);
            setCopyStatus("ID copied.");
        } catch {
            setCopyStatus("ID could not be copied.");
        }
    };

    return (
        <div className={cn("flex flex-col gap-2", className)}>
            <Label htmlFor={id} className="text-sm font-medium">
                ID
            </Label>
            <div className="flex items-center gap-2">
                <Input
                    id={id}
                    name="slug"
                    value={slug}
                    readOnly
                    aria-describedby={descriptionId}
                />
                <Button
                    type="button"
                    variant="outline"
                    aria-label="Copy ID"
                    disabled={!slug}
                    onClick={() => void copySlug()}>
                    Copy
                </Button>
            </div>
            <div id={descriptionId} className="text-sm text-title-secondary">
                <p>
                    {nameEditable
                        ? "This ID cannot be changed later. The name can be changed."
                        : "This ID and deployment-managed name cannot be changed here."}
                </p>
                <p role="status" aria-live="polite">
                    {copyStatus}
                </p>
            </div>
        </div>
    );
};
