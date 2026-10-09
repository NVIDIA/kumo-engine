// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useMemo, useRef, useState } from "react";

import { Add, CloseSmall } from "@/icons";
import { cn } from "@/lib/utils";
import { Title } from "@/components/ui/title";
import { TagInput } from "@/components/ui/tag-input";

export interface TagProps {
    tagKey?: string;
    value?: string | number | React.ReactNode;
    mode?: "display" | "add" | "edit";
    onRemove?: () => void;
    onSave?: (value: string) => void;
    onCancel?: () => void;
    onError?: (error: string | null) => void;
    existingTags?: Record<string, string>;
    protectedKeys?: Set<string>;
}

// Validation pattern for tag keys and values
const VALID_TAG_PATTERN = /^[a-zA-Z0-9_-]+$/;

function Tag({
    tagKey,
    value,
    mode = "display",
    onRemove,
    onSave,
    onCancel,
    onError,
    existingTags,
    protectedKeys
}: TagProps) {
    const [keyInput, setKeyInput] = useState("");
    const [valueInput, setValueInput] = useState("");
    const containerRef = useRef<HTMLDivElement>(null);

    const isAddMode = useMemo(() => mode === "add", [mode]);
    const isEditMode = useMemo(() => mode === "edit", [mode]);
    const isProtectedKey = useMemo(
        () => tagKey && protectedKeys?.has(tagKey),
        [tagKey, protectedKeys]
    );

    const validateString = useCallback(
        (str: string): boolean => VALID_TAG_PATTERN.test(str),
        []
    );

    const checkDuplicateKey = useCallback(
        (key: string): string | null => {
            if (!existingTags) return null;

            // Check if key already exists
            if (key in existingTags) {
                return `Tag key "${key}" already exists`;
            }

            return null;
        },
        [existingTags]
    );

    const handleSubmit = useCallback(() => {
        const trimmedKey = keyInput.trim();
        const trimmedValue = valueInput.trim();

        // Check if both fields are empty (cancel action)
        if (!trimmedKey && !trimmedValue) {
            onError?.(null);
            onCancel?.();
            return;
        }

        // Validate that both key and value are provided
        if (!trimmedKey) {
            onError?.("Key is required");
            return;
        }

        if (!trimmedValue) {
            onError?.("Value is required");
            return;
        }

        // Validate key format
        if (!validateString(trimmedKey)) {
            onError?.(
                "Only alphanumeric characters, underscores (_), or hyphens (-) are allowed in key"
            );
            return;
        }

        // Validate value format
        if (!validateString(trimmedValue)) {
            onError?.(
                "Only alphanumeric characters, underscores (_), or hyphens (-) are allowed in value"
            );
            return;
        }

        // Check for duplicate key
        const duplicateError = checkDuplicateKey(trimmedKey);
        if (duplicateError) {
            onError?.(duplicateError);
            return;
        }

        // All validations passed
        onError?.(null);
        onSave?.(`${trimmedKey}=${trimmedValue}`);
        setKeyInput("");
        setValueInput("");
    }, [
        keyInput,
        valueInput,
        onSave,
        onCancel,
        validateString,
        onError,
        checkDuplicateKey
    ]);

    const handleKeyDown = useCallback(
        (e: React.KeyboardEvent<HTMLInputElement>) => {
            if (e.key === "Enter") {
                handleSubmit();
            }
        },
        [handleSubmit]
    );

    const handleInputChange = useCallback(
        (
            setter: React.Dispatch<React.SetStateAction<string>>,
            fieldName: string
        ) =>
            (e: React.ChangeEvent<HTMLInputElement>) => {
                const newValue = e.target.value;
                setter(newValue);

                // Real-time validation as user types
                if (newValue.trim()) {
                    if (!validateString(newValue)) {
                        onError?.(
                            `Only alphanumeric characters, underscores (_), or hyphens (-) are allowed in ${fieldName}`
                        );
                    } else {
                        onError?.(null);
                    }
                } else {
                    onError?.(null);
                }
            },
        [onError, validateString]
    );

    const handleKeyChange = useMemo(
        () => handleInputChange(setKeyInput, "key"),
        [handleInputChange]
    );

    const handleValueChange = useMemo(
        () => handleInputChange(setValueInput, "value"),
        [handleInputChange]
    );

    const handleBlur = useCallback(() => {
        // Use setTimeout to allow the browser to update document.activeElement
        setTimeout(() => {
            // Check if the new focus is still within our container
            if (
                containerRef.current &&
                !containerRef.current.contains(document.activeElement)
            ) {
                // Focus has left the entire tag component
                handleSubmit();
            }
        }, 0);
    }, [handleSubmit]);

    if (isEditMode) {
        return (
            <div
                data-slot="tag"
                ref={containerRef}
                className="bg-kumo-grey-100 flex items-center gap-1 rounded-xl px-4 py-0.5">
                <div className="flex items-center gap-0">
                    <Title children={"Key="} variant="section-header" />
                    <TagInput
                        value={keyInput}
                        onChange={handleKeyChange}
                        onKeyDown={handleKeyDown}
                        onBlur={handleBlur}
                        placeholder="key"
                        autoFocus
                    />
                </div>
                <div className="flex items-center gap-0">
                    <Title children={"Value="} variant="section-header" />
                    <TagInput
                        value={valueInput}
                        onChange={handleValueChange}
                        onKeyDown={handleKeyDown}
                        onBlur={handleBlur}
                        placeholder="value"
                    />
                </div>
            </div>
        );
    }

    return (
        <div
            data-slot="tag"
            className={cn(
                "flex items-center rounded-xl py-1",
                isAddMode
                    ? "shrink-0 cursor-pointer gap-1 whitespace-nowrap px-0"
                    : "bg-kumo-grey-100 shrink-0 gap-0 whitespace-nowrap px-4"
            )}>
            {isAddMode ? (
                <>
                    <Add className="size-3.5! fill-title-secondary" />

                    <Title children={"Add tag"} variant="section-header" />
                </>
            ) : (
                <>
                    {tagKey && value !== "" && value !== undefined && (
                        <Title
                            children={`${tagKey}=`}
                            variant="section-header"
                        />
                    )}
                    {value === "" || value === undefined ? (
                        <Title children={tagKey} variant="section-header" />
                    ) : (
                        <span className="text-title-primary text-base font-medium leading-5">
                            {value}
                        </span>
                    )}
                    {onRemove && !isProtectedKey && (
                        <>
                            <span className="inline-block w-1" />
                            <div
                                className="cursor-pointer"
                                onClick={onRemove}
                                data-testid="tag-remove-icon">
                                <CloseSmall className="size-4!" />
                            </div>
                        </>
                    )}
                </>
            )}
        </div>
    );
}

export { Tag };
