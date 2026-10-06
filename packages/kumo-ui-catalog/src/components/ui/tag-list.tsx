// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useState } from "react";

import { Description } from "@/components/ui/description";
import { Tag } from "@/components/ui/tag";

interface TagListProps {
    tags?: Record<string, string>;
    onTagAdd?: (key: string, value: string | undefined) => void;
    onTagRemove?: (key: string) => void;
    protectedKeys?: Set<string>;
    showHeader?: boolean;
    noWrap?: boolean;
}

function TagList({
    tags,
    onTagAdd,
    onTagRemove,
    protectedKeys,
    showHeader = true,
    noWrap = false
}: TagListProps) {
    const [isAddingTag, setIsAddingTag] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const handleRemoveTag = useCallback(
        (tagKey: string) => {
            onTagRemove?.(tagKey);
        },
        [onTagRemove]
    );

    const handleSaveTag = useCallback(
        (value: string) => {
            if (!onTagAdd) return;

            // Split by first '=' to create key-value pair
            const equalIndex = value.indexOf("=");
            let key: string, val: string | undefined;

            if (equalIndex !== -1) {
                // If '=' exists, split into key and value
                key = value.substring(0, equalIndex).trim();
                val = value.substring(equalIndex + 1).trim();
            } else {
                // If no '=', use the value as key with undefined value
                key = value;
                val = undefined;
            }

            onTagAdd(key, val);
            setIsAddingTag(false);
        },
        [onTagAdd]
    );

    return (
        <div data-slot="tag-list" className="flex flex-col">
            {showHeader && (
                <Description
                    size="sm"
                    variant="secondary"
                    className="py-0 font-medium">
                    {"Tags"}
                </Description>
            )}
            {showHeader && <div className="h-2" />}
            <div
                className={`flex gap-2 ${noWrap ? "flex-nowrap overflow-x-auto" : "flex-wrap items-start"}`}>
                {tags &&
                    Object.entries(tags).map(([key, value]) => (
                        <Tag
                            key={"tags-" + key}
                            tagKey={key}
                            value={value}
                            onRemove={() => handleRemoveTag(key)}
                            protectedKeys={protectedKeys}
                        />
                    ))}
                {isAddingTag && (
                    <Tag
                        mode="edit"
                        onSave={handleSaveTag}
                        onCancel={() => setIsAddingTag(false)}
                        onError={setError}
                        existingTags={tags}
                    />
                )}
                <div onClick={() => setIsAddingTag(true)}>
                    <Tag mode="add" />
                </div>
            </div>
            {error && (
                <div className="mt-1">
                    <span className="text-destructive text-xs">{error}</span>
                </div>
            )}
            {showHeader && <div className="h-2" />}
        </div>
    );
}

export { TagList };
export type { TagListProps };
