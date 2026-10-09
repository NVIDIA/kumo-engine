// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Input } from "@/components/ui/input";

interface TagInputProps {
    value: string;
    onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
    onKeyDown: (e: React.KeyboardEvent<HTMLInputElement>) => void;
    onBlur: () => void;
    placeholder: string;
    autoFocus?: boolean;
}

function TagInput({
    value,
    onChange,
    onKeyDown,
    onBlur,
    placeholder,
    autoFocus = false
}: TagInputProps) {
    return (
        <Input
            data-slot="tag-input"
            type="text"
            value={value}
            onChange={onChange}
            onKeyDown={onKeyDown}
            onBlur={onBlur}
            className="text-base! text-title-primary h-auto w-12 min-w-0 border-none bg-transparent p-0 font-medium shadow-none outline-none focus-visible:ring-0"
            placeholder={placeholder}
            autoFocus={autoFocus}
        />
    );
}

export { TagInput };
export type { TagInputProps };
