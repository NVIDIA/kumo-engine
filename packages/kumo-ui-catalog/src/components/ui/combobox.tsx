// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2023 shadcn
// SPDX-License-Identifier: Apache-2.0 AND MIT

// src/components/ui/combobox.tsx
"use client";

import { ChevronsUpDown } from "lucide-react";
import * as React from "react";

import { Button } from "@/components/ui/button";
import {
    Command,
    CommandEmpty,
    CommandGroup,
    CommandInput,
    CommandItem,
    CommandList
} from "@/components/ui/command";
import {
    Popover,
    PopoverContent,
    PopoverTrigger
} from "@/components/ui/popover";
import { Check } from "@/icons";
import { cn } from "@/lib/utils";

export type ComboboxOption = {
    value: string;
    label: string;
    disabled?: boolean;
};

interface ComboboxProps {
    options: ComboboxOption[];
    value?: string;
    onValueChange?: (value: string) => void;
    placeholder?: string;
    emptyText?: string;
    disabled?: boolean;
    searchPlaceholder?: string;
    className?: string;
    popoverClassName?: string;
    triggerClassName?: string;
    "aria-label"?: string;
}

export function Combobox({
    options,
    value,
    onValueChange,
    placeholder = "Select option",
    emptyText = "No results found.",
    disabled = false,
    searchPlaceholder = "Search...",
    className,
    popoverClassName,
    triggerClassName,
    "aria-label": ariaLabel
}: ComboboxProps) {
    const [open, setOpen] = React.useState(false);
    const [selectedValue, setSelectedValue] = React.useState(value || "");

    React.useEffect(() => {
        if (value !== undefined) {
            setSelectedValue(value);
        }
    }, [value]);

    const handleSelect = React.useCallback(
        (value: string) => {
            setSelectedValue(value);
            onValueChange?.(value);
            setOpen(false);
        },
        [onValueChange]
    );

    const selectedOption = React.useMemo(
        () => options.find((option) => option.value === selectedValue),
        [options, selectedValue]
    );

    return (
        <Popover open={open} onOpenChange={setOpen}>
            <PopoverTrigger asChild>
                <Button
                    variant="light-outline"
                    role="combobox"
                    aria-expanded={open}
                    aria-label={ariaLabel || placeholder}
                    disabled={disabled}
                    className={cn(
                        "w-full justify-between font-normal",
                        !selectedValue && "text-text-secondary",
                        triggerClassName
                    )}>
                    {selectedOption ? selectedOption.label : placeholder}
                    <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
                </Button>
            </PopoverTrigger>
            <PopoverContent
                className={cn("w-[200px] p-0", popoverClassName)}
                align="start">
                <Command className={className}>
                    <CommandInput placeholder={searchPlaceholder} />
                    <CommandList>
                        <CommandEmpty>{emptyText}</CommandEmpty>
                        <CommandGroup>
                            {options.map((option) => (
                                <CommandItem
                                    key={option.value}
                                    value={option.value}
                                    disabled={option.disabled}
                                    onSelect={handleSelect}>
                                    <Check
                                        className={cn(
                                            "mr-2 size-4",
                                            selectedValue === option.value
                                                ? "opacity-100"
                                                : "opacity-0"
                                        )}
                                    />
                                    {option.label}
                                </CommandItem>
                            ))}
                        </CommandGroup>
                    </CommandList>
                </Command>
            </PopoverContent>
        </Popover>
    );
}
