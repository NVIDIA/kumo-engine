"use client";
// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2023 shadcn
// SPDX-License-Identifier: Apache-2.0 AND MIT


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
import { ChevronDown } from "@/icons";
import { cn } from "@/lib/utils";
import { Check } from "lucide-react";
import { Checkbox } from "@/components/ui/checkbox";
import { AddCircle } from "@/icons";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useVirtualizer } from "@tanstack/react-virtual";
import { Description } from "./description";
import { SelectOption } from "@/types/select";

interface SelectProps {
    id?: string;
    options: SelectOption[];
    selectedValues?: string[];
    onValuesChange?: (values: string[]) => void;
    placeholder?: string | React.ReactNode;
    emptyText?: string;
    search?: boolean;
    searchPlaceholder?: string;
    disabled?: boolean;
    className?: string;
    popoverClassName?: string;
    mode?: "single" | "multi";
    variant?: "default" | "secondary" | "nooutline";
    size?: "md" | "sm" | "xs";
    multiModeText?: string | React.ReactNode;
    sortOptions?: boolean;
    isInValid?: boolean;
    selectedOptionPrefix?: string;
    showSelectedCount?: boolean;
    popoverTriggerClassName?: string;
    popoverSide?: "top" | "bottom" | "left" | "right";
    collisionBoundary?: Element | Element[] | null;
    maxPopoverHeight?: number;
    showCheckbox?: boolean;
    onOpenChange?: (open: boolean) => void;
}

export function Select({
    id = "combobox",
    options,
    selectedValues: externalSelectedValues,
    onValuesChange,
    placeholder = "Select options",
    emptyText = "No options found.",
    search = true,
    searchPlaceholder = "Search...",
    disabled = false,
    className,
    popoverClassName = "w-fit",
    popoverTriggerClassName,
    mode = "multi",
    variant = "default",
    multiModeText = "selected",
    size = "sm",
    sortOptions = true,
    isInValid = false,
    selectedOptionPrefix = "",
    showSelectedCount = true,
    popoverSide,
    collisionBoundary,
    maxPopoverHeight,
    showCheckbox = false,
    onOpenChange
}: SelectProps) {
    const ADD_KUMO_ENTITY = "<add_kumo_entity>";
    // Check if popoverClassName contains w-fit
    const isWidthFit = popoverClassName?.includes("w-fit");

    // Internal state for selected values (use external if provided)
    const [internalSelectedValues, setInternalSelectedValues] = useState<
        string[]
    >([]);
    const [open, setOpen] = useState(false);
    const [searchValue, setSearchValue] = useState("");

    // Ref for the scrollable parent element
    const parentRef = useRef<HTMLDivElement>(null);

    // Ref for trigger button to measure available space
    const triggerRef = useRef<HTMLButtonElement>(null);
    const [computedMaxHeight, setComputedMaxHeight] = useState<number>(
        maxPopoverHeight ?? 300
    );

    // Ref to track user scroll activity
    const userScrollingRef = useRef(false);
    const scrollTimeoutRef = useRef<NodeJS.Timeout | null>(null);
    const prevOpenRef = useRef(false);

    const defaultVariant =
        "bg-white text-title-primary border border-border-input shadow-xs hover:bg-kumo-grey-50 [&_svg]:fill-title-secondary";
    const secondaryVariant =
        "bg-transparent border border-border-secondary text-title-secondary shadow-xs hover:bg-kumo-grey-50 focus:bg-kumo-grey-50 [&_svg]:fill-title-secondary";
    const noOutlineVariant =
        "bg-transparent text-text-primary border-0 hover:bg-kumo-grey-50 [&_svg]:fill-text-secondary";

    // Use either controlled or uncontrolled values
    const selectedValues =
        externalSelectedValues !== undefined
            ? externalSelectedValues
            : internalSelectedValues;

    // Update internal state if external values change
    useEffect(() => {
        if (externalSelectedValues !== undefined) {
            setInternalSelectedValues(externalSelectedValues);
        }
    }, [externalSelectedValues]);

    // Sort options - "add_kumo_entity" at top, then selected options, then alphabetically.
    const sortedOptions = useMemo(() => {
        if (sortOptions === false) {
            return [...options];
        }
        // Create a copy of options to avoid mutating the original
        return [...options].sort((a, b) => {
            // Always prioritize "add_kumo_entity" option
            if (a.value === ADD_KUMO_ENTITY) return -1;
            if (b.value === ADD_KUMO_ENTITY) return 1;

            const aSelected = selectedValues.includes(a.value);
            const bSelected = selectedValues.includes(b.value);

            // If one is selected and the other isn't, selected comes first
            if (aSelected && !bSelected) return -1;
            if (!aSelected && bSelected) return 1;

            // If both are selected or both are unselected, sort by label
            return a.label.localeCompare(b.label);
        });
    }, [options, selectedValues, sortOptions]);

    // Filter options based on search value
    const filteredOptions = useMemo(() => {
        if (!searchValue) {
            return sortedOptions;
        }
        const lowerSearch = searchValue.toLowerCase();
        return sortedOptions.filter((option) => {
            const extended = `${option.value} ${option.label}`.toLowerCase();
            return extended.includes(lowerSearch);
        });
    }, [sortedOptions, searchValue]);

    // Set up virtualizer for efficient rendering of large lists with dynamic measurement
    const rowVirtualizer = useVirtualizer({
        count: filteredOptions.length,
        getScrollElement: () => parentRef.current,
        estimateSize: (index) => {
            // Estimate based on whether the option has a description
            const option = filteredOptions[index];
            if (!option?.labelDescription) {
                return 32; // Simple items: py-1.5 (12px) + text (~20px) = ~32px
            }
            // For items with descriptions, estimate based on description length
            // Assuming ~60 characters per line and ~18px per line for better accuracy
            const descLength = option.labelDescription.length;
            const estimatedLines = Math.ceil(descLength / 60);
            // Base: 32px (label) + 4px (description py-1) + (lines * 18px) + 2px extra gap
            return 38 + estimatedLines * 18;
        },
        overscan: 5, // Render 5 extra items above/below viewport for smooth scrolling
        measureElement:
            typeof window !== "undefined" &&
            navigator.userAgent.indexOf("Firefox") === -1
                ? (element) => element?.getBoundingClientRect().height
                : undefined // Dynamic measurement for accurate heights (disabled in Firefox for performance)
    });

    // Add scroll event listener to detect user scrolling
    useEffect(() => {
        const scrollElement = parentRef.current;
        if (!scrollElement || !open) return;

        const handleUserScroll = () => {
            userScrollingRef.current = true;
        };

        scrollElement.addEventListener("scroll", handleUserScroll, {
            passive: true
        });

        return () => {
            scrollElement.removeEventListener("scroll", handleUserScroll);
        };
    }, [open]);

    // Force virtualizer to recalculate when popover opens or closes
    useEffect(() => {
        // Only run scroll logic when popover transitions from closed to open
        const isOpening = open && !prevOpenRef.current;
        prevOpenRef.current = open;

        if (isOpening) {
            // Reset user scrolling flag when popover opens
            userScrollingRef.current = false;

            // Use requestAnimationFrame to ensure DOM is updated
            requestAnimationFrame(() => {
                requestAnimationFrame(() => {
                    if (parentRef.current) {
                        // Temporarily disable smooth scrolling for instant reset
                        parentRef.current.style.scrollBehavior = "auto";
                    }

                    // First measure to get accurate positions
                    rowVirtualizer.measure();

                    // Small delay to ensure measure completes
                    setTimeout(() => {
                        // Re-measure to be sure
                        rowVirtualizer.measure();
                        // Determine scroll target based on mode and sortOptions
                        let scrollToIndex = 0;
                        let shouldScrollToSelected = false;

                        // In single mode with sortOptions=false, scroll to selected option
                        if (
                            mode === "single" &&
                            sortOptions === false &&
                            selectedValues.length > 0
                        ) {
                            const selectedIndex = filteredOptions.findIndex(
                                (opt) => opt.value === selectedValues[0]
                            );
                            if (selectedIndex !== -1) {
                                scrollToIndex = selectedIndex;
                                shouldScrollToSelected = true;
                            }
                        }

                        // Scroll to target index (either 0 or selected option)
                        if (shouldScrollToSelected) {
                            rowVirtualizer.scrollToIndex(scrollToIndex, {
                                align: "start"
                            });
                        } else {
                            // Scroll to top
                            if (parentRef.current) {
                                parentRef.current.scrollTop = 0;
                            }
                            rowVirtualizer.scrollToOffset(0);
                        }
                    }, 0);

                    // Restore smooth scrolling after a brief delay
                    if (parentRef.current) {
                        setTimeout(() => {
                            if (parentRef.current) {
                                parentRef.current.style.scrollBehavior = "";
                            }
                        }, 50);
                    }
                });
            });
        } else if (!open) {
            // Reset scroll position when closing to ensure clean state for next open
            if (parentRef.current) {
                parentRef.current.style.scrollBehavior = "auto";
                parentRef.current.scrollTop = 0;
            }
        }
    }, [
        open,
        rowVirtualizer,
        mode,
        sortOptions,
        selectedValues,
        filteredOptions
    ]);

    // Toggle selection of a value - modified to handle single and multi mode
    const toggleValue = useCallback(
        (value: string) => {
            let newValues: string[];

            if (mode === "single") {
                // In single mode, replace the selection
                newValues = [value];

                // Close the dropdown when in single mode after selection
                setOpen(false);
            } else {
                // In multi mode, toggle the selection
                newValues = selectedValues.includes(value)
                    ? selectedValues.filter((v) => v !== value)
                    : [...selectedValues, value];

                // In multi mode with sortOptions, scroll to top when selection changes
                // (because selected items move to top due to sorting)
                if (sortOptions) {
                    // Clear any existing scroll timeout
                    if (scrollTimeoutRef.current) {
                        clearTimeout(scrollTimeoutRef.current);
                    }

                    // Reset user scrolling flag
                    userScrollingRef.current = false;

                    scrollTimeoutRef.current = setTimeout(() => {
                        // Only scroll to top if user hasn't scrolled
                        if (!userScrollingRef.current && parentRef.current) {
                            // Temporarily disable smooth scrolling for instant reset
                            parentRef.current.style.scrollBehavior = "auto";
                            parentRef.current.scrollTop = 0;
                            // Restore smooth scrolling
                            setTimeout(() => {
                                if (parentRef.current) {
                                    parentRef.current.style.scrollBehavior = "";
                                }
                            }, 50);
                            rowVirtualizer.scrollToOffset(0);
                        }
                    }, 0);
                }
            }
            if (value === ADD_KUMO_ENTITY) {
                setOpen(false);
            }

            // Update internal state
            setInternalSelectedValues(newValues);

            // Call external handler if provided
            onValuesChange?.(newValues);
        },
        [selectedValues, onValuesChange, mode, sortOptions, rowVirtualizer]
    );

    // Get the count of selected options for display
    const selectedCount = selectedValues.length;

    // Get the label of the selected option for single mode
    const selectedOptionLabel = useMemo(() => {
        if (mode === "single" && selectedCount === 1) {
            const selectedOption = options.find(
                (opt) => opt.value === selectedValues[0]
            );
            return selectedOption?.label;
        }
        return null;
    }, [mode, selectedCount, selectedValues, options]);

    // Calculate the width needed for the longest option when using w-fit
    const calculatedWidth = useMemo(() => {
        if (!isWidthFit) return undefined;

        // Find the longest label to determine minimum width needed
        const longestLabel = filteredOptions.reduce((longest, option) => {
            const labelLength =
                option.label.length + (option.labelDescription ? 30 : 0);
            return labelLength > longest ? labelLength : longest;
        }, 0);

        // Better approximation: 9px per character + padding for icon/checkmark (60px) + scrollbar (20px)
        // Remove max limit to allow full expansion
        const estimatedWidth = Math.max(250, longestLabel * 9 + 80);
        return `${estimatedWidth}px`;
    }, [isWidthFit, filteredOptions]);

    return (
        <Popover
            open={open}
            onOpenChange={(isOpen) => {
                setOpen(isOpen);
                onOpenChange?.(isOpen);
                if (!isOpen) {
                    setSearchValue("");
                } else if (triggerRef.current && collisionBoundary) {
                    // Measure available space above and below trigger
                    // within the collision boundary
                    const triggerRect =
                        triggerRef.current.getBoundingClientRect();
                    const boundaryEl = Array.isArray(collisionBoundary)
                        ? collisionBoundary[0]
                        : collisionBoundary;
                    if (boundaryEl) {
                        const boundaryRect = boundaryEl.getBoundingClientRect();
                        const spaceBelow =
                            boundaryRect.bottom - triggerRect.bottom - 8;
                        const spaceAbove =
                            triggerRect.top - boundaryRect.top - 8;
                        const available =
                            popoverSide === "top" || popoverSide === "left"
                                ? spaceAbove
                                : popoverSide
                                  ? spaceBelow
                                  : Math.max(spaceBelow, spaceAbove);
                        const cap = maxPopoverHeight ?? 300;
                        // Subtract ~40px for search input + padding
                        setComputedMaxHeight(
                            Math.min(cap, Math.max(100, available - 40))
                        );
                    }
                }
            }}
            modal={false}>
            <PopoverTrigger asChild>
                <button
                    ref={triggerRef}
                    type="button"
                    id={id}
                    data-testid={`testid-${id}`}
                    role="button"
                    disabled={disabled}
                    className={cn(
                        "w-full items-center justify-between",
                        "rounded-sm! focus-visible:border-ring aria-invalid:border-destructive aria-invalid:ring-destructive/20 dark:aria-invalid:ring-destructive/40 inline-flex cursor-pointer gap-2 whitespace-nowrap font-medium outline-none focus-visible:ring-[3px] disabled:pointer-events-none disabled:opacity-50",
                        "text-sm [&_svg]:size-5 [&_svg]:shrink-0",
                        "aria-invalid:border-destructive aria-invalid:ring-destructive/20 dark:aria-invalid:ring-destructive/40",
                        variant === "default"
                            ? defaultVariant
                            : variant === "secondary"
                              ? secondaryVariant
                              : noOutlineVariant,
                        size === "md"
                            ? "h-10 px-3 py-2"
                            : size === "sm"
                              ? "h-8 px-3 py-1"
                              : "text-sm! [&_svg]:size-4! h-7 px-3 py-1",
                        !selectedCount && "text-title-secondary",
                        popoverTriggerClassName
                    )}
                    onClick={(e) => e.stopPropagation()}
                    data-state="closed"
                    aria-invalid={isInValid}>
                    <div className="flex items-center gap-2 overflow-hidden text-ellipsis">
                        {selectedCount > 0 ? (
                            mode === "single" ? (
                                `${selectedOptionPrefix ? `${selectedOptionPrefix} ` : ""}${selectedOptionLabel}`
                            ) : showSelectedCount ? (
                                `${selectedCount} ${multiModeText}`
                            ) : (
                                <>{multiModeText}</>
                            )
                        ) : (
                            placeholder
                        )}
                    </div>
                    <ChevronDown />
                </button>
            </PopoverTrigger>
            <PopoverContent
                className={cn("z-1500 p-0", popoverClassName)}
                align="start"
                side={popoverSide}
                sideOffset={popoverSide ? 8 : 4}
                avoidCollisions={!popoverSide}
                collisionBoundary={collisionBoundary}
                forceMount>
                <Command
                    className={cn(
                        isWidthFit ? "" : "w-full",
                        "rounded-br-none!",
                        className
                    )}
                    style={
                        isWidthFit
                            ? {
                                  width: calculatedWidth
                              }
                            : undefined
                    }
                    shouldFilter={false}>
                    {search === true && (
                        <CommandInput
                            className={
                                "focus-within:!border-border-secondary hover:!border-border-secondary hover:!bg-kumo-grey-50"
                            }
                            placeholder={searchPlaceholder}
                            value={searchValue}
                            onValueChange={setSearchValue}
                        />
                    )}
                    <CommandList className="w-full max-w-none overflow-visible rounded-b-sm">
                        {filteredOptions.length === 0 ? (
                            <CommandEmpty>{emptyText}</CommandEmpty>
                        ) : (
                            <CommandGroup
                                data-testid={"listbox"}
                                className="p-0! w-full">
                                <div
                                    ref={parentRef}
                                    className={
                                        "[&::-webkit-scrollbar-thumb]:bg-(--scrollbar) [&::-webkit-scrollbar-track]:bg-secondary scroll-py-1 overflow-auto scroll-smooth p-1 [-ms-overflow-style:auto] [&::-webkit-scrollbar-thumb]:rounded-md [&::-webkit-scrollbar:horizontal]:h-3 [&::-webkit-scrollbar:vertical]:w-3 [&::-webkit-scrollbar]:opacity-100 [&::-webkit-scrollbar]:transition-opacity [&::-webkit-scrollbar]:duration-1000 [&::-webkit-scrollbar]:ease-in-out"
                                    }
                                    style={{
                                        maxHeight: `${computedMaxHeight}px`,
                                        height: "fit-content"
                                    }}>
                                    <div
                                        style={{
                                            height: `${rowVirtualizer.getTotalSize()}px`,
                                            width: "100%",
                                            position: "relative"
                                        }}>
                                        {rowVirtualizer
                                            .getVirtualItems()
                                            .map((virtualRow) => {
                                                const option =
                                                    filteredOptions[
                                                        virtualRow.index
                                                    ];
                                                const isSelected =
                                                    selectedValues.includes(
                                                        option.value
                                                    );
                                                return (
                                                    <CommandItem
                                                        key={option.value}
                                                        value={option.value}
                                                        disabled={
                                                            option.disabled
                                                        }
                                                        data-testid={`${id}-option-${virtualRow.index}`}
                                                        data-index={
                                                            virtualRow.index
                                                        }
                                                        ref={
                                                            rowVirtualizer.measureElement
                                                        }
                                                        onSelect={() =>
                                                            toggleValue(
                                                                option.value
                                                            )
                                                        }
                                                        style={{
                                                            position:
                                                                "absolute",
                                                            top: 0,
                                                            left: 0,
                                                            width: "100%",
                                                            transform: `translateY(${virtualRow.start}px)`
                                                        }}
                                                        className={cn(
                                                            "w-full cursor-pointer justify-between font-medium",
                                                            "text-title-primary",
                                                            option.disabled &&
                                                                "cursor-not-allowed opacity-50",
                                                            "hover:!bg-kumo-grey-50 focus:!bg-kumo-grey-50"
                                                        )}>
                                                        <div className="flex w-full min-w-0 flex-col">
                                                            <div className="flex flex-row items-center justify-between gap-2">
                                                                {showCheckbox &&
                                                                    option.value !==
                                                                        ADD_KUMO_ENTITY &&
                                                                    option.showCheckbox !==
                                                                        false && (
                                                                        <Checkbox
                                                                            checked={
                                                                                isSelected
                                                                            }
                                                                            onCheckedChange={() => {}}
                                                                            className="data-[state=checked]:border-kumo-grey-950 data-[state=checked]:bg-kumo-grey-950 pointer-events-none shrink-0 [&[data-state=checked]_svg]:stroke-white [&[data-state=checked]_svg]:stroke-[3px]"
                                                                        />
                                                                    )}
                                                                <span className="min-w-0 flex-1 whitespace-normal break-words text-left">
                                                                    {
                                                                        option.label
                                                                    }
                                                                </span>
                                                                <span className="shrink-0">
                                                                    {option.value ===
                                                                    ADD_KUMO_ENTITY ? (
                                                                        <AddCircle className="ml-2 size-4 shrink-0" />
                                                                    ) : !showCheckbox ? (
                                                                        <Check
                                                                            className={cn(
                                                                                "ml-2 h-4 w-4 shrink-0",
                                                                                isSelected
                                                                                    ? "text-green-500"
                                                                                    : "text-transparent opacity-0 group-hover:bg-slate-100"
                                                                            )}
                                                                        />
                                                                    ) : null}
                                                                </span>
                                                            </div>
                                                            {option?.labelDescription && (
                                                                <Description
                                                                    variant="secondary"
                                                                    size="sm"
                                                                    className="mb-0! max-w-120 wrap-break-word whitespace-normal text-wrap">
                                                                    {
                                                                        option?.labelDescription
                                                                    }
                                                                </Description>
                                                            )}
                                                        </div>
                                                    </CommandItem>
                                                );
                                            })}
                                    </div>
                                </div>
                            </CommandGroup>
                        )}
                    </CommandList>
                </Command>
            </PopoverContent>
        </Popover>
    );
}
