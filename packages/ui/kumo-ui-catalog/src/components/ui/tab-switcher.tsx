// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import * as React from "react";

import { cn } from "@/lib/utils";

interface TabSwitcherProps {
    tabs: Array<{ label: string; value: string }>;
    value?: string;
    defaultValue?: string;
    onValueChange?: (value: string) => void;
    className?: string;
}

function TabSwitcher({
    tabs,
    value: controlledValue,
    defaultValue,
    onValueChange,
    className
}: TabSwitcherProps) {
    const [internalValue, setInternalValue] = React.useState(
        defaultValue ?? tabs[0]?.value
    );

    const isControlled = controlledValue !== undefined;
    const activeValue = isControlled ? controlledValue : internalValue;

    const handleTabClick = (tabValue: string) => {
        if (!isControlled) {
            setInternalValue(tabValue);
        }
        onValueChange?.(tabValue);
    };

    return (
        <div data-slot="tab-switcher" className={cn("w-full", className)}>
            <div
                data-slot="tab-switcher-list"
                role="tablist"
                className="bg-surface-secondary inline-flex items-center gap-2 rounded-xl p-1">
                {tabs.map((tab) => (
                    <button
                        key={tab.value}
                        data-slot="tab-switcher-trigger"
                        role="tab"
                        type="button"
                        aria-selected={activeValue === tab.value}
                        data-state={
                            activeValue === tab.value ? "active" : "inactive"
                        }
                        onClick={() => handleTabClick(tab.value)}
                        className={cn(
                            "cursor-pointer rounded-xl px-4 py-2 text-base font-medium leading-none transition-colors duration-75",
                            "focus-visible:outline-none focus-visible:ring-[3px]",
                            activeValue === tab.value
                                ? "text-title-primary bg-white"
                                : "text-title-secondary"
                        )}>
                        {tab.label}
                    </button>
                ))}
            </div>
        </div>
    );
}

export { TabSwitcher };
