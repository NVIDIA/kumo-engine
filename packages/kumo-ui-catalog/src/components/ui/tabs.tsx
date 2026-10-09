// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2023 shadcn
// SPDX-License-Identifier: Apache-2.0 AND MIT

import * as TabsPrimitive from "@radix-ui/react-tabs";
import * as React from "react";

import { cn } from "@/lib/utils";

function Tabs({
    className,
    ...props
}: React.ComponentProps<typeof TabsPrimitive.Root>) {
    return (
        <TabsPrimitive.Root
            data-slot="tabs"
            className={cn("", className)}
            {...props}
        />
    );
}

function TabsList({
    className,
    ...props
}: React.ComponentProps<typeof TabsPrimitive.List>) {
    return (
        <TabsPrimitive.List
            data-slot="tabs-list"
            // KUI primary tabs: faint 2px base underline under the whole tab row
            className={cn(
                "flex gap-6 shadow-[inset_0_-2px_0_0_var(--kui-border-base)]",
                className
            )}
            {...props}
        />
    );
}

function TabsTrigger({
    className,
    ...props
}: React.ComponentProps<typeof TabsPrimitive.Trigger>) {
    return (
        <TabsPrimitive.Trigger
            data-slot="tabs-trigger"
            className={cn(
                // Remove default button look
                "m-0 cursor-pointer appearance-none border-none bg-transparent p-0 outline-none",

                // KUI tab geometry: 40px tall (spacing*10), text vertically centered,
                // 4px padding-block (spacing*1) — this is what creates the space
                // between the label and the underline.
                "relative flex h-10 items-center justify-center py-1",
                // Resting (unselected): app text token (consistent with the rest of the UI).
                // NOTE: KUI uses neutral grays (#636363/#000000); kept on app tokens for
                // now — full KUI typography migration is a separate PR.
                "text-title-secondary text-base font-medium transition-colors",

                // Hover (unselected) → text darkens (app token)
                "hover:text-title-primary",

                // Disabled
                "disabled:pointer-events-none disabled:opacity-50",

                // Selected → darker text (app token)
                "data-[state=active]:text-title-primary",

                // Underline indicator sits ON the bottom edge (over the grey base line):
                // transparent → 2px hover (KUI color/interaction/hover/border = 40% black,
                // light mirror of the dark #ffffff66) → 4px brand-green when selected (KUI).
                "after:absolute after:bottom-0 after:left-0 after:right-0",
                "after:h-[2px] after:bg-transparent after:content-['']",
                "after:origin-left after:scale-x-0 after:transition-transform",
                "hover:after:bg-kui-hover-border hover:after:scale-x-100",
                "data-[state=active]:after:bg-brand-green data-[state=active]:after:h-[4px] data-[state=active]:after:scale-x-100",

                className
            )}
            {...props}
        />
    );
}

function TabsContent({
    className,
    ...props
}: React.ComponentProps<typeof TabsPrimitive.Content>) {
    return (
        <TabsPrimitive.Content
            data-slot="tabs-content"
            className={cn("flex-1 py-[16px] outline-none", className)}
            {...props}
        />
    );
}

export { Tabs, TabsContent, TabsList, TabsTrigger };
