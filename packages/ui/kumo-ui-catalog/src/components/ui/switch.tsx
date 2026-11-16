// SPDX-FileCopyrightText: Copyright (c) 2023 shadcn
// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0 AND MIT

import * as React from "react";
import * as SwitchPrimitive from "@radix-ui/react-switch";

import { cn } from "@/lib/utils";

interface SwitchProps
    extends React.ComponentProps<typeof SwitchPrimitive.Root> {
    // Kept for backwards compatibility with existing call sites. Both values now
    // render the single KUI `.nv-switch` style — there is no separate look.
    variant?: "default" | "primary";
}

// All colors, sizes, and timing come from the switch-scoped `--switch-*` tokens
// (see kumo-ui-catalog.css) so this matches KUI `.nv-switch` exactly and stays
// tokenized. Written as literal Tailwind arbitrary values so the JIT generates them.

function Switch({
    className,
    // Accepted for backwards compatibility but no longer changes the look;
    // destructured here only to keep it off the underlying DOM element.
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    variant,
    ...props
}: SwitchProps) {
    return (
        <SwitchPrimitive.Root
            data-slot="switch"
            className={cn(
                // Track: KUI medium switch pill. Ring toggles instantly (matches KUI).
                "bg-switch-off-background peer relative isolate inline-flex h-[var(--switch-track-height)] w-[var(--switch-track-width)] shrink-0 cursor-pointer items-center overflow-hidden rounded-full border-0 p-0 outline-none disabled:cursor-not-allowed",
                // Focus ring (KUI: 2px outline, 2px offset).
                "focus-visible:outline-2 focus-visible:outline-offset-2",
                // Off state: 2px inset ring; removed when checked. Disabled uses a lighter ring.
                "shadow-[inset_0_0_0_2px_var(--switch-track-ring)] data-[state=checked]:shadow-none",
                "disabled:shadow-[inset_0_0_0_2px_var(--switch-track-ring-disabled)]",
                // On state: brand-green fill wipes in from the left (KUI ::after), behind the thumb.
                "after:bg-switch-on-background after:absolute after:inset-0 after:-z-10 after:rounded-full after:transition-transform after:duration-[var(--switch-duration-fill)] after:ease-[var(--switch-ease)] after:[transform:translateX(-100%)] data-[state=checked]:after:[transform:translateX(0)]",
                className
            )}
            {...props}>
            <SwitchPrimitive.Thumb
                data-slot="switch-thumb"
                className={cn(
                    // Thumb: slides on toggle (via `translate`) and recolors dark -> black.
                    // Transition `translate` (not `transform`) — Tailwind v4 sets the CSS translate property.
                    "pointer-events-none relative z-10 ml-[var(--switch-thumb-margin)] size-[var(--switch-thumb-size)] rounded-full shadow-sm transition-[translate,background-color] duration-[var(--switch-duration-thumb)] ease-[var(--switch-ease)]",
                    "bg-switch-off-thumb data-[state=checked]:bg-switch-on-thumb",
                    "data-[state=checked]:translate-x-[var(--switch-thumb-travel)]",
                    "data-[disabled]:bg-[var(--switch-thumb-disabled)]"
                )}
            />
        </SwitchPrimitive.Root>
    );
}

export { Switch };
