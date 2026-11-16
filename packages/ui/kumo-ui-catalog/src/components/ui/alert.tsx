// SPDX-FileCopyrightText: Copyright (c) 2023 shadcn
// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0 AND MIT

import { cva, type VariantProps } from "class-variance-authority";
import * as React from "react";

import { BrightnessAlert, InfoFilled } from "@/icons";
import { Close } from "@/icons/MaterialIcons/Close";
import { WarningIcon } from "@/icons/MaterialIcons/WarningIcon";
import { cn, createScrollbarInjector } from "@/lib/utils";
import { useState } from "react";

// Inject scrollbar styles once globally
const injectScrollbarStyles = createScrollbarInjector("alert-scrollable");

const alertVariants = cva(
    //"var(--font-inter)  rounded-md border px-4 py-4  items-start grid grid-cols-32 grid-rows-1 gap-1",
    "alert-scrollable var(--font-inter) justify-items-start rounded-md border py-4 pr-3 items-start grid grid-rows-1 overflow-y-auto max-h-[15rem]",
    {
        variants: {
            variant: {
                default:
                    "pl-6 grid-cols-1 bg-slate-100 text-text-secondary border-transparent",
                warn: "p-4 has-[>svg]:grid-cols-[24px_auto_1fr] grid-cols-[24px_1fr] gap-2 bg-amber-50 border-amber-500 text-title-primary" /* "bg-card text-card-foreground" */,
                destructive:
                    "p-6 has-[>svg]:grid-cols-[24px_auto_1fr] grid-cols-[24px_1fr] gap-2 bg-red-50 border-red-500 text-title-primary",
                info: "p-4 grid-cols-[16px_1fr] gap-x-4 gap-y-2 bg-kumo-grey-50 border-kumo-grey-500 text-title-primary rounded-lg"
            }
        },
        defaultVariants: {
            variant: "default"
        }
    }
);

function Alert({
    className,
    variant,
    enableClose = true,
    onClose = undefined,
    ...props
}: React.ComponentProps<"div"> &
    VariantProps<typeof alertVariants> & {
        enableClose?: boolean;
        onClose?: () => void;
    }) {
    const [closed, setClosed] = useState<boolean>(false);
    const childClass =
        variant === "destructive" || variant === "warn" || variant === "info"
            ? "col-start-2 row-1"
            : "row-1";

    // Inject scrollbar styles once on first render
    injectScrollbarStyles();

    if (closed) return <></>;
    return (
        <div
            data-slot="alert"
            role="alert"
            tabIndex={0}
            className={cn(alertVariants({ variant }), className)}
            {...props}>
            {variant === "destructive" && (
                <div className="row-1 col-start-1 flex items-start [&_svg]:fill-red-500">
                    <BrightnessAlert className="size-5.5" />
                </div>
            )}
            {variant === "warn" && (
                <div className="row-1 col-start-1 mt-0.5 flex items-start [&_svg]:fill-amber-500">
                    <WarningIcon fill="true" className="size-5.5" />
                </div>
            )}
            {variant === "info" && (
                <div className="row-1 [&_svg]:fill-kumo-grey-500 col-start-1 mt-0.5 flex items-start">
                    <InfoFilled className="size-4" />
                </div>
            )}
            {enableClose && (
                <div className="row-1 col-start-[-1]">
                    <button
                        type="button"
                        aria-label="Close alert"
                        className="flex cursor-pointer appearance-none border-0 bg-transparent p-0"
                        data-testid="alert-close"
                        onClick={() => {
                            setClosed(true);
                            if (onClose) onClose();
                        }}>
                        <Close className="fill-text-title-primary size-4" />
                    </button>
                </div>
            )}
            <div className={childClass}>{props.children}</div>
        </div>
    );
}

function AlertTitle({ className, ...props }: React.ComponentProps<"div">) {
    return (
        <>
            <div
                data-slot="alert-title"
                className={cn("pb-1 text-base font-semibold", className)}
                {...props}></div>
        </>
    );
}

function AlertDescription({
    className,
    ...props
}: React.ComponentProps<"div">) {
    return (
        <div
            data-slot="alert-description"
            className={cn(
                "w-full break-words pt-1 text-sm font-normal first:pt-0",
                className
            )}
            {...props}
        />
    );
}

export { Alert, AlertDescription, AlertTitle };
