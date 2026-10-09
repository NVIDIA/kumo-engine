// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import * as React from "react";
import { cn } from "@/lib/utils";

interface FormProps extends React.ComponentProps<"form"> {
    onSubmit?: (e: React.FormEvent<HTMLFormElement>) => void;
}

function Form({ className, ...props }: FormProps) {
    return (
        <form
            data-slot="form"
            className={cn("w-full space-y-4", className)}
            {...props}
        />
    );
}

function FormField({ className, ...props }: React.ComponentProps<"div">) {
    return (
        <div
            data-slot="form-field"
            className={cn("flex flex-col gap-1.5", className)}
            {...props}
        />
    );
}

function FormDescription({ className, ...props }: React.ComponentProps<"p">) {
    return (
        <p
            data-slot="form-description"
            className={cn("text-muted-foreground text-xs", className)}
            {...props}
        />
    );
}

function FormError({ className, ...props }: React.ComponentProps<"p">) {
    return (
        <p
            data-slot="form-error"
            className={cn("text-destructive text-xs font-medium", className)}
            {...props}
        />
    );
}

function FormWrapper({
    className,
    ...props
}: React.ComponentProps<"fieldset">) {
    return (
        <fieldset
            data-slot="form-wrapper"
            className={cn("space-y-4", className)}
            {...props}
        />
    );
}

type FormControlProps = {
    as?: "input" | "textarea";
    children?: React.ReactNode;
} & (
    | React.InputHTMLAttributes<HTMLInputElement>
    | React.TextareaHTMLAttributes<HTMLTextAreaElement>
);

function FormControl({
    className,
    as = "input",
    children,
    ...props
}: FormControlProps) {
    const baseStyles = cn(
        "w-full rounded-md border border-gray-300",
        "px-3 py-2",
        "text-sm",
        "transition-colors",
        "focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent",
        "disabled:bg-gray-100 disabled:cursor-not-allowed disabled:opacity-50",
        "placeholder:text-gray-400",
        children && "pl-8",
        className
    );

    if (!children) {
        // No icon - render input/textarea only
        if (as === "textarea") {
            return (
                <textarea
                    data-slot="form-control"
                    className={baseStyles}
                    {...(props as React.TextareaHTMLAttributes<HTMLTextAreaElement>)}
                />
            );
        }

        return (
            <input
                data-slot="form-control"
                className={baseStyles}
                {...(props as React.InputHTMLAttributes<HTMLInputElement>)}
            />
        );
    }

    // With icon - render wrapper with input and icon
    return (
        <div className="relative w-full">
            <div className="pointer-events-none absolute left-2.5 top-1/2 flex -translate-y-1/2 items-center">
                {children}
            </div>
            {as === "textarea" ? (
                <textarea
                    data-slot="form-control"
                    className={baseStyles}
                    {...(props as React.TextareaHTMLAttributes<HTMLTextAreaElement>)}
                />
            ) : (
                <input
                    data-slot="form-control"
                    className={baseStyles}
                    {...(props as React.InputHTMLAttributes<HTMLInputElement>)}
                />
            )}
        </div>
    );
}

function FormSection({ className, ...props }: React.ComponentProps<"div">) {
    return (
        <div
            data-slot="form-section"
            className={cn("space-y-3", className)}
            {...props}
        />
    );
}

function FormActions({ className, ...props }: React.ComponentProps<"div">) {
    return (
        <div
            data-slot="form-actions"
            className={cn(
                "flex items-center justify-end gap-2 border-t pt-4",
                className
            )}
            {...props}
        />
    );
}

export {
    Form,
    FormField,
    FormDescription,
    FormError,
    FormControl,
    FormWrapper,
    FormSection,
    FormActions
};
