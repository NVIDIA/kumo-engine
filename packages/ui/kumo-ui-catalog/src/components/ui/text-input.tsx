// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React, { ChangeEvent, FocusEvent } from "react";

import AnimatedDone from "@/animations/animated-done";
import Spinner from "@/animations/spinner";

import { Description } from "./description";
import { Input } from "./input";
import { Label } from "./label";
import { Tooltip, TooltipContent, TooltipTrigger } from "./tooltip";

export interface TextInputProps {
    id: string;
    name?: string;
    type?: string;
    label?: string | React.ReactNode;
    labelToolTip?: string;
    labelClassname?: string;
    onChange: (event: ChangeEvent<HTMLInputElement>) => void;
    value: string;
    isInvalid?: boolean;
    errorMsg?: string;
    placeholder?: string;
    onSubmit?: () => void;
    readOnly?: boolean;
    onBlur?: (event: FocusEvent<HTMLInputElement>) => void;
    inputToolTip?: string;
    labelVariant?: "default" | "secondary";
    state?: InputState;
    showState?: boolean;
    className?: string;
}

export enum InputState {
    None = "none",
    Validating = "validating",
    Valid = "valid",
    InValid = "invalid",
    Warning = "warning",
    ValidationError = "validationerror"
}

export const TextInput: React.FC<TextInputProps> = ({
    id,
    name,
    type = "text",
    label,
    labelToolTip,
    labelClassname,
    onChange,
    value,
    isInvalid = false,
    errorMsg,
    placeholder = "",
    onSubmit = () => {},
    readOnly = false,
    onBlur,
    labelVariant = "default",
    state,
    showState = false,
    className = ""
}) => {
    return (
        <>
            {label &&
                (typeof label === "string" ? (
                    labelToolTip ? (
                        <div style={{ width: "fit-content" }}>
                            <Tooltip>
                                <TooltipTrigger>
                                    <span>
                                        <Label
                                            htmlFor={id}
                                            className={
                                                (labelVariant === "default"
                                                    ? "text-text-primary"
                                                    : "text-text-secondary") +
                                                (labelClassname
                                                    ? ` ${labelClassname}`
                                                    : "")
                                            }>
                                            {label}
                                        </Label>
                                    </span>
                                </TooltipTrigger>
                                <TooltipContent
                                    arrow={false}
                                    side={"right"}
                                    sideOffset={5}
                                    className="max-w-100 justify-around">
                                    {labelToolTip}
                                </TooltipContent>
                            </Tooltip>
                        </div>
                    ) : (
                        <Label
                            htmlFor={id}
                            className={
                                (labelVariant === "default"
                                    ? "text-text-primary"
                                    : "text-text-secondary") +
                                (labelClassname ? ` ${labelClassname}` : "")
                            }>
                            {label}
                        </Label>
                    )
                ) : (
                    label
                ))}
            <div className="flex flex-row">
                <Input
                    id={id}
                    name={name}
                    type={type}
                    placeholder={placeholder}
                    value={value}
                    aria-invalid={isInvalid}
                    aria-label={!label ? placeholder || id : undefined}
                    readOnly={readOnly}
                    disabled={readOnly}
                    onChange={onChange}
                    onBlur={onBlur}
                    onKeyDown={(e) => {
                        if (readOnly) {
                            e?.preventDefault();
                            return;
                        }
                        if (e?.key === "Enter" && onSubmit) {
                            e?.preventDefault();
                            onSubmit();
                        }
                    }}
                    className={className}
                />
                {showState && (
                    <div className="mx-2 flex size-8 items-center justify-center py-2">
                        {state === InputState.Validating && (
                            <Spinner size={28} />
                        )}
                        {state === InputState.Valid && (
                            <AnimatedDone size={28} />
                        )}
                    </div>
                )}
            </div>
            {isInvalid && errorMsg && (
                <Description
                    id={`error${id}`}
                    size="sm"
                    variant="destructive"
                    className="!mb-0 pt-2">
                    {errorMsg}
                </Description>
            )}
        </>
    );
};
