// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React from "react";

import { GridCol, GridRow, TextInput } from "@kumo-ai/kumo-ui-catalog";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";

export interface PredictionTextInputProps {
    id: string;
    title: string;
    placeholder: string;
    value: string;
    isInvalid: boolean;
    errorMsg?: string;
    onChange: (threshold: string) => void;
}

const PredictionTextInput: React.FC<PredictionTextInputProps> = ({
    id,
    title,
    placeholder,
    value,
    isInvalid,
    errorMsg,
    onChange
}) => {
    return (
        <>
            <GridRow xs={12} sm={12} md={12}>
                <GridCol xs={12} sm={12}>
                    <TextInput
                        id={id}
                        label={title}
                        placeholder={placeholder}
                        value={value}
                        isInvalid={isInvalid}
                        errorMsg={errorMsg}
                        onChange={(e) => onChange(e.target.value)}
                    />
                </GridCol>
            </GridRow>
        </>
    );
};

export default WithErrorBoundary(PredictionTextInput);
