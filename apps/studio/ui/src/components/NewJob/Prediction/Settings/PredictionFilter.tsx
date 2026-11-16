// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React from "react";

import { GridCol, GridRow, Label, TextArea } from "@kumo-ai/kumo-ui-catalog";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";

import { TaskType } from "types/predictiveQueries";

export interface PredictionFilterProps {
    title: string;
    taskType?: TaskType;
    value: string | null;
    isPQLANG_V2?: boolean;
    isInvalid?: boolean;
    onChange: (value: string) => void;
}

const PredictionFilter: React.FC<PredictionFilterProps> = ({
    title,
    value,
    onChange
}) => {
    return (
        <>
            <GridRow xs={12} sm={12} md={12}>
                <GridCol xs={12} sm={12}>
                    <Label htmlFor={title}>{title}</Label>
                    <TextArea
                        data-testid="prediction-filter"
                        value={value as string}
                        id={"prediction-filter"}
                        onChange={(e) => onChange(e?.target?.value)}
                        rows={2}
                        cols={0}
                    />
                </GridCol>
            </GridRow>
        </>
    );
};

export default WithErrorBoundary(PredictionFilter);
