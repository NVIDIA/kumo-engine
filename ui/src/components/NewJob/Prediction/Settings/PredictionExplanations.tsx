// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React from "react";
import Spacer from "react-spacer";

import { GridCol, GridRow, Label, Switch } from "@kumo-ai/kumo-ui-catalog";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";

export interface PredictionExplanationsProps {
    isToggled: boolean;
    disabled?: boolean;
    onChange: () => void;
}

const PredictionExplanations: React.FC<PredictionExplanationsProps> = ({
    isToggled,
    disabled,
    onChange
}) => {
    return (
        <>
            <GridRow xs={12} sm={12} md={12} className="mt-1">
                <GridCol xs={12} sm={12} md={9}>
                    <Label htmlFor="Enable Explanations">
                        Enable Explanations
                    </Label>
                </GridCol>
                <GridCol
                    xs={12}
                    sm={12}
                    md={3}
                    className="flex justify-end mt-1">
                    <Switch
                        id="explanations-toggle"
                        checked={isToggled}
                        onCheckedChange={() => {
                            onChange();
                        }}
                        disabled={disabled}
                    />
                </GridCol>
            </GridRow>

            <Spacer height={8} />
        </>
    );
};

export default WithErrorBoundary(PredictionExplanations);
