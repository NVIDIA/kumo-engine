// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React from "react";
import Spacer from "react-spacer";

import {
    DatePicker,
    DocumentationLink,
    GridCol,
    GridRow,
    Info,
    Label,
    Tooltip,
    TooltipContent,
    TooltipTrigger
} from "@kumo-ai/kumo-ui-catalog";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";

export interface PredictionAnchorTimeProps {
    onChange: (date: Date) => void;
}

const PredictionAnchorTime: React.FC<PredictionAnchorTimeProps> = ({
    onChange
}) => {
    return (
        <>
            <GridRow xs={12} sm={12} md={12}>
                <GridCol xs={12} sm={12}>
                    <Label htmlFor="Prediction Anchor Time">
                        Prediction Anchor Time
                        <Tooltip id={"prediction-anchor-time"}>
                            <TooltipTrigger>
                                <span>
                                    <Info className="size-3.5 mx-0" />
                                </span>
                            </TooltipTrigger>
                            <TooltipContent
                                arrow={false}
                                sideOffset={5}
                                className="max-w-112.5">
                                {
                                    "Anchor time represents a specific point in time from which you want to predict the future. To learn more, visit the "
                                }
                                <span className="whitespace-nowrap">
                                    <DocumentationLink
                                        id="prediction-anchor-link"
                                        url={
                                            "https://kumo.ai/docs/troubleshooting/what-is-anchor-time-and-why-is-it-important/"
                                        }
                                        linkText="docs"
                                        className="text-blue-300! hover:text-blue-400!"
                                        showExternalIcon={true}
                                    />
                                    {"."}
                                </span>
                            </TooltipContent>
                        </Tooltip>
                    </Label>
                    <DatePicker onDateChange={onChange} />
                </GridCol>
            </GridRow>
            <Spacer height={16} />
        </>
    );
};

export default WithErrorBoundary(PredictionAnchorTime);
