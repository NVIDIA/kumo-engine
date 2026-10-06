// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React from "react";

import { GridRow, Skeleton, Symbol } from "@kumo-ai/kumo-ui-catalog";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";

import { RegressionEntityOutput } from "types/explorer";

const BinaryClassificationPrediction: React.FC<{
    entityOutput: RegressionEntityOutput;
    isDataLoading: boolean;
    query: string;
    anchorTime: string;
    showGroundTruth?: boolean;
}> = ({ entityOutput, isDataLoading, query, anchorTime, showGroundTruth }) => {
    if (!entityOutput) {
        return null;
    }
    const regex = /PREDICT\s+(.*?)(?=\s+FOR EACH)/;
    const match = query?.match(regex);

    return (
        <GridRow
            xs={12}
            sm={12}
            md="auto"
            className="flex flex-row items-center gap-16">
            <div className="flex flex-col m-0">
                <label className="mb-1 text-title-secondary text-sm font-medium">
                    Output{" "}
                </label>
                {isDataLoading ? (
                    <Skeleton className="w-40 h-6" />
                ) : !match || match?.length < 2 ? (
                    <Symbol type="emdash" />
                ) : (
                    <label className="text-title-primary text-base font-medium">
                        {match[1]}
                    </label>
                )}
            </div>
            <div className="flex flex-col m-0">
                <label className="mb-1 text-title-secondary text-sm font-medium">
                    {" "}
                    Prediction{" "}
                </label>
                {isDataLoading ? (
                    <Skeleton className="w-20 h-6" />
                ) : (
                    <label className="text-title-primary text-base font-medium">
                        {entityOutput?.prediction?.toFixed(2)}
                    </label>
                )}
            </div>
            {showGroundTruth && (
                <div className="flex flex-col m-0">
                    <label className="mb-1 text-title-secondary text-sm font-medium">
                        {" "}
                        Ground Truth{" "}
                    </label>
                    {isDataLoading ? (
                        <Skeleton className="w-20 h-6" />
                    ) : (
                        <label className="text-title-primary text-base font-medium">
                            {entityOutput?.target?.toFixed(2)}
                        </label>
                    )}
                </div>
            )}

            <div className="flex flex-col m-0">
                <label className="mb-1 text-title-secondary text-sm font-medium">
                    Anchor Time
                </label>
                <label className="text-title-primary text-base font-medium">
                    {anchorTime}
                </label>
            </div>
        </GridRow>
    );
};

export default WithErrorBoundary(BinaryClassificationPrediction);
