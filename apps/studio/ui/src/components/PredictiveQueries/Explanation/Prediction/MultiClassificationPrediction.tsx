// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React from "react";

import {
    DataTable,
    GridCol,
    GridRow,
    Skeleton
} from "@kumo-ai/kumo-ui-catalog";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";
import { MultiClassificationPredictionColumns } from "components/@ui/TableColumns/MultiClassificationPrediction";

import { ClassificationEntityOutput } from "types/explorer";

import { getKeys } from "utils/Helpers/objectHelpers";

const MultiClassificationPrediction: React.FC<{
    entityOutput: ClassificationEntityOutput;
    isDataLoading: boolean;
    showGroundTruth?: boolean;
}> = ({ entityOutput, isDataLoading, showGroundTruth }) => {
    // Prepare data for the table
    const prepareTableData = (data: { [key: string]: number }) => {
        return getKeys(data || {}).map((label) => ({
            label: label,
            value: data[label].toFixed(2)
        }));
    };

    // Define columns for the tables
    const columns = MultiClassificationPredictionColumns();

    if (!entityOutput) {
        return null;
    }

    const predictionTableData = prepareTableData(entityOutput?.prediction);
    const targetTableData = prepareTableData(entityOutput?.target || {});

    return (
        <GridRow xs={12} sm={12} md={12}>
            <GridCol xs={12} sm={12} md={6}>
                <h3>Predictions</h3>
                {!entityOutput.prediction ? (
                    <Skeleton className="w-12 h-6" />
                ) : (
                    <DataTable
                        id="prediction-table"
                        columns={columns}
                        data={predictionTableData}
                        enableGlobalFilter={false}
                        defaultPageSize={10}
                        disableRowSelectionHover={true}
                        isLoading={isDataLoading}
                        pageCount={
                            predictionTableData?.length > 0
                                ? Math.ceil(predictionTableData?.length / 10)
                                : 1
                        }
                    />
                )}
            </GridCol>
            {showGroundTruth && (
                <GridCol xs={12} sm={12} md={6}>
                    <h3>Ground Truth</h3>
                    {!entityOutput.target ? (
                        <Skeleton className="w-12 h-6" />
                    ) : (
                        <DataTable
                            id="ground-truth-table"
                            columns={columns}
                            disableRowSelectionHover={true}
                            data={targetTableData}
                            enableGlobalFilter={false}
                            defaultPageSize={10}
                            isLoading={isDataLoading}
                            pageCount={
                                targetTableData?.length > 0
                                    ? Math.ceil(targetTableData?.length / 10)
                                    : 1
                            }
                        />
                    )}
                </GridCol>
            )}
        </GridRow>
    );
};

export default WithErrorBoundary(MultiClassificationPrediction);
