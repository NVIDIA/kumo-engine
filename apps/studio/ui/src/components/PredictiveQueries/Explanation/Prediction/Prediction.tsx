// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React from "react";

import { Alert, AlertDescription, Skeleton } from "@kumo-ai/kumo-ui-catalog";
import { EntityOutput, HistoricalItems } from "gen/v0/types";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";

import {
    ClassificationEntityOutput,
    LinkPredictionEntityOutput,
    RegressionEntityOutput
} from "types/explorer";
import { TaskType } from "types/predictiveQueries";

import BinaryClassificationPrediction from "./BinaryClassificationPrediction";
import LinkPrediction from "./LinkPrediction";
import MultiClassificationPrediction from "./MultiClassificationPrediction";
import RegressionPrediction from "./RegressionPrediction";

const NoDataAlert = () => (
    <Alert variant="default" enableClose={false}>
        <AlertDescription>
            No prediction data or explainability scores are available for this
            entity at the moment. If this issue persists across all entities,
            please consider retraining the model or contact your support team at
            Kumo for further assistance.{" "}
        </AlertDescription>
    </Alert>
);

export interface PredictionDisplayProps {
    entityOutput: EntityOutput | undefined;
    taskType: TaskType;
    historicalItems?: HistoricalItems;
    isDataLoading: boolean;
    query: string;
    anchorTime: string;
    showGroundTruth?: boolean;
    selectedColumns?: Array<string>;
    setSelectedColumns?: (columns: Array<string>) => void;
}

/**
 * Renders the PredictionDisplay component, which displays the prediction details
 * based on the task type.
 *
 * @component
 * @param {EntityOutputData} props.entityOutput - The entity output data.
 * @param {string} props.taskType - The task type.
 * @param {HistoricalItems} props.historicalItems - The historical items.
 * @param {boolean} props.isDataLoading - Indicates whether the data is loading.
 * @returns {JSX.Element} The rendered PredictionDisplay component.
 */

const Prediction: React.FC<PredictionDisplayProps> = ({
    entityOutput,
    taskType,
    historicalItems,
    isDataLoading,
    query,
    anchorTime,
    showGroundTruth,
    selectedColumns,
    setSelectedColumns
}) => {
    if (entityOutput?.prediction === null && entityOutput?.target === null) {
        return <NoDataAlert />;
    }
    if (isDataLoading && !entityOutput) {
        return <Skeleton className="w-full h-10" />;
    }
    return (
        <>
            {taskType === TaskType.LINK_PREDICTION ||
            taskType === TaskType.TEMPORAL_LINK_PREDICTION ||
            taskType === TaskType.STATIC_LINK_PREDICTION ? (
                <LinkPrediction
                    entityOutput={entityOutput as LinkPredictionEntityOutput}
                    historicalItems={historicalItems as HistoricalItems}
                    isDataLoading={isDataLoading}
                    showGroundTruth={showGroundTruth}
                    selectedColumns={selectedColumns}
                    setSelectedColumns={setSelectedColumns}
                />
            ) : taskType === TaskType.REGRESSION ? (
                <RegressionPrediction
                    entityOutput={entityOutput as RegressionEntityOutput}
                    isDataLoading={isDataLoading}
                    query={query}
                    anchorTime={anchorTime}
                    showGroundTruth={showGroundTruth}
                />
            ) : taskType === TaskType.BINARY_CLASSIFICATION ? (
                <BinaryClassificationPrediction
                    entityOutput={entityOutput as RegressionEntityOutput}
                    isDataLoading={isDataLoading}
                    query={query}
                    anchorTime={anchorTime}
                    showGroundTruth={showGroundTruth}
                />
            ) : taskType === TaskType.MULTICLASS_CLASSIFICATION ||
              taskType === TaskType.MULTILABEL_CLASSIFICATION ? (
                <MultiClassificationPrediction
                    entityOutput={entityOutput as ClassificationEntityOutput}
                    isDataLoading={isDataLoading}
                    showGroundTruth={showGroundTruth}
                />
            ) : null}
        </>
    );
};

export default WithErrorBoundary(Prediction);
