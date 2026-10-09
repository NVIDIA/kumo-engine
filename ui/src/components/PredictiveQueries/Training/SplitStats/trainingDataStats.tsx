// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useState } from "react";

import { SplitTableStats, SplitTrainTableStats } from "gen/v0/types";

import StatsChart, {
    isClassificationTargetDistribution
} from "components/@ui/Chart/StatsChart";

import { TaskType } from "types/predictiveQueries";
import {
    ClassificationTargetDistribution,
    RegressionTargetDistribution,
    TimeFrameStats
} from "types/stats";

import { parseNumbers } from "utils/Helpers/numberHelpers";
import { isLinkPredictionType } from "utils/Helpers/typeGuards";
import { parseDateTime } from "utils/timeConverter";

export interface TransposedRow {
    metric: string;
    training: string | number | JSX.Element;
    validation: string | number | JSX.Element;
    holdout: string | number | JSX.Element;
}

function toTimeFrameStats(
    stat: SplitTableStats,
    split: "Training" | "Validation" | "Holdout"
): TimeFrameStats {
    return {
        ...stat,
        data_split: split,
        isActive: false,
        min_date: parseDateTime(stat.min_date as string, false, true),
        max_date: parseDateTime(stat.max_date as string, false, true),
        anchor_date: parseDateTime(stat.anchor_date as string, false, true),
        target_distribution: stat.target_distribution as
            | ClassificationTargetDistribution
            | RegressionTargetDistribution,
        num_targets: stat.num_targets ?? 0
    };
}

const formatPredictionWindow = (
    stats: SplitTableStats | null | undefined
): string => {
    if (!stats?.min_date || !stats?.max_date) return "-";
    return `${parseDateTime(stats.min_date, false, true)} - ${parseDateTime(stats.max_date, false, true)}`;
};

const calculatePercentage = (
    numRows: number | undefined,
    totalRows: number
): string => {
    if (!numRows || totalRows === 0) return "-";
    const percentage = (numRows / totalRows) * 100;
    return `${percentage.toFixed(2)}%`;
};

const formatValue = (value: number | undefined): string | number => {
    if (value === undefined || value === null) return "-";
    const parsed = parseNumbers(value, { decimals: 1 });
    return parsed !== null ? parsed : "-";
};

const formatSample = (
    count: number | undefined,
    total: number | undefined
): string => {
    if (!count || !total) return "-";
    const percentage = (count / total) * 100;
    return `${count} (${parseNumbers(percentage, { decimals: 1 })}%)`;
};

const formatMode = (dist: RegressionTargetDistribution | undefined): string => {
    if (!dist?.target_mode) return "-";
    const percentage = dist.target_mode_percentage
        ? dist.target_mode_percentage * 100
        : 0;
    return `${parseNumbers(dist.target_mode)} (${parseNumbers(percentage, { decimals: 1 })}%)`;
};

const hasDistribution = (
    stats: SplitTableStats | null | undefined
): boolean => {
    if (!stats?.target_distribution) return false;

    // Check for regression distribution (has histogram)
    const regressionDist =
        stats.target_distribution as RegressionTargetDistribution;
    if (regressionDist?.histogram) return true;

    // Check for classification distribution (has count)
    const classificationDist =
        stats.target_distribution as ClassificationTargetDistribution;
    if (classificationDist?.count) return true;

    return false;
};

const calculateChartWidth = (
    stats: SplitTableStats | null | undefined
): string => {
    if (!stats?.target_distribution) return "300px";

    let count = 0;
    const dist = stats.target_distribution as
        | ClassificationTargetDistribution
        | RegressionTargetDistribution;

    // Check if it's a classification distribution
    if (isClassificationTargetDistribution(dist)) {
        count = Object.keys(dist.count).length;
    }
    // Check if it's a regression distribution
    else if ((dist as RegressionTargetDistribution)?.histogram?.bins) {
        count = (dist as RegressionTargetDistribution).histogram.bins.length;
    }

    // Calculate width: minimum 300px, 30px per data point, maximum 600px

    const calculatedWidth = Math.max(300, Math.min(600, count * 30));

    return `${calculatedWidth}px`;
};

const renderStatsChart = (
    stats: SplitTableStats | null | undefined,
    split: "Training" | "Validation" | "Holdout",
    hasChart: boolean
): JSX.Element | string => {
    if (!hasChart || !stats) return "-";
    const width = calculateChartWidth(stats);
    const height = "260px";
    return (
        <StatsChart
            data={toTimeFrameStats(stats, split)}
            title={""}
            width={width}
            height={height}
        />
    );
};

const createRow = (
    metric: string,
    training: string | number | JSX.Element,
    validation: string | number | JSX.Element,
    holdout: string | number | JSX.Element
): TransposedRow => ({
    metric,
    training,
    validation,
    holdout
});

// Task-specific row builders
const addBinaryClassificationRows = (
    transposedData: TransposedRow[],
    training: SplitTableStats | null | undefined,
    validation: SplitTableStats | null | undefined,
    holdout: SplitTableStats | null | undefined
): void => {
    const trainingDist = training?.target_distribution as
        | ClassificationTargetDistribution
        | undefined;
    const validationDist = validation?.target_distribution as
        | ClassificationTargetDistribution
        | undefined;
    const holdoutDist = holdout?.target_distribution as
        | ClassificationTargetDistribution
        | undefined;

    transposedData.push(
        createRow(
            "Positive samples",
            formatSample(trainingDist?.count?.True, training?.num_train_rows),
            formatSample(
                validationDist?.count?.True,
                validation?.num_train_rows
            ),
            formatSample(holdoutDist?.count?.True, holdout?.num_train_rows)
        ),
        createRow(
            "Negative samples",
            formatSample(trainingDist?.count?.False, training?.num_train_rows),
            formatSample(
                validationDist?.count?.False,
                validation?.num_train_rows
            ),
            formatSample(holdoutDist?.count?.False, holdout?.num_train_rows)
        )
    );
};

const addRegressionRows = (
    transposedData: TransposedRow[],
    training: SplitTableStats | null | undefined,
    validation: SplitTableStats | null | undefined,
    holdout: SplitTableStats | null | undefined,
    distributionAvailable: boolean,
    trainingHasDistribution: boolean,
    validationHasDistribution: boolean,
    holdoutHasDistribution: boolean
): void => {
    const trainingDist = training?.target_distribution as
        | RegressionTargetDistribution
        | undefined;
    const validationDist = validation?.target_distribution as
        | RegressionTargetDistribution
        | undefined;
    const holdoutDist = holdout?.target_distribution as
        | RegressionTargetDistribution
        | undefined;

    transposedData.push(
        createRow(
            "Mean",
            formatValue(trainingDist?.target_mean),
            formatValue(validationDist?.target_mean),
            formatValue(holdoutDist?.target_mean)
        ),
        createRow(
            "Mode",
            formatMode(trainingDist),
            formatMode(validationDist),
            formatMode(holdoutDist)
        ),
        createRow(
            "Min",
            formatValue(trainingDist?.target_min),
            formatValue(validationDist?.target_min),
            formatValue(holdoutDist?.target_min)
        ),
        createRow(
            "25%",
            formatValue(trainingDist?.target_percentile_25),
            formatValue(validationDist?.target_percentile_25),
            formatValue(holdoutDist?.target_percentile_25)
        ),
        createRow(
            "Median",
            formatValue(trainingDist?.target_median),
            formatValue(validationDist?.target_median),
            formatValue(holdoutDist?.target_median)
        ),
        createRow(
            "75%",
            formatValue(trainingDist?.target_percentile_75),
            formatValue(validationDist?.target_percentile_75),
            formatValue(holdoutDist?.target_percentile_75)
        ),
        createRow(
            "Max",
            formatValue(trainingDist?.target_max),
            formatValue(validationDist?.target_max),
            formatValue(holdoutDist?.target_max)
        )
    );

    if (distributionAvailable) {
        transposedData.push(
            createRow(
                "Label distribution",
                renderStatsChart(training, "Training", trainingHasDistribution),
                renderStatsChart(
                    validation,
                    "Validation",
                    validationHasDistribution
                ),
                renderStatsChart(holdout, "Holdout", holdoutHasDistribution)
            )
        );
    }
};

export const useRestructureStats = (
    taskType: TaskType,
    splitStats: SplitTrainTableStats | null,
    isTrainingEvaluation: boolean = false
) => {
    const [allStats, setAllStats] = useState<SplitTrainTableStats | null>(null);
    const [tableData, setTableData] = useState<TransposedRow[]>([]);

    const restructureStats = useCallback(
        (stats: SplitTrainTableStats) => {
            const training = stats.train_table_stats;
            const validation = stats.val_table_stats;
            const holdout = stats.test_table_stats;

            // Calculate total rows for percentage calculation
            const totalRows =
                (training?.num_train_rows || 0) +
                (validation?.num_train_rows || 0) +
                (holdout?.num_train_rows || 0);

            const transposedData: TransposedRow[] = [];

            transposedData.push(
                createRow(
                    "Prediction window",
                    formatPredictionWindow(training),
                    formatPredictionWindow(validation),
                    formatPredictionWindow(holdout)
                )
            );

            transposedData.push(
                createRow(
                    "Split distribution",
                    calculatePercentage(training?.num_train_rows, totalRows),
                    calculatePercentage(validation?.num_train_rows, totalRows),
                    calculatePercentage(holdout?.num_train_rows, totalRows)
                ),
                createRow(
                    "Total rows",
                    training?.num_train_rows || 0,
                    validation?.num_train_rows || 0,
                    holdout?.num_train_rows || 0
                )
            );

            // If isTrainingEvaluation is true, only show the basic rows
            if (isTrainingEvaluation) {
                setAllStats(stats);
                setTableData(transposedData);
                return;
            }

            // Add Unique Entities for non-training evaluation view
            transposedData.push(
                createRow(
                    "Unique entities",
                    training?.num_entities || 0,
                    validation?.num_entities || 0,
                    holdout?.num_entities || 0
                )
            );

            // Check if distribution is available
            const trainingHasDistribution = hasDistribution(training);
            const validationHasDistribution = hasDistribution(validation);
            const holdoutHasDistribution = hasDistribution(holdout);

            const distributionAvailable =
                trainingHasDistribution ||
                validationHasDistribution ||
                holdoutHasDistribution;

            // Add taskType-specific rows
            if (taskType === TaskType.BINARY_CLASSIFICATION) {
                addBinaryClassificationRows(
                    transposedData,
                    training,
                    validation,
                    holdout
                );
            } else if (
                taskType === TaskType.REGRESSION ||
                taskType === TaskType.FORECASTING
            ) {
                addRegressionRows(
                    transposedData,
                    training,
                    validation,
                    holdout,
                    distributionAvailable,
                    trainingHasDistribution,
                    validationHasDistribution,
                    holdoutHasDistribution
                );
            } else {
                if (distributionAvailable && !isLinkPredictionType(taskType)) {
                    // Show charts for multiclass and other non-link-prediction types
                    transposedData.push(
                        createRow(
                            "Label distribution",
                            renderStatsChart(
                                training,
                                "Training",
                                trainingHasDistribution
                            ),
                            renderStatsChart(
                                validation,
                                "Validation",
                                validationHasDistribution
                            ),
                            renderStatsChart(
                                holdout,
                                "Holdout",
                                holdoutHasDistribution
                            )
                        )
                    );
                }
            }

            setAllStats(stats);
            setTableData(transposedData);
        },
        [taskType, isTrainingEvaluation]
    );

    useEffect(() => {
        if (splitStats) {
            restructureStats(splitStats);
        }
    }, [restructureStats, splitStats]);

    return { tableData, allStats };
};
