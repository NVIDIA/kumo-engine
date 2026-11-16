// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
    DistributionChart,
    EmptyTable,
    GridCol,
    GridRow
} from "@kumo-ai/kumo-ui-catalog";

import { TaskType } from "types/predictiveQueries";
import {
    ClassificationTargetDistribution,
    RegressionTargetDistribution,
    TimeFrameStats
} from "types/stats";

import { jobStatus } from "utils/Constants";
import { getKeys } from "utils/Helpers/objectHelpers";
import { isLinkPredictionType, isValid } from "utils/Helpers/typeGuards";

/**
 * Component to show Predictive Query training timeframe line chart *
 * @param queryStatus Status of the Query *
 * @param taskType Type of Query (BC, regression, etc) *
 * @param data state information from parent component which hold stats data
 * @param show show/hide this stats
 * @returns ```<TimeframeChart />```
 */

export interface TimeframeChartProps {
    queryStatus: string;
    taskType: string;
    timeFrameStats: Array<TimeFrameStats>;
    noData: boolean;
    show: boolean;
}

const TimeframeChart = ({
    queryStatus,
    taskType,
    timeFrameStats,
    noData,
    show
}: TimeframeChartProps) => {
    const dataValues = timeFrameStats?.slice().reverse();
    const xAxisValues = dataValues?.map(
        (v) => v?.min_date + " - " + v?.max_date
    );
    const y2AxisValues: { [k: string]: number[] } = {};
    y2AxisValues["Number of Entities"] = dataValues?.map(
        (val) => val?.num_entities
    );
    const stats = dataValues?.map((v) => {
        if (
            taskType === TaskType.BINARY_CLASSIFICATION ||
            taskType === TaskType.MULTICLASS_CLASSIFICATION ||
            taskType === TaskType.MULTILABEL_CLASSIFICATION
        ) {
            return (v?.target_distribution as ClassificationTargetDistribution)
                .count;
        } else if (isLinkPredictionType(taskType)) {
            return null;
        } else {
            return v?.target_distribution as RegressionTargetDistribution;
        }
    });

    const keys =
        stats?.length > 0 && isValid(stats[0])
            ? Array.from(new Set(stats.flatMap((obj) => getKeys(obj))))
            : [];

    let y1AxisValues: { [k: string]: Array<number> } = {};

    keys.forEach((key: string) => {
        if (key !== "histogram") {
            let newKey = key;
            if (key === "True") {
                newKey = "Count Positive";
            } else if (key === "False") {
                newKey = "Count Negative";
            } else if (key?.includes("_percentile_")) {
                newKey = key.split("_percentile_").pop() + "%";
            } else if (key?.includes("target_")) {
                if (key.split("_").length < 3) {
                    newKey = key.split("_").pop() as string;
                } else {
                    return;
                }
            }
            y1AxisValues[newKey] = stats?.map((stat) => stat?.[key] as number);
        }
    });

    let yAxisTitle = "";
    let showSecondGraph = false;
    if (isLinkPredictionType(taskType)) {
        yAxisTitle = "Number of Entities";
        y1AxisValues = y2AxisValues as { [k: string]: number[] };
    }
    if (taskType === TaskType.REGRESSION || taskType === TaskType.FORECASTING) {
        yAxisTitle =
            taskType === TaskType.REGRESSION
                ? "Distribution of Regression Values"
                : "Distribution of Forecasting Values";
        showSecondGraph = true;
    }
    if (
        taskType === TaskType.BINARY_CLASSIFICATION ||
        taskType === TaskType.MULTICLASS_CLASSIFICATION
    ) {
        yAxisTitle = "Count of Entities with a Given Value";
    }
    if (taskType === TaskType.MULTILABEL_CLASSIFICATION) {
        yAxisTitle = "Distribution of Labels";
        showSecondGraph = true;
    }
    return !show ? null : (
        <GridRow className="flex flex-col">
            <GridCol className="overflow-auto">
                {timeFrameStats?.length > 0 ? (
                    <div>
                        <DistributionChart
                            id={"train-table-stat-chart"}
                            title={"Training Table Statistics over Time"}
                            xAxisLabel={"Time Frames"}
                            yAxisLabel={yAxisTitle}
                            xAxisData={xAxisValues}
                            chartData={y1AxisValues}
                            lineChart={
                                taskType === TaskType.REGRESSION ||
                                taskType === TaskType.FORECASTING
                            }
                        />
                        {showSecondGraph && (
                            <DistributionChart
                                id={"train-table-stat-chart-entities"}
                                title={""}
                                xAxisLabel={"Time Frames"}
                                yAxisLabel={"Number of Entities"}
                                xAxisData={xAxisValues}
                                chartData={y2AxisValues}
                            />
                        )}
                    </div>
                ) : (
                    <div>
                        <EmptyTable
                            enableCard={false}
                            id="no-data"
                            label={"No data found!"}
                            showLoading={
                                queryStatus !== jobStatus.Cancelled &&
                                queryStatus !== jobStatus.Running &&
                                queryStatus !== jobStatus.Queued &&
                                noData
                            }
                        />
                    </div>
                )}
            </GridCol>
        </GridRow>
    );
};

export default TimeframeChart;
