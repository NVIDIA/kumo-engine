// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useState } from "react";

import {
    Button,
    Card,
    ChartLabel,
    Description,
    GridCol,
    GridRow,
    HeatMapChart,
    HeatMapData,
    Select,
    Skeleton,
    Title
} from "@kumo-ai/kumo-ui-catalog";
import {
    RegressionHeatmapResponse,
    RegressionResiduals,
    TrainingStage
} from "gen/v0/types";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";

import { useRegressionHeatmap } from "hooks/jobs/training/eval/regression";

import { dataSplitKeys } from "utils/Constants";
import { getValues, isObjectNotEmpty } from "utils/Helpers/objectHelpers";
import { isValid } from "utils/Helpers/typeGuards";

// Number of entries below which we count as outliers
const OUTLIER_THRESHOLD = 10;

const DATA_SPLIT_OPTIONS = [
    { value: dataSplitKeys.HoldOut, label: "Holdout" },
    { value: dataSplitKeys.Validation, label: "Validation" },
    { value: dataSplitKeys.Training, label: "Training" }
] as const;

const parseAxisValues = (
    values: (string | number | null | undefined)[] | undefined
): number[] =>
    values
        ?.map((val) => {
            if (val == null) return NaN;
            if (typeof val === "string") {
                const num = Number(val.replace(/,/g, ""));
                return Number.isFinite(num) ? parseFloat(num.toFixed(3)) : NaN;
            }
            return Number.isFinite(val) ? parseFloat(val.toFixed(3)) : NaN;
        })
        .filter((v) => Number.isFinite(v)) ?? [];

/**
 * Heatmap Component for Regression stats.
 *
 * @param {object} xyzData Selected data splits
 *   {
 *      x_values:[x-axis values],
 *      y_values:[y-axis values],
 *      buckets:[[samples],[samples]]
 *   }
 *
 * @returns `<HeatmapStats />`
 */
export interface RegressionHeatMapProps {
    queryId: string;
    trainingJobId: string;
    dataSplit: string;
    handleError: (err: Error, title?: string) => void;
}

const RegressionHeatMap = ({
    queryId,
    trainingJobId,
    dataSplit,
    handleError
}: RegressionHeatMapProps) => {
    const [isLoading, setLoading] = useState<boolean>(true);
    const [regHeatmapStats, setRegHeatmapStats] = useState<
        RegressionHeatmapResponse["residuals_chart"]
    >({});
    const [hideOutliers, setHideOutliers] = useState<boolean>(false);
    const [shouldRenderChart, setShouldRenderChart] = useState<boolean>(false);
    const [statskey, setStatsKey] = useState(dataSplit);

    let stats =
        isValid(statskey) &&
        isObjectNotEmpty(regHeatmapStats) &&
        isObjectNotEmpty(regHeatmapStats[statskey as TrainingStage])
            ? regHeatmapStats[statskey as TrainingStage]
            : null;

    const processSplit = useCallback((stats: RegressionResiduals) => {
        const data: RegressionResiduals = {
            x_values: stats?.x_values,
            y_values: stats?.y_values,
            buckets: stats?.buckets
        };

        // Align the axes
        if (!data?.x_values?.length || !data?.y_values?.length) {
            return data; // or return a safe empty object
        }
        const isPredSmaller = data.y_values[0] < data.x_values[0];

        let count = 0;
        if (isPredSmaller) {
            for (let i = 0; i < data?.y_values?.length - 1; i++) {
                if (data.y_values[i] >= data.x_values[0]) {
                    break;
                } else {
                    count++;
                }
            }
            // Padding - If pred is smaller we add count number of values to x axis
            // ( label_values ) and correspondingly to the bucket entries.
            for (let i = 0; i < count; i++) {
                if (!data?.x_values?.includes(data?.y_values[0])) {
                    data?.x_values?.unshift(data.y_values[0]);
                    data?.buckets?.forEach((bucket) => bucket?.unshift(0));
                }
            }
        } else {
            for (let i = 0; i < data?.x_values?.length - 1; i++) {
                if (data.x_values[i] >= data.y_values[0]) {
                    break;
                } else {
                    count++;
                }
            }
            // Padding - If pred is equal or greater we add count number of values
            // to y axis ( pred_values ) and a corresponding new entry to data.buckets.
            const bucketSize = data?.buckets[0]?.length;
            for (let i = 0; i < count; i++) {
                if (!data?.y_values?.includes(data?.x_values[0])) {
                    data?.y_values.unshift(data?.x_values[0]);
                    const tmp = Array(bucketSize).fill(0);
                    data?.buckets.unshift(tmp);
                }
            }
        }

        return data;
    }, []);

    const sumArray = useCallback((arr: number[]) => {
        return arr?.reduce((partialSum, a) => partialSum + a, 0);
    }, []);

    const transpose = useCallback((matrix: number[][]) => {
        return matrix[0]?.map((_col: number, i: number) =>
            matrix?.map((row) => row[i])
        );
    }, []);

    const findLastOutlierIndex = useCallback((counts: number[]) => {
        const maxIdx = counts?.length;
        let retVal = maxIdx;
        counts.reverse().forEach((count, idx) => {
            if (count > OUTLIER_THRESHOLD && retVal === maxIdx) {
                retVal = counts?.length - idx;
            }
        });
        return retVal;
    }, []);

    const { data: regressionHeatmapResponse, isLoading: isDataLoading } =
        useRegressionHeatmap({
            queryId,
            trainingJobId,
            handleError
        });

    useEffect(() => {
        if (regressionHeatmapResponse?.residuals_chart) {
            const splits = getValues(dataSplitKeys) as TrainingStage[];
            const chartData: RegressionHeatmapResponse["residuals_chart"] = {};
            splits.forEach((split) => {
                const splitKey = split as TrainingStage;
                const splitStats =
                    regressionHeatmapResponse.residuals_chart[splitKey];
                if (splitStats && splitStats.x_values && splitStats.y_values) {
                    chartData[split] = processSplit(splitStats);
                }
            });
            setRegHeatmapStats(chartData);
            setLoading(false);
        }
    }, [regressionHeatmapResponse, processSplit]);

    useEffect(() => {
        setShouldRenderChart(false);
        const timer = setTimeout(() => {
            setShouldRenderChart(true);
        }, 100);
        return () => clearTimeout(timer);
    }, [stats, hideOutliers]);

    const hideOutliersfunc = useCallback(
        (stats: RegressionResiduals) => {
            const predictionRowCounts = stats?.buckets?.map((bucket) =>
                sumArray(bucket)
            );
            const labelColumnCounts = transpose(stats?.buckets).map((bucket) =>
                sumArray(bucket)
            );
            const predictionLastOutlierIndex =
                findLastOutlierIndex(predictionRowCounts);
            const labelLastOutlierIndex =
                findLastOutlierIndex(labelColumnCounts);

            return {
                y_values: stats?.y_values?.slice(0, predictionLastOutlierIndex),
                x_values: stats?.x_values?.slice(0, labelLastOutlierIndex),
                buckets: stats?.buckets
                    .slice(0, predictionLastOutlierIndex)
                    .map((bucket) => bucket?.slice(0, labelLastOutlierIndex))
            };
        },
        [findLastOutlierIndex, sumArray, transpose]
    );

    if (isLoading || isDataLoading) {
        return <Skeleton className="mb-4 mt-2.5 h-12.5" />;
    }
    if (
        regressionHeatmapResponse &&
        getValues(regressionHeatmapResponse?.residuals_chart).length < 1
    ) {
        return null;
    }

    if (hideOutliers && stats) {
        stats = hideOutliersfunc(stats);
    }

    const validX = parseAxisValues((stats as RegressionResiduals)?.x_values);
    const validY = parseAxisValues((stats as RegressionResiduals)?.y_values);

    const y_set = new Set(validY);
    // Find the points where x == y to draw straight line
    const matchingPoints =
        validX?.length > 0 && validY?.length
            ? validX?.filter((x) => y_set.has(x))
            : [];

    // Extra polate the line
    if (matchingPoints?.length > 1) {
        if (validX?.length > matchingPoints?.length) {
            const index = validX.indexOf(
                matchingPoints[matchingPoints?.length - 1]
            );

            if (index !== -1 && index < validX?.length - 1) {
                const nextElement = validX[index + 1];
                matchingPoints.push(nextElement);
            }
        } else {
            const diff =
                matchingPoints[matchingPoints?.length - 1] -
                matchingPoints[matchingPoints?.length - 2];
            const extraPolate = parseFloat(
                (
                    matchingPoints[matchingPoints?.length - 1] +
                    parseFloat(diff.toFixed(2))
                ).toFixed(2)
            );
            matchingPoints.push(extraPolate);
        }
    }

    // Echart draws ideal case line only upto point where both x and y cordinates are
    // present. To extend ideal case line over heatmap, add last missing cordinate in
    // either x or y axis so that it don't stop at center of heatmap data.
    if (
        !validX?.includes(matchingPoints[matchingPoints?.length - 1]) &&
        validX[validX?.length - 1] !==
            matchingPoints[matchingPoints?.length - 1]
    ) {
        validX.push(matchingPoints[matchingPoints?.length - 1]);
    }

    if (
        !validY?.includes(matchingPoints[matchingPoints?.length - 1]) &&
        validY[validY?.length - 1] !==
            matchingPoints[matchingPoints?.length - 1]
    ) {
        validY.push(matchingPoints[matchingPoints?.length - 1]);
    }

    const hmap = {
        z: stats?.buckets,
        x: validX,
        y: validY
    };

    const diagonalLine = {
        x: matchingPoints,
        y: matchingPoints
    };

    return (
        <Card className="rounded-sm shadow-none p-6 mb-0">
            <GridRow className="flex justify-between">
                <GridCol>
                    <Title
                        variant="tertiary-heading"
                        children={"Predicted vs Actual Values"}
                    />
                </GridCol>
                <GridCol>
                    <div className="flex gap-2 justify-end">
                        {stats && (
                            <Button
                                id={"hide-outliers"}
                                onClick={() => {
                                    setHideOutliers(!hideOutliers);
                                }}
                                variant="light-outline"
                                size="sm">
                                {hideOutliers
                                    ? "Show Outliers"
                                    : "Hide Outliers"}
                            </Button>
                        )}
                        <Select
                            id={"select-split"}
                            selectedValues={[statskey]}
                            popoverClassName="w-32"
                            options={[...DATA_SPLIT_OPTIONS]}
                            onValuesChange={(newValue) => {
                                setStatsKey(newValue[0] as string);
                            }}
                            search={false}
                            mode="single"
                            placeholder={"Select"}
                            variant="secondary"
                            size="sm"
                            isInValid={false}
                        />
                    </div>
                </GridCol>
            </GridRow>
            {!stats ? (
                <Description
                    variant="secondary"
                    size="sm"
                    className="leading-5">
                    The distribution heatmap for this query is not available for
                    the selected data split.
                </Description>
            ) : (
                stats &&
                shouldRenderChart &&
                hmap.x?.length > 0 &&
                hmap.y?.length > 0 &&
                (hmap.z?.length ?? 0) > 0 && (
                    <div className="min-w-200 max-w-250 overflow-auto">
                        <div className="flex justify-end me-4 pe-2">
                            <ChartLabel
                                iconClassName="bg-pink-600"
                                text={"Ideal Case"}
                            />
                        </div>
                        <HeatMapChart
                            id={"RegressionHeatmap"}
                            heatmapData={hmap as HeatMapData}
                            heatmapLineData={diagonalLine}
                            xAxisLabel={"Actual Values"}
                            yAxisLabel={"Predicted Values"}
                            width={"auto"}
                        />
                    </div>
                )
            )}
        </Card>
    );
};

export default WithErrorBoundary(RegressionHeatMap);
