// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React, { useMemo } from "react";

import { BarChart, Card, CardContent } from "@kumo-ai/kumo-ui-catalog";

import {
    ClassificationTargetDistribution,
    RegressionTargetDistribution,
    TimeFrameStats
} from "types/stats";

import { getKeys, getValues } from "utils/Helpers/objectHelpers";

export const isClassificationTargetDistribution = (
    target_distribution:
        | ClassificationTargetDistribution
        | RegressionTargetDistribution
): target_distribution is ClassificationTargetDistribution => {
    return (
        (target_distribution as ClassificationTargetDistribution).count !==
        undefined
    );
};

export interface StatsChartProps {
    data: TimeFrameStats;
    title: string;
    width?: string | number;
    height?: string | number;
}

const StatsChart: React.FC<StatsChartProps> = ({
    data,
    title = "",
    width,
    height
}) => {
    const distribution = data?.target_distribution;

    const { xAxisLabels, dataSet } = useMemo(() => {
        if (distribution && isClassificationTargetDistribution(distribution)) {
            return {
                xAxisLabels: getKeys(distribution.count),
                dataSet: getValues(distribution.count)
            };
        }
        if (distribution?.histogram?.bins && distribution?.histogram?.counts) {
            return {
                xAxisLabels: distribution.histogram.bins.map((b) =>
                    b.toFixed(3)
                ),
                dataSet: distribution.histogram.counts
            };
        }
        return { xAxisLabels: [] as string[], dataSet: [] as number[] };
    }, [distribution]);

    if (!distribution) {
        return <></>;
    }
    return (
        <Card className="flex-grow border-none overflow-y-auto px-0 py-0">
            <CardContent className="px-0">
                <BarChart
                    id={"bar-chart"}
                    title={title}
                    xAxisLabel=""
                    dataSetLabel={""}
                    yAxisLabel=""
                    legend={{
                        display: false
                    }}
                    xAxisTickLabels={xAxisLabels}
                    dataSet={dataSet}
                    trimXAxisLabel={true}
                    customToolTip={true}
                    toolTipText={"Count"}
                    width={width}
                    height={height}
                />
            </CardContent>
        </Card>
    );
};

export default StatsChart;
