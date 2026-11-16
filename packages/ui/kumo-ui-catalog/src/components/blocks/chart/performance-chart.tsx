// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React from "react";
import { useEChartsReady } from "@/hooks/use-echarts-ready";
import ReactECharts from "echarts-for-react";
import { createDiagonalPattern } from "@/utils/helper";
import { useChartConfig } from "@/hooks/use-chart-config";
import { BarChartData, jobStatus, QueryProgressStats } from "@/types/chart";
import type { EChartsOption, SeriesOption } from "echarts";

/**
 * Line chart for Training Progress Performance
 *
 * @param id Id for chart
 * @param progressData data
 * @param size size for the  chart
 * @param jobStatus query status
 * @returns ```<PerformanceChart />
 */

interface PerformanceChartProps {
    id?: string;
    progressData: QueryProgressStats;
    width?: string | number;
    height?: string | number;
    jobstatus: jobStatus;
    showTitle?: boolean;
}

export const PerformanceChart: React.FC<PerformanceChartProps> = ({
    id = "performance-chart",
    progressData,
    width = "480px",
    height = "380px",
    jobstatus,
    showTitle = true
}) => {
    const chartConfig = useChartConfig();
    const { isReady, containerRef } = useEChartsReady();
    const barData: BarChartData[] = [];
    const bestLineData: number[] = [];
    const isTrainComplete = jobstatus === jobStatus.Done;
    const mode = progressData?.mode || "max";
    const lineColor = progressData?.datasets?.[0]?.color;

    let maxVal = 0;
    let currBest = 0;

    const BORDER_COLOR = chartConfig.colors.barBorderColor;
    const BACKGROUND_COLOR = chartConfig.colors.barBackground;

    progressData?.datasets?.[0]?.value?.forEach((val, index) => {
        const parsed = parseFloat(val?.toFixed(3));
        if (index === 0) currBest = parsed;
        else {
            currBest =
                mode === "max"
                    ? Math.max(currBest, parsed)
                    : Math.min(currBest, parsed);
        }
        bestLineData.push(currBest);
        maxVal = Math.max(parsed, maxVal);
        const isLast = index === progressData?.datasets?.[0]?.value?.length - 1;
        const usePattern = !isTrainComplete && isLast;

        barData.push({
            value: parsed,
            itemStyle: {
                color: usePattern
                    ? {
                          image: createDiagonalPattern(
                              BACKGROUND_COLOR,
                              chartConfig.colors.tooltipBackground,
                              16
                          ),
                          repeat: "repeat"
                      }
                    : BACKGROUND_COLOR,
                borderColor: BORDER_COLOR,
                borderWidth: 0.7
            }
        });
    });

    const xAxisTickLabels = progressData?.xAxisLabels || [];
    const xAxisLabel = progressData?.xLabel || "";
    const yAxisLabel = progressData?.yLabel || "";

    const option: EChartsOption = {
        tooltip: {
            trigger: "item",
            confine: true,
            backgroundColor: chartConfig.colors.tooltipBackground,
            borderColor: chartConfig.colors.tooltipBorderColor,
            borderWidth: Number(chartConfig.borders.tooltipBorderWidth),
            borderRadius: Number(chartConfig.borders.tooltipBorderRadius),
            padding: [
                Number(chartConfig.paddings.tooltipPaddingY),
                Number(chartConfig.paddings.tooltipPaddingX)
            ],
            textStyle: {
                color: chartConfig.colors.tooltipForeground,
                fontFamily: chartConfig.fonts.font,
                fontSize: Number(chartConfig.fonts.tooltipFontSize),
                fontWeight: Number(chartConfig.fonts.tooltipFontWeight)
            },
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            formatter: (params: any) => {
                const dataIndex = params?.dataIndex;
                const displayLabel =
                    isTrainComplete || dataIndex !== xAxisTickLabels.length - 1
                        ? `${dataIndex + 1}`
                        : "(In Progress)";
                const barValue = barData[dataIndex]?.value;
                const lineValue = bestLineData[dataIndex];

                const tooltipItems = [];
                tooltipItems.push(`<strong>${displayLabel}</strong>`);

                // Bar series tooltip
                if (barValue != null) {
                    const isPatternedBar =
                        !isTrainComplete &&
                        dataIndex ===
                            progressData?.datasets?.[0]?.value?.length - 1;
                    const barColor = isPatternedBar
                        ? BACKGROUND_COLOR
                        : barData[dataIndex]?.itemStyle?.color ||
                          BACKGROUND_COLOR;
                    const barBorderColor =
                        barData[dataIndex]?.itemStyle?.borderColor ||
                        BORDER_COLOR;

                    const barMarker = `<span style="
                        display:inline-block;
                        margin-right:6px;
                        width:14px;
                        height:14px;
                        background-color:${barColor};
                        border:2px solid ${barBorderColor};
                        vertical-align:middle;
                    "></span>`;

                    tooltipItems.push(
                        `${barMarker}Best value in Experiment: ${barValue}`
                    );
                }

                // Line series tooltip
                if (lineValue !== null && lineValue !== undefined) {
                    const lineMarker = `<span style="
                        display:inline-block;
                        margin-right:6px;
                        width:14px;
                        height:14px;
                        background-color:${BACKGROUND_COLOR};
                        border:2px solid ${lineColor};
                        vertical-align:middle;
                    "></span>`;

                    tooltipItems.push(
                        `${lineMarker}Best value across all Experiments: ${lineValue}`
                    );
                }

                return tooltipItems.join("<br/>");
            }
        },
        grid: {
            top: showTitle ? 50 : 40,
            left: 30,
            right: 10,
            bottom: xAxisLabel ? 30 : 10,
            containLabel: true
        },
        animation: false,
        legend: {
            show: false
        },
        xAxis: [
            {
                type: "category",
                name: xAxisLabel,
                nameLocation: "middle",
                nameGap: 30,
                nameTextStyle: {
                    color: chartConfig.colors.axisName,
                    fontFamily: chartConfig.fonts.font,
                    fontSize: chartConfig.fonts.axisNameFontSize,
                    fontWeight: Number(chartConfig.fonts.axisNameFontWeight)
                },
                data: xAxisTickLabels,
                splitArea: {
                    show: false
                },
                splitLine: {
                    show: false
                },
                axisLine: {
                    lineStyle: {
                        color: chartConfig.colors.axisLine
                    }
                },
                axisLabel: {
                    interval: 0,
                    margin: 10,
                    formatter: (_value: string, index: number) =>
                        isTrainComplete || index !== xAxisTickLabels.length - 1
                            ? `${index + 1}`
                            : "(In Progress)",
                    color: chartConfig.colors.axisLabel
                },
                axisTick: {
                    show: false
                }
            }
        ],
        yAxis: [
            {
                type: "value",
                name: yAxisLabel,
                nameLocation: "middle",
                nameTextStyle: {
                    color: chartConfig.colors.axisName,
                    fontFamily: chartConfig.fonts.font,
                    fontSize: chartConfig.fonts.axisNameFontSize,
                    fontWeight: Number(chartConfig.fonts.axisNameFontWeight)
                },
                nameGap: 45,
                min: 0,
                max: maxVal,
                axisLabel: {
                    color: chartConfig.colors.axisLabel,
                    fontFamily: chartConfig.fonts.font,
                    fontSize: chartConfig.fonts.axisLabelFontSize,
                    fontWeight: Number(chartConfig.fonts.axisLabelFontWeight)
                },
                axisLine: {
                    show: true,
                    lineStyle: {
                        color: chartConfig.colors.axisLine
                    }
                },
                splitArea: {
                    show: false
                },
                splitLine: {
                    show: false
                }
            }
        ],
        series: [
            {
                name: "Best value in Experiment",
                type: "bar",
                data: barData,
                barMaxWidth: 50,
                barMinHeight: 4,
                emphasis: {
                    disabled: true
                },
                z: 1
            },
            {
                name: "Best value across all Experiments",
                type: "line",
                data: bestLineData.map((value) => ({
                    value,
                    itemStyle: {
                        color: lineColor
                    }
                })),
                symbolSize: 4,
                connectNulls: false,
                lineStyle: {
                    color: lineColor,
                    width: 2
                },
                yAxisIndex: 0,
                emphasis: {
                    focus: "series",
                    scale: true,
                    symbolSize: 10,
                    itemStyle: {
                        borderWidth: 6
                    }
                },
                z: 0
            }
        ] as SeriesOption[],
        title: {
            text: showTitle ? "Best Performance So Far" : "",
            left: "center"
        }
    };

    return (
        <div
            id={id}
            className="col-span-2 sm:col-span-4"
            ref={containerRef}
            style={{ height, width }}>
            {isReady && (
                <ReactECharts
                    option={option}
                    style={{
                        height: "100%",
                        width: "100%"
                    }}
                    notMerge={true}
                    lazyUpdate={true}
                    opts={{
                        renderer: "svg"
                    }}
                />
            )}
        </div>
    );
};
