// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { TargetHistogram } from "@/types/chart";
import { useEChartsReady } from "@/hooks/use-echarts-ready";
import {
    default as EChartsReact,
    default as ReactECharts
} from "echarts-for-react";
import type { ECElementEvent } from "echarts/types/dist/shared";
import { useMemo, useRef, useState } from "react";

import { useChartConfig } from "@/hooks/use-chart-config";
import type { BarSeriesOption, LineSeriesOption } from "echarts/charts";
import type {
    GridComponentOption,
    LegendComponentOption,
    TooltipComponentOption
} from "echarts/components";
import type { ComposeOption } from "echarts/core";
import type { CallbackDataParams } from "echarts/types/dist/shared";

// Union of multiple series tooltip params
type TooltipFormatterParams = CallbackDataParams | CallbackDataParams[];

// Combine types into one proper `EChartsOption` type
type CustomEChartsOption = ComposeOption<
    | BarSeriesOption
    | LineSeriesOption
    | GridComponentOption
    | TooltipComponentOption
    | LegendComponentOption
>;
type CustomSeries = BarSeriesOption | LineSeriesOption;

interface HistogramChartProps {
    id: string;
    className?: string;
    predData: TargetHistogram;
    labelData: TargetHistogram;
    xAxisLabel: string;
    yAxisLabel: string;
}
export const HistogramChart = ({
    id,
    className,
    predData,
    labelData,
    xAxisLabel,
    yAxisLabel
}: HistogramChartProps) => {
    const chartConfig = useChartConfig();
    const { isReady, containerRef } = useEChartsReady();
    const chartRef = useRef<EChartsReact | null>(null);
    const barDataPredicted = useMemo(() => {
        return predData?.counts || [];
    }, [predData?.counts]);

    const barDataActuals = useMemo(() => {
        return labelData?.counts || [];
    }, [labelData?.counts]);
    const [legendState, setLegendState] = useState<Record<string, boolean>>({
        Predictions: true,
        Actuals: true
    });

    function setTransparency(hex: string, alpha: number) {
        const r = parseInt(hex.slice(1, 3), 16);
        const g = parseInt(hex.slice(3, 5), 16);
        const b = parseInt(hex.slice(5, 7), 16);
        return `rgba(${r}, ${g}, ${b}, ${alpha})`;
    }

    function getStrikethroughText(text: string) {
        const stroke = "\u0336";
        return text
            .split("")
            .map((c) => c + stroke)
            .join("");
    }

    const series: CustomSeries[] = useMemo(() => {
        return [
            {
                name: "Predictions",
                type: "line",
                data: legendState["Predictions"] ? barDataPredicted : [],
                symbol: "none",
                smooth: true,
                lineStyle: {
                    color: setTransparency(chartConfig.colors.primary, 0.8),
                    width: 1
                    //opacity: legendState["Predictions"] ? 1 : 0
                },
                z: 3
            },
            {
                name: "Actuals",
                type: "line",
                data: legendState["Actuals"] ? barDataActuals : [],
                symbol: "none",
                smooth: true,
                lineStyle: {
                    color: chartConfig.colors.slate400,
                    width: 1
                    //opacity: legendState["Actuals"] ? 1 : 0
                },
                z: 3
            },
            {
                type: "bar",
                barWidth: "100%",
                barCategoryGap: "0%",
                data: barDataActuals,
                itemStyle: {
                    color: setTransparency(chartConfig.colors.slate400, 0.3),
                    borderColor: setTransparency(
                        chartConfig.colors.slate400,
                        0.8
                    ),
                    borderWidth: 0.1
                },
                z: 1
            },
            {
                type: "bar",
                barWidth: "100%",
                barGap: "-100%",
                data: barDataPredicted,
                itemStyle: {
                    color: setTransparency(chartConfig.colors.primary, 0.2),
                    borderColor: setTransparency(
                        chartConfig.colors.primary,
                        0.4
                    ),
                    borderWidth: 0.1
                },
                z: 2
            }
        ];
    }, [
        legendState,
        barDataPredicted,
        chartConfig.colors.primary,
        chartConfig.colors.slate400,
        barDataActuals
    ]);

    const legendFormatter = useMemo(() => {
        return (name: string) =>
            legendState[name] ? name : getStrikethroughText(name);
    }, [legendState]);

    const legendData = useMemo(() => {
        return ["Predictions", "Actuals"].map((name) => ({
            name,
            icon: "rect",
            itemStyle: {
                color:
                    name === "Predictions"
                        ? chartConfig.colors.primary
                        : chartConfig.colors.slate400,
                borderWidth: 0
            }
        }));
    }, [chartConfig.colors.primary, chartConfig.colors.slate400]);

    function getYAxisNameGap(data: number[]): number {
        // Estimate the widest tick label
        const maxVal = Math.max(...data);
        const digits = Math.floor(Math.log10(maxVal || 1)) + 1;
        const approxCharWidth = 8; // pixels per digit (approx for 12px font)
        const buffer = 16; // extra padding

        return Math.max(30, digits * approxCharWidth + buffer);
    }

    function getGridLeft(data: number[]): number {
        const maxVal = Math.max(...data);
        const digits = Math.floor(Math.log10(maxVal || 1)) + 1;
        const approxCharWidth = 8; // pixels per digit

        return Math.max(40, digits * approxCharWidth);
    }

    const option: CustomEChartsOption = useMemo(() => {
        const xAxisData = predData?.bins || [];

        return {
            tooltip: {
                trigger: "axis",
                axisPointer: {
                    type: "shadow"
                },
                position: "top",
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
                formatter: (params: TooltipFormatterParams) => {
                    if (!Array.isArray(params)) return "";

                    const axisValueLabel =
                        "axisValueLabel" in params[0]
                            ? params[0].axisValueLabel
                            : "";
                    if (typeof axisValueLabel !== "string") return "";

                    const currentBinValue = parseFloat(axisValueLabel);
                    if (isNaN(currentBinValue)) return "";

                    const bins = predData?.bins || [];
                    const index = bins.indexOf(currentBinValue);
                    const nextBinValue =
                        index >= 0 && index + 1 < bins.length
                            ? bins[index + 1]
                            : null;
                    const rangeLabel =
                        nextBinValue != null
                            ? `${currentBinValue.toFixed(3)} to ${nextBinValue.toFixed(3)}`
                            : `${currentBinValue.toFixed(3)}`;

                    const pred = params.find(
                        (p) => p.seriesName === "Predictions"
                    );
                    const actual = params.find(
                        (p) => p.seriesName === "Actuals"
                    );
                    const predCount =
                        pred?.value ?? barDataPredicted[index] ?? "-";
                    const actualCount =
                        actual?.value ?? barDataActuals[index] ?? "-";

                    return `
                        <div style="font-size: 12px; line-height: 1.3; ">
                        <div><strong>Range:</strong> ${rangeLabel}</div>
                        <div><strong style="color: ${chartConfig.colors.primary};">Prediction Count:</strong> ${predCount}</div>
                        <div><strong style="color: ${chartConfig.colors.slate400};">Actual Count:</strong> ${actualCount}</div>
                        </div>
                    `;
                }
            },
            legend: {
                top: 10,
                right: 10,
                orient: "horizontal", // or 'vertical' if you prefer
                textStyle: {
                    fontSize: chartConfig.fonts.axisLabelFontSize, //12px
                    color: chartConfig.colors.slate500
                },
                itemWidth: 38, // size of legend color box
                itemHeight: 12,
                data: legendData,
                selected: {
                    Predictions: true,
                    Actuals: true
                },
                formatter: legendFormatter
            },
            grid: {
                left: getGridLeft([...barDataActuals, ...barDataPredicted]),
                right: 20,
                top: 60,
                bottom: 40,
                containLabel: true
            },
            xAxis: [
                {
                    name: xAxisLabel,
                    nameLocation: "middle",
                    nameGap: 60,
                    nameTextStyle: {
                        color: chartConfig.colors.axisName,
                        fontFamily: chartConfig.fonts.font,
                        fontSize: chartConfig.fonts.axisNameFontSize,
                        fontWeight: Number(chartConfig.fonts.axisNameFontWeight)
                    },
                    type: "category",
                    data: xAxisData,

                    axisTick: {
                        show: false
                    },
                    axisLabel: {
                        rotate: 45, // or 30–60 for more/less tilt
                        align: "center",
                        verticalAlign: "top",
                        margin: 30,
                        interval: "auto", // Let ECharts decide which to skip
                        color: chartConfig.colors.axisLabel,
                        fontFamily: chartConfig.fonts.font,
                        fontSize: chartConfig.fonts.axisLabelFontSize,
                        fontWeight: Number(
                            chartConfig.fonts.axisLabelFontWeight
                        ),
                        formatter: (value: string | number) => {
                            const numValue =
                                typeof value === "string"
                                    ? parseFloat(value)
                                    : value;
                            // Show up to 3 decimal places, removing trailing zeros
                            return parseFloat(numValue.toFixed(4)).toString();
                        }
                    },
                    axisLine: {
                        show: true, // This enables the x-axis line
                        lineStyle: {
                            color: chartConfig.colors.axisLine, // Set x-axis line color here
                            width: 1
                        }
                    }
                }
            ],
            yAxis: [
                {
                    name: yAxisLabel,
                    nameLocation: "middle",
                    nameGap: getYAxisNameGap([
                        ...barDataActuals,
                        ...barDataPredicted
                    ]),
                    nameTextStyle: {
                        color: chartConfig.colors.axisName,
                        fontFamily: chartConfig.fonts.font,
                        fontSize: chartConfig.fonts.axisNameFontSize,
                        fontWeight: Number(chartConfig.fonts.axisNameFontWeight)
                    },
                    axisLabel: {
                        color: chartConfig.colors.axisLabel,
                        fontFamily: chartConfig.fonts.font,
                        fontSize: chartConfig.fonts.axisLabelFontSize,
                        fontWeight: Number(
                            chartConfig.fonts.axisLabelFontWeight
                        )
                    },
                    type: "value",
                    axisLine: {
                        show: true, // This enables the y-axis line
                        lineStyle: {
                            color: chartConfig.colors.axisLine, // Set y-axis line color here
                            width: 1
                        }
                    },
                    splitLine: {
                        show: false // Disable horizontal grid lines (split lines)
                    }
                }
            ],
            series
        };
    }, [
        predData?.bins,
        chartConfig.colors.tooltipBackground,
        chartConfig.colors.tooltipBorderColor,
        chartConfig.colors.tooltipForeground,
        chartConfig.colors.slate500,
        chartConfig.colors.axisName,
        chartConfig.colors.axisLabel,
        chartConfig.colors.axisLine,
        chartConfig.colors.primary,
        chartConfig.colors.slate400,
        chartConfig.borders.tooltipBorderWidth,
        chartConfig.borders.tooltipBorderRadius,
        chartConfig.paddings.tooltipPaddingY,
        chartConfig.paddings.tooltipPaddingX,
        chartConfig.fonts.font,
        chartConfig.fonts.tooltipFontSize,
        chartConfig.fonts.tooltipFontWeight,
        chartConfig.fonts.axisLabelFontSize,
        chartConfig.fonts.axisNameFontSize,
        chartConfig.fonts.axisNameFontWeight,
        chartConfig.fonts.axisLabelFontWeight,
        legendData,
        legendFormatter,
        barDataActuals,
        barDataPredicted,
        xAxisLabel,
        yAxisLabel,
        series
    ]);

    return (
        <div
            id={id}
            className={className}
            ref={containerRef}
            style={{ height: "560px", width: "100%" }}>
            {isReady && (
                <ReactECharts
                    key={`chart-${legendState.Predictions}-${legendState.Actuals}`}
                    option={option}
                    ref={chartRef}
                    onEvents={{
                        legendselectchanged: (params: ECElementEvent) => {
                            const name = params.name;

                            setLegendState((prev) => {
                                const updated = {
                                    ...prev,
                                    [name]: !prev[name]
                                };

                                // Update legend formatter manually via setOption
                                chartRef.current
                                    ?.getEchartsInstance()
                                    .setOption({
                                        legend: {
                                            formatter: (legendName: string) =>
                                                updated[legendName]
                                                    ? legendName
                                                    : getStrikethroughText(
                                                          legendName
                                                      )
                                        }
                                    });

                                return updated;
                            });

                            // Prevent ECharts from toggling visibility
                            chartRef.current
                                ?.getEchartsInstance()
                                .dispatchAction({
                                    type: params.selected[name]
                                        ? "legendUnSelect"
                                        : "legendSelect",
                                    name
                                });
                        },
                        updateAxisPointer: (params: {
                            axesInfo?: { value: number }[];
                        }) => {
                            const xIndex = params?.axesInfo?.[0]?.value;
                            if (xIndex == null || xIndex === -1) return;

                            const predictedY = barDataPredicted[xIndex];
                            const actualY = barDataActuals[xIndex];

                            const chart =
                                chartRef.current?.getEchartsInstance();

                            const updatedSeries = [];
                            if (legendState["Predictions"]) {
                                updatedSeries.push({
                                    name: "Predictions",
                                    markPoint: {
                                        data: [
                                            {
                                                coord: [xIndex, predictedY],
                                                symbol: "circle",
                                                symbolSize: 10,
                                                itemStyle: {
                                                    color: chartConfig.colors
                                                        .primary
                                                }
                                            }
                                        ]
                                    }
                                });
                            }

                            if (legendState["Actuals"]) {
                                updatedSeries.push({
                                    name: "Actuals",
                                    markPoint: {
                                        data: [
                                            {
                                                coord: [xIndex, actualY],
                                                symbol: "circle",
                                                symbolSize: 10,
                                                itemStyle: {
                                                    color: chartConfig.colors
                                                        .slate400
                                                }
                                            }
                                        ]
                                    }
                                });
                            }

                            chart?.setOption({ series: updatedSeries });
                        },
                        globalout: () => {
                            const clearedSeries = [];
                            if (legendState["Predictions"]) {
                                clearedSeries.push({
                                    name: "Predictions",
                                    markPoint: { data: [] }
                                });
                            }

                            if (legendState["Actuals"]) {
                                clearedSeries.push({
                                    name: "Actuals",
                                    markPoint: { data: [] }
                                });
                            }

                            chartRef.current
                                ?.getEchartsInstance()
                                .setOption({ series: clearedSeries });
                        }
                    }}
                    style={{ height: "100%", width: "100%" }}
                    notMerge={false}
                    lazyUpdate={true}
                    opts={{
                        renderer: "svg"
                    }}
                />
            )}
        </div>
    );
};
