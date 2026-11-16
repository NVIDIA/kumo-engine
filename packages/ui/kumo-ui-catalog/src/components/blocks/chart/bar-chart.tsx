// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useChartConfig } from "@/hooks/use-chart-config";
import { useEChartsReady } from "@/hooks/use-echarts-ready";
import { parseNumbers } from "@/utils/helper";
import type { EChartsOption, SeriesOption } from "echarts";
import ReactECharts from "echarts-for-react";
export interface BarChartProps {
    id: string;
    title?: string;
    xAxisLabel?: string;
    yAxisLabel?: string;
    dataSet: (number | string)[];
    dataSetLabel?: string;
    yAxisMin?: number;
    yAxisMax?: number;
    step?: number;
    xAxisTickLabels?: (number | string)[];
    legend?: { [key: string]: number | string | boolean | null };
    height?: string | number;
    width?: string | number;
    trimXAxisLabel?: boolean;
    customToolTip?: boolean;
    enableDataLabel?: boolean;
    customLabel?: boolean;
    toolTipText?: string;
}

export const BarChart: React.FC<BarChartProps> = ({
    id,
    title = "",
    xAxisLabel,
    yAxisLabel,
    dataSet,
    dataSetLabel = "",
    yAxisMin,
    yAxisMax,
    step,
    xAxisTickLabels = [],
    legend = {},
    height = "560px",
    width = "580px",
    trimXAxisLabel = false,
    customToolTip = false,
    enableDataLabel = false,
    customLabel = false,
    toolTipText = ""
}) => {
    const chartConfig = useChartConfig();
    const { isReady, containerRef } = useEChartsReady();
    const limitedXAxisTickLabels =
        xAxisTickLabels.length > 20
            ? xAxisTickLabels.slice(0, 20)
            : xAxisTickLabels;
    const shouldRotateLabels =
        (limitedXAxisTickLabels?.some((label) => label.toString().length > 5) &&
            limitedXAxisTickLabels?.length > 5) ||
        !customLabel;

    const option: EChartsOption = {
        title: {
            text: title,
            left: "center",
            top: 20
        },
        grid: {
            left: shouldRotateLabels ? 65 : 30,
            right: 20,
            bottom: xAxisLabel ? 20 : 10,
            containLabel: true
        },
        legend: {
            // echarts 5 defaulted the legend to the top; echarts 6 changed the
            // default to the bottom, where it collides with the x-axis label.
            // Pin it to the top unless the caller anchors it explicitly.
            ...(legend.top == null && legend.bottom == null ? { top: 0 } : {}),
            ...legend
        },
        tooltip: {
            confine: true,
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
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            formatter: function (params: any) {
                const param = Array.isArray(params) ? params[0] : params;
                const label = param?.name ?? "";
                const value = param?.value ?? "";
                const color = param?.color ?? "";

                if (customToolTip) {
                    const formattedValue = parseNumbers(value?.toString(), {
                        decimals: 2
                    });
                    const unit = toolTipText === "Percent Share" ? "%" : "";
                    const tooltipLabel = customToolTip
                        ? `${toolTipText}: ${formattedValue}${unit}`
                        : `${value}`;
                    return `
                        <div style="display: flex; flex-direction: column; align-items: flex-start;">
                            <strong style="margin-bottom: 4px;">${label}</strong>
                            <div style="display: flex; align-items: center;">
                                <span style="
                                        display:inline-block;
                                        margin-right:6px;
                                        width:14px;
                                        height:14px;
                                        background-color:${color};
                                        border: 2px solid #62748e;
                                "></span>
                                <span>${tooltipLabel}</span>
                            </div>
                        </div>`;
                }
                return `<strong>${label}</strong>: ${value}`;
            }
        },
        animation: false,
        xAxis: {
            type: "category",
            name: xAxisLabel,
            nameGap: 80,
            nameLocation: "middle",
            nameTextStyle: {
                color: chartConfig.colors.axisName,
                fontFamily: chartConfig.fonts.font,
                fontSize: chartConfig.fonts.axisNameFontSize,
                fontWeight: Number(chartConfig.fonts.axisNameFontWeight)
            },
            data: limitedXAxisTickLabels,
            splitArea: {
                show: false
            },
            axisTick: {
                show: false
            },
            axisLine: {
                lineStyle: {
                    color: chartConfig.colors.axisLine
                }
            },
            axisLabel: {
                interval: 0, // Show all labels
                rotate: shouldRotateLabels ? 45 : 0, // Tilt labels to avoid overlap
                margin: 10,
                color: chartConfig.colors.axisLabel,
                fontFamily: chartConfig.fonts.font,
                fontSize: chartConfig.fonts.axisLabelFontSize,
                fontWeight: Number(chartConfig.fonts.axisLabelFontWeight),
                formatter: function (_value: string, index: number) {
                    let label = xAxisTickLabels[index]?.toString() || "";
                    if (customLabel) {
                        const splitLabels = label.split(" - ");
                        if (index === 0) {
                            label = splitLabels[0] ?? "";
                        } else if (
                            index === xAxisTickLabels.length - 1 &&
                            splitLabels?.length > 1
                        ) {
                            label = splitLabels[1];
                        } else {
                            label = "";
                        }
                    }

                    if (trimXAxisLabel && label.length > 15) {
                        label = label.substring(0, 15) + "...";
                    }

                    return label;
                }
            }
        },
        yAxis: {
            type: "value",
            name: yAxisLabel,
            nameLocation: "middle",
            nameGap: 30,
            nameRotate: 90,
            nameTextStyle: {
                color: chartConfig.colors.axisName,
                fontFamily: chartConfig.fonts.font,
                fontSize: chartConfig.fonts.axisNameFontSize,
                fontWeight: Number(chartConfig.fonts.axisNameFontWeight)
            },
            min: yAxisMin,
            max: yAxisMax,
            interval: step,
            splitArea: {
                show: false
            },
            splitLine: {
                show: false
            },
            axisLine: {
                show: true,
                lineStyle: {
                    color: chartConfig.colors.axisLine
                }
            },
            axisTick: {
                show: false
            },
            axisLabel: {
                color: chartConfig.colors.axisLabel,
                fontFamily: chartConfig.fonts.font,
                fontSize: chartConfig.fonts.axisLabelFontSize,
                fontWeight: Number(chartConfig.fonts.axisLabelFontWeight)
            }
        },
        series: [
            {
                name: dataSetLabel,
                type: "bar",
                data: dataSet.length > 20 ? dataSet.slice(0, 20) : dataSet,
                itemStyle: {
                    color: chartConfig.colors.primary
                },
                label: {
                    show: enableDataLabel,
                    position: "top",
                    color: "black",
                    fontSize: 12,
                    formatter: function (params: { value: number }) {
                        const param = Array.isArray(params)
                            ? params[0]
                            : params;
                        const value = param?.value ?? "";
                        return Math.round(value).toString();
                    }
                },
                barWidth: "40%",
                barMinHeight: 2,
                emphasis: {
                    disabled: true
                }
            }
        ] as SeriesOption[]
    };

    return (
        <div id={id} ref={containerRef} style={{ width, height }}>
            {isReady && (
                <ReactECharts
                    option={option}
                    style={{ width: "100%", height: "100%" }}
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
