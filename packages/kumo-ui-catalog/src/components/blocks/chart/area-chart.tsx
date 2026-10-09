// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useChartConfig } from "@/hooks/use-chart-config";
import { useEChartsReady } from "@/hooks/use-echarts-ready";
import { parseNumbers } from "@/utils/helper";
import type { EChartsOption, SeriesOption } from "echarts";
import ReactECharts from "echarts-for-react";

export interface AreaChartProps {
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
    customToolTip?: boolean;
    enableDataLabel?: boolean;
    customLabel?: boolean;
    toolTipText?: string;
}

export const AreaChart: React.FC<AreaChartProps> = ({
    id,
    title = "",
    xAxisLabel = "",
    yAxisLabel = "",
    dataSet,
    dataSetLabel = "",
    yAxisMin,
    yAxisMax,
    step,
    height = "520px",
    width = "580px",
    xAxisTickLabels = [],
    legend = {},
    customToolTip = false,
    toolTipText = ""
}) => {
    const chartConfig = useChartConfig();
    const { isReady, containerRef } = useEChartsReady();
    const option: EChartsOption = {
        title: {
            text: title,
            left: "center",
            top: 20
        },
        grid: {
            left: 30,
            right: 20,
            bottom: xAxisLabel ? 30 : 10,
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
            trigger: "item",
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
            axisPointer: {
                type: "none"
            },
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            formatter: function (params: any) {
                const param = Array.isArray(params) ? params[0] : params;
                const label = param?.name ?? "";
                const value = param?.value ?? "";
                let backgroundColor = "";
                const borderColor = chartConfig.colors.primaryDark;
                if (param.seriesName === dataSetLabel) {
                    backgroundColor = "rgba(19, 78, 74, 0.3)";
                }

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
                                    background-color:${backgroundColor};
                                    border: 2px solid ${borderColor};
                                "></span>
                                <span>${tooltipLabel}</span>
                            </div>
                        </div>`;
            }
        },
        animation: false,
        xAxis: {
            type: "category",
            name: xAxisLabel,
            nameLocation: "middle",
            nameTextStyle: {
                color: chartConfig.colors.axisName,
                fontFamily: chartConfig.fonts.font,
                fontSize: chartConfig.fonts.axisNameFontSize,
                fontWeight: Number(chartConfig.fonts.axisNameFontWeight)
            },
            nameGap: 30,
            data: xAxisTickLabels,
            axisTick: {
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
                rotate: xAxisTickLabels?.length > 18 ? 45 : 0,
                color: chartConfig.colors.axisLabel,
                fontFamily: chartConfig.fonts.font,
                fontSize: chartConfig.fonts.axisLabelFontSize,
                fontWeight: Number(chartConfig.fonts.axisLabelFontWeight)
            },
            boundaryGap: false
        },
        yAxis: {
            type: "value",
            name: yAxisLabel,
            nameLocation: "middle",
            nameGap: 25,
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
            splitLine: {
                show: true
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
                type: "line",
                data: dataSet,
                smooth: true,
                symbol: "circle",
                symbolSize: 5,
                lineStyle: {
                    color: chartConfig.colors.primaryDark,
                    width: 3
                },
                itemStyle: {
                    color: chartConfig.colors.primary
                },
                areaStyle: {
                    color: chartConfig.colors.primary,
                    opacity: 0.3
                },
                label: {
                    show: false
                },
                emphasis: {
                    focus: "series",
                    scale: true,
                    itemStyle: {
                        borderWidth: 3,
                        borderColor: chartConfig.colors.primaryDark
                    }
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
