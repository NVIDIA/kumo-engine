// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React from "react";
import { useEChartsReady } from "@/hooks/use-echarts-ready";
import ReactECharts from "echarts-for-react";
import type {
    EChartsOption,
    SeriesOption,
    VisualMapComponentOption
} from "echarts";
import { HeatMapData, HeatMapDataPoint, HeatMapLineData } from "@/types/chart";
import { useChartConfig } from "@/hooks/use-chart-config";

export interface HeatMapChartProps {
    id: string;
    heatmapData: HeatMapData;
    heatmapLineData: HeatMapLineData;
    xAxisLabel: string;
    yAxisLabel: string;
    width?: string | number;
    height?: string | number;
    visualMapWidth?: number;
    visualMapHeight?: number;
}

export const HeatMapChart: React.FC<HeatMapChartProps> = ({
    id,
    heatmapData,
    heatmapLineData,
    xAxisLabel,
    yAxisLabel,
    width = "800px",
    height = "600px",
    visualMapWidth = 450,
    visualMapHeight = 30
}) => {
    const chartConfig = useChartConfig();
    const { isReady, containerRef } = useEChartsReady();
    const colorsScale = [
        chartConfig.colors.secondary,
        chartConfig.colors.primary
    ];

    // Convert heat map data to ECharts heatmap format
    const heatmapSeriesData: HeatMapDataPoint[] = [];
    for (let yIndex = 0; yIndex < heatmapData.y.length; yIndex++) {
        for (let xIndex = 0; xIndex < heatmapData.x.length; xIndex++) {
            if (
                heatmapData.z[yIndex] &&
                heatmapData.z[yIndex][xIndex] !== undefined
            ) {
                const value = heatmapData.z[yIndex][xIndex];
                if (value === 0) {
                    // Add the cell but with a special marker
                    heatmapSeriesData.push({
                        value: [xIndex, yIndex, value],
                        itemStyle: {
                            color: "rgba(0, 0, 0, 0)",
                            borderColor: "rgba(0, 0, 0, 0)"
                        }
                    });
                } else {
                    heatmapSeriesData.push([xIndex, yIndex, value]);
                }
            }
        }
    }

    // Convert diagonal line data to ECharts format
    const diagonalSeriesData: [number | string, number | string][] = [];

    const lineLength = Math.min(
        heatmapLineData.x.length,
        heatmapLineData.y.length
    );
    for (let i = 0; i < lineLength; i++) {
        const xValue = heatmapLineData.x[i];
        const yValue = heatmapLineData.y[i];

        const xIndex = heatmapData.x.findIndex(
            (val) => Math.abs(Number(val) - xValue) < 0.0001
        );
        const yIndex = heatmapData.y.findIndex(
            (val) => Math.abs(Number(val) - yValue) < 0.0001
        );

        if (xIndex !== -1 && yIndex !== -1) {
            diagonalSeriesData.push([xIndex, yIndex]);
        }
    }

    // Find min and max values for visualMap
    const flatZ: number[] = heatmapData.z.flat();
    const validValues = flatZ.filter(
        (v) => !isNaN(v) && v !== null && v !== undefined
    );

    const minValue: number = Math.min(...validValues);
    const maxValue: number = Math.max(...validValues);

    // Create visualMap configuration
    const visualMapConfig: VisualMapComponentOption = {
        min: minValue,
        max: maxValue,
        type: "continuous",
        calculable: false,
        text: [String(maxValue), String(minValue)],
        orient: "vertical",
        top: "0",
        right: "20",
        itemHeight: visualMapWidth,
        itemWidth: visualMapHeight,
        seriesIndex: [0],
        textStyle: {
            color: chartConfig.colors.axisName,
            fontFamily: chartConfig.fonts.font,
            fontSize: chartConfig.fonts.axisNameFontSize,
            fontWeight: Number(chartConfig.fonts.axisNameFontWeight)
        },
        inRange: {
            color: colorsScale
        }
    };

    const formatLabel = (val: string | number): string => {
        if (typeof val === "number") {
            return val.toFixed(3);
        }
        return String(val);
    };

    const option: EChartsOption = {
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
                if (params.seriesType === "heatmap") {
                    let xIndex: number, yIndex: number, value: number;

                    if (params.data.value) {
                        xIndex = params.data.value[0];
                        yIndex = params.data.value[1];
                        value = params.data.value[2];
                    } else {
                        xIndex = params.data[0];
                        yIndex = params.data[1];
                        value = params.data[2];
                    }

                    const xLabel = heatmapData.x[xIndex];
                    const yLabel = heatmapData.y[yIndex];
                    return `X: ${formatLabel(xLabel)}, Y: ${formatLabel(yLabel)}, Z: ${value}`;
                }
                const xIndex = params.data[0];
                const yIndex = params.data[1];
                const xVal = heatmapData.x[xIndex];
                const yVal = heatmapData.y[yIndex];
                return `${formatLabel(xVal)}, ${formatLabel(yVal)}`;
            }
        },
        grid: {
            height: "85%",
            width: "80%",
            top: "0%",
            left: "10%",
            backgroundColor: "transparent"
        },
        xAxis: {
            type: "category",
            name: xAxisLabel,
            nameGap: 60,
            nameLocation: "middle",
            nameTextStyle: {
                color: chartConfig.colors.axisName,
                fontFamily: chartConfig.fonts.font,
                fontSize: chartConfig.fonts.axisNameFontSize,
                fontWeight: Number(chartConfig.fonts.axisNameFontWeight)
            },
            data: heatmapData.x,
            axisLine: {
                lineStyle: {
                    color: chartConfig.colors.axisLine
                }
            },
            axisTick: {
                show: false
            },
            axisLabel: {
                interval: Math.floor(heatmapData.x.length / 30),
                rotate: heatmapData.x.length > 15 ? 45 : 0,
                color: chartConfig.colors.axisLabel,
                fontFamily: chartConfig.fonts.font,
                fontSize: chartConfig.fonts.axisLabelFontSize,
                fontWeight: Number(chartConfig.fonts.axisLabelFontWeight)
            }
        },
        yAxis: {
            type: "category",
            name: yAxisLabel,
            nameGap: 60,
            nameLocation: "middle",
            nameTextStyle: {
                color: chartConfig.colors.axisName,
                fontFamily: chartConfig.fonts.font,
                fontSize: chartConfig.fonts.axisNameFontSize,
                fontWeight: Number(chartConfig.fonts.axisNameFontWeight)
            },
            data: heatmapData.y,
            axisLine: {
                lineStyle: {
                    color: chartConfig.colors.axisLine
                }
            },
            axisTick: {
                show: false
            },
            axisLabel: {
                interval: Math.floor(heatmapData.y.length / 30),
                color: chartConfig.colors.axisLabel,
                fontFamily: chartConfig.fonts.font,
                fontSize: chartConfig.fonts.axisLabelFontSize,
                fontWeight: Number(chartConfig.fonts.axisLabelFontWeight)
            }
        },
        backgroundColor: "transparent",
        visualMap: visualMapConfig,
        series: [
            {
                name: "Heatmap",
                type: "heatmap",
                data: heatmapSeriesData,
                emphasis: {
                    itemStyle: {
                        shadowBlur: 10,
                        shadowColor: "rgba(0, 0, 0, 0.5)"
                    }
                },
                z: 0
            },
            {
                name: "Diagonal Line",
                type: "line",
                data: diagonalSeriesData,
                coordinateSystem: "cartesian2d",
                lineStyle: {
                    color: chartConfig.colors.ideal,
                    width: 3,
                    type: "solid"
                },
                itemStyle: {
                    color: chartConfig.colors.ideal,
                    borderColor: chartConfig.colors.ideal,
                    borderWidth: 1
                },
                symbol: "circle",
                symbolSize: 4,
                z: 1
            }
        ] as SeriesOption[]
    };

    return (
        <div id={id} ref={containerRef} style={{ width, height }}>
            {isReady && (
                <ReactECharts
                    option={option}
                    style={{
                        height: height,
                        width: width
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
