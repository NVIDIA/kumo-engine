// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useMemo, useCallback } from "react";
import { useEChartsReady } from "@/hooks/use-echarts-ready";
import ReactECharts from "echarts-for-react";
import { SeriesOption } from "echarts";
import { hasOnlyZeroOnePoints, isRocCurve } from "@/utils/chart";
import { ScatterLineData, ScatterPointData } from "@/types/chart";
import { useChartConfig } from "@/hooks/use-chart-config";

interface ScatterPointLineProps {
    id: string;
    scatterPointData: Array<ScatterPointData>;
    scatterLineData: Array<ScatterLineData>;
    currentIndex: number;
    xAxisLabel: string;
    yAxisLabel: string;
    width?: string | number;
    height?: string | number;
}

export const ScatterPointLineChart: React.FC<ScatterPointLineProps> = ({
    id,
    scatterPointData,
    scatterLineData,
    currentIndex,
    xAxisLabel = "",
    yAxisLabel = "",
    width = "400px",
    height = "350px"
}) => {
    const chartConfig = useChartConfig();
    const { isReady, containerRef } = useEChartsReady();
    let showSteppedLine = true;

    // If the chart is a ROC curve and the points are only
    // (0,0)s and (1,1)s, we cannot make the chart stepped
    // and show the line because it will be a horizontal
    // line with at y = 1 or same with the random guess line.
    if (isRocCurve(xAxisLabel, yAxisLabel)) {
        // Convert the x and y values to numbers as now
        // they are strings
        for (let i = 0; i < scatterPointData.length; i++) {
            scatterPointData[i].x = Number(scatterPointData[i].x);
            scatterPointData[i].y = Number(scatterPointData[i].y);
        }
        if (hasOnlyZeroOnePoints(scatterPointData)) {
            showSteppedLine = false;
        }
    }

    // Convert data to ECharts format
    const chartData = useMemo(() => {
        return scatterPointData.map(
            (point: { x: number; y: number }, index: number) => [
                point.x,
                point.y,
                index
            ]
        );
    }, [scatterPointData]);

    // Create series with markLine and markPoint for crosshairs and highlighting
    const createSeries = useCallback(() => {
        const series: SeriesOption[] = [
            {
                type: "line",
                data: showSteppedLine ? chartData : [],
                step: "end",
                showSymbol: false,
                lineStyle: {
                    color: chartConfig.colors.primary,
                    width: 1.5
                },
                emphasis: {
                    disabled: true
                }
            }
        ];

        // Add custom lines as separate line series
        scatterLineData.forEach((line) => {
            // Normalize coordinates from 0-100 range to 0-1 range
            const normalizedCoordinates = line.coordinates.map((coord) => [
                coord.x / 100,
                coord.y / 100
            ]);

            series.push({
                type: "line",
                data: normalizedCoordinates,
                showSymbol: false,
                lineStyle: {
                    color: line.lineColor,
                    width: line?.lineWidth || 1
                },
                emphasis: {
                    disabled: true
                }
            });
        });

        // Add crosshairs and highlight for current point
        if (currentIndex >= 0 && currentIndex < scatterPointData.length) {
            const currentPoint = scatterPointData[currentIndex];
            const pointerX = parseFloat(currentPoint.x.toString());
            const pointerY = parseFloat(currentPoint.y.toString());

            if (pointerX >= 0 && pointerY >= 0) {
                series[0].markLine = {
                    silent: true,
                    symbol: "none",
                    lineStyle: {
                        color: "grey",
                        width: 0.7,
                        type: "solid"
                    },
                    data: [
                        [
                            { coord: [0, pointerY] },
                            { coord: [pointerX, pointerY] }
                        ],
                        [
                            { coord: [pointerX, 0] },
                            { coord: [pointerX, pointerY] }
                        ]
                    ]
                };

                series[0].markPoint = {
                    silent: true,
                    symbol: "circle",
                    symbolSize: 7,
                    itemStyle: {
                        color: "white",
                        borderColor: "black",
                        borderWidth: 1
                    },
                    label: {
                        show: true,
                        position: "top",
                        formatter: () => {
                            return `(${pointerX} , ${pointerY})`;
                        },
                        color: chartConfig.colors.primary,
                        fontSize: chartConfig.fonts.axisLabelFontSize,
                        offset: [4, 4]
                    },
                    data: [{ coord: [pointerX, pointerY] }]
                };
            }
        }

        return series;
    }, [
        showSteppedLine,
        chartData,
        chartConfig.colors.primary,
        chartConfig.fonts.axisLabelFontSize,
        scatterLineData,
        currentIndex,
        scatterPointData
    ]);

    const option = useMemo(() => {
        return {
            grid: {
                height: "80%",
                width: "70%",
                top: "5%",
                left: "12%",
                bottom: "2%",
                backgroundColor: "transparent"
            },
            xAxis: {
                type: "value",
                name: xAxisLabel,
                nameLocation: "middle",
                nameGap: 30,
                nameTextStyle: {
                    color: chartConfig.colors.axisName,
                    fontFamily: chartConfig.fonts.font,
                    fontSize: chartConfig.fonts.axisNameFontSize,
                    fontWeight: Number(chartConfig.fonts.axisNameFontWeight)
                },
                min: 0,
                max: 1,
                axisLine: {
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
                },
                splitLine: {
                    show: false
                }
            },
            yAxis: {
                type: "value",
                name: yAxisLabel,
                nameLocation: "middle",
                nameGap: 30,
                nameTextStyle: {
                    color: chartConfig.colors.axisName,
                    fontFamily: chartConfig.fonts.font,
                    fontSize: chartConfig.fonts.axisNameFontSize,
                    fontWeight: Number(chartConfig.fonts.axisNameFontWeight)
                },
                min: 0,
                max: 1,
                axisLine: {
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
                },
                splitLine: {
                    show: false
                }
            },
            series: createSeries(),
            tooltip: {
                show: false
            },
            legend: {
                show: false
            },
            animation: false
        };
    }, [
        xAxisLabel,
        chartConfig.colors.axisName,
        chartConfig.colors.axisLine,
        chartConfig.colors.axisLabel,
        chartConfig.fonts.font,
        chartConfig.fonts.axisNameFontSize,
        chartConfig.fonts.axisNameFontWeight,
        chartConfig.fonts.axisLabelFontSize,
        chartConfig.fonts.axisLabelFontWeight,
        yAxisLabel,
        createSeries
    ]);

    return (
        <div id={id} ref={containerRef} style={{ width, height }}>
            {isReady && (
                <ReactECharts
                    option={option}
                    style={{
                        width: width,
                        height: height,
                        minWidth: 400,
                        minHeight: 330
                    }}
                    opts={{
                        renderer: "svg"
                    }}
                />
            )}
        </div>
    );
};
