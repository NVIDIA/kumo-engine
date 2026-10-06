// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React from "react";
import ReactECharts from "echarts-for-react";
import { useEChartsReady } from "@/hooks/use-echarts-ready";
import { parseNumbers } from "@/utils/helper";
import {
    Pagination,
    PaginationContent,
    PaginationItem,
    PaginationLink,
    PaginationNext,
    PaginationPrevious
} from "@/components/ui/pagination";
import { CardFooter } from "@/components/ui/card";
import { useChartConfig } from "@/hooks/use-chart-config";

interface DivergingBarChartProps {
    id: string;
    xAxisLabels: (string | number)[];
    dataSetTruth: (string | number)[];
    dataSetPred: (string | number)[];
    populationRatio?: (string | number)[];
    globalAvgScore: number;
    totalRecords: number;
    pageSize: number;
    pageIndex: number;
    onGotoPage: (page: number) => void;
    dropdown_menus?: React.ReactNode;
}

export const DivergingBarChart: React.FC<DivergingBarChartProps> = ({
    id,
    xAxisLabels = [],
    dataSetTruth,
    dataSetPred,
    populationRatio = [],
    globalAvgScore = 0.0,
    totalRecords = 0,
    pageSize = 0,
    pageIndex = 0,
    onGotoPage = () => {},
    dropdown_menus
}) => {
    const chartConfig = useChartConfig();
    const { isReady, containerRef } = useEChartsReady();
    const truth = dataSetTruth?.map((num) => Number(num) - globalAvgScore);
    const pred = dataSetPred?.map((num) => Number(num) - globalAvgScore);

    let rawMin = 0;
    let rawMax = 0;

    truth?.forEach((value, index) => {
        rawMin = Math.min(rawMin, value, pred[index]);
        rawMax = Math.max(rawMax, value, pred[index]);
    });

    // Inline "nice scale" calculation (like ECharts)
    const ticks = 5;
    const range = rawMax - rawMin;

    const safeRange = range <= 0 ? 1 : range;
    const rawInterval = safeRange / ticks;
    const safeRawInterval = rawInterval <= 0 ? 1 : rawInterval;

    const magnitude = Math.pow(10, Math.floor(Math.log10(safeRawInterval)));
    const norm = safeRawInterval / magnitude;

    let interval: number;
    if (norm <= 1) {
        interval = 1;
    } else if (norm <= 2) {
        interval = 2;
    } else if (norm <= 5) {
        interval = 5;
    } else {
        interval = 10;
    }
    interval = interval * magnitude;

    const min = Math.floor(rawMin / interval) * interval;
    const max = Math.ceil(rawMax / interval) * interval;
    const labelLength = xAxisLabels
        ? Math.max(...xAxisLabels.map((l) => l?.toString()?.length))
        : 1;

    // Prepare data for ECharts
    const reversedLabels = [...xAxisLabels].reverse();
    const reversedTruthData = [...truth].reverse();
    const reversedPredData = [...pred].reverse();
    const reversedPopulation = [...populationRatio].reverse();

    const yAxisData = reversedLabels.map((_label, index) => {
        return `category_${index}`;
    });

    // Generate horizontal lines (above first bar and between all bars)
    const horizontalLines = [];
    for (let i = -1; i < yAxisData.length - 1; i++) {
        horizontalLines.push({
            yAxis: i + 0.5,
            lineStyle: {
                color: chartConfig.colors.horizontalLine,
                width: 1,
                type: "solid"
            }
        });
    }

    const option = {
        grid: {
            left: Math.max(Math.min(labelLength * 10, 60), 30),
            right: 30,
            bottom: 90,
            top: 20,
            containLabel: true
        },
        xAxis: {
            min: min,
            max: max,
            interval: interval,
            type: "value",
            axisLine: {
                show: false
            },
            axisTick: {
                show: false
            },
            axisLabel: {
                formatter: (value: number) => {
                    return parseNumbers(value + globalAvgScore, {
                        decimals: 1
                    });
                },
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
            type: "category",
            data: yAxisData,
            axisLine: {
                show: false
            },
            axisTick: {
                show: false
            },
            axisLabel: {
                show: true,
                color: chartConfig.colors.axisLabel,
                // eslint-disable-next-line @typescript-eslint/no-unused-vars
                formatter: (value: string, _index: number) => {
                    const actualIndex = parseInt(value.split("_")[1]);
                    const label = reversedLabels[actualIndex];
                    const ratio = reversedPopulation[actualIndex];

                    const populationValue = parseFloat(ratio as string);
                    let fractionPopulation;
                    if (Number.isInteger(populationValue)) {
                        fractionPopulation = `${Math.floor(populationValue)}%`;
                    } else {
                        fractionPopulation = `${populationValue.toFixed(2)}%`;
                    }

                    const labelStr = label?.toString();
                    const truncatedLabel =
                        labelStr?.length > 24
                            ? labelStr.slice(0, 23) + "..."
                            : labelStr;

                    return `{label|${truncatedLabel}}\n{ratio|${fractionPopulation}}`;
                },
                rich: {
                    label: {
                        color: chartConfig.colors.axisLabel,
                        lineHeight: 16,
                        fontFamily: chartConfig.fonts.font,
                        fontSize: chartConfig.fonts.axisLabelFontSize,
                        fontWeight: Number(
                            chartConfig.fonts.axisLabelFontWeight
                        )
                    },
                    ratio: {
                        color: chartConfig.colors.axisLabel,
                        lineHeight: 16,
                        fontFamily: chartConfig.fonts.font,
                        fontSize: chartConfig.fonts.axisLabelFontSize,
                        fontWeight: Number(
                            chartConfig.fonts.axisLabelFontWeight
                        )
                    }
                }
            }
        },
        series: [
            {
                name: "Truth",
                type: "bar",
                barGap: 0,
                barCategoryGap: "30%",
                barMaxWidth: 30,
                data: reversedTruthData.map((value, index) => ({
                    value: value,
                    itemStyle: {
                        color:
                            value < 0
                                ? chartConfig.colors.violet300
                                : chartConfig.colors.rose300
                    },
                    originalValue: dataSetTruth[dataSetTruth.length - 1 - index]
                })),
                markLine: {
                    symbol: "none",
                    label: {
                        show: false
                    },
                    lineStyle: {
                        color: chartConfig.colors.axisLine,
                        width: 0.5,
                        type: "solid"
                    },
                    data: [
                        {
                            xAxis: 0,
                            label: {
                                show: true,
                                position: "start",
                                formatter: "population avg. score",
                                rotate: 30,
                                offset: [-60, 20]
                            }
                        }, // Central line at global average with label
                        { xAxis: min } // Line at minimum value
                    ],
                    silent: true
                }
            },
            {
                name: "Prediction",
                type: "bar",
                barGap: 0,
                barCategoryGap: "30%",
                barMaxWidth: 30,
                data: reversedPredData.map((value, index) => ({
                    value: value,
                    itemStyle: {
                        color:
                            value < 0
                                ? chartConfig.colors.violet500
                                : chartConfig.colors.rose400
                    },
                    originalValue: dataSetPred[dataSetPred.length - 1 - index]
                }))
            },
            // Add horizontal lines between bars
            {
                name: "HorizontalLines",
                type: "line",
                data: [],
                markLine: {
                    symbol: "none",
                    label: {
                        show: false
                    },
                    data: horizontalLines,
                    silent: true
                }
            }
        ],
        tooltip: {
            trigger: "item",
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
                const seriesName = params.seriesName;
                const originalValue = params.data.originalValue;
                const color = params.color;
                const dataIndex = params.dataIndex;
                const categoryName = reversedLabels[dataIndex];

                return `
                    <div style="display: flex; align-items: center;">
                        <div style="width: 12px; height: 12px; background-color: ${color}; margin-right: 8px;"></div>
                        ${seriesName} - ${categoryName}: <strong style="margin-left: 4px;">${parseNumbers(originalValue)}</strong>
                    </div>
                `;
            }
        }
    };

    const chartHeight =
        xAxisLabels?.length < 2
            ? 240
            : xAxisLabels?.length <= 5
              ? 480
              : xAxisLabels?.length <= 8
                ? 740
                : xAxisLabels?.length * 70;

    return (
        <div>
            {dropdown_menus && (
                <div className="flex justify-end">{dropdown_menus}</div>
            )}
            <div
                ref={containerRef}
                style={{
                    minWidth: 600,
                    height: chartHeight
                }}
                data-testid={id}>
                {isReady && (
                    <ReactECharts
                        option={option}
                        style={{
                            height: chartHeight,
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
            {totalRecords > pageSize && (
                <CardFooter className="flex justify-center">
                    <Pagination>
                        <PaginationContent>
                            <PaginationItem>
                                <PaginationPrevious
                                    onClick={() => onGotoPage(-2)}
                                    className={
                                        pageIndex === 0
                                            ? "pointer-events-none opacity-50"
                                            : "cursor-pointer"
                                    }
                                />
                            </PaginationItem>

                            {Array.from({
                                length: Math.ceil(totalRecords / pageSize)
                            }).map((_, i) => (
                                <PaginationItem key={i}>
                                    <PaginationLink
                                        isActive={i === pageIndex}
                                        onClick={() => onGotoPage(i)}
                                        className={
                                            i === pageIndex
                                                ? "bg-secondary cursor-pointer rounded-sm text-white"
                                                : "cursor-pointer"
                                        }>
                                        {i + 1}
                                    </PaginationLink>
                                </PaginationItem>
                            ))}
                            <PaginationItem>
                                <PaginationNext
                                    onClick={() => onGotoPage(-1)}
                                    className={
                                        totalRecords <=
                                        (pageIndex + 1) * pageSize
                                            ? "pointer-events-none opacity-50"
                                            : "cursor-pointer"
                                    }
                                />
                            </PaginationItem>
                        </PaginationContent>
                    </Pagination>
                </CardFooter>
            )}
        </div>
    );
};
