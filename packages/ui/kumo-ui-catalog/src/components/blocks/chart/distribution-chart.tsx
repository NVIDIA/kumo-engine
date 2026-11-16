// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Checkbox } from "@/components/ui/checkbox";
import { useEChartsReady } from "@/hooks/use-echarts-ready";
import { useChartConfig } from "@/hooks/use-chart-config";
import { getKeys, getValues } from "@/lib/utils";
import { lineTypes } from "@/types/chart";
import ReactECharts from "echarts-for-react";
import type { CallbackDataParams as TooltipParams } from "echarts/types/dist/shared";

import { useCallback, useMemo, useRef, useState } from "react";

const MAX_NUM_CLASSES = 10;
const sum = (arr: Array<number>) =>
    arr?.filter((a) => a).reduce((a, b) => a + b, 0);

interface DistributionChartProps {
    id: string;
    title: string;
    xAxisLabel: string;
    yAxisLabel: string;
    chartData: Record<string, (number | null)[]>;
    xAxisData: string[];
    lineChart?: boolean;
}
export const DistributionChart = ({
    id,
    title,
    xAxisLabel,
    yAxisLabel,
    chartData,
    xAxisData,
    lineChart = false
}: DistributionChartProps) => {
    const chartConfig = useChartConfig();
    const { isReady, containerRef } = useEChartsReady();
    // Color options when color and line type combinations
    // are used in line charts.
    const colorOptions = useMemo(() => {
        return [
            chartConfig.colors.primary,
            chartConfig.colors.lime400, // tertiary
            chartConfig.colors.ideal,
            chartConfig.colors.primaryLight
        ];
    }, [
        chartConfig.colors.ideal,
        chartConfig.colors.lime400,
        chartConfig.colors.primary,
        chartConfig.colors.primaryLight
    ]);

    // Color options when only colors are used in bar charts.
    const largeColorOptions = useMemo(() => {
        return [
            chartConfig.colors.primary,
            chartConfig.colors.lime400, //tertiary
            chartConfig.colors.ideal,
            chartConfig.colors.primaryLight,
            chartConfig.colors.amber800, //brown
            chartConfig.colors.lime700, //olive
            chartConfig.colors.red600,
            chartConfig.colors.orange400,
            chartConfig.colors.yellow300,
            chartConfig.colors.green600,
            chartConfig.colors.blue600,
            chartConfig.colors.fuschia700
        ];
    }, [
        chartConfig.colors.amber800,
        chartConfig.colors.blue600,
        chartConfig.colors.fuschia700,
        chartConfig.colors.green600,
        chartConfig.colors.ideal,
        chartConfig.colors.lime400,
        chartConfig.colors.lime700,
        chartConfig.colors.orange400,
        chartConfig.colors.primary,
        chartConfig.colors.primaryLight,
        chartConfig.colors.red600,
        chartConfig.colors.yellow300
    ]);

    const keyIndex = useCallback(
        (i: number) => Math.floor(i / colorOptions.length) % lineTypes.length,
        [colorOptions.length]
    );

    const chartRef = useRef<ReactECharts>(null);
    const chartDataNew = { ...chartData };
    // Limit the number of classes we show
    const chartDataArray = getKeys(chartData).map((key) => ({
        name: key,
        vals: chartData[key]
    }));
    // Sort to get the largest classes at the beginning of the array
    chartDataArray?.sort(
        (a, b) => sum(b?.vals as number[]) - sum(a?.vals as number[])
    );
    const bottomClasses = chartDataArray?.slice(MAX_NUM_CLASSES);

    // If we need to collapse some classes to "Other"
    if (!lineChart && MAX_NUM_CLASSES < getKeys(chartDataNew).length) {
        chartDataNew["Other"] = [];
        // Populate the "Other" class
        xAxisData.forEach((_, i) => {
            chartDataNew["Other"].push(
                sum(bottomClasses?.map(({ vals }) => vals[i] as number))
            );
        });
        // Remove all the bottomClassess from the dataset
        getKeys(chartDataNew).forEach((key) => {
            if (bottomClasses?.map(({ name }) => name).includes(key)) {
                delete chartDataNew[key];
            }
        });
    }

    const rawData: (number | null)[][] = getValues(chartDataNew);
    const keyData: (string | null)[] = getKeys(chartDataNew);

    const displayName = (key: string | null) =>
        !key || key.trim() === "" || key === null ? "<MISSING>" : key;

    //Hide these plots for line chart only.
    const hiddenPlots = ["max", "min", "median", "mode"];

    const [selectedSeries, setSelectedSeries] = useState<
        Record<string, boolean>
    >(
        Object.fromEntries(
            keyData?.map((key) => {
                const name = displayName(key);
                const isHidden = lineChart && hiddenPlots?.includes(name);
                return [name, !isHidden];
            })
        )
    );

    const toggleSeries = (key: string) => {
        const name = displayName(key);
        const next = !selectedSeries[name];
        setSelectedSeries((prev) => ({ ...prev, [name]: next }));
        const chart = chartRef?.current?.getEchartsInstance?.();

        if (!chart) return;
        chart.dispatchAction({
            type: next ? "legendSelect" : "legendUnSelect",
            name
        });
    };

    // This filters out deselected series before creating chart option
    const series = useMemo(() => {
        return keyData
            .map((key, index) => ({ key, index }))
            .filter(({ key }) => selectedSeries?.[displayName(key)])
            .map(({ key, index }) => ({
                name: displayName(key),
                type: lineChart ? "line" : "bar",
                connectNulls: lineChart, // connect nulls for line charts only
                symbol: lineChart ? "circle" : "none",
                z: index, // ensures consistent layering
                lineStyle: lineChart
                    ? {
                          type: lineTypes[keyIndex(index)],
                          color: colorOptions[index % colorOptions.length],
                          width: 2
                      }
                    : {},
                stack: lineChart ? undefined : "total", // <-- only stack bars
                barWidth: lineChart ? undefined : "40%",
                data: rawData[index],
                itemStyle: {
                    color: lineChart
                        ? colorOptions[index % colorOptions.length]
                        : largeColorOptions[index % largeColorOptions.length]
                }
            }));
    }, [
        colorOptions,
        keyData,
        keyIndex,
        largeColorOptions,
        lineChart,
        rawData,
        selectedSeries
    ]);

    // To handle '<' and '>' in "<MISSING>" string which is considered as html tags
    // in the tooltip formatter function.
    const escapeHtml = (str: string) =>
        str?.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

    const option = useMemo(
        () => ({
            title: {
                text: title,
                left: "center",
                top: 10,
                textStyle: {
                    fontSize: 16,
                    fontWeight: 500
                }
            },
            legend: {
                show: false
            },
            grid: {
                left: 70,
                right: 20,
                top: 60,
                bottom: 100,
                containLabel: true
            },
            yAxis: {
                type: "value",
                name: yAxisLabel,
                nameLocation: "middle",
                nameGap: 50,
                nameTextStyle: {
                    fontSize: chartConfig.fonts.axisNameFontSize,
                    fontWeight: chartConfig.fonts.axisNameFontWeight,
                    fontFamily: chartConfig.fonts.font,
                    color: chartConfig.colors.axisName
                },

                axisLine: {
                    show: true // This enables the y-axis line
                },
                axisLabel: {
                    color: chartConfig.colors.axisLabel,
                    fontFamily: chartConfig.fonts.font,
                    fontSize: chartConfig.fonts.axisLabelFontSize,
                    fontWeight: Number(chartConfig.fonts.axisLabelFontWeight),
                    formatter: (value: number) => {
                        if (value >= 1_000_000)
                            return (value / 1_000_000).toFixed(1) + "M";
                        if (value >= 1_000)
                            return (value / 1_000).toFixed(1) + "K";
                        return value.toString();
                    }
                }
            },
            xAxis: {
                type: "category",
                boundaryGap: !lineChart, // disable gap for line charts only
                name: xAxisLabel, // <-- X-axis label
                nameLocation: "middle",
                nameGap: 60,
                nameTextStyle: {
                    fontSize: chartConfig.fonts.axisNameFontSize,
                    fontWeight: chartConfig.fonts.axisNameFontWeight,
                    fontFamily: chartConfig.fonts.font,
                    color: chartConfig.colors.axisName
                },
                data: xAxisData,
                axisTick: {
                    alignWithLabel: true
                },
                axisLabel: {
                    color: chartConfig.colors.axisLabel,
                    fontFamily: chartConfig.fonts.font,
                    fontSize: chartConfig.fonts.axisLabelFontSize,
                    fontWeight: Number(chartConfig.fonts.axisLabelFontWeight),
                    formatter: (val: string) => {
                        if (val?.includes(" - ")) {
                            const [start, end] = val.split(" - ");
                            return `${start}\nto\n${end}`; // Clean vertical layout
                        }
                        return val;
                    }
                }
            },
            series: series,
            tooltip: {
                trigger: "item",
                axisPointer: {
                    type: lineChart ? "line" : "shadow"
                },
                //position: "top",
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
                formatter: (params: TooltipParams) => {
                    const p = params;
                    const name = escapeHtml(
                        displayName(p?.seriesName as string)
                    );
                    const value =
                        p?.value != null && typeof p?.value === "number"
                            ? p?.value.toLocaleString()
                            : "N/A";
                    return `
                        <strong>Time Range: ${p?.name}</strong><br/>
                        <span class="inline-block mr-1  w-2.5 h-2.5" style="background-color:${p?.color};"></span>
                        ${name}: ${value}
                        `;
                }
            }
        }),
        [
            chartConfig.borders.tooltipBorderRadius,
            chartConfig.borders.tooltipBorderWidth,
            chartConfig.colors.axisLabel,
            chartConfig.colors.axisName,
            chartConfig.colors.tooltipBackground,
            chartConfig.colors.tooltipBorderColor,
            chartConfig.colors.tooltipForeground,
            chartConfig.fonts.axisLabelFontSize,
            chartConfig.fonts.axisLabelFontWeight,
            chartConfig.fonts.axisNameFontSize,
            chartConfig.fonts.axisNameFontWeight,
            chartConfig.fonts.font,
            chartConfig.fonts.tooltipFontSize,
            chartConfig.fonts.tooltipFontWeight,
            chartConfig.paddings.tooltipPaddingX,
            chartConfig.paddings.tooltipPaddingY,
            lineChart,
            series,
            title,
            xAxisData,
            xAxisLabel,
            yAxisLabel
        ]
    );

    return (
        <div id={id} className="flex h-[80vh] flex-row-reverse p-4">
            <div className="flex flex-col items-start gap-2 pt-16">
                {keyData.map((key, index) => {
                    const name = displayName(key);
                    return (
                        <label
                            key={"legend-" + key}
                            className="flex items-center gap-2 whitespace-nowrap text-sm">
                            <Checkbox
                                id={`${id}-legend-${index}`}
                                data-testid={"legend-checkbox-" + index}
                                checked={selectedSeries[name]}
                                onCheckedChange={() =>
                                    toggleSeries(key as string)
                                }
                            />

                            {lineChart ? (
                                <span
                                    className="relative inline-block h-0.5 w-6"
                                    aria-hidden="true"
                                    style={{
                                        backgroundColor: "transparent",
                                        borderTop: `2px ${lineTypes[keyIndex(index)]} ${colorOptions[index % colorOptions.length]}`
                                    }}>
                                    <span
                                        className="absolute left-1/2 top-1/2 h-2 w-2 rounded-full"
                                        style={{
                                            backgroundColor:
                                                colorOptions[
                                                    index % colorOptions.length
                                                ],
                                            transform: "translate(-50%, -60%)"
                                        }}
                                    />
                                </span>
                            ) : (
                                <span
                                    className="inline-block h-3 w-3"
                                    aria-hidden="true"
                                    style={{
                                        backgroundColor: lineChart
                                            ? colorOptions[
                                                  index % colorOptions.length
                                              ]
                                            : largeColorOptions[
                                                  index %
                                                      largeColorOptions.length
                                              ]
                                    }}
                                />
                            )}
                            <span
                                className={
                                    name === "<MISSING>" ? "text-slate-400" : ""
                                }>
                                {name}
                            </span>
                        </label>
                    );
                })}
            </div>
            <div ref={containerRef} className="w-full">
                {isReady && (
                    <ReactECharts
                        ref={chartRef}
                        option={option}
                        style={{ height: "100%", width: "100%" }}
                        notMerge={true}
                        lazyUpdate={true}
                        opts={{
                            renderer: "svg"
                        }}
                    />
                )}
            </div>
        </div>
    );
};
