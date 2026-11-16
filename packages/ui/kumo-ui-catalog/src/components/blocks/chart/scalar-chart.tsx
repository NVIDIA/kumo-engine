// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Checkbox } from "@/components/ui/checkbox";
import { useEChartsReady } from "@/hooks/use-echarts-ready";
import { useChartConfig } from "@/hooks/use-chart-config";
import ReactECharts from "echarts-for-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { getKeys } from "@/lib/utils";
import { lineTypes, ScalarChartData } from "@/types/chart";
import type {
    GridComponentOption,
    SeriesOption,
    TitleComponentOption,
    XAXisComponentOption,
    YAXisComponentOption
} from "echarts";
import { CallbackDataParams } from "echarts/types/dist/shared";
import { parseNumbers } from "@/utils/helper";

interface ScalarChartProps {
    id: string;
    tag: string;
    xAxisLabel: string;
    yAxisLabel: string;
    dataDict: ScalarChartData;
}

export const ScalarChart = ({
    id,
    tag,
    xAxisLabel,
    yAxisLabel,
    dataDict
}: ScalarChartProps) => {
    const chartConfig = useChartConfig();
    const { isReady, containerRef } = useEChartsReady();
    const chartRef = useRef<ReactECharts>(null);

    const colorOptions = useMemo(
        () => [
            chartConfig.colors.primary,
            chartConfig.colors.lime400,
            chartConfig.colors.ideal,
            chartConfig.colors.primaryLight
        ],
        [
            chartConfig.colors.ideal,
            chartConfig.colors.lime400,
            chartConfig.colors.primary,
            chartConfig.colors.primaryLight
        ]
    );

    const keyIndex = useCallback(
        (i: number) => Math.floor(i / colorOptions.length) % lineTypes.length,
        [colorOptions.length]
    );

    const chartKeys = getKeys(dataDict);
    const experimentKeys = Array.from(
        new Set(chartKeys.flatMap((name) => getKeys(dataDict[name])))
    ).sort((a, b) => Number(a) - Number(b));

    const legendData = experimentKeys.map(
        (k) => `Experiment ${parseInt(k, 10) + 1}`
    );

    const [selectedSeries, setSelectedSeries] = useState<
        Record<string, boolean>
    >(Object.fromEntries(legendData?.map((name) => [name, true])));

    const toggleSeries = (name: string) => {
        const next = !selectedSeries[name];
        setSelectedSeries((prev) => ({ ...prev, [name]: next }));
        const chart = chartRef?.current?.getEchartsInstance?.();
        if (!chart) return;
        chart.dispatchAction({
            type: next ? "legendSelect" : "legendUnSelect",
            name
        });
    };

    const toggleAllSeries = () => {
        const allSelected = legendData.every((name) => selectedSeries[name]);
        const newState = !allSelected;
        const newSelectedSeries = Object.fromEntries(
            legendData.map((name) => [name, newState])
        );
        setSelectedSeries(newSelectedSeries);

        const chart = chartRef?.current?.getEchartsInstance?.();
        if (!chart) return;

        legendData.forEach((name) => {
            chart.dispatchAction({
                type: newState ? "legendSelect" : "legendUnSelect",
                name
            });
        });
    };

    const allSelected = useMemo(
        () => legendData.every((name) => selectedSeries[name]),
        [legendData, selectedSeries]
    );

    useEffect(() => {
        const selectedKeys = Object.keys(selectedSeries);
        if (legendData?.length !== selectedKeys?.length) {
            const newSelected = legendData.map((name) =>
                selectedKeys.includes(name)
                    ? [name, selectedSeries[name]]
                    : [name, true]
            );
            setSelectedSeries(Object.fromEntries(newSelected));
        }
    }, [legendData, selectedSeries]);

    const option = useMemo(() => {
        const grids: GridComponentOption[] = [];
        const xAxes: XAXisComponentOption[] = [];
        const yAxes: YAXisComponentOption[] = [];
        const titles: TitleComponentOption[] = [];
        const series: SeriesOption[] = [];

        // Compute global min and max for yData. This is to ensure that
        // y-axis tick values are same for all charts.
        let globalMin = Infinity;
        let globalMax = -Infinity;
        chartKeys?.forEach((chartName) => {
            experimentKeys?.forEach((expKey) => {
                const exp = dataDict[chartName]?.[expKey];
                if (!exp) return;
                const yVals = (exp.yData || []).filter(
                    (v) => v != null
                ) as number[];
                if (!yVals.length) return;
                const localMin = Math.min(...yVals);
                const localMax = Math.max(...yVals);
                if (localMin < globalMin) globalMin = localMin;
                if (localMax > globalMax) globalMax = localMax;
            });
        });

        // Add a small margin (so lines don't touch border)
        globalMin = globalMin - 0.02;
        globalMax = globalMax + 0.02;

        // Calculate dynamic nameGap based on the maximum tick value length BEFORE creating grids
        const roundTo = (v: number, d = 3) => Number(v.toFixed(d));
        const finalMin = roundTo(globalMin, 3);
        const finalMax = roundTo(globalMax, 3);
        const maxTickValue = Math.max(Math.abs(finalMin), Math.abs(finalMax));
        const tickValueString = parseNumbers(maxTickValue) as string;
        const estimatedTickWidth = tickValueString.length * 6.5; // Approximate 6.5px per character
        const dynamicNameGap = Math.max(50, estimatedTickWidth + 20); // At least 50px, or width + 20px padding

        const gapPercent = 12; // horizontal gap between charts in %
        const gridBasePercent = 80; // make charts take less horizontal space
        const gridWidth =
            (gridBasePercent - gapPercent * (chartKeys.length - 1)) /
            chartKeys.length;
        const totalWidth =
            chartKeys.length * gridWidth + (chartKeys.length - 1) * gapPercent;

        // Calculate additional left offset needed for y-axis label (convert px to %)
        const chartContainerWidth =
            chartRef.current?.getEchartsInstance().getWidth() ?? 1200;
        const additionalLeftPx = Math.max(0, dynamicNameGap - 50); // Extra space beyond default 50px
        const additionalLeftPercent =
            (additionalLeftPx / chartContainerWidth) * 100;
        const leftOffset = (100 - totalWidth) / 2 + additionalLeftPercent;

        chartKeys.forEach((chartName, chartIndex) => {
            const gridLeft = leftOffset + chartIndex * (gridWidth + gapPercent);

            grids.push({
                left: gridLeft + "%",
                top: 60,
                width: gridWidth + "%",
                height: "65%"
            });
            // 1) ALL categories for this chart (union of all experiments)
            const allXRaw = Array.from(
                new Set(
                    experimentKeys
                        .filter((expKey) => {
                            const name = `Experiment ${parseInt(expKey, 10) + 1}`;
                            return selectedSeries[name] ?? true; // only include if selected
                        })
                        .flatMap((k) => dataDict[chartName]?.[k]?.xData ?? [])
                )
            ).sort((a, b) => Number(a) - Number(b));
            // ECharts category axis matches by string
            const allX: string[] = allXRaw.map((x) => String(x));

            // 2) Visible categories (union of experiments that are currently selected)
            const visibleCats = new Set<string>(
                experimentKeys
                    .filter((expKey) => {
                        const name = `Experiment ${parseInt(expKey, 10) + 1}`;
                        return selectedSeries[name] ?? true;
                    })
                    .flatMap((expKey) =>
                        (dataDict[chartName]?.[expKey]?.xData ?? []).map((x) =>
                            String(x)
                        )
                    )
            );

            xAxes.push({
                gridIndex: chartIndex,
                type: "category",
                data: allX, // <- full union; do NOT depend on selection
                name: xAxisLabel,
                nameLocation: "middle",
                nameGap: 40,
                boundaryGap: true, // small padding on both ends
                nameTextStyle: {
                    fontSize: chartConfig.fonts.axisNameFontSize,
                    fontWeight: Number(chartConfig.fonts.axisNameFontWeight),
                    color: chartConfig.colors.axisName,
                    fontFamily: chartConfig.fonts.font
                },
                axisLabel: {
                    color: chartConfig.colors.axisLabel,
                    fontFamily: chartConfig.fonts.font,
                    fontSize: chartConfig.fonts.axisLabelFontSize,
                    fontWeight: Number(chartConfig.fonts.axisLabelFontWeight),
                    // hide labels that have no visible series point
                    interval: (_: number, value: string) =>
                        visibleCats.has(value),
                    formatter: (value: string) =>
                        visibleCats.has(value) ? value : ""
                },
                axisTick: {
                    alignWithLabel: true,
                    // hide ticks that have no visible series point
                    interval: (_: number, value: string) =>
                        visibleCats.has(value)
                },
                splitLine: { show: false }
            });

            // --- Y axis uniform ticks (same as before) ---
            const chartHeight = Math.max(
                400,
                chartRef.current?.getEchartsInstance().getHeight() ?? 400
            );
            const desiredSplits = Math.max(2, Math.round(chartHeight / 80));
            const EPS = 1e-9;
            const gMin = Number(globalMin);
            const gMax = Number(globalMax);
            let interval = roundTo(
                (gMax - gMin) / Math.max(1, desiredSplits),
                3
            );
            if (interval <= 0) interval = Math.max(1e-3, gMax - gMin || 1e-3);
            const finalInterval =
                Math.abs(interval) > EPS
                    ? interval
                    : Math.max(1e-3, Math.abs(gMax - gMin));

            yAxes.push({
                gridIndex: chartIndex,
                type: "value",
                name: yAxisLabel.toUpperCase(),
                nameLocation: "middle",
                nameGap: dynamicNameGap,
                nameTextStyle: {
                    fontSize: chartConfig.fonts.axisNameFontSize,
                    fontWeight: Number(chartConfig.fonts.axisNameFontWeight),
                    color: chartConfig.colors.axisName,
                    fontFamily: chartConfig.fonts.font
                },
                axisLabel: {
                    color: chartConfig.colors.axisLabel,
                    fontFamily: chartConfig.fonts.font,
                    fontSize: chartConfig.fonts.axisLabelFontSize,
                    fontWeight: Number(chartConfig.fonts.axisLabelFontWeight),
                    formatter: (v: number) => parseNumbers(v) as string
                },
                axisLine: { show: true },
                axisTick: { show: true },
                min: finalMin,
                max: finalMax,
                interval: finalInterval,
                scale: true,
                splitLine: { show: false }
            });

            titles.push({
                text:
                    chartName.charAt(0).toUpperCase() +
                    chartName.slice(1) +
                    " " +
                    tag.toUpperCase(),
                left: `${gridLeft + gridWidth / 2}%`, // <-- true center
                top: 10, //10px from the top of the chart
                textAlign: "center"
            });

            // --- Series (align to allX; fill gaps with null) ---
            experimentKeys.forEach((expKey, index) => {
                const seriesName = `Experiment ${parseInt(expKey, 10) + 1}`;
                const exp = dataDict[chartName]?.[expKey];
                if (!exp) return;

                const xAsStr = (exp.xData ?? []).map((x) => String(x));

                const dataY = allX.map((cat) => {
                    const idx = xAsStr.indexOf(cat);
                    return idx >= 0 ? exp.yData[idx] : null;
                });

                if (selectedSeries[seriesName] ?? true) {
                    series.push({
                        name: seriesName,
                        type: "line",
                        xAxisIndex: chartIndex,
                        yAxisIndex: chartIndex,
                        data: dataY, // aligned to category axis
                        showSymbol: true,
                        symbol: "circle",
                        symbolSize: 6,
                        connectNulls: false,
                        lineStyle: {
                            width: 2,
                            type: lineTypes[keyIndex(index)] as
                                | "solid"
                                | "dashed"
                                | "dotted"
                        },
                        itemStyle: {
                            color: colorOptions[index % colorOptions.length]
                        },
                        z: 2, // baseline z-order (above grids/axes)
                        emphasis: {
                            focus: "series", // dim other series
                            lineStyle: {
                                width: 2.5 // make hovered line a bit thicker
                            },
                            itemStyle: {
                                borderWidth: 2,
                                borderColor: "#fff" // white border so circles "pop"
                            }
                        },
                        blur: {
                            lineStyle: { opacity: 0.4 },
                            itemStyle: { opacity: 0.4 }
                        }
                    });
                }
            });
        });

        return {
            title: titles,
            tooltip: {
                trigger: "item", // only when hovering the symbol
                formatter: (p: CallbackDataParams) => {
                    const x = String(p.name ?? "");
                    const y =
                        typeof p.value === "number"
                            ? parseNumbers(p.value)
                            : Array.isArray(p.value) &&
                                typeof p.value[1] === "number"
                              ? parseNumbers(p.value[1])
                              : "NA";
                    return `${p.marker} (${x}, ${y}) : ${p.seriesName}`;
                }
            },
            legend: { data: legendData, show: false },
            color: colorOptions,
            grid: grids,
            xAxis: xAxes,
            yAxis: yAxes,
            series
        };
    }, [
        chartKeys,
        legendData,
        colorOptions,
        experimentKeys,
        dataDict,
        xAxisLabel,
        chartConfig.fonts.axisNameFontSize,
        chartConfig.fonts.axisNameFontWeight,
        chartConfig.fonts.font,
        chartConfig.fonts.axisLabelFontSize,
        chartConfig.fonts.axisLabelFontWeight,
        chartConfig.colors.axisName,
        chartConfig.colors.axisLabel,
        yAxisLabel,
        tag,
        selectedSeries,
        keyIndex
    ]);

    return (
        <div id={id} className="flex w-full flex-col">
            <div ref={containerRef} className="h-[400px] w-full">
                {isReady && (
                    <ReactECharts
                        ref={chartRef}
                        option={option}
                        style={{ height: "100%", width: "100%" }}
                        notMerge={true}
                        lazyUpdate={true}
                        opts={{ renderer: "svg" }}
                    />
                )}
            </div>
            {/* Legend checkboxes below the chart */}
            <div className="ml-6 flex flex-col items-start gap-3">
                {legendData && legendData.length > 0 && (
                    <label className="flex items-center gap-2 text-sm font-semibold">
                        <Checkbox
                            id={`${id}-select-all`}
                            checked={allSelected}
                            onCheckedChange={toggleAllSeries}
                        />
                        <span>Select All</span>
                    </label>
                )}
                <div className="flex flex-wrap gap-x-6 gap-y-2">
                    {legendData?.map((name, index) => (
                        <label
                            key={"legend-" + name}
                            className="flex w-44 items-center gap-2 text-sm">
                            <Checkbox
                                id={`${id}-legend-${index}`}
                                checked={selectedSeries[name]}
                                onCheckedChange={() => toggleSeries(name)}
                            />
                            <span
                                className="relative inline-block h-0.5 w-6 shrink-0"
                                aria-hidden="true"
                                style={{
                                    backgroundColor: "transparent",
                                    borderTop: `2px ${lineTypes[keyIndex(index)]} ${
                                        colorOptions[
                                            index % colorOptions.length
                                        ]
                                    }`
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
                            <span className="truncate">{name}</span>
                        </label>
                    ))}
                </div>
            </div>
        </div>
    );
};
