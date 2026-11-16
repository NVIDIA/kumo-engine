// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ChartLabel } from "@/components/blocks/chart/chart-label";
import { ScatterPointLineChart } from "@/components/blocks/chart/scatter-point-line";
import { SectionHeader } from "@/components/ui/section-header";
import { Slider } from "@/components/ui/slider";
import { useChartConfig } from "@/hooks/use-chart-config";
import type { ScatterPointData } from "@/types/chart";
import type { Meta, StoryObj } from "@storybook/react-vite";
import { useCallback, useState } from "react";
import { userEvent, waitFor } from "storybook/test";

const meta: Meta<typeof ScatterPointLineChart> = {
    title: "Blocks/ScatterPointLineChart",
    component: ScatterPointLineChart,
    tags: ["autodocs"]
};

export default meta;
type Story = StoryObj<typeof ScatterPointLineChart>;

const thresholdData = [
    0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12,
    0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25,
    0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38,
    0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51,
    0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6, 0.61, 0.62, 0.63, 0.64,
    0.65, 0.66, 0.67, 0.68, 0.69, 0.7, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77,
    0.78, 0.79, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9,
    0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1
];

const rocData: ScatterPointData[] = [
    { x: 1, y: 1 },
    { x: 1, y: 1 },
    { x: 1, y: 1 },
    { x: 1, y: 1 },
    { x: 1, y: 1 },
    { x: 1, y: 1 },
    { x: 1, y: 1 },
    { x: 1, y: 1 },
    { x: 1, y: 1 },
    { x: 1, y: 1 },
    { x: 1, y: 1 },
    { x: 1, y: 1 },
    { x: 1, y: 1 },
    { x: 1, y: 1 },
    { x: 1, y: 1 },
    { x: 1, y: 1 },
    { x: 1, y: 1 },
    { x: 1, y: 1 },
    { x: 1, y: 1 },
    { x: 1, y: 1 },
    { x: 1, y: 1 },
    { x: 1, y: 1 },
    { x: 1, y: 1 },
    { x: 1, y: 1 },
    { x: 1, y: 1 },
    { x: 0.96, y: 1 },
    { x: 0.96, y: 1 },
    { x: 0.96, y: 1 },
    { x: 0.96, y: 1 },
    { x: 0.96, y: 1 },
    { x: 0.96, y: 1 },
    { x: 0.96, y: 1 },
    { x: 0.96, y: 1 },
    { x: 0.91, y: 1 },
    { x: 0.91, y: 1 },
    { x: 0.91, y: 1 },
    { x: 0.87, y: 1 },
    { x: 0.87, y: 0.99 },
    { x: 0.87, y: 0.99 },
    { x: 0.87, y: 0.97 },
    { x: 0.87, y: 0.97 },
    { x: 0.87, y: 0.97 },
    { x: 0.87, y: 0.97 },
    { x: 0.87, y: 0.97 },
    { x: 0.87, y: 0.97 },
    { x: 0.87, y: 0.97 },
    { x: 0.87, y: 0.96 },
    { x: 0.87, y: 0.96 },
    { x: 0.87, y: 0.96 },
    { x: 0.74, y: 0.95 },
    { x: 0.74, y: 0.95 },
    { x: 0.7, y: 0.92 },
    { x: 0.7, y: 0.91 },
    { x: 0.65, y: 0.91 },
    { x: 0.65, y: 0.91 },
    { x: 0.65, y: 0.91 },
    { x: 0.61, y: 0.9 },
    { x: 0.61, y: 0.87 },
    { x: 0.61, y: 0.87 },
    { x: 0.61, y: 0.86 },
    { x: 0.61, y: 0.86 },
    { x: 0.57, y: 0.86 },
    { x: 0.48, y: 0.86 },
    { x: 0.48, y: 0.84 },
    { x: 0.48, y: 0.82 },
    { x: 0.43, y: 0.82 },
    { x: 0.39, y: 0.78 },
    { x: 0.39, y: 0.78 },
    { x: 0.39, y: 0.78 },
    { x: 0.39, y: 0.78 },
    { x: 0.3, y: 0.78 },
    { x: 0.26, y: 0.78 },
    { x: 0.26, y: 0.77 },
    { x: 0.26, y: 0.75 },
    { x: 0.26, y: 0.75 },
    { x: 0.26, y: 0.71 },
    { x: 0.26, y: 0.7 },
    { x: 0.26, y: 0.7 },
    { x: 0.26, y: 0.7 },
    { x: 0.22, y: 0.68 },
    { x: 0.22, y: 0.66 },
    { x: 0.22, y: 0.6 },
    { x: 0.22, y: 0.6 },
    { x: 0.17, y: 0.57 },
    { x: 0.17, y: 0.56 },
    { x: 0.17, y: 0.52 },
    { x: 0.09, y: 0.51 },
    { x: 0.09, y: 0.48 },
    { x: 0.04, y: 0.34 },
    { x: 0.04, y: 0.26 },
    { x: 0.04, y: 0.23 },
    { x: 0, y: 0.22 },
    { x: 0, y: 0.16 },
    { x: 0, y: 0.06 },
    { x: 0, y: 0.01 },
    { x: 0, y: 0 },
    { x: 0, y: 0 },
    { x: 0, y: 0 },
    { x: 0, y: 0 },
    { x: 0, y: 0 },
    { x: 0, y: 0 }
];

const prData: ScatterPointData[] = [
    { x: 1.0, y: 0.77 },
    { x: 1.0, y: 0.77 },
    { x: 1.0, y: 0.77 },
    { x: 1.0, y: 0.77 },
    { x: 1.0, y: 0.77 },
    { x: 1.0, y: 0.77 },
    { x: 1.0, y: 0.77 },
    { x: 1.0, y: 0.77 },
    { x: 1.0, y: 0.77 },
    { x: 1.0, y: 0.77 },
    { x: 1.0, y: 0.77 },
    { x: 1.0, y: 0.77 },
    { x: 1.0, y: 0.77 },
    { x: 1.0, y: 0.77 },
    { x: 1.0, y: 0.77 },
    { x: 1.0, y: 0.77 },
    { x: 1.0, y: 0.77 },
    { x: 1.0, y: 0.77 },
    { x: 1.0, y: 0.77 },
    { x: 1.0, y: 0.77 },
    { x: 1.0, y: 0.77 },
    { x: 1.0, y: 0.77 },
    { x: 1.0, y: 0.77 },
    { x: 1.0, y: 0.77 },
    { x: 1.0, y: 0.77 },
    { x: 1.0, y: 0.78 },
    { x: 1.0, y: 0.78 },
    { x: 1.0, y: 0.78 },
    { x: 1.0, y: 0.78 },
    { x: 1.0, y: 0.78 },
    { x: 1.0, y: 0.78 },
    { x: 1.0, y: 0.78 },
    { x: 1.0, y: 0.78 },
    { x: 1.0, y: 0.79 },
    { x: 1.0, y: 0.79 },
    { x: 1.0, y: 0.79 },
    { x: 1.0, y: 0.79 },
    { x: 0.99, y: 0.79 },
    { x: 0.99, y: 0.79 },
    { x: 0.97, y: 0.79 },
    { x: 0.97, y: 0.79 },
    { x: 0.97, y: 0.79 },
    { x: 0.97, y: 0.79 },
    { x: 0.97, y: 0.79 },
    { x: 0.97, y: 0.79 },
    { x: 0.97, y: 0.79 },
    { x: 0.96, y: 0.79 },
    { x: 0.96, y: 0.79 },
    { x: 0.96, y: 0.79 },
    { x: 0.95, y: 0.81 },
    { x: 0.95, y: 0.81 },
    { x: 0.92, y: 0.82 },
    { x: 0.91, y: 0.81 },
    { x: 0.91, y: 0.82 },
    { x: 0.91, y: 0.82 },
    { x: 0.91, y: 0.82 },
    { x: 0.9, y: 0.83 },
    { x: 0.87, y: 0.83 },
    { x: 0.87, y: 0.83 },
    { x: 0.86, y: 0.82 },
    { x: 0.86, y: 0.82 },
    { x: 0.86, y: 0.84 },
    { x: 0.86, y: 0.86 },
    { x: 0.84, y: 0.86 },
    { x: 0.82, y: 0.85 },
    { x: 0.82, y: 0.86 },
    { x: 0.78, y: 0.87 },
    { x: 0.78, y: 0.87 },
    { x: 0.78, y: 0.87 },
    { x: 0.78, y: 0.87 },
    { x: 0.78, y: 0.9 },
    { x: 0.78, y: 0.91 },
    { x: 0.77, y: 0.91 },
    { x: 0.75, y: 0.91 },
    { x: 0.75, y: 0.91 },
    { x: 0.71, y: 0.9 },
    { x: 0.7, y: 0.9 },
    { x: 0.7, y: 0.9 },
    { x: 0.7, y: 0.9 },
    { x: 0.68, y: 0.91 },
    { x: 0.66, y: 0.91 },
    { x: 0.6, y: 0.9 },
    { x: 0.6, y: 0.9 },
    { x: 0.57, y: 0.92 },
    { x: 0.56, y: 0.91 },
    { x: 0.52, y: 0.91 },
    { x: 0.51, y: 0.95 },
    { x: 0.48, y: 0.95 },
    { x: 0.34, y: 0.96 },
    { x: 0.26, y: 0.95 },
    { x: 0.23, y: 0.95 },
    { x: 0.22, y: 1.0 },
    { x: 0.16, y: 1.0 },
    { x: 0.06, y: 1.0 },
    { x: 0.01, y: 1.0 },
    { x: 0.0, y: 1.0 },
    { x: 0.0, y: 1.0 },
    { x: 0.0, y: 1.0 },
    { x: 0.0, y: 1.0 },
    { x: 0.0, y: 1.0 },
    { x: 0.0, y: 1.0 }
];

const gainData: ScatterPointData[] = [
    { x: 1.0, y: 1.0 },
    { x: 1.0, y: 1.0 },
    { x: 1.0, y: 1.0 },
    { x: 1.0, y: 1.0 },
    { x: 1.0, y: 1.0 },
    { x: 1.0, y: 1.0 },
    { x: 1.0, y: 1.0 },
    { x: 1.0, y: 1.0 },
    { x: 1.0, y: 1.0 },
    { x: 1.0, y: 1.0 },
    { x: 1.0, y: 1.0 },
    { x: 1.0, y: 1.0 },
    { x: 1.0, y: 1.0 },
    { x: 1.0, y: 1.0 },
    { x: 1.0, y: 1.0 },
    { x: 1.0, y: 1.0 },
    { x: 1.0, y: 1.0 },
    { x: 1.0, y: 1.0 },
    { x: 1.0, y: 1.0 },
    { x: 1.0, y: 1.0 },
    { x: 1.0, y: 1.0 },
    { x: 1.0, y: 1.0 },
    { x: 1.0, y: 1.0 },
    { x: 1.0, y: 1.0 },
    { x: 1.0, y: 1.0 },
    { x: 0.99, y: 1.0 },
    { x: 0.99, y: 1.0 },
    { x: 0.99, y: 1.0 },
    { x: 0.99, y: 1.0 },
    { x: 0.99, y: 1.0 },
    { x: 0.99, y: 1.0 },
    { x: 0.99, y: 1.0 },
    { x: 0.99, y: 1.0 },
    { x: 0.98, y: 1.0 },
    { x: 0.98, y: 1.0 },
    { x: 0.98, y: 1.0 },
    { x: 0.97, y: 1.0 },
    { x: 0.96, y: 0.99 },
    { x: 0.96, y: 0.99 },
    { x: 0.95, y: 0.97 },
    { x: 0.95, y: 0.97 },
    { x: 0.95, y: 0.97 },
    { x: 0.95, y: 0.97 },
    { x: 0.95, y: 0.97 },
    { x: 0.95, y: 0.97 },
    { x: 0.95, y: 0.97 },
    { x: 0.94, y: 0.96 },
    { x: 0.94, y: 0.96 },
    { x: 0.94, y: 0.96 },
    { x: 0.9, y: 0.95 },
    { x: 0.9, y: 0.95 },
    { x: 0.87, y: 0.92 },
    { x: 0.86, y: 0.91 },
    { x: 0.85, y: 0.91 },
    { x: 0.85, y: 0.91 },
    { x: 0.85, y: 0.91 },
    { x: 0.83, y: 0.9 },
    { x: 0.81, y: 0.87 },
    { x: 0.81, y: 0.87 },
    { x: 0.8, y: 0.86 },
    { x: 0.8, y: 0.86 },
    { x: 0.79, y: 0.86 },
    { x: 0.77, y: 0.86 },
    { x: 0.76, y: 0.84 },
    { x: 0.74, y: 0.82 },
    { x: 0.73, y: 0.82 },
    { x: 0.69, y: 0.78 },
    { x: 0.69, y: 0.78 },
    { x: 0.69, y: 0.78 },
    { x: 0.69, y: 0.78 },
    { x: 0.67, y: 0.78 },
    { x: 0.66, y: 0.78 },
    { x: 0.65, y: 0.77 },
    { x: 0.64, y: 0.75 },
    { x: 0.64, y: 0.75 },
    { x: 0.61, y: 0.71 },
    { x: 0.6, y: 0.7 },
    { x: 0.6, y: 0.7 },
    { x: 0.6, y: 0.7 },
    { x: 0.57, y: 0.68 },
    { x: 0.56, y: 0.66 },
    { x: 0.51, y: 0.6 },
    { x: 0.51, y: 0.6 },
    { x: 0.48, y: 0.57 },
    { x: 0.47, y: 0.56 },
    { x: 0.44, y: 0.52 },
    { x: 0.41, y: 0.51 },
    { x: 0.39, y: 0.48 },
    { x: 0.27, y: 0.34 },
    { x: 0.21, y: 0.26 },
    { x: 0.19, y: 0.23 },
    { x: 0.17, y: 0.22 },
    { x: 0.12, y: 0.16 },
    { x: 0.05, y: 0.06 },
    { x: 0.01, y: 0.01 },
    { x: 0.0, y: 0.0 },
    { x: 0.0, y: 0.0 },
    { x: 0.0, y: 0.0 },
    { x: 0.0, y: 0.0 },
    { x: 0.0, y: 0.0 },
    { x: 0.0, y: 0.0 }
];

const pprIdeal = 0.77;
const datalength = rocData.length - 1;

// Custom hook to manage threshold slider state
function useThresholdSlider() {
    const [sliderPoint, setSliderPoint] = useState<Array<number>>([0.0]);
    const [threshold, setThreshold] = useState<number>(0);
    const [currentIndex, setCurrentIndex] = useState<number>(0);

    const sliderRangeUpdate = useCallback((sliderPoint: number[]) => {
        if (sliderPoint) {
            const pos = Math.round(sliderPoint[0] * thresholdData.length);
            const threshold = thresholdData?.[pos];
            setSliderPoint(sliderPoint);
            setCurrentIndex(pos);
            setThreshold(threshold as number);
        }
    }, []);

    return { sliderPoint, threshold, currentIndex, sliderRangeUpdate };
}

// Reusable threshold slider component
function ThresholdSlider({
    sliderPoint,
    threshold,
    onValueChange
}: {
    sliderPoint: number[];
    threshold: number;
    onValueChange: (value: number[]) => void;
}) {
    return (
        <div
            className="mx-4 mb-8 mt-3 pt-6"
            style={{ fontFamily: "Source Code Pro" }}>
            <Slider
                min={0}
                max={1}
                step={1 / (thresholdData.length - 1)}
                value={sliderPoint}
                onValueChange={onValueChange}
                defaultValue={Array.of(threshold)}
            />
        </div>
    );
}

export const ROCCurve: Story = {
    render: function ROCCurveStory() {
        const chartConfig = useChartConfig();
        const { sliderPoint, threshold, currentIndex, sliderRangeUpdate } =
            useThresholdSlider();

        return (
            <>
                <ThresholdSlider
                    sliderPoint={sliderPoint}
                    threshold={threshold}
                    onValueChange={sliderRangeUpdate}
                />
                <div style={{ width: "500px" }}>
                    <SectionHeader
                        title="ROC Curve"
                        titleClassName="leading-none"
                        titleSize="sm"
                        titleLetterSpacing={0.1}
                    />

                    <div style={{ height: 350 }}>
                        <ScatterPointLineChart
                            id="roc-curve"
                            scatterPointData={rocData}
                            yAxisLabel="True Positive Rate"
                            xAxisLabel="False Positive Rate"
                            currentIndex={currentIndex}
                            width={400}
                            height={350}
                            scatterLineData={[
                                {
                                    coordinates: [
                                        { x: 0, y: 0 },
                                        { x: datalength, y: datalength }
                                    ],
                                    lineColor: chartConfig.colors.slate400,
                                    lineWidth: 1
                                }
                            ]}
                        />
                    </div>
                    <div className="mt-2">
                        <ChartLabel
                            iconClassName="bg-teal-800"
                            text="Predictive Query Performance"
                        />
                        <ChartLabel
                            iconClassName="bg-slate-200"
                            text="Random Guesses"
                        />
                    </div>
                </div>
            </>
        );
    },
    play: async ({ canvasElement }) => {
        const svg = await waitFor(() => {
            const el = canvasElement.querySelector("svg");
            if (!el) throw new Error("svg not found");
            return el;
        });
        await userEvent.pointer({ target: svg, coords: { x: 150, y: 150 } });
        await userEvent.pointer({ target: svg, coords: { x: 200, y: 100 } });
        await userEvent.pointer({ target: svg, coords: { x: 0, y: 0 } });
    }
};

// Random guess is typically the proportion of positives
const randomGuess = 0.77;

export const PrecisionVsRecall: Story = {
    render: function PrecisionVsRecallStory() {
        const chartConfig = useChartConfig();
        const { sliderPoint, threshold, currentIndex, sliderRangeUpdate } =
            useThresholdSlider();

        return (
            <>
                <ThresholdSlider
                    sliderPoint={sliderPoint}
                    threshold={threshold}
                    onValueChange={sliderRangeUpdate}
                />
                <div style={{ width: "500px" }}>
                    <SectionHeader
                        titleClassName="leading-none"
                        title="Precision VS Recall"
                        titleSize="sm"
                        titleLetterSpacing={0.1}
                    />
                    <div style={{ height: 350 }}>
                        <ScatterPointLineChart
                            id="pr-curve"
                            scatterPointData={prData}
                            yAxisLabel="Precision"
                            xAxisLabel="Recall"
                            currentIndex={currentIndex}
                            width={400}
                            height={350}
                            scatterLineData={[
                                {
                                    coordinates: [
                                        { x: 0, y: randomGuess * datalength },
                                        {
                                            x: datalength,
                                            y: randomGuess * datalength
                                        }
                                    ],
                                    lineColor: chartConfig.colors.slate400,
                                    lineWidth: 1.5
                                }
                            ]}
                        />
                    </div>
                    <div className="mt-2">
                        <ChartLabel
                            iconClassName="bg-teal-800"
                            text="Predictive Query Performance"
                        />
                        <ChartLabel
                            iconClassName="bg-slate-200"
                            text="Random Guesses"
                        />
                    </div>
                </div>
            </>
        );
    },
    play: async ({ canvasElement }) => {
        const svg = await waitFor(() => {
            const el = canvasElement.querySelector("svg");
            if (!el) throw new Error("svg not found");
            return el;
        });
        await userEvent.pointer({ target: svg, coords: { x: 150, y: 150 } });
        await userEvent.pointer({ target: svg, coords: { x: 200, y: 100 } });
        await userEvent.pointer({ target: svg, coords: { x: 0, y: 0 } });
    }
};

export const CumulativeGain: Story = {
    render: function CumulativeGainStory() {
        const chartConfig = useChartConfig();
        const { sliderPoint, threshold, currentIndex, sliderRangeUpdate } =
            useThresholdSlider();

        return (
            <>
                <ThresholdSlider
                    sliderPoint={sliderPoint}
                    threshold={threshold}
                    onValueChange={sliderRangeUpdate}
                />
                <div style={{ width: "500px" }}>
                    <SectionHeader
                        title="Cumulative Gain"
                        titleClassName="leading-none"
                        titleSize="sm"
                        titleLetterSpacing={0.1}
                    />
                    <div style={{ height: 350 }}>
                        <ScatterPointLineChart
                            id="gain-curve"
                            scatterPointData={gainData}
                            yAxisLabel="True Positive Rate"
                            xAxisLabel="Rate of Positive Predictions"
                            currentIndex={currentIndex}
                            width={400}
                            height={350}
                            scatterLineData={[
                                {
                                    coordinates: [
                                        { x: 0, y: 0 },
                                        { x: datalength, y: datalength }
                                    ],
                                    lineColor: chartConfig.colors.lime400
                                },
                                {
                                    coordinates: [
                                        { x: 0, y: 0 },
                                        {
                                            x: pprIdeal * datalength,
                                            y: datalength
                                        },
                                        { x: datalength, y: datalength }
                                    ],
                                    lineColor: chartConfig.colors.pink600
                                }
                            ]}
                        />
                    </div>
                    <div className="mt-2">
                        <ChartLabel
                            iconClassName="bg-pink-600"
                            text="Ideal Case"
                        />
                        <ChartLabel
                            iconClassName="bg-teal-800"
                            text="Lift Curve"
                        />
                        <ChartLabel
                            iconClassName="bg-lime-400"
                            text="Baseline"
                        />
                    </div>
                </div>
            </>
        );
    },
    play: async ({ canvasElement }) => {
        const svg = await waitFor(() => {
            const el = canvasElement.querySelector("svg");
            if (!el) throw new Error("svg not found");
            return el;
        });
        await userEvent.pointer({ target: svg, coords: { x: 150, y: 150 } });
        await userEvent.pointer({ target: svg, coords: { x: 200, y: 100 } });
        await userEvent.pointer({ target: svg, coords: { x: 0, y: 0 } });
    }
};
