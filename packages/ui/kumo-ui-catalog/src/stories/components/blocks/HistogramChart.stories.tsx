// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { HistogramChart } from "@/components/blocks/chart/histogram-chart";
import { TargetHistogram } from "@/types/chart";
import type { Meta, StoryObj } from "@storybook/react-vite";
import { userEvent, waitFor } from "storybook/test";

const meta: Meta<typeof HistogramChart> = {
    title: "Blocks/HistogramChart",
    component: HistogramChart,
    tags: ["autodocs"]
};

export default meta;
type Story = StoryObj<typeof HistogramChart>;

const label: TargetHistogram = {
    "bins": [
        17, 18.36, 19.72, 21.08, 22.44, 23.8, 25.16, 26.52, 27.88, 29.24, 30.6,
        31.96, 33.32, 34.68, 36.04, 37.4, 38.76, 40.12, 41.48, 42.84, 44.2,
        45.56, 46.92, 48.28, 49.64, 51, 52.36, 53.72, 55.08, 56.44, 57.8, 59.16,
        60.52, 61.88, 63.24, 64.6, 65.96, 67.32, 68.68, 70.04, 71.4, 72.76,
        74.12, 75.48, 76.84, 78.2, 79.56, 80.92, 82.28, 83.64, 85
    ],
    "counts": [
        32, 30, 102, 50, 52, 93, 36, 45, 74, 34, 24, 46, 20, 33, 11, 10, 28, 18,
        10, 19, 21, 19, 44, 28, 45, 28, 20, 28, 10, 17, 25, 9, 9, 20, 7, 5, 11,
        5, 10, 3, 4, 3, 1, 0, 3, 0, 0, 1, 0, 1
    ],
    "bin_width": 1.36,
    "num_bins": 50
};

const pred: TargetHistogram = {
    "bins": [
        17, 18.36, 19.72, 21.08, 22.44, 23.8, 25.16, 26.52, 27.88, 29.24, 30.6,
        31.96, 33.32, 34.68, 36.04, 37.4, 38.76, 40.12, 41.48, 42.84, 44.2,
        45.56, 46.92, 48.28, 49.64, 51, 52.36, 53.72, 55.08, 56.44, 57.8, 59.16,
        60.52, 61.88, 63.24, 64.6, 65.96, 67.32, 68.68, 70.04, 71.4, 72.76,
        74.12, 75.48, 76.84, 78.2, 79.56, 80.92, 82.28, 83.64, 85
    ],
    "counts": [
        0, 0, 2, 5, 10, 15, 35, 63, 99, 113, 95, 104, 94, 84, 64, 55, 63, 50,
        30, 33, 33, 21, 12, 15, 10, 6, 10, 9, 3, 2, 3, 3, 1, 1, 0, 0, 0, 0, 0,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ],
    "bin_width": 1.36,
    "num_bins": 50
};

export const Histogram: Story = {
    render: function DefaultFunc() {
        return (
            <HistogramChart
                id={"barchart"}
                className="flex h-[80vh]"
                predData={pred}
                labelData={label}
                xAxisLabel="Actual or Predicted Value"
                yAxisLabel="Number of Occurrances"
            />
        );
    },
    play: async ({ canvasElement }) => {
        const svg = await waitFor(() => {
            const el = canvasElement.querySelector("svg");
            if (!el) throw new Error("svg not found");
            return el;
        });
        await userEvent.pointer({ target: svg, coords: { x: 150, y: 150 } });
        await userEvent.pointer({ target: svg, coords: { x: 200, y: 120 } });
        await userEvent.pointer({ target: svg, coords: { x: 0, y: 0 } });
    }
};

const smallLabel: TargetHistogram = {
    bins: [16.32, 17.68, 19.04, 20.4],
    counts: [32, 50, 20],
    bin_width: 1.36,
    num_bins: 3
};

const smallPred: TargetHistogram = {
    bins: [16.32, 17.68, 19.04, 20.4],
    counts: [5, 25, 40],
    bin_width: 1.36,
    num_bins: 3
};

export const HistogramSmallDataset: Story = {
    render: function DefaultFunc() {
        return (
            <HistogramChart
                id={"histogram-small"}
                className="flex h-[60vh]"
                predData={smallPred}
                labelData={smallLabel}
                xAxisLabel="Value"
                yAxisLabel="Count"
            />
        );
    },
    play: async ({ canvasElement }) => {
        const svg = await waitFor(() => {
            const el = canvasElement.querySelector("svg");
            if (!el) throw new Error("svg not found");
            return el;
        });
        await userEvent.pointer({ target: svg, coords: { x: 150, y: 150 } });
        await userEvent.pointer({ target: svg, coords: { x: 200, y: 120 } });
        await userEvent.pointer({ target: svg, coords: { x: 0, y: 0 } });
    }
};

const noCounts: TargetHistogram = {
    bins: [1, 2, 3, 4],
    counts: [],
    bin_width: 1,
    num_bins: 3
};

export const HistogramEmptyCounts: Story = {
    render: function DefaultFunc() {
        return (
            <HistogramChart
                id={"histogram-empty-counts"}
                className="flex h-[60vh]"
                predData={noCounts}
                labelData={noCounts}
                xAxisLabel="Value"
                yAxisLabel="Count"
            />
        );
    }
};

const zeroCounts: TargetHistogram = {
    bins: [0, 1, 2, 3],
    counts: [0, 0, 0],
    bin_width: 1,
    num_bins: 3
};

export const HistogramZeroCounts: Story = {
    render: function DefaultFunc() {
        return (
            <HistogramChart
                id={"histogram-zero-counts"}
                className="flex h-[60vh]"
                predData={zeroCounts}
                labelData={zeroCounts}
                xAxisLabel="Value"
                yAxisLabel="Count"
            />
        );
    }
};

export const HistogramWithLegendToggle: Story = {
    render: function DefaultFunc() {
        return (
            <HistogramChart
                id={"histogram-legend-toggle"}
                className="flex h-[80vh]"
                predData={pred}
                labelData={label}
                xAxisLabel="Value"
                yAxisLabel="Count"
            />
        );
    },
    play: async ({ canvasElement }) => {
        const svg = await waitFor(() => {
            const el = canvasElement.querySelector("svg");
            if (!el) throw new Error("svg not found");
            return el;
        });

        const predLegend = canvasElement.querySelector(
            "text[data-z-level]"
        ) as SVGTextElement | null;
        if (predLegend) await userEvent.click(predLegend);

        await userEvent.pointer({ target: svg, coords: { x: 150, y: 150 } });
        await userEvent.pointer({ target: svg, coords: { x: 0, y: 0 } });

        const legendTexts = canvasElement.querySelectorAll("text");
        for (const t of legendTexts) {
            if (t.textContent?.includes("Actuals")) {
                await userEvent.click(t);
                break;
            }
        }
        await userEvent.pointer({ target: svg, coords: { x: 200, y: 150 } });
        await userEvent.pointer({ target: svg, coords: { x: 0, y: 0 } });
    }
};
