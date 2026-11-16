// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { BarChart, BarChartProps } from "@/components/blocks/chart/bar-chart";
import { Meta, StoryObj } from "@storybook/react-vite";
import { userEvent, waitFor } from "storybook/test";

export default {
    title: "Blocks/BarChart",
    component: BarChart
} as Meta<typeof BarChart>;

const xAxisTickLabels = [
    "Solid",
    "All over pattern",
    "Melange",
    "Stripe",
    "Denim",
    "Front print",
    "Placement print",
    "Check",
    "Lace",
    "Colour blocking"
];

// Convert the values into heatmap format [x, y, value]
// For a single row heatmap (1 row), y = 0
const dataSet = [45.87, 15.15, 6.82, 5.59, 4.66, 3.25, 3.06, 1.97, 1.91, 1.74];
const customToolTip = true;
const toolTipText = "Percent Share";
const xAxisLabel = "Category";
const yAxisLabel = "test";
const enableDataLabel = true;
type Story = StoryObj<typeof BarChart>;

export const BarChartExample: Story = {
    args: {
        title: "Fabric Pattern Popularity",
        dataSet,
        xAxisTickLabels,
        xAxisLabel,
        yAxisLabel,
        customToolTip,
        toolTipText,
        enableDataLabel
    } as BarChartProps,
    play: async ({ canvasElement }) => {
        const svg = await waitFor(() => {
            const el = canvasElement.querySelector("svg");
            if (!el) throw new Error("svg not found");
            return el;
        });
        await userEvent.pointer({ target: svg, coords: { x: 100, y: 100 } });
        await userEvent.pointer({ target: svg, coords: { x: 120, y: 80 } });
        await userEvent.pointer({ target: svg, coords: { x: 0, y: 0 } });
    }
};

export const BarChartWithCustomLabel: Story = {
    args: {
        title: "Range Labels",
        dataSet: [10, 25, 15, 30, 20],
        xAxisTickLabels: ["0 - 10", "10 - 20", "20 - 30", "30 - 40", "40 - 50"],
        xAxisLabel: "Range",
        yAxisLabel: "Count",
        customToolTip: false,
        enableDataLabel: true,
        customLabel: true
    } as BarChartProps
};

export const BarChartWithTrimmedLabels: Story = {
    args: {
        title: "Long Category Names",
        dataSet: [40, 30, 20, 10, 5],
        xAxisTickLabels: [
            "Very Long Category Name One",
            "Another Long Category Name",
            "Yet Another Long Label Here",
            "Short",
            "Medium Label Text"
        ],
        xAxisLabel: "Category",
        yAxisLabel: "Value",
        customToolTip: false,
        enableDataLabel: false,
        trimXAxisLabel: true
    } as BarChartProps
};

export const BarChartPercentShareTooltip: Story = {
    args: {
        title: "Percent Share Tooltip",
        dataSet,
        xAxisTickLabels,
        xAxisLabel,
        yAxisLabel,
        customToolTip: true,
        toolTipText: "Percent Share",
        enableDataLabel: false
    } as BarChartProps,
    play: async ({ canvasElement }) => {
        const svg = await waitFor(() => {
            const el = canvasElement.querySelector("svg");
            if (!el) throw new Error("svg not found");
            return el;
        });
        await userEvent.pointer({ target: svg, coords: { x: 100, y: 100 } });
        await userEvent.pointer({ target: svg, coords: { x: 0, y: 0 } });
    }
};

export const BarChartMoreThan20: Story = {
    args: {
        title: "More Than 20 Items",
        dataSet: Array.from({ length: 25 }, (_, i) => i + 1),
        xAxisTickLabels: Array.from({ length: 25 }, (_, i) => `Cat ${i + 1}`),
        xAxisLabel: "Category",
        yAxisLabel: "Value",
        customToolTip: false,
        enableDataLabel: false
    } as BarChartProps
};

export const BarChartCustomLabelMiddle: Story = {
    args: {
        title: "Custom Label Middle",
        dataSet: [10, 25, 15, 30, 20, 5],
        xAxisTickLabels: [
            "0 - 10",
            "10 - 20",
            "20 - 30",
            "30 - 40",
            "40 - 50",
            "50 - 60"
        ],
        xAxisLabel: "Range",
        yAxisLabel: "Count",
        customToolTip: false,
        enableDataLabel: true,
        customLabel: true
    } as BarChartProps
};

export const BarChartDefaultTooltip: Story = {
    args: {
        title: "Default Tooltip",
        dataSet,
        xAxisTickLabels,
        xAxisLabel,
        yAxisLabel,
        customToolTip: false,
        enableDataLabel: false
    } as BarChartProps,
    play: async ({ canvasElement }) => {
        const svg = await waitFor(() => {
            const el = canvasElement.querySelector("svg");
            if (!el) throw new Error("svg not found");
            return el;
        });
        await userEvent.pointer({ target: svg, coords: { x: 100, y: 100 } });
        await userEvent.pointer({ target: svg, coords: { x: 0, y: 0 } });
    }
};
