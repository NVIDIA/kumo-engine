// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
    AreaChart,
    AreaChartProps
} from "@/components/blocks/chart/area-chart";
import { Meta, StoryObj } from "@storybook/react-vite";
import { userEvent } from "storybook/test";

const meta: Meta<typeof AreaChart> = {
    title: "Blocks/AreaChart",
    component: AreaChart,
    tags: ["autodocs"]
};

export default meta;

type Story = StoryObj<typeof AreaChart>;

const xAxisTickLabels = [
    "0",
    "8",
    "16",
    "24",
    "32",
    "40",
    "48",
    "56",
    "64",
    "72",
    "80",
    "88",
    "96",
    "104",
    "112",
    "120",
    "128",
    "136",
    "144"
];

const dataSet = [
    9.09090909, 9.09090909, 7.79220779, 9.09090909, 10.38961039, 2.5974026,
    10.38961039, 9.09090909, 9.09090909, 5.19480519, 7.79220779, 3.8961039,
    3.8961039, 0, 0, 0, 0, 1.2987013, 1.2987013
];

export const BasicAreaChart: Story = {
    args: {
        id: "storybook-area-chart",
        title: "Sample Area Chart",
        xAxisLabel: "Range",
        yAxisLabel: "Frequency",
        xAxisTickLabels,
        dataSet,
        dataSetLabel: "Distribution",
        toolTipText: "Value",
        customToolTip: true,
        legend: {
            show: false
        }
    } as AreaChartProps
};

export const AreaChartNonPercentTooltip: Story = {
    args: {
        id: "storybook-area-chart-count",
        title: "Count Area Chart",
        xAxisLabel: "Range",
        yAxisLabel: "Count",
        xAxisTickLabels,
        dataSet,
        dataSetLabel: "Distribution",
        toolTipText: "Count",
        customToolTip: true,
        legend: {
            show: false
        }
    } as AreaChartProps,
    play: async ({ canvasElement }) => {
        for (let x = 50; x < 560; x += 30) {
            await userEvent.pointer({
                target: canvasElement,
                coords: { x, y: 200 }
            });
        }
        await userEvent.pointer({
            target: canvasElement,
            coords: { x: 0, y: 0 }
        });
    }
};

export const AreaChartRawTooltip: Story = {
    args: {
        id: "storybook-area-chart-raw",
        title: "Raw Tooltip Area Chart",
        xAxisLabel: "Range",
        yAxisLabel: "Frequency",
        xAxisTickLabels,
        dataSet,
        dataSetLabel: "Distribution",
        toolTipText: "Percent Share",
        customToolTip: false,
        legend: {
            show: false
        }
    } as AreaChartProps,
    play: async ({ canvasElement }) => {
        for (let x = 50; x < 560; x += 30) {
            await userEvent.pointer({
                target: canvasElement,
                coords: { x, y: 200 }
            });
        }
        await userEvent.pointer({
            target: canvasElement,
            coords: { x: 0, y: 0 }
        });
    }
};

export const AreaChartPercentShare: Story = {
    args: {
        id: "storybook-area-chart-percent",
        title: "Percent Share",
        xAxisLabel: "Range",
        yAxisLabel: "Percent",
        xAxisTickLabels,
        dataSet,
        dataSetLabel: "Percent Share",
        toolTipText: "Percent Share",
        customToolTip: true,
        legend: {
            show: true
        }
    } as AreaChartProps,
    play: async ({ canvasElement }) => {
        for (let x = 50; x < 560; x += 30) {
            await userEvent.pointer({
                target: canvasElement,
                coords: { x, y: 200 }
            });
        }
        await userEvent.pointer({
            target: canvasElement,
            coords: { x: 0, y: 0 }
        });
    }
};
