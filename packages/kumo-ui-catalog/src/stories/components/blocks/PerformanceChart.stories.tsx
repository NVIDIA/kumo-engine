// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Meta, StoryObj } from "@storybook/react-vite";

import { PerformanceChart } from "@/components/blocks/chart/performance-chart";
import { jobStatus } from "@/types/chart";

const meta: Meta<typeof PerformanceChart> = {
    title: "Blocks/PerformanceChart",
    component: PerformanceChart,
    tags: ["autodocs"]
};
export default meta;

const mockProgressData = {
    xAxisLabels: [1, 2, 3, 4],
    xLabel: "Experiments",
    yLabel: "map@10",
    mode: "max",
    datasets: [
        {
            label: "Best validation metric",
            color: "#ff6666",
            value: [0.0379, 0.0384, 0.0357, 0.0368]
        }
    ]
};

type Story = StoryObj<typeof PerformanceChart>;

export const Default: Story = {
    args: {
        id: "default-chart",
        progressData: mockProgressData,
        jobstatus: jobStatus.Done,
        showTitle: true
    }
};

export const InProgress: Story = {
    args: {
        id: "in-progress-chart",
        progressData: mockProgressData,
        jobstatus: jobStatus.Running,
        showTitle: true
    }
};

export const WithoutTitle: Story = {
    args: {
        id: "no-title-chart",
        progressData: mockProgressData,
        jobstatus: jobStatus.Done,
        showTitle: false
    }
};

export const InProgressMultiExperiment: Story = {
    args: {
        id: "in-progress-multi-chart",
        progressData: {
            xAxisLabels: [1, 2, 3, 4, 5, 6, 7, 8],
            xLabel: "Experiments",
            yLabel: "map@10",
            mode: "max",
            datasets: [
                {
                    label: "Best validation metric",
                    color: "#4f86c6",
                    value: [0.02, 0.035, 0.04, 0.038, 0.045, 0.042, 0.05, 0.048]
                }
            ]
        },
        jobstatus: jobStatus.Running,
        showTitle: true
    }
};

export const MinModePerformance: Story = {
    args: {
        id: "min-mode-chart",
        progressData: {
            xAxisLabels: [1, 2, 3, 4, 5],
            xLabel: "Experiments",
            yLabel: "RMSE",
            mode: "min",
            datasets: [
                {
                    label: "Best validation metric",
                    color: "#e07b54",
                    value: [0.9, 0.75, 0.6, 0.72, 0.55]
                }
            ]
        },
        jobstatus: jobStatus.Done,
        showTitle: true
    }
};
