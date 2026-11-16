// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Meta, StoryObj } from "@storybook/react-vite";
import { ConfusionMetricTable } from "@/components/ui/confusion-metrics";

const meta: Meta<typeof ConfusionMetricTable> = {
    title: "UI/ConfusionMetricTable",
    component: ConfusionMetricTable
};

export default meta;

type Story = StoryObj<typeof ConfusionMetricTable>;

const baseData = {
    classes: ["False", "True"],
    title: "CONFUSION MATRIX"
};

export const Default: Story = {
    args: {
        ...baseData,
        values: [
            [
                { count: 31297, percentage: 99.8 },
                { count: 65, percentage: 0.2 }
            ],
            [
                { count: 203, percentage: 59.5 },
                { count: 138, percentage: 40.5 }
            ]
        ],
        sumsRight: [31297, 203],
        sumsBottom: [31362, 138]
    }
};

export const PerfectPrediction: Story = {
    args: {
        ...baseData,
        values: [
            [
                { count: 100, percentage: 100 },
                { count: 0, percentage: 0 }
            ],
            [
                { count: 0, percentage: 0 },
                { count: 100, percentage: 100 }
            ]
        ],
        sumsRight: [100, 0],
        sumsBottom: [100, 100]
    }
};

export const AllIncorrect: Story = {
    args: {
        ...baseData,
        values: [
            [
                { count: 0, percentage: 0 },
                { count: 100, percentage: 100 }
            ],
            [
                { count: 100, percentage: 100 },
                { count: 0, percentage: 0 }
            ]
        ],
        sumsRight: [0, 100],
        sumsBottom: [100, 100]
    }
};

export const ImbalancedClasses: Story = {
    args: {
        ...baseData,
        values: [
            [
                { count: 9500, percentage: 95 },
                { count: 500, percentage: 5 }
            ],
            [
                { count: 100, percentage: 10 },
                { count: 900, percentage: 90 }
            ]
        ],
        sumsRight: [9500, 100],
        sumsBottom: [9600, 1400]
    }
};
