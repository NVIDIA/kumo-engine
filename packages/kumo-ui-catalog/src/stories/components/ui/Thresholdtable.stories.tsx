// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Meta, StoryObj } from "@storybook/react-vite";
import { BinaryClassificationMetrics } from "@/types/threshold-metrics";
import { ThresholdMetricTable } from "@/components/ui/threshold-metrics";

const meta: Meta<typeof ThresholdMetricTable> = {
    title: "UI/ThresholdMetricTable",
    component: ThresholdMetricTable
};

export default meta;

type Story = StoryObj<typeof ThresholdMetricTable>;

const mockData: BinaryClassificationMetrics = {
    precision: [0.95, 0.8, 1],
    recall: [0.9, 0.75, 1],
    accuracy: [0.92, 0.78, 1],
    predicted_pos_rate: [0.5, 0.6, 0.4],
    f1: [0.925, 0.775, 1],
    true_positive: [10, 5, 0],
    false_positive: [0, 2, 0]
};

export const Default: Story = {
    args: {
        data: mockData,
        currentPosition: 0,
        title: "Threshold Metrics"
    }
};

export const SecondThreshold: Story = {
    args: {
        data: mockData,
        currentPosition: 1,
        title: "Threshold Metrics"
    }
};

export const EdgeCasePrecisionNA: Story = {
    args: {
        data: mockData,
        currentPosition: 2,
        title: "Threshold Metrics"
    }
};
