// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MetricCard } from "@/components/ui/metric-card";
import { MetricItem } from "@/types/metric-card";
import { Meta, StoryObj } from "@storybook/react-vite";

const meta: Meta<typeof MetricCard> = {
    title: "UI/MetricCard",
    component: MetricCard,
    parameters: {
        layout: "centered"
    },
    tags: ["autodocs"],
    argTypes: {}
};

export default meta;
type Story = StoryObj<typeof MetricCard>;

const metricList: MetricItem[] = [
    {
        name: "AP",
        val: 0.5
    },
    {
        name: "AUPRC",
        val: 0.88
    },
    {
        name: "AUPRC",
        val: "1.067"
    }
];

export const MetricCarddefault: Story = {
    render: () => <MetricCard metricList={metricList} id={"Metric-card"} />
};
