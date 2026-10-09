// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { TextScrollArea } from "@/components/ui/text-scroll-area";
import type { Meta, StoryObj } from "@storybook/react-vite";

const meta: Meta<typeof TextScrollArea> = {
    title: "UI/TextScrollArea",
    component: TextScrollArea,
    tags: ["autodocs"]
};

export default meta;
type Story = StoryObj<typeof TextScrollArea>;

export const Default: Story = {
    render: () => (
        <TextScrollArea
            text={
                "training_job:\n\tnum_experiments: 1\n\tmetrics:\n\t\t- mae\n\t\t- mse"
            }
        />
    )
};

export const Height_Scroll: Story = {
    render: () => (
        <TextScrollArea
            text={
                "training_job:\n\tnum_experiments: 1\n\tmetrics:\n\t\t- mae\n\t\t- mse"
            }
            height={64}
        />
    )
};

export const Loading: Story = {
    render: () => <TextScrollArea id={"loading"} loading={true} text={""} />
};
