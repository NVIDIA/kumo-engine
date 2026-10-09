// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Meta, StoryObj } from "@storybook/react-vite";
import { ValidationCard } from "@/components/blocks/card/validation-card";

const meta = {
    title: "Components/Blocks/ValidationCard",
    component: ValidationCard,
    tags: ["autodocs"],
    parameters: { layout: "centered" },
    argTypes: {
        variant: { control: "select", options: ["error", "warning"] }
    },
    args: {
        variant: "error",
        data: [
            {
                title: "Missing required field",
                description:
                    "The 'email' field is required and cannot be empty."
            }
        ]
    }
} satisfies Meta<typeof ValidationCard>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {};

export const Error: Story = {
    args: {
        variant: "error",
        data: [
            {
                title: "Invalid configuration",
                description:
                    "The model configuration is invalid.\nPlease check your settings and try again."
            },
            {
                title: "Missing required field",
                description: "The 'target' field is required."
            }
        ]
    }
};

export const Warning: Story = {
    args: {
        variant: "warning",
        data: [
            {
                title: "Low confidence score",
                description:
                    "The prediction confidence is below the recommended threshold of 0.8."
            },
            {
                title: "Sparse data",
                description:
                    "Some features have less than 100 training samples."
            }
        ]
    }
};

export const NoTitle: Story = {
    args: {
        variant: "error",
        data: [
            { description: "An unexpected error occurred during processing." },
            { description: "Please retry the operation or contact support." }
        ]
    }
};

export const SingleItem: Story = {
    args: {
        variant: "warning",
        data: [
            {
                title: "Data drift detected",
                description:
                    "Feature distribution has shifted significantly from training data."
            }
        ]
    }
};

export const AllVariants: Story = {
    render: () => (
        <div className="flex flex-col gap-4">
            <ValidationCard
                variant="error"
                data={[
                    {
                        title: "Error variant",
                        description: "This is an error validation message."
                    }
                ]}
            />
            <ValidationCard
                variant="warning"
                data={[
                    {
                        title: "Warning variant",
                        description: "This is a warning validation message."
                    }
                ]}
            />
        </div>
    )
};
