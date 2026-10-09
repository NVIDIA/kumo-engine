// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Meta, StoryObj } from "@storybook/react-vite";
import { expect, fn, within } from "storybook/test";

import {
    PropertiesOverviewCard,
    type OverviewProperties
} from "@/components/blocks/card/properties-overview-card";

const sampleProperties: OverviewProperties[] = [
    { name: "Project ID", value: "proj_8f2a91", copyText: "proj_8f2a91" },
    { name: "Owner", value: "harshithv@kumo.ai" },
    { name: "Status", value: "Active" },
    {
        name: "Region",
        value: "us-east-1",
        tooltip: "Primary deployment region"
    },
    { name: "Created", value: "2026-01-15" }
];

const meta = {
    title: "Components/Blocks/PropertiesOverviewCard",
    component: PropertiesOverviewCard,
    tags: ["autodocs"],
    parameters: { layout: "padded" },
    args: {
        properties: sampleProperties,
        onTagAdd: fn(),
        onTagRemove: fn()
    },
    argTypes: {
        variant: {
            control: "select",
            options: ["properties", "project"]
        },
        titleSize: {
            control: "select",
            options: ["sm", "md", "lg", "xl"]
        },
        hideTitle: { control: "boolean" },
        loading: { control: "boolean" },
        asCol: { control: "boolean" }
    }
} satisfies Meta<typeof PropertiesOverviewCard>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
    args: { title: "Overview" },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await expect(canvas.getByText("Overview")).toBeInTheDocument();
        await expect(canvas.getByText("Project ID")).toBeInTheDocument();
    }
};

export const WithQuery: Story = {
    args: {
        title: "Overview",
        queryContent: <code>SELECT * FROM events WHERE status = 'active'</code>
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await expect(canvas.getByText("Query")).toBeInTheDocument();
    }
};

export const WithTags: Story = {
    args: {
        title: "Overview",
        tags: {
            environment: "production",
            team: "ml-platform"
        }
    }
};

export const Project: Story = {
    //  titleSize="md" matches how the project variant is used in production
    // (ProjectSettings, ProjectRestore): an h4 title above h5 property labels
    // keeps the heading order valid. The default "xl" would render an h2 title,
    // skipping to h5 and tripping the a11y heading-order rule.
    args: { variant: "project", title: "Project summary", titleSize: "md" },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await expect(canvas.getByText("Project summary")).toBeInTheDocument();
        await expect(canvas.getByText("Owner")).toBeInTheDocument();
    }
};

export const ProjectAsColumn: Story = {
    args: {
        variant: "project",
        title: "Project summary",
        titleSize: "md",
        asCol: true
    }
};

export const HideTitle: Story = {
    args: { variant: "project", hideTitle: true },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await expect(canvas.queryByText("Project summary")).toBeNull();
        await expect(canvas.getByText("Owner")).toBeInTheDocument();
    }
};

export const Loading: Story = {
    args: { variant: "project", loading: true }
};
