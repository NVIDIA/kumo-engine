// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DocumentationLink } from "@/components/ui/documentation-link";
import type { Meta, StoryObj } from "@storybook/react-vite";
import { expect, within } from "storybook/test";

const meta: Meta<typeof DocumentationLink> = {
    title: "UI/DocumentationLink",
    component: DocumentationLink,
    tags: ["autodocs"],
    args: {
        id: "sample-docs-link",
        url: "https://kumo.ai/docs/connectors/bigquery/",
        linkText: "here",
        showExternalIcon: false
    }
};

export default meta;

type Story = StoryObj<typeof DocumentationLink>;

export const Default: Story = {
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const link = canvas.getByRole("link", { name: /here/i });

        await expect(link).toHaveAttribute(
            "href",
            "https://kumo.ai/docs/connectors/bigquery/"
        );
        await expect(link).toHaveAttribute("target", "_blank");
        await expect(link).toHaveAttribute("rel", "noreferrer");
    }
};

export const WithExternalIcon: Story = {
    args: {
        showExternalIcon: true,
        linkText: "Documentation"
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const link = canvas.getByRole("link", { name: /documentation/i });

        await expect(link).toBeVisible();
        // External-link arrow is decorative
        const arrow = link.querySelector("[aria-hidden='true']");
        await expect(arrow).not.toBeNull();
    }
};

export const InlineWithCopy: Story = {
    args: {
        linkText: "here",
        showExternalIcon: true
    },
    render: (args) => (
        <p className="text-title-secondary text-sm">
            For more information, refer to the documentation{" "}
            <DocumentationLink {...args} />.
        </p>
    )
};

export const CustomStyling: Story = {
    args: {
        className: "font-semibold underline",
        linkText: "styled link"
    }
};

export const WithPageLink: Story = {
    args: {
        url: undefined,
        pageLink: "connectors/snowflake/",
        linkText: "Snowflake connector"
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const link = canvas.getByRole("link", { name: /snowflake connector/i });

        await expect(link).toHaveAttribute(
            "href",
            "https://kumo.ai/docs/connectors/snowflake/"
        );
    }
};
