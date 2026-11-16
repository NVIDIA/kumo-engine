// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { PageNotFound } from "@/components/ui/page-not-found";
import type { Meta, StoryObj } from "@storybook/react-vite";
import { expect, within } from "storybook/test";

const meta: Meta<typeof PageNotFound> = {
    title: "Components/UI/PageNotFound",
    component: PageNotFound,
    tags: ["autodocs"],
    decorators: [
        (Story) => (
            <div className="h-96 w-full">
                <Story />
            </div>
        )
    ],
    argTypes: {
        title: {
            control: "text",
            description: "Title of the page"
        },
        redirectLink: {
            control: "text",
            description: 'Path the "here" link navigates to'
        }
    }
};

export default meta;
type Story = StoryObj<typeof PageNotFound>;

export const Default: Story = {
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await expect(
            canvas.getByText("404 - Page Not Found")
        ).toBeInTheDocument();
        await expect(
            canvas.getByText("The page you are looking for does not exist.")
        ).toBeInTheDocument();
        await expect(
            canvas.getByRole("link", { name: /here/i })
        ).toHaveAttribute("href", "/");
    }
};

export const CustomTitleAndRedirect: Story = {
    args: {
        title: "404 - Not Available",
        redirectLink: "/connectors"
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await expect(
            canvas.getByText("404 - Not Available")
        ).toBeInTheDocument();
        await expect(
            canvas.getByRole("link", { name: /here/i })
        ).toHaveAttribute("href", "/connectors");
    }
};
