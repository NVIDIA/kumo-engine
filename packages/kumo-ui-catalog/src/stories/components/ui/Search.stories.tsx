// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { SearchBar } from "@/components/ui/search";
import { Meta, StoryObj } from "@storybook/react-vite";
import { expect, fn, userEvent, within } from "storybook/test";

const meta = {
    title: "UI/SearchBar",
    component: SearchBar,
    parameters: {
        layout: "centered"
    },
    tags: ["autodocs"],
    args: {
        onChange: fn()
    },
    argTypes: {
        variant: {
            control: "select",
            options: ["default", "secondary"],
            description: "Visual style variant"
        },
        position: {
            control: "select",
            options: ["default", "top"],
            description: "Position style (top removes bottom border radius)"
        },
        placeholder: {
            control: "text",
            description: "Placeholder text"
        },
        disabled: {
            control: "boolean",
            description: "Whether the input is disabled"
        },
        className: {
            control: "text",
            description: "Additional CSS classes"
        }
    }
} satisfies Meta<typeof SearchBar>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
    args: {
        placeholder: "Search...",
        className: "w-[300px]",
        "aria-label": "Search"
    },
    play: async ({ canvasElement, args }) => {
        const canvas = within(canvasElement);
        const input = canvas.getByRole("textbox", { name: /search/i });

        await userEvent.type(input, "hello");
        await expect(input).toHaveValue("hello");
        await expect(args.onChange).toHaveBeenCalled();
    }
};

export const Secondary: Story = {
    args: {
        variant: "secondary",
        placeholder: "Search...",
        className: "w-[300px]",
        "aria-label": "Search"
    }
};

export const PositionTop: Story = {
    args: {
        position: "top",
        placeholder: "Search...",
        className: "w-[300px]",
        "aria-label": "Search"
    }
};

export const WithValue: Story = {
    args: {
        defaultValue: "Example search term",
        className: "w-[300px]",
        "aria-label": "Search"
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const input = canvas.getByRole("textbox", { name: /search/i });
        await expect(input).toHaveValue("Example search term");
    }
};

export const Disabled: Story = {
    args: {
        placeholder: "Search disabled...",
        disabled: true,
        className: "w-[300px]",
        "aria-label": "Search"
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await expect(
            canvas.getByRole("textbox", { name: /search/i })
        ).toBeDisabled();
    }
};

export const AllVariants: Story = {
    render: () => (
        <div className="flex w-[300px] flex-col gap-4">
            <SearchBar
                variant="default"
                placeholder="Default variant"
                aria-label="Default variant search"
            />
            <SearchBar
                variant="secondary"
                placeholder="Secondary variant"
                aria-label="Secondary variant search"
            />
            <SearchBar
                position="top"
                placeholder="Position top"
                aria-label="Position top search"
            />
        </div>
    )
};
