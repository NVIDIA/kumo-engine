// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Button } from "@/components/ui/button";
import { Meta, StoryObj } from "@storybook/react-vite";
import { expect, fn, userEvent, within } from "storybook/test";
import { ChevronRight, CloseSmall } from "@/icons";
import { AddCircle } from "@/icons";
import { Replay } from "@/icons";

// Define the metadata for the Button component
const meta = {
    title: "UI/Button",
    component: Button,
    parameters: {
        layout: "centered"
    },
    tags: ["autodocs"],
    argTypes: {
        onClick: { action: "clicked" },
        variant: {
            control: "select",
            options: [
                "primary",
                "secondary",
                "outline",
                "light-outline",
                "destructive",
                "no-outline",
                "link",
                "underline"
            ]
        },
        size: {
            control: "select",
            options: ["default", "xs", "sm", "md", "lg"]
        },
        disabled: {
            control: { type: "boolean" },
            defaultValue: false
        },
        className: {
            control: { type: "text" },
            defaultValue: ""
        }
    }
} satisfies Meta<typeof Button>;

export default meta;
type Story = StoryObj<typeof meta>;

// Comprehensive Showcase
export const AllVariants: Story = {
    render: () => (
        <div className="flex flex-wrap items-center gap-4">
            <Button variant="primary" size="sm">
                Primary
            </Button>
            <Button variant="secondary" size="sm">
                Secondary
            </Button>
            <Button variant="outline" size="sm">
                Outline
            </Button>
            <Button variant="destructive" size="sm">
                Destructive
            </Button>
            <Button variant="light-outline" size="sm">
                Light Outline
            </Button>
            <Button variant="no-outline" size="sm">
                No Outline
            </Button>
            <Button variant="link" size="sm">
                Link
            </Button>
            <Button variant="underline" size="sm">
                Underline
            </Button>
        </div>
    )
};

export const AllSizes: Story = {
    render: () => (
        <div className="flex items-center gap-2">
            <Button variant="primary" size="xs">
                XS
            </Button>
            <Button variant="primary" size="sm">
                SM
            </Button>
            <Button variant="primary" size="default">
                Default
            </Button>
            <Button variant="primary" size="md">
                MD
            </Button>
            <Button variant="primary" size="lg">
                LG
            </Button>
        </div>
    )
};

// Variant Stories
export const Primary: Story = {
    args: {
        children: "Primary Button",
        variant: "primary",
        size: "sm",
        onClick: fn()
    },
    play: async ({ canvasElement, args }) => {
        const canvas = within(canvasElement);
        const button = canvas.getByRole("button", { name: /primary button/i });
        await expect(button).toBeVisible();
        await userEvent.click(button);
        await expect(args.onClick).toHaveBeenCalledTimes(1);
    }
};

export const Secondary: Story = {
    args: {
        children: "Secondary Button",
        variant: "secondary",
        size: "sm"
    }
};

export const Outline: Story = {
    args: {
        children: "Outline Button",
        variant: "outline",
        size: "sm"
    }
};

export const SecondaryOutline: Story = {
    args: {
        children: "Secondary Outline Button",
        variant: "light-outline",
        size: "sm"
    }
};

export const Destructive: Story = {
    args: {
        children: "Delete",
        variant: "destructive",
        size: "sm"
    }
};

export const NoOutline: Story = {
    args: {
        children: "No Outline Button",
        variant: "no-outline",
        size: "sm"
    }
};

export const Link: Story = {
    args: {
        children: "Link Button",
        variant: "link",
        size: "sm"
    }
};

export const Underline: Story = {
    args: {
        children: "Underline Button",
        variant: "underline",
        size: "sm"
    }
};

// Size Stories
export const SizeXS: Story = {
    args: {
        children: "Extra Small",
        variant: "primary",
        size: "xs"
    }
};

export const SizeSM: Story = {
    args: {
        children: "Small",
        variant: "primary",
        size: "sm"
    }
};

export const SizeDefault: Story = {
    args: {
        children: "Default Size",
        variant: "primary",
        size: "default"
    }
};

export const SizeMD: Story = {
    args: {
        children: "Medium",
        variant: "primary",
        size: "md"
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await expect(
            canvas.getByRole("button", { name: /medium/i })
        ).toBeVisible();
    }
};

export const SizeLG: Story = {
    args: {
        children: "Large",
        variant: "primary",
        size: "lg"
    }
};

// Icon Examples
export const WithLeadingIcon: Story = {
    args: {
        children: (
            <>
                <AddCircle />
                Add Table
            </>
        ),
        variant: "primary",
        size: "sm"
    }
};

export const WithIconSecondary: Story = {
    args: {
        children: (
            <>
                <AddCircle />
                Add Table
            </>
        ),
        variant: "secondary",
        size: "sm"
    }
};

export const WithTrailingIcon: Story = {
    args: {
        children: (
            <>
                Start predicting
                <ChevronRight />
            </>
        ),
        variant: "primary",
        size: "sm"
    }
};

export const IconOnly: Story = {
    args: {
        children: <Replay />,
        variant: "primary",
        size: "sm",
        "aria-label": "Replay"
    }
};

export const IconOnlyFlat: Story = {
    args: {
        children: <CloseSmall />,
        variant: "primary",
        size: "sm",
        className: "!shadow-none !rounded-none !border-none",
        "aria-label": "Close"
    }
};

// State Examples
export const Disabled: Story = {
    args: {
        children: "Disabled Button",
        variant: "primary",
        size: "sm",
        disabled: true,
        onClick: fn()
    },
    play: async ({ canvasElement, args }) => {
        const canvas = within(canvasElement);
        const button = canvas.getByRole("button", { name: /disabled button/i });
        await expect(button).toBeDisabled();
        await expect(args.onClick).not.toHaveBeenCalled();
    }
};

export const AsChild: Story = {
    render: () => (
        <Button asChild variant="light-outline" size="sm">
            <a href="#">Link as Button</a>
        </Button>
    ),
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const link = canvas.getByRole("link", { name: /link as button/i });
        await expect(link).toBeVisible();
        await expect(link).toHaveAttribute("data-slot", "button");
    }
};
