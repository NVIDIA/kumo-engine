// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Meta, StoryObj } from "@storybook/react-vite";
import { expect, fn, userEvent, within } from "storybook/test";

const meta = {
    title: "UI/RadioGroup",
    component: RadioGroup,
    parameters: {
        layout: "centered"
    },
    tags: ["autodocs"],
    args: {
        onValueChange: fn()
    },
    argTypes: {
        defaultValue: {
            control: "text",
            description: "The default selected value"
        },
        disabled: {
            control: "boolean",
            description: "Whether the entire group is disabled"
        }
    }
} satisfies Meta<typeof RadioGroup>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
    render: () => (
        <RadioGroup defaultValue="option-1">
            <div className="flex items-center space-x-2">
                <RadioGroupItem value="option-1" id="option-1" />
                <Label htmlFor="option-1">Option 1</Label>
            </div>
            <div className="flex items-center space-x-2">
                <RadioGroupItem value="option-2" id="option-2" />
                <Label htmlFor="option-2">Option 2</Label>
            </div>
            <div className="flex items-center space-x-2">
                <RadioGroupItem value="option-3" id="option-3" />
                <Label htmlFor="option-3">Option 3</Label>
            </div>
        </RadioGroup>
    ),
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);

        const radio1 = canvas.getByRole("radio", { name: /option 1/i });
        const radio2 = canvas.getByRole("radio", { name: /option 2/i });
        await expect(radio1).toBeChecked();

        await userEvent.click(radio2);
        await expect(radio2).toBeChecked();
        await expect(radio1).not.toBeChecked();
    }
};

export const Horizontal: Story = {
    render: () => (
        <RadioGroup defaultValue="small" className="flex flex-row gap-4">
            <div className="flex items-center space-x-2">
                <RadioGroupItem value="small" id="small" />
                <Label htmlFor="small">Small</Label>
            </div>
            <div className="flex items-center space-x-2">
                <RadioGroupItem value="medium" id="medium" />
                <Label htmlFor="medium">Medium</Label>
            </div>
            <div className="flex items-center space-x-2">
                <RadioGroupItem value="large" id="large" />
                <Label htmlFor="large">Large</Label>
            </div>
        </RadioGroup>
    ),
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const small = canvas.getByRole("radio", { name: /small/i });
        const medium = canvas.getByRole("radio", { name: /medium/i });
        await expect(small).toBeChecked();
        await userEvent.click(medium);
        await expect(medium).toBeChecked();
        await expect(small).not.toBeChecked();
    }
};

export const WithDisabledOption: Story = {
    render: () => (
        <RadioGroup defaultValue="active">
            <div className="flex items-center space-x-2">
                <RadioGroupItem value="active" id="active" />
                <Label htmlFor="active">Active</Label>
            </div>
            <div className="flex items-center space-x-2">
                <RadioGroupItem value="disabled" id="disabled" disabled />
                <Label htmlFor="disabled" className="opacity-50">
                    Disabled
                </Label>
            </div>
            <div className="flex items-center space-x-2">
                <RadioGroupItem value="archived" id="archived" />
                <Label htmlFor="archived">Archived</Label>
            </div>
        </RadioGroup>
    ),
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const active = canvas.getByRole("radio", { name: /active/i });
        const disabled = canvas.getByRole("radio", { name: /disabled/i });
        const archived = canvas.getByRole("radio", { name: /archived/i });
        await expect(active).toBeChecked();
        await expect(disabled).toBeDisabled();
        await userEvent.click(archived);
        await expect(archived).toBeChecked();
        await expect(active).not.toBeChecked();
    }
};

export const DisabledGroup: Story = {
    render: () => (
        <RadioGroup defaultValue="option-1" disabled>
            <div className="flex items-center space-x-2">
                <RadioGroupItem value="option-1" id="disabled-1" />
                <Label htmlFor="disabled-1" className="opacity-50">
                    Option 1
                </Label>
            </div>
            <div className="flex items-center space-x-2">
                <RadioGroupItem value="option-2" id="disabled-2" />
                <Label htmlFor="disabled-2" className="opacity-50">
                    Option 2
                </Label>
            </div>
        </RadioGroup>
    ),
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const radios = canvas.getAllByRole("radio");
        for (const radio of radios) {
            await expect(radio).toBeDisabled();
        }
        await expect(radios[0]).toBeChecked();
    }
};

export const WithDescriptions: Story = {
    render: () => (
        <RadioGroup defaultValue="comfortable">
            <div className="flex items-start space-x-2">
                <RadioGroupItem value="compact" id="compact" className="mt-1" />
                <div>
                    <Label htmlFor="compact">Compact</Label>
                    <p className="text-muted-foreground text-sm">
                        Minimal spacing between items.
                    </p>
                </div>
            </div>
            <div className="flex items-start space-x-2">
                <RadioGroupItem
                    value="comfortable"
                    id="comfortable"
                    className="mt-1"
                />
                <div>
                    <Label htmlFor="comfortable">Comfortable</Label>
                    <p className="text-muted-foreground text-sm">
                        Default spacing for most use cases.
                    </p>
                </div>
            </div>
            <div className="flex items-start space-x-2">
                <RadioGroupItem
                    value="spacious"
                    id="spacious"
                    className="mt-1"
                />
                <div>
                    <Label htmlFor="spacious">Spacious</Label>
                    <p className="text-muted-foreground text-sm">
                        Extra room for readability.
                    </p>
                </div>
            </div>
        </RadioGroup>
    )
};

export const NoDefaultSelection: Story = {
    render: () => (
        <RadioGroup>
            <div className="flex items-center space-x-2">
                <RadioGroupItem value="yes" id="yes" />
                <Label htmlFor="yes">Yes</Label>
            </div>
            <div className="flex items-center space-x-2">
                <RadioGroupItem value="no" id="no" />
                <Label htmlFor="no">No</Label>
            </div>
        </RadioGroup>
    ),
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const yes = canvas.getByRole("radio", { name: /yes/i });
        const no = canvas.getByRole("radio", { name: /no/i });

        await expect(yes).not.toBeChecked();
        await expect(no).not.toBeChecked();

        await userEvent.click(yes);
        await expect(yes).toBeChecked();

        await userEvent.click(no);
        await expect(no).toBeChecked();
        await expect(yes).not.toBeChecked();
    }
};
