// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Meta, StoryObj } from "@storybook/react-vite";
import { expect, userEvent, within } from "storybook/test";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";

const Logo = ({ label }: { label: string }) => (
    <div className="flex items-center gap-2">
        <div className="h-5 w-5 rounded-full bg-gray-300" />
        <span>{label}</span>
    </div>
);

const meta: Meta<typeof ToggleGroup> = {
    title: "UI/ToggleGroup",
    component: ToggleGroup,
    tags: ["autodocs"],
    args: {
        type: "single",
        variant: "outline",
        size: "lg",
        defaultValue: "snowflake"
    },
    argTypes: {
        type: {
            control: "select",
            options: ["single", "multiple"]
        },
        variant: {
            control: "select",
            options: ["default", "outline"]
        },
        size: {
            control: "select",
            options: ["sm", "default", "lg"]
        }
    }
};

export default meta;
type Story = StoryObj<typeof ToggleGroup>;

// Exercises variant="default" branch and size="sm"/"default" on ToggleGroupItem
export const DefaultVariant: Story = {
    args: {
        type: "single",
        variant: "default",
        size: "default",
        defaultValue: "a"
    },
    render: (args) => (
        <ToggleGroup {...args}>
            <ToggleGroupItem value="a" isFirst>
                Option A
            </ToggleGroupItem>
            <ToggleGroupItem value="b">Option B</ToggleGroupItem>
            <ToggleGroupItem value="c" isLast>
                Option C
            </ToggleGroupItem>
        </ToggleGroup>
    ),
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const b = canvas.getByRole("radio", { name: /option b/i });
        await userEvent.click(b);
        await expect(b).toHaveAttribute("data-state", "on");
    }
};

// Exercises size="sm" variant on ToggleGroupItem (line 101: context.size || size)
export const SmallSize: Story = {
    args: {
        type: "multiple",
        variant: "outline",
        size: "sm"
    },
    render: (args) => (
        <ToggleGroup {...args}>
            <ToggleGroupItem value="x" isFirst>
                X
            </ToggleGroupItem>
            <ToggleGroupItem value="y" isLast>
                Y
            </ToggleGroupItem>
        </ToggleGroup>
    )
};

export const ItemLevelVariant: Story = {
    render: () => (
        <ToggleGroup type="single" defaultValue="a">
            <ToggleGroupItem value="a" variant="outline" size="sm" isFirst>
                A
            </ToggleGroupItem>
            <ToggleGroupItem value="b" variant="outline" size="sm" isLast>
                B
            </ToggleGroupItem>
        </ToggleGroup>
    ),
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const a = canvas.getByRole("radio", { name: /^a$/i });
        await expect(a).toHaveAttribute("data-variant", "outline");
        await expect(a).toHaveAttribute("data-size", "sm");
        await userEvent.click(canvas.getByRole("radio", { name: /^b$/i }));
        await expect(
            canvas.getByRole("radio", { name: /^b$/i })
        ).toHaveAttribute("data-state", "on");
    }
};

export const DataSources: Story = {
    render: (args) => {
        const items = [
            { value: "snowflake", label: "Snowflake" },
            { value: "s3", label: "Amazon S3" },
            { value: "bigquery", label: "BigQuery (Disabled)", disabled: true },
            { value: "databricks", label: "Databricks" },
            { value: "local-upload", label: "Local Upload" }
        ];

        return (
            <ToggleGroup {...args} className="flex-wrap gap-1">
                {items.map((item, index) => (
                    <ToggleGroupItem
                        key={item.value}
                        value={item.value}
                        disabled={item.disabled}
                        isFirst={index === 0}
                        isLast={index === items.length - 1}>
                        <Logo label={item.label} />
                    </ToggleGroupItem>
                ))}
            </ToggleGroup>
        );
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);

        const snowflake = canvas.getByRole("radio", { name: /snowflake/i });
        await expect(snowflake).toHaveAttribute("data-state", "on");

        const s3 = canvas.getByRole("radio", { name: /amazon s3/i });
        await userEvent.click(s3);
        await expect(s3).toHaveAttribute("data-state", "on");
        await expect(snowflake).toHaveAttribute("data-state", "off");

        const bigquery = canvas.getByRole("radio", { name: /bigquery/i });
        await expect(bigquery).toBeDisabled();

        const databricks = canvas.getByRole("radio", {
            name: /databricks/i
        });
        await userEvent.click(databricks);
        await expect(databricks).toHaveAttribute("data-state", "on");
        await expect(s3).toHaveAttribute("data-state", "off");

        const localUpload = canvas.getByRole("radio", {
            name: /local upload/i
        });
        await userEvent.click(localUpload);
        await expect(localUpload).toHaveAttribute("data-state", "on");
        await expect(databricks).toHaveAttribute("data-state", "off");
    }
};
