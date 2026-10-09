// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// MultiSelectCombobox.stories.tsx
import type { Meta, StoryObj } from "@storybook/react-vite";
import { expect, userEvent, waitFor, within } from "storybook/test";
import { Select } from "@/components/ui/select";

const meta: Meta<typeof Select> = {
    title: "UI/Select",
    component: Select,
    tags: ["autodocs"],
    parameters: {
        layout: "centered"
    },
    argTypes: {
        options: {
            description: "Array of options to select from",
            control: "object"
        },
        selectedValues: {
            description: "Array of selected values (controlled)",
            control: "object"
        },
        onValuesChange: { action: "values changed" },
        placeholder: {
            description: "Placeholder text when no options are selected",
            control: "text"
        },
        emptyText: {
            description: "Text to display when no results match the search",
            control: "text"
        },
        searchPlaceholder: {
            description: "Placeholder text for the search input",
            control: "text"
        },
        disabled: {
            description: "Whether the combobox is disabled",
            control: "boolean"
        }
    }
};

export default meta;
type Story = StoryObj<typeof Select>;

const defaultOptions = [
    { value: "durian", label: "Durian" },
    { value: "elderberry", label: "Elderberry" },
    { value: "apple", label: "Apple" },
    { value: "banana", label: "Banana" },
    { value: "cherry", label: "Cherry" },

    { value: "fig", label: "Fig" },
    { value: "grape", label: "Grape" },
    { value: "honeydew", label: "Honeydew" }
];

// Default story with basic options
export const Default: Story = {
    args: {
        options: defaultOptions,
        placeholder: "Select fruits...",
        emptyText: "No fruits found.",
        search: true,
        searchPlaceholder: "Search fruits...",
        variant: "default",
        popoverClassName: "w-200",
        sortOptions: false
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const trigger = canvas.getByTestId("testid-combobox");
        await userEvent.click(trigger);
        const body = within(document.body);
        await waitFor(() => {
            expect(body.getByPlaceholderText("Search fruits...")).toBeVisible();
        });
        await waitFor(() => {
            expect(body.getByRole("option", { name: /apple/i })).toBeVisible();
        });
        await userEvent.click(body.getByRole("option", { name: /apple/i }));
        await userEvent.click(body.getByRole("option", { name: /cherry/i }));
        await waitFor(() => {
            expect(trigger).toHaveTextContent("2 selected");
        });
        await userEvent.keyboard("{Escape}");
        await waitFor(() => {
            expect(
                body.queryByRole("option", { name: /apple/i })
            ).not.toBeInTheDocument();
        });
    }
};

export const Secondary: Story = {
    args: {
        options: defaultOptions,
        placeholder: "Select fruits...",
        emptyText: "No fruits found.",
        search: true,
        searchPlaceholder: "Search fruits...",
        variant: "secondary"
    }
};

// Story with pre-selected values
export const WithPreselectedValues: Story = {
    args: {
        ...Default.args,
        selectedValues: ["apple", "cherry"],
        search: false
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await expect(canvas.getByTestId("testid-combobox")).toHaveTextContent(
            "2 selected"
        );
    }
};

export const SingleModeWithPreselectedValues: Story = {
    args: {
        ...Default.args,
        selectedValues: ["apple"],
        search: false,
        mode: "single"
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const trigger = canvas.getByTestId("testid-combobox");
        await expect(trigger).toHaveTextContent("Apple");
        await userEvent.click(trigger);
        const body = within(document.body);
        await waitFor(() => {
            expect(body.getByRole("option", { name: /banana/i })).toBeVisible();
        });
        await userEvent.keyboard("{Escape}");
        await waitFor(() => {
            expect(
                body.queryByRole("option", { name: /banana/i })
            ).not.toBeInTheDocument();
        });
    }
};

// Story with disabled state
export const Disabled: Story = {
    args: {
        ...Default.args,
        disabled: true
    }
};

// Story with many options to demonstrate scrolling
export const ManyOptions: Story = {
    args: {
        ...Default.args,
        options: [
            ...defaultOptions,
            { value: "kiwi", label: "Kiwi" },
            { value: "lemon", label: "Lemon" },
            { value: "mango", label: "Mango" },
            { value: "nectarine", label: "Nectarine" },
            { value: "orange", label: "Orange" },
            { value: "papaya", label: "Papaya" },
            { value: "quince", label: "Quince" },
            { value: "raspberry", label: "Raspberry" },
            { value: "strawberry", label: "Strawberry" },
            { value: "tangerine", label: "Tangerine" }
        ],
        search: true
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await userEvent.click(canvas.getByTestId("testid-combobox"));
        const body = within(document.body);
        await waitFor(() => {
            expect(body.getByPlaceholderText("Search fruits...")).toBeVisible();
        });
        await userEvent.type(
            body.getByPlaceholderText("Search fruits..."),
            "mango"
        );
        await waitFor(() => {
            expect(body.getByRole("option", { name: /mango/i })).toBeVisible();
        });
        await userEvent.keyboard("{Escape}");
        await waitFor(() => {
            expect(
                body.queryByRole("option", { name: /mango/i })
            ).not.toBeInTheDocument();
        });
    }
};

// Story with some disabled options
export const WithDisabledOptions: Story = {
    args: {
        ...Default.args,
        options: [
            { value: "apple", label: "Apple" },
            { value: "banana", label: "Banana", disabled: true },
            { value: "cherry", label: "Cherry" },
            { value: "durian", label: "Durian", disabled: true },
            { value: "elderberry", label: "Elderberry" }
        ],
        search: true
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await userEvent.click(canvas.getByTestId("testid-combobox"));
        const body = within(document.body);
        await waitFor(() => {
            expect(body.getByRole("option", { name: /apple/i })).toBeVisible();
            expect(
                body.getByRole("option", { name: /banana/i })
            ).toHaveAttribute("data-disabled");
            expect(
                body.getByRole("option", { name: /durian/i })
            ).toHaveAttribute("data-disabled");
        });
        await userEvent.keyboard("{Escape}");
        await waitFor(() => {
            expect(
                body.queryByRole("option", { name: /apple/i })
            ).not.toBeInTheDocument();
        });
    }
};

export const WithEntity: Story = {
    args: {
        ...Default.args,
        options: [
            { value: "<add_kumo_entity>", label: "Add New table" },
            {
                value: "banana",
                label: "pqueryf100ef7ed0ea2cb652d77d5f427c5f95f100ef7ed0ea2cb652d77d5f427c5f95"
            },
            { value: "cherry", label: "Cherry" },
            { value: "durian", label: "Durian" },
            { value: "elderberry", label: "Elderberry" }
        ],
        search: true,
        multiModeText: "Tags",
        popoverClassName: "w-[450px]"
    }
};

export const SingleSelectWithEntity: Story = {
    args: {
        ...Default.args,
        options: [
            { value: "<add_kumo_entity>", label: "Add New table" },
            { value: "banana", label: "Banana" },
            { value: "cherry", label: "Cherry" },
            { value: "durian", label: "Durian" },
            { value: "elderberry", label: "Elderberry" }
        ],
        search: true,
        mode: "single"
    }
};

export const SingleSelectSecondaryWithEntity: Story = {
    args: {
        ...Default.args,
        options: [
            { value: "<add_kumo_entity>", label: "Add New table" },
            { value: "banana", label: "Banana" },
            { value: "cherry", label: "Cherry" },
            { value: "durian", label: "Durian" },
            { value: "elderberry", label: "Elderberry" }
        ],
        search: true,
        mode: "single",
        variant: "secondary"
    }
};

export const WithCheckbox: Story = {
    args: {
        ...Default.args,
        showCheckbox: true,
        search: false
    }
};
