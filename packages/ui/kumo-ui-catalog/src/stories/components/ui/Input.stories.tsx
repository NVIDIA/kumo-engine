// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// src/components/ui/input.stories.tsx
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { CalendarBasic } from "@/icons";
import type { Meta, StoryObj } from "@storybook/react-vite";
import { Lock, Search, User, X } from "lucide-react";
import { useState } from "react";
import { expect, userEvent, within } from "storybook/test";

const meta: Meta<typeof Input> = {
    title: "UI/Input",
    component: Input,
    tags: ["autodocs"],
    argTypes: {
        disabled: {
            control: "boolean",
            description: "Whether the input is disabled"
        },
        type: {
            control: "select",
            options: [
                "text",
                "password",
                "email",
                "number",
                "tel",
                "url",
                "search",
                "date"
            ],
            description: "The type of input field"
        },
        placeholder: {
            control: "text",
            description: "Placeholder text for the input"
        }
    }
};

export default meta;
type Story = StoryObj<typeof Input>;

// Basic Input
export const Basic: Story = {
    args: {
        placeholder: "Enter your name",
        "aria-label": "Enter your name"
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const input = canvas.getByRole("textbox", {
            name: /enter your name/i
        });
        await expect(input).toBeVisible();
        await userEvent.type(input, "John Doe");
        await expect(input).toHaveValue("John Doe");
    }
};

// Input with Label
export const WithLabel: Story = {
    render: () => (
        <div className="grid w-full max-w-sm items-center gap-1.5">
            <Label htmlFor="name">Name</Label>
            <Input id="name" placeholder="Enter your name" />
        </div>
    ),
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const input = canvas.getByRole("textbox", { name: /name/i });
        await expect(input).toBeVisible();
        await userEvent.type(input, "Jane");
        await expect(input).toHaveValue("Jane");
    }
};

// Disabled Input
export const Disabled: Story = {
    args: {
        disabled: true,
        placeholder: "This input is disabled",
        value: "Disabled value"
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const input = canvas.getByDisplayValue("Disabled value");
        await expect(input).toBeDisabled();
    }
};

// Input with Icon
export const WithIcon: Story = {
    render: () => (
        <div className="grid w-full max-w-sm gap-4">
            <div className="relative">
                <Search className="text-muted-foreground absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2" />
                <Input
                    className="pl-9"
                    placeholder="Search..."
                    aria-label="Search"
                />
            </div>

            <div className="relative">
                <CalendarBasic className="absolute left-3 top-1/2 size-4 -translate-y-1/2" />
                <Input className="pl-9" type="date" aria-label="Select date" />
            </div>
        </div>
    ),
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);

        const searchInput = canvas.getByRole("textbox", { name: /search/i });
        await expect(searchInput).toBeVisible();

        await userEvent.type(searchInput, "test query");
        await expect(searchInput).toHaveValue("test query");
    }
};

// Input Sizes
export const Sizes: Story = {
    render: () => (
        <div className="grid w-full max-w-sm gap-4">
            <Input
                className="h-8 text-xs"
                placeholder="Small input"
                aria-label="Small input"
            />
            <Input placeholder="Default input" aria-label="Default input" />
            <Input
                className="h-12 text-lg"
                placeholder="Large input"
                aria-label="Large input"
            />
        </div>
    ),
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const smallInput = canvas.getByRole("textbox", {
            name: /small input/i
        });
        const defaultInput = canvas.getByRole("textbox", {
            name: /default input/i
        });
        const largeInput = canvas.getByRole("textbox", {
            name: /large input/i
        });
        await expect(smallInput).toBeVisible();
        await expect(defaultInput).toBeVisible();
        await expect(largeInput).toBeVisible();
    }
};

// Input with Validation
export const WithValidation: Story = {
    render: function ValidationInput() {
        const [value, setValue] = useState("");
        const [isValid, setIsValid] = useState(true);

        const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
            const newValue = e.target.value;
            setValue(newValue);
            setIsValid(newValue.length >= 3);
        };

        return (
            <div className="grid w-full max-w-sm items-center gap-1.5">
                <Label htmlFor="validation">Username (min 3 characters)</Label>
                <div className="relative">
                    <Input
                        id="validation"
                        value={value}
                        onChange={handleChange}
                        className={`${!isValid && value ? "border-red-500 focus-visible:ring-red-500" : ""}`}
                        placeholder="Enter username"
                    />
                    {!isValid && value && (
                        <p className="mt-1 text-xs text-red-500">
                            Username must be at least 3 characters
                        </p>
                    )}
                </div>
            </div>
        );
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const input = canvas.getByPlaceholderText("Enter username");
        await userEvent.type(input, "ab");
        await expect(
            canvas.getByText("Username must be at least 3 characters")
        ).toBeVisible();
        await userEvent.type(input, "c");
        await expect(
            canvas.queryByText("Username must be at least 3 characters")
        ).not.toBeInTheDocument();
    }
};

// Input with Clear Button
export const WithClearButton: Story = {
    render: function ClearableInput() {
        const [value, setValue] = useState("");

        const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
            setValue(e.target.value);
        };

        const handleClear = () => {
            setValue("");
        };

        return (
            <div className="grid w-full max-w-sm items-center gap-1.5">
                <Label htmlFor="clearable">Clearable Input</Label>
                <div className="relative">
                    <Input
                        id="clearable"
                        value={value}
                        onChange={handleChange}
                        placeholder="Type something..."
                    />
                    {value && (
                        <button
                            type="button"
                            aria-label="Clear input"
                            className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-500 hover:text-gray-700"
                            onClick={handleClear}>
                            <X className="h-4 w-4" />
                        </button>
                    )}
                </div>
            </div>
        );
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const input = canvas.getByPlaceholderText("Type something...");
        await userEvent.type(input, "Hello");
        await expect(input).toHaveValue("Hello");

        const clearButton = canvas.getByRole("button", {
            name: /clear input/i
        });
        await userEvent.click(clearButton);
        await expect(input).toHaveValue("");
    }
};

// Input Form Group
export const FormGroup: Story = {
    render: () => (
        <div className="w-full max-w-md rounded-lg border p-6">
            <h3 className="mb-4 text-lg font-medium">Sign In</h3>
            <div className="space-y-4">
                <div className="space-y-2">
                    <Label htmlFor="email-form">Email</Label>
                    <div className="relative">
                        <User className="text-muted-foreground absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2" />
                        <Input
                            id="email-form"
                            className="pl-9"
                            type="email"
                            placeholder="you@example.com"
                        />
                    </div>
                </div>

                <div className="space-y-2">
                    <Label htmlFor="password-form">Password</Label>
                    <div className="relative">
                        <Lock className="text-muted-foreground absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2" />
                        <Input
                            id="password-form"
                            className="pl-9"
                            type="password"
                            placeholder="••••••••"
                        />
                    </div>
                </div>
            </div>
        </div>
    ),
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const emailInput = canvas.getByLabelText("Email");
        const passwordInput = canvas.getByLabelText("Password");
        await userEvent.type(emailInput, "test@example.com");
        await expect(emailInput).toHaveValue("test@example.com");
        await userEvent.type(passwordInput, "secret123");
        await expect(passwordInput).toHaveValue("secret123");
    }
};

// Input Types
export const InputTypes: Story = {
    render: () => (
        <div className="grid w-full max-w-sm gap-4">
            <div className="space-y-2">
                <Label htmlFor="text">Text</Label>
                <Input id="text" type="text" placeholder="Text input" />
            </div>

            <div className="space-y-2">
                <Label htmlFor="number">Number</Label>
                <Input id="number" type="number" placeholder="Number input" />
            </div>

            <div className="space-y-2">
                <Label htmlFor="date">Date</Label>
                <Input id="date" type="date" />
            </div>

            <div className="space-y-2">
                <Label htmlFor="time">Time</Label>
                <Input id="time" type="time" />
            </div>

            <div className="space-y-2">
                <Label htmlFor="file">File</Label>
                <Input id="file" type="file" />
            </div>

            <div className="space-y-2">
                <Label htmlFor="color">Color</Label>
                <Input id="color" type="color" className="h-10 px-1 py-1" />
            </div>
        </div>
    ),
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const textInput = canvas.getByLabelText("Text");
        await userEvent.type(textInput, "Hello");
        await expect(textInput).toHaveValue("Hello");
        const numberInput = canvas.getByLabelText("Number");
        await userEvent.type(numberInput, "42");
        await expect(numberInput).toHaveValue(42);
    }
};

// Custom Styled Input
export const CustomStyled: Story = {
    render: () => (
        <div className="grid w-full max-w-sm gap-4">
            <Input
                placeholder="Primary input"
                aria-label="Primary input"
                className="border-primary text-primary focus-visible:ring-primary"
            />

            <Input
                placeholder="Success input"
                aria-label="Success input"
                className="border-green-500 focus-visible:ring-green-500"
            />

            <Input
                placeholder="Warning input"
                aria-label="Warning input"
                className="border-yellow-500 focus-visible:ring-yellow-500"
            />

            <Input
                placeholder="Danger input"
                aria-label="Danger input"
                className="border-red-500 focus-visible:ring-red-500"
            />

            <Input
                placeholder="No border input"
                aria-label="No border input"
                className="focus-visible:border-b-primary rounded-none border-0 border-b border-b-gray-300 px-0 focus-visible:ring-0"
            />
        </div>
    ),
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const primaryInput = canvas.getByRole("textbox", {
            name: /primary input/i
        });
        await expect(primaryInput).toBeVisible();
        await userEvent.type(primaryInput, "styled");
        await expect(primaryInput).toHaveValue("styled");
    }
};
