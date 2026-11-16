// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { TextInput } from "@/components/ui/text-input";
import { Meta, StoryObj } from "@storybook/react-vite";
import { useState } from "react";
import { expect, fn, userEvent, within } from "storybook/test";

type InputStateType = React.ComponentProps<typeof TextInput>["state"];

const meta = {
    title: "UI/TextInput",
    component: TextInput,
    parameters: {
        layout: "centered"
    },
    tags: ["autodocs"],
    args: {
        onChange: fn()
    },
    argTypes: {
        labelVariant: {
            control: "select",
            options: ["default", "secondary"]
        },
        state: {
            control: "select",
            options: [
                "none",
                "validating",
                "valid",
                "invalid",
                "warning",
                "validationerror"
            ]
        }
    },
    decorators: [
        (Story) => (
            <div className="w-80">
                <Story />
            </div>
        )
    ]
} satisfies Meta<typeof TextInput>;

export default meta;
type Story = StoryObj<typeof meta>;

// Helper to create stateful render function
const createStatefulRender = (initialValue: string = "") => {
    return function Render(args: React.ComponentProps<typeof TextInput>) {
        const [value, setValue] = useState(initialValue);
        return (
            <TextInput
                {...args}
                value={value}
                onChange={(e) => {
                    setValue(e.target.value);
                    args.onChange?.(e);
                }}
            />
        );
    };
};

export const Default: Story = {
    args: {
        id: "default-input",
        label: "Name",
        placeholder: "Enter your name",
        value: ""
    },
    render: createStatefulRender(),
    play: async ({ canvasElement, args }) => {
        const canvas = within(canvasElement);
        const input = canvas.getByPlaceholderText("Enter your name");

        await expect(input).toHaveValue("");

        await userEvent.type(input, "John Doe");
        await expect(input).toHaveValue("John Doe");

        await expect(args.onChange).toHaveBeenCalled();
    }
};

export const WithValue: Story = {
    args: {
        id: "with-value-input",
        label: "Email",
        placeholder: "Enter your email",
        value: "john.doe@example.com"
    },
    render: createStatefulRender("john.doe@example.com")
};

export const WithLabelTooltip: Story = {
    args: {
        id: "tooltip-input",
        label: "API Key",
        labelToolTip:
            "Your unique API key for authentication. Keep this secret!",
        placeholder: "Enter API key",
        value: ""
    },
    render: createStatefulRender()
};

export const SecondaryLabel: Story = {
    args: {
        id: "secondary-label-input",
        label: "Optional Field",
        labelVariant: "secondary",
        placeholder: "Enter value",
        value: ""
    },
    render: createStatefulRender()
};

export const Invalid: Story = {
    args: {
        id: "invalid-input",
        label: "Email",
        placeholder: "Enter your email",
        value: "invalid-email",
        isInvalid: true,
        errorMsg: "Please enter a valid email address"
    },
    render: createStatefulRender("invalid-email"),
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const input = canvas.getByPlaceholderText("Enter your email");

        await expect(input).toHaveAttribute("aria-invalid", "true");

        await expect(
            canvas.getByText("Please enter a valid email address")
        ).toBeVisible();
    }
};

export const Disabled: Story = {
    args: {
        id: "disabled-input",
        label: "Disabled Field",
        placeholder: "Cannot edit this",
        value: "Disabled content",
        readOnly: true
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const input = canvas.getByDisplayValue("Disabled content");

        await expect(input).toBeDisabled();
    }
};

export const ReadOnly: Story = {
    args: {
        id: "readonly-input",
        label: "User ID",
        value: "usr_12345abcde",
        readOnly: true
    }
};

export const Validating: Story = {
    args: {
        id: "validating-input",
        label: "Username",
        placeholder: "Enter username",
        value: "johndoe",
        showState: true,
        state: "validating" as InputStateType
    },
    render: createStatefulRender("johndoe")
};

export const Valid: Story = {
    args: {
        id: "valid-input",
        label: "Username",
        placeholder: "Enter username",
        value: "johndoe",
        showState: true,
        state: "valid" as InputStateType
    },
    render: createStatefulRender("johndoe")
};

export const Warning: Story = {
    args: {
        id: "warning-input",
        label: "Password",
        placeholder: "Enter password",
        value: "weak",
        showState: true,
        state: "warning" as InputStateType
    },
    render: createStatefulRender("weak")
};

export const ValidationError: Story = {
    args: {
        id: "validation-error-input",
        label: "Email",
        placeholder: "Enter email",
        value: "not-an-email",
        showState: true,
        state: "validationerror" as InputStateType
    },
    render: createStatefulRender("not-an-email")
};

export const PasswordType: Story = {
    args: {
        id: "password-input",
        label: "Password",
        type: "password",
        placeholder: "Enter your password",
        value: ""
    },
    render: createStatefulRender()
};

export const WithoutLabel: Story = {
    args: {
        id: "no-label-input",
        placeholder: "Search...",
        value: ""
    },
    render: createStatefulRender()
};
