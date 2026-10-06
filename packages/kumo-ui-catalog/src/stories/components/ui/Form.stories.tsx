// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Meta, StoryObj } from "@storybook/react-vite";
import { expect, userEvent, within } from "storybook/test";
import {
    Form,
    FormField,
    FormDescription,
    FormError,
    FormControl,
    FormWrapper,
    FormSection,
    FormActions
} from "@/components/ui/form";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Switch } from "@/components/ui/switch";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { useState } from "react";
import { Eye, EyeOff } from "lucide-react";
import { Search } from "@/icons/MaterialIcons/Search";

const meta: Meta<typeof Form> = {
    title: "UI/Form",
    component: Form,
    tags: ["autodocs"],
    parameters: {
        layout: "padded"
    }
};

export default meta;
type Story = StoryObj<typeof meta>;

// Basic Form
export const Basic: Story = {
    render: () => (
        <Form>
            <FormField>
                <Label htmlFor="name">Name</Label>
                <FormControl id="name" placeholder="Enter your name" />
            </FormField>
        </Form>
    ),
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const input = canvas.getByLabelText("Name");
        await userEvent.type(input, "John Doe");
        await expect(input).toHaveValue("John Doe");
    }
};

// Form with Description
export const WithDescription: Story = {
    render: () => (
        <Form>
            <FormField>
                <Label htmlFor="email">Email</Label>
                <FormControl
                    id="email"
                    type="email"
                    placeholder="you@example.com"
                />
                <FormDescription>
                    We'll never share your email with anyone else.
                </FormDescription>
            </FormField>
        </Form>
    )
};

// Form with Icon
export const WithIcon: Story = {
    render: () => (
        <Form>
            <FormField>
                <FormControl id="search" placeholder="Search...">
                    <Search className="text-muted-foreground h-4 w-4" />
                </FormControl>
            </FormField>
        </Form>
    )
};

// Password Toggle Example
export const PasswordToggle: Story = {
    render: function PasswordToggleComponent() {
        const [showPassword, setShowPassword] = useState(false);

        return (
            <Form>
                <FormField>
                    <Label htmlFor="password-toggle">Password</Label>
                    <div className="relative">
                        <FormControl
                            id="password-toggle"
                            type={showPassword ? "text" : "password"}
                            placeholder="Enter password"
                            className="pr-10"
                        />
                        <button
                            type="button"
                            aria-label={
                                showPassword ? "Hide password" : "Show password"
                            }
                            onClick={() => setShowPassword(!showPassword)}
                            className="text-muted-foreground hover:text-foreground absolute right-3 top-1/2 -translate-y-1/2 transition-colors">
                            {showPassword ? (
                                <EyeOff className="h-4 w-4" />
                            ) : (
                                <Eye className="h-4 w-4" />
                            )}
                        </button>
                    </div>
                    <FormDescription>
                        Click the icon to toggle password visibility
                    </FormDescription>
                </FormField>
            </Form>
        );
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);

        const input = canvas.getByLabelText("Password");
        await expect(input).toHaveAttribute("type", "password");

        const toggleButton = canvas.getByRole("button", {
            name: /Show password/i
        });
        await userEvent.click(toggleButton);
        await expect(input).toHaveAttribute("type", "text");

        const hideButton = canvas.getByRole("button", {
            name: /Hide password/i
        });
        await userEvent.click(hideButton);
        await expect(input).toHaveAttribute("type", "password");
    }
};

// Form with Error
export const WithError: Story = {
    render: () => (
        <Form>
            <FormField>
                <Label htmlFor="username">Username</Label>
                <FormControl
                    id="username"
                    placeholder="Enter username"
                    className="border-destructive"
                />
                <FormError>
                    Username must be at least 3 characters long.
                </FormError>
            </FormField>
        </Form>
    ),
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await expect(
            canvas.getByText("Username must be at least 3 characters long.")
        ).toBeVisible();
    }
};

// Form with Textarea
export const WithTextarea: Story = {
    render: () => (
        <Form>
            <FormField>
                <Label htmlFor="message">Message</Label>
                <FormControl
                    as="textarea"
                    id="message"
                    placeholder="Enter your message"
                />
                <FormDescription>Maximum 500 characters.</FormDescription>
            </FormField>
        </Form>
    )
};

// Multiple Fields
export const MultipleFields: Story = {
    render: () => (
        <Form>
            <FormField>
                <Label htmlFor="firstname">First Name</Label>
                <FormControl id="firstname" placeholder="John" />
            </FormField>

            <FormField>
                <Label htmlFor="lastname">Last Name</Label>
                <FormControl id="lastname" placeholder="Doe" />
            </FormField>

            <FormField>
                <Label htmlFor="email-multi">Email</Label>
                <FormControl
                    id="email-multi"
                    type="email"
                    placeholder="john.doe@example.com"
                />
            </FormField>
        </Form>
    )
};

// Form with Wrapper (Fieldset)
export const WithWrapper: Story = {
    render: () => (
        <Form>
            <FormWrapper>
                <legend className="mb-2 text-lg font-semibold">
                    Personal Information
                </legend>

                <FormField>
                    <Label htmlFor="name-wrapper">Name</Label>
                    <FormControl
                        id="name-wrapper"
                        placeholder="Enter your name"
                    />
                </FormField>

                <FormField>
                    <Label htmlFor="email-wrapper">Email</Label>
                    <FormControl
                        id="email-wrapper"
                        type="email"
                        placeholder="you@example.com"
                    />
                </FormField>
            </FormWrapper>
        </Form>
    )
};

// Form with Sections
export const WithSections: Story = {
    render: () => (
        <Form>
            <FormSection>
                <h3 className="text-base font-semibold">Account Details</h3>
                <FormField>
                    <Label htmlFor="username-section">Username</Label>
                    <FormControl id="username-section" placeholder="johndoe" />
                </FormField>

                <FormField>
                    <Label htmlFor="password-section">Password</Label>
                    <FormControl
                        id="password-section"
                        type="password"
                        placeholder="••••••••"
                    />
                </FormField>
            </FormSection>

            <FormSection>
                <h3 className="text-base font-semibold">Contact Information</h3>
                <FormField>
                    <Label htmlFor="phone">Phone</Label>
                    <FormControl
                        id="phone"
                        type="tel"
                        placeholder="+1 (555) 000-0000"
                    />
                </FormField>

                <FormField>
                    <Label htmlFor="address">Address</Label>
                    <FormControl
                        as="textarea"
                        id="address"
                        placeholder="Enter your address"
                    />
                </FormField>
            </FormSection>
        </Form>
    )
};

// Form with Actions
export const WithActions: Story = {
    render: () => (
        <Form>
            <FormField>
                <Label htmlFor="title">Title</Label>
                <FormControl id="title" placeholder="Enter title" />
            </FormField>

            <FormField>
                <Label htmlFor="description">Description</Label>
                <FormControl
                    as="textarea"
                    id="description"
                    placeholder="Enter description"
                />
            </FormField>

            <FormActions>
                <Button variant="light-outline" type="button">
                    Cancel
                </Button>
                <Button type="submit">Save</Button>
            </FormActions>
        </Form>
    )
};

// Complete Form Example with Validation
export const CompleteFormExample: Story = {
    render: function CompleteFormComponent() {
        const [formData, setFormData] = useState({
            name: "",
            email: "",
            bio: ""
        });

        const [errors, setErrors] = useState({
            name: "",
            email: "",
            bio: ""
        });

        const [submitted, setSubmitted] = useState(false);

        const handleChange = (
            e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>
        ) => {
            const { id, value } = e.target;
            setFormData((prev) => ({ ...prev, [id]: value }));
            setErrors((prev) => ({ ...prev, [id]: "" }));
        };

        const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
            e.preventDefault();
            const newErrors = { name: "", email: "", bio: "" };

            if (!formData.name) {
                newErrors.name = "Name is required";
            } else if (formData.name.length < 2) {
                newErrors.name = "Name must be at least 2 characters";
            }

            if (!formData.email) {
                newErrors.email = "Email is required";
            } else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(formData.email)) {
                newErrors.email = "Please enter a valid email address";
            }

            if (formData.bio && formData.bio.length > 200) {
                newErrors.bio = "Bio must not exceed 200 characters";
            }

            if (newErrors.name || newErrors.email || newErrors.bio) {
                setErrors(newErrors);
                return;
            }

            setSubmitted(true);
            console.log("Form submitted:", formData);
        };

        const handleReset = () => {
            setFormData({ name: "", email: "", bio: "" });
            setErrors({ name: "", email: "", bio: "" });
            setSubmitted(false);
        };

        return (
            <div className="mx-auto max-w-2xl rounded-lg border p-6">
                <h2 className="mb-4 text-xl font-bold">User Registration</h2>

                {submitted && (
                    <div className="mb-4 rounded border border-green-200 bg-green-50 p-4">
                        <p className="text-green-800">
                            Form submitted successfully!
                        </p>
                    </div>
                )}

                <Form onSubmit={handleSubmit}>
                    <FormField>
                        <Label htmlFor="name">Full Name *</Label>
                        <FormControl
                            id="name"
                            value={formData.name}
                            onChange={handleChange}
                            placeholder="Enter your full name"
                            className={errors.name ? "border-destructive" : ""}
                        />
                        {errors.name ? (
                            <FormError>{errors.name}</FormError>
                        ) : (
                            <FormDescription>
                                Your name as it appears on official documents
                            </FormDescription>
                        )}
                    </FormField>

                    <FormField>
                        <Label htmlFor="email">Email Address *</Label>
                        <FormControl
                            id="email"
                            type="email"
                            value={formData.email}
                            onChange={handleChange}
                            placeholder="you@example.com"
                            className={errors.email ? "border-destructive" : ""}
                        />
                        {errors.email ? (
                            <FormError>{errors.email}</FormError>
                        ) : (
                            <FormDescription>
                                We'll use this for account notifications
                            </FormDescription>
                        )}
                    </FormField>

                    <FormField>
                        <Label htmlFor="bio">Bio (Optional)</Label>
                        <FormControl
                            as="textarea"
                            id="bio"
                            value={formData.bio}
                            onChange={handleChange}
                            placeholder="Tell us about yourself"
                            className={errors.bio ? "border-destructive" : ""}
                        />
                        {errors.bio ? (
                            <FormError>{errors.bio}</FormError>
                        ) : (
                            <FormDescription>
                                {formData.bio.length}/200 characters
                            </FormDescription>
                        )}
                    </FormField>

                    <FormActions>
                        <Button
                            variant="light-outline"
                            type="button"
                            onClick={handleReset}>
                            Reset
                        </Button>
                        <Button type="submit">Submit</Button>
                    </FormActions>
                </Form>
            </div>
        );
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);

        const submitButton = canvas.getByRole("button", { name: /Submit/i });
        await userEvent.click(submitButton);
        await expect(canvas.getByText("Name is required")).toBeVisible();
        await expect(canvas.getByText("Email is required")).toBeVisible();

        const nameInput = canvas.getByLabelText("Full Name *");
        await userEvent.type(nameInput, "A");
        await userEvent.click(submitButton);
        await expect(
            canvas.getByText("Name must be at least 2 characters")
        ).toBeVisible();

        await userEvent.clear(nameInput);
        await userEvent.type(nameInput, "John Doe");
        const emailInput = canvas.getByLabelText("Email Address *");
        await userEvent.type(emailInput, "a@b");
        await userEvent.click(submitButton);
        await expect(
            canvas.getByText("Please enter a valid email address")
        ).toBeVisible();

        await userEvent.clear(emailInput);
        await userEvent.type(emailInput, "john@example.com");
        await userEvent.click(submitButton);
        await expect(
            canvas.getByText("Form submitted successfully!")
        ).toBeVisible();

        const resetButton = canvas.getByRole("button", { name: /Reset/i });
        await userEvent.click(resetButton);
        await expect(nameInput).toHaveValue("");
        await expect(emailInput).toHaveValue("");
    }
};

// Disabled Form Fields
export const DisabledFields: Story = {
    render: () => (
        <Form>
            <FormField>
                <Label htmlFor="disabled-name">Name</Label>
                <FormControl
                    id="disabled-name"
                    disabled
                    value="John Doe"
                    placeholder="Enter your name"
                />
                <FormDescription>This field is disabled</FormDescription>
            </FormField>

            <FormField>
                <Label htmlFor="disabled-email">Email</Label>
                <FormControl
                    id="disabled-email"
                    type="email"
                    disabled
                    value="john@example.com"
                    placeholder="you@example.com"
                />
            </FormField>
        </Form>
    ),
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await expect(canvas.getByLabelText("Name")).toBeDisabled();
        await expect(canvas.getByLabelText("Email")).toBeDisabled();
    }
};

// Form with Different Input Types
export const DifferentInputTypes: Story = {
    render: () => (
        <Form>
            <FormSection>
                <FormField>
                    <Label htmlFor="text-input">Text Input</Label>
                    <FormControl
                        id="text-input"
                        type="text"
                        placeholder="Text"
                    />
                </FormField>

                <FormField>
                    <Label htmlFor="number-input">Number Input</Label>
                    <FormControl
                        id="number-input"
                        type="number"
                        placeholder="123"
                    />
                </FormField>

                <FormField>
                    <Label htmlFor="date-input">Date Input</Label>
                    <FormControl id="date-input" type="date" />
                </FormField>

                <FormField>
                    <Label htmlFor="time-input">Time Input</Label>
                    <FormControl id="time-input" type="time" />
                </FormField>

                <FormField>
                    <Label htmlFor="password-input">Password Input</Label>
                    <FormControl
                        id="password-input"
                        type="password"
                        placeholder="••••••••"
                    />
                </FormField>

                <FormField>
                    <Label htmlFor="url-input">URL Input</Label>
                    <FormControl
                        id="url-input"
                        type="url"
                        placeholder="https://example.com"
                    />
                </FormField>
            </FormSection>
        </Form>
    )
};

// Form with Checkbox
export const WithCheckbox: Story = {
    render: () => (
        <Form>
            <FormField>
                <div className="flex items-center space-x-2">
                    <Checkbox id="terms" />
                    <Label htmlFor="terms">
                        I accept the terms and conditions
                    </Label>
                </div>
            </FormField>

            <FormField>
                <div className="flex items-center space-x-2">
                    <Checkbox id="marketing" />
                    <Label htmlFor="marketing">Send me marketing emails</Label>
                </div>
                <FormDescription className="ml-6">
                    You can unsubscribe at any time.
                </FormDescription>
            </FormField>
        </Form>
    ),
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);

        const checkboxes = canvas.getAllByRole("checkbox");
        const termsCheckbox = checkboxes[0];
        const marketingCheckbox = checkboxes[1];

        await expect(termsCheckbox).not.toBeChecked();
        await expect(marketingCheckbox).not.toBeChecked();

        await userEvent.click(termsCheckbox);
        await expect(termsCheckbox).toBeChecked();

        await userEvent.click(marketingCheckbox);
        await expect(marketingCheckbox).toBeChecked();

        await userEvent.click(termsCheckbox);
        await expect(termsCheckbox).not.toBeChecked();
    }
};

// Form with Switch
export const WithSwitch: Story = {
    render: () => (
        <Form>
            <FormField>
                <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                        <Label htmlFor="notifications">
                            Email Notifications
                        </Label>
                        <FormDescription>
                            Receive emails about your account activity.
                        </FormDescription>
                    </div>
                    <Switch id="notifications" />
                </div>
            </FormField>

            <FormField>
                <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                        <Label htmlFor="marketing-emails">
                            Marketing Emails
                        </Label>
                        <FormDescription>
                            Receive emails about new products and features.
                        </FormDescription>
                    </div>
                    <Switch id="marketing-emails" />
                </div>
            </FormField>
        </Form>
    ),
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);

        const switches = canvas.getAllByRole("switch");
        const notificationsSwitch = switches[0];
        const marketingSwitch = switches[1];

        await expect(notificationsSwitch).not.toBeChecked();
        await expect(marketingSwitch).not.toBeChecked();

        await userEvent.click(notificationsSwitch);
        await expect(notificationsSwitch).toBeChecked();

        await userEvent.click(marketingSwitch);
        await expect(marketingSwitch).toBeChecked();

        await userEvent.click(notificationsSwitch);
        await expect(notificationsSwitch).not.toBeChecked();
    }
};

// Form with Switch Primary Variant
export const WithSwitchPrimary: Story = {
    render: () => (
        <Form>
            <FormField>
                <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                        <Label htmlFor="notifications-primary">
                            Email Notifications
                        </Label>
                        <FormDescription>
                            Switch with primary (purple) color variant.
                        </FormDescription>
                    </div>
                    <Switch id="notifications-primary" variant="primary" />
                </div>
            </FormField>

            <FormField>
                <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                        <Label htmlFor="marketing-primary">
                            Marketing Emails
                        </Label>
                        <FormDescription>
                            Primary variant with white thumb.
                        </FormDescription>
                    </div>
                    <Switch
                        id="marketing-primary"
                        variant="primary"
                        defaultChecked
                    />
                </div>
            </FormField>
        </Form>
    )
};

// Form with Radio Group
export const WithRadioGroup: Story = {
    render: () => (
        <Form>
            <FormField>
                <Label className="mb-3 block">Select your plan</Label>
                <RadioGroup defaultValue="basic">
                    <div className="flex items-center space-x-2">
                        <RadioGroupItem value="basic" id="plan-basic" />
                        <Label htmlFor="plan-basic">Basic</Label>
                    </div>
                    <div className="flex items-center space-x-2">
                        <RadioGroupItem value="pro" id="plan-pro" />
                        <Label htmlFor="plan-pro">Pro</Label>
                    </div>
                    <div className="flex items-center space-x-2">
                        <RadioGroupItem
                            value="enterprise"
                            id="plan-enterprise"
                        />
                        <Label htmlFor="plan-enterprise">Enterprise</Label>
                    </div>
                </RadioGroup>
                <FormDescription>
                    Choose the plan that best fits your needs.
                </FormDescription>
            </FormField>
        </Form>
    ),
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);

        const basicRadio = canvas.getByRole("radio", { name: /Basic/i });
        const proRadio = canvas.getByRole("radio", { name: /Pro/i });
        const enterpriseRadio = canvas.getByRole("radio", {
            name: /Enterprise/i
        });

        await expect(basicRadio).toBeChecked();
        await expect(proRadio).not.toBeChecked();

        await userEvent.click(proRadio);
        await expect(proRadio).toBeChecked();
        await expect(basicRadio).not.toBeChecked();

        await userEvent.click(enterpriseRadio);
        await expect(enterpriseRadio).toBeChecked();
        await expect(proRadio).not.toBeChecked();
    }
};

// Complete Form with All Components
export const CompleteFormWithAllComponents: Story = {
    render: function CompleteFormWithAllComponentsExample() {
        const [formData, setFormData] = useState({
            name: "",
            email: "",
            plan: "basic",
            notifications: false,
            marketing: false,
            terms: false
        });

        const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
            e.preventDefault();
            console.log("Form submitted:", formData);
        };

        return (
            <div className="mx-auto max-w-2xl rounded-lg border p-6">
                <h2 className="mb-4 text-xl font-bold">
                    Complete Registration
                </h2>

                <Form onSubmit={handleSubmit}>
                    <FormSection>
                        <h3 className="text-base font-semibold">
                            Personal Information
                        </h3>
                        <FormField>
                            <Label htmlFor="complete-name">Full Name</Label>
                            <FormControl
                                id="complete-name"
                                value={formData.name}
                                onChange={(
                                    e: React.ChangeEvent<HTMLInputElement>
                                ) =>
                                    setFormData({
                                        ...formData,
                                        name: e.target.value
                                    })
                                }
                                placeholder="Enter your name"
                            />
                        </FormField>

                        <FormField>
                            <Label htmlFor="complete-email">Email</Label>
                            <FormControl
                                id="complete-email"
                                type="email"
                                value={formData.email}
                                onChange={(
                                    e: React.ChangeEvent<HTMLInputElement>
                                ) =>
                                    setFormData({
                                        ...formData,
                                        email: e.target.value
                                    })
                                }
                                placeholder="you@example.com"
                            />
                        </FormField>
                    </FormSection>

                    <FormSection>
                        <h3 className="text-base font-semibold">
                            Plan Selection
                        </h3>
                        <FormField>
                            <RadioGroup
                                value={formData.plan}
                                onValueChange={(value) =>
                                    setFormData({ ...formData, plan: value })
                                }>
                                <div className="flex items-center space-x-2">
                                    <RadioGroupItem
                                        value="basic"
                                        id="final-basic"
                                    />
                                    <Label htmlFor="final-basic">
                                        Basic - Free
                                    </Label>
                                </div>
                                <div className="flex items-center space-x-2">
                                    <RadioGroupItem
                                        value="pro"
                                        id="final-pro"
                                    />
                                    <Label htmlFor="final-pro">
                                        Pro - $9/month
                                    </Label>
                                </div>
                                <div className="flex items-center space-x-2">
                                    <RadioGroupItem
                                        value="enterprise"
                                        id="final-enterprise"
                                    />
                                    <Label htmlFor="final-enterprise">
                                        Enterprise - $29/month
                                    </Label>
                                </div>
                            </RadioGroup>
                        </FormField>
                    </FormSection>

                    <FormSection>
                        <h3 className="text-base font-semibold">Preferences</h3>
                        <FormField>
                            <div className="flex items-center justify-between">
                                <div className="space-y-0.5">
                                    <Label htmlFor="complete-notifications">
                                        Email Notifications
                                    </Label>
                                    <FormDescription>
                                        Receive emails about your account
                                        activity.
                                    </FormDescription>
                                </div>
                                <Switch
                                    id="complete-notifications"
                                    variant="primary"
                                    checked={formData.notifications}
                                    onCheckedChange={(checked) =>
                                        setFormData({
                                            ...formData,
                                            notifications: !!checked
                                        })
                                    }
                                />
                            </div>
                        </FormField>

                        <FormField>
                            <div className="flex items-center justify-between">
                                <div className="space-y-0.5">
                                    <Label htmlFor="complete-marketing">
                                        Marketing Emails
                                    </Label>
                                    <FormDescription>
                                        Receive emails about new products.
                                    </FormDescription>
                                </div>
                                <Switch
                                    id="complete-marketing"
                                    checked={formData.marketing}
                                    onCheckedChange={(checked) =>
                                        setFormData({
                                            ...formData,
                                            marketing: !!checked
                                        })
                                    }
                                />
                            </div>
                        </FormField>
                    </FormSection>

                    <FormField>
                        <div className="flex items-start space-x-2">
                            <Checkbox
                                id="complete-terms"
                                checked={formData.terms}
                                onCheckedChange={(checked) =>
                                    setFormData({
                                        ...formData,
                                        terms: !!checked
                                    })
                                }
                                className="mt-1"
                            />
                            <div className="flex flex-col gap-1">
                                <Label htmlFor="complete-terms">
                                    I accept the terms and conditions
                                </Label>
                                <FormDescription>
                                    You must accept the terms to continue.
                                </FormDescription>
                            </div>
                        </div>
                    </FormField>

                    <FormActions>
                        <Button variant="light-outline" type="button">
                            Cancel
                        </Button>
                        <Button type="submit" disabled={!formData.terms}>
                            Create Account
                        </Button>
                    </FormActions>
                </Form>
            </div>
        );
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const submitButton = canvas.getByRole("button", {
            name: /Create Account/i
        });
        await expect(submitButton).toBeDisabled();

        const nameInput = canvas.getByLabelText("Full Name");
        await userEvent.type(nameInput, "Jane Doe");
        await expect(nameInput).toHaveValue("Jane Doe");

        const emailInput = canvas.getByLabelText("Email");
        await userEvent.type(emailInput, "jane@example.com");
        await expect(emailInput).toHaveValue("jane@example.com");

        const proRadio = canvas.getByRole("radio", {
            name: /Pro - \$9\/month/i
        });
        await userEvent.click(proRadio);
        await expect(proRadio).toBeChecked();

        const switches = canvas.getAllByRole("switch");
        await userEvent.click(switches[0]);
        await expect(switches[0]).toBeChecked();

        const termsCheckbox = canvas.getByRole("checkbox");
        await userEvent.click(termsCheckbox);
        await expect(termsCheckbox).toBeChecked();
        await expect(submitButton).toBeEnabled();
    }
};
