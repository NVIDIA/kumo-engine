// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// src/components/ui/select/select.stories.tsx
import type { Meta, StoryObj } from "@storybook/react-vite";
import { expect, userEvent, waitFor, within } from "storybook/test";
import { Label } from "@/components/ui/label";

import {
    Select,
    SelectContent,
    SelectGroup,
    SelectItem,
    SelectLabel,
    SelectTrigger,
    SelectValue,
    SelectSeparator
} from "@/components/ui/singleselect";

const meta: Meta<typeof Select> = {
    title: "UI/SingleSelect",
    component: Select,
    tags: ["autodocs"],
    argTypes: {},
    decorators: [
        (Story) => (
            <div className="flex items-center justify-center p-8">
                <div className="w-[240px]">
                    <Story />
                </div>
            </div>
        )
    ]
};

export default meta;
type Story = StoryObj<typeof Select>;

// Basic Select
export const Basic: Story = {
    render: () => (
        <Select>
            <SelectTrigger aria-label="Select a fruit">
                <SelectValue placeholder="Select a fruit" />
            </SelectTrigger>
            <SelectContent className="w-[350px] p-4">
                <SelectItem value="apple">Apple</SelectItem>
                <SelectItem value="banana">Banana</SelectItem>
                <SelectItem value="orange">Orange</SelectItem>
                <SelectItem value="pear">Pear</SelectItem>
                <SelectItem value="strawberry">Strawberry</SelectItem>
            </SelectContent>
        </Select>
    ),
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await userEvent.click(
            canvas.getByRole("combobox", { name: /select a fruit/i })
        );
        const body = within(document.body);
        await waitFor(() => {
            expect(body.getByRole("option", { name: /apple/i })).toBeVisible();
        });

        await userEvent.click(body.getByRole("option", { name: /orange/i }));
        await waitFor(() => {
            expect(canvas.getByRole("combobox")).toHaveTextContent("Orange");
        });
    }
};

// Select with Label
export const WithLabel: Story = {
    render: () => (
        <div className="space-y-2">
            <Label htmlFor="fruit">Favorite Fruit</Label>
            <Select>
                <SelectTrigger id="fruit">
                    <SelectValue placeholder="Select a fruit" />
                </SelectTrigger>
                <SelectContent>
                    <SelectItem value="apple">Apple</SelectItem>
                    <SelectItem value="banana">Banana</SelectItem>
                    <SelectItem value="orange">Orange</SelectItem>
                    <SelectItem value="pear">Pear</SelectItem>
                    <SelectItem value="strawberry">Strawberry</SelectItem>
                </SelectContent>
            </Select>
        </div>
    )
};

// Grouped Select
export const Grouped: Story = {
    render: () => (
        <Select>
            <SelectTrigger aria-label="Select a food">
                <SelectValue placeholder="Select a food" />
            </SelectTrigger>
            <SelectContent>
                <SelectGroup>
                    <SelectLabel>Fruits</SelectLabel>
                    <SelectItem value="apple">Apple</SelectItem>
                    <SelectItem value="banana">Banana</SelectItem>
                    <SelectItem value="orange">Orange</SelectItem>
                </SelectGroup>
                <SelectSeparator />
                <SelectGroup>
                    <SelectLabel>Vegetables</SelectLabel>
                    <SelectItem value="carrot">Carrot</SelectItem>
                    <SelectItem value="potato">Potato</SelectItem>
                    <SelectItem value="broccoli">Broccoli</SelectItem>
                </SelectGroup>
            </SelectContent>
        </Select>
    )
};

// Disabled Select
export const Disabled: Story = {
    render: () => (
        <Select disabled>
            <SelectTrigger aria-label="Select a fruit">
                <SelectValue placeholder="Select a fruit" />
            </SelectTrigger>
            <SelectContent>
                <SelectItem value="apple">Apple</SelectItem>
                <SelectItem value="banana">Banana</SelectItem>
                <SelectItem value="orange">Orange</SelectItem>
            </SelectContent>
        </Select>
    ),
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);

        await expect(
            canvas.getByRole("combobox", { name: /select a fruit/i })
        ).toBeDisabled();
    }
};

// Disabled Items
export const DisabledItems: Story = {
    render: () => (
        <Select>
            <SelectTrigger aria-label="Select a fruit">
                <SelectValue placeholder="Select a fruit" />
            </SelectTrigger>
            <SelectContent>
                <SelectItem value="apple">Apple</SelectItem>
                <SelectItem value="banana" disabled>
                    Banana
                </SelectItem>
                <SelectItem value="orange">Orange</SelectItem>
                <SelectItem value="pear" disabled>
                    Pear
                </SelectItem>
                <SelectItem value="strawberry">Strawberry</SelectItem>
            </SelectContent>
        </Select>
    ),
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await userEvent.click(
            canvas.getByRole("combobox", { name: /select a fruit/i })
        );
        const body = within(document.body);
        await waitFor(() => {
            expect(
                body.getByRole("option", { name: /banana/i })
            ).toHaveAttribute("data-disabled");
            expect(body.getByRole("option", { name: /pear/i })).toHaveAttribute(
                "data-disabled"
            );
        });
        await userEvent.keyboard("{Escape}");
    }
};

// With Default Value
export const WithDefaultValue: Story = {
    render: () => (
        <Select defaultValue="orange">
            <SelectTrigger aria-label="Select a fruit">
                <SelectValue />
            </SelectTrigger>
            <SelectContent>
                <SelectItem value="apple">Apple</SelectItem>
                <SelectItem value="banana">Banana</SelectItem>
                <SelectItem value="orange">Orange</SelectItem>
                <SelectItem value="pear">Pear</SelectItem>
                <SelectItem value="strawberry">Strawberry</SelectItem>
            </SelectContent>
        </Select>
    ),
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await expect(
            canvas.getByRole("combobox", { name: /select a fruit/i })
        ).toHaveTextContent("Orange");
    }
};

// Large List with Scroll
export const LargeList: Story = {
    render: () => (
        <Select>
            <SelectTrigger aria-label="Select a country">
                <SelectValue placeholder="Select a country" />
            </SelectTrigger>
            <SelectContent>
                <SelectGroup>
                    <SelectLabel>North America</SelectLabel>
                    <SelectItem value="us">United States</SelectItem>
                    <SelectItem value="ca">Canada</SelectItem>
                    <SelectItem value="mx">Mexico</SelectItem>
                </SelectGroup>
                <SelectSeparator />
                <SelectGroup>
                    <SelectLabel>Europe</SelectLabel>
                    <SelectItem value="uk">United Kingdom</SelectItem>
                    <SelectItem value="fr">France</SelectItem>
                    <SelectItem value="de">Germany</SelectItem>
                    <SelectItem value="it">Italy</SelectItem>
                    <SelectItem value="es">Spain</SelectItem>
                    <SelectItem value="pt">Portugal</SelectItem>
                    <SelectItem value="nl">Netherlands</SelectItem>
                    <SelectItem value="be">Belgium</SelectItem>
                    <SelectItem value="ch">Switzerland</SelectItem>
                    <SelectItem value="at">Austria</SelectItem>
                    <SelectItem value="se">Sweden</SelectItem>
                    <SelectItem value="no">Norway</SelectItem>
                    <SelectItem value="dk">Denmark</SelectItem>
                    <SelectItem value="fi">Finland</SelectItem>
                </SelectGroup>
                <SelectSeparator />
                <SelectGroup>
                    <SelectLabel>Asia</SelectLabel>
                    <SelectItem value="cn">China</SelectItem>
                    <SelectItem value="jp">Japan</SelectItem>
                    <SelectItem value="kr">South Korea</SelectItem>
                    <SelectItem value="in">India</SelectItem>
                    <SelectItem value="sg">Singapore</SelectItem>
                    <SelectItem value="th">Thailand</SelectItem>
                </SelectGroup>
            </SelectContent>
        </Select>
    )
};

// Large List with Long Text
export const LargeListLongText: Story = {
    render: () => (
        <Select>
            <SelectTrigger aria-label="Select a lengthy policy">
                <SelectValue placeholder="Select a lengthy policy" />
            </SelectTrigger>
            <SelectContent className="w-[350px]">
                <SelectGroup>
                    <SelectLabel>Privacy Policies</SelectLabel>
                    <SelectItem value="policy1">
                        Data Protection Policy with Comprehensive Guidelines for
                        User Information Handling and Processing in Compliance
                        with International Regulations
                    </SelectItem>
                    <SelectItem value="policy2">
                        Terms and Conditions Agreement for Software Usage
                        Including Licensing Restrictions and Warranty
                        Disclaimers for Enterprise Applications
                    </SelectItem>
                    <SelectItem value="policy3">
                        Cookie Consent Implementation Strategy with Third-Party
                        Tracking Limitations and User Opt-Out Mechanisms for
                        E-commerce Platforms
                    </SelectItem>
                    <SelectItem value="policy4">
                        Data Retention Schedule with Automated Purging
                        Guidelines and Legal Hold Exceptions for Financial
                        Institutions
                    </SelectItem>
                    <SelectItem value="policy5">
                        Cross-Border Data Transfer Agreement with Standard
                        Contractual Clauses and Regional Compliance
                        Specifications for Multinational Organizations
                    </SelectItem>
                </SelectGroup>
                <SelectSeparator />
                <SelectGroup>
                    <SelectLabel>Security Protocols</SelectLabel>
                    <SelectItem value="security1">
                        Multi-Factor Authentication Implementation Guidelines
                        with Hardware Token Integration and Biometric
                        Verification for Secure Access Management
                    </SelectItem>
                    <SelectItem value="security2">
                        Encryption Standards for Data-at-Rest with Key Rotation
                        Policies and Hardware Security Module Integration for
                        Sensitive Information
                    </SelectItem>
                    <SelectItem value="security3">
                        Incident Response Protocol with Severity Classification
                        Matrix and Stakeholder Communication Templates for
                        Security Breach Management
                    </SelectItem>
                    <SelectItem value="security4">
                        Vulnerability Assessment Schedule with Critical Systems
                        Prioritization and Remediation Timeline Requirements for
                        IT Infrastructure
                    </SelectItem>
                    <SelectItem value="security5">
                        Physical Security Guidelines with Access Control
                        Specifications and Environmental Monitoring Requirements
                        for Data Center Facilities
                    </SelectItem>
                </SelectGroup>
                <SelectSeparator />
                <SelectGroup>
                    <SelectLabel>Compliance Frameworks</SelectLabel>
                    <SelectItem value="compliance1">
                        General Data Protection Regulation Implementation
                        Strategy with Data Subject Rights Fulfillment Procedures
                        and Documentation Requirements
                    </SelectItem>
                    <SelectItem value="compliance2">
                        Health Insurance Portability and Accountability Act
                        Compliance Program with Protected Health Information
                        Safeguards and Breach Notification Procedures
                    </SelectItem>
                    <SelectItem value="compliance3">
                        Payment Card Industry Data Security Standard Adherence
                        Guidelines with Cardholder Data Environment Segmentation
                        and Quarterly Scanning Requirements
                    </SelectItem>
                    <SelectItem value="compliance4">
                        California Consumer Privacy Act Implementation Framework
                        with Consumer Request Handling Procedures and
                        Do-Not-Sell Mechanism Specifications
                    </SelectItem>
                    <SelectItem value="compliance5">
                        System and Organization Controls Audit Preparation Guide
                        with Trust Services Criteria Documentation and Evidence
                        Collection Methodology
                    </SelectItem>
                </SelectGroup>
            </SelectContent>
        </Select>
    )
};

export const LongUnbrokenStrings: Story = {
    render: () => (
        <Select>
            <SelectTrigger aria-label="Select a long string">
                <SelectValue placeholder="Select a long string" />
            </SelectTrigger>
            <SelectContent className="w-[300px]">
                <SelectItem value="short">Normal short option</SelectItem>
                <SelectItem value="long1">
                    Thisisanextremelylongwordwithoutanyspacesorbreakstodemonstratehowtheselectcomponenthandleslongcontinuousstringsofcharacterswithoutanynaturalbreakpointsorwordwrappingopportunities
                </SelectItem>
                <SelectItem value="long2">
                    Supercalifragilisticexpialidocioussupercalifragilisticexpialidocioussupercalifragilisticexpialidocioussupercalifragilisticexpialidocioussupercalifragilisticexpialidocious
                </SelectItem>
                <SelectItem value="long3">
                    ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZ
                </SelectItem>
                <SelectItem value="long4">
                    ThisIsACombinationOfCamelCaseAndPascalCaseWithNoSpacesOrSeparatorsToTestHowTheSelectComponentHandlesTextWithoutExplicitBreakpointsButWithCaseChangesAsSubtleVisualDividers
                </SelectItem>
                <SelectItem value="long5">
                    12345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890
                </SelectItem>
            </SelectContent>
        </Select>
    )
};

// Custom Styled Select
export const CustomStyled: Story = {
    render: () => (
        <Select>
            <SelectTrigger
                aria-label="Select a theme"
                className="border-primary bg-primary/5 text-primary">
                <SelectValue placeholder="Select a theme" />
            </SelectTrigger>
            <SelectContent className="border-primary/20">
                <SelectItem
                    value="light"
                    className="text-primary hover:text-primary">
                    Light
                </SelectItem>
                <SelectItem
                    value="dark"
                    className="text-primary hover:text-primary">
                    Dark
                </SelectItem>
                <SelectItem
                    value="system"
                    className="text-primary hover:text-primary">
                    System
                </SelectItem>
            </SelectContent>
        </Select>
    )
};

// Dark Mode Variant
export const DarkMode: Story = {
    render: () => (
        <div className="dark rounded-lg bg-slate-950 p-8">
            <Select>
                <SelectTrigger aria-label="Select a fruit">
                    <SelectValue placeholder="Select a fruit" />
                </SelectTrigger>
                <SelectContent>
                    <SelectItem value="apple">Apple</SelectItem>
                    <SelectItem value="banana">Banana</SelectItem>
                    <SelectItem value="orange">Orange</SelectItem>
                    <SelectItem value="pear">Pear</SelectItem>
                    <SelectItem value="strawberry">Strawberry</SelectItem>
                </SelectContent>
            </Select>
        </div>
    )
};

// Form Like Example
export const FormExample: Story = {
    render: () => (
        <div className="space-y-4 rounded-md border p-4">
            <div className="space-y-2">
                <Label htmlFor="category">Category</Label>
                <Select>
                    <SelectTrigger id="category">
                        <SelectValue placeholder="Select category" />
                    </SelectTrigger>
                    <SelectContent>
                        <SelectItem value="electronics">Electronics</SelectItem>
                        <SelectItem value="clothing">Clothing</SelectItem>
                        <SelectItem value="books">Books</SelectItem>
                        <SelectItem value="home">Home & Kitchen</SelectItem>
                    </SelectContent>
                </Select>
            </div>

            <div className="space-y-2">
                <Label htmlFor="priority">Priority</Label>
                <Select defaultValue="medium">
                    <SelectTrigger id="priority">
                        <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                        <SelectItem value="low">Low</SelectItem>
                        <SelectItem value="medium">Medium</SelectItem>
                        <SelectItem value="high">High</SelectItem>
                    </SelectContent>
                </Select>
            </div>
        </div>
    )
};

// Sizes
export const Sizes: Story = {
    render: () => (
        <div className="flex flex-col space-y-4">
            <Select>
                <SelectTrigger
                    aria-label="Small size select"
                    className="h-8 text-xs">
                    <SelectValue placeholder="Small" />
                </SelectTrigger>
                <SelectContent>
                    <SelectItem value="option1">Option 1</SelectItem>
                    <SelectItem value="option2">Option 2</SelectItem>
                </SelectContent>
            </Select>

            <Select>
                <SelectTrigger aria-label="Default size select">
                    <SelectValue placeholder="Default" />
                </SelectTrigger>
                <SelectContent>
                    <SelectItem value="option1">Option 1</SelectItem>
                    <SelectItem value="option2">Option 2</SelectItem>
                </SelectContent>
            </Select>

            <Select>
                <SelectTrigger
                    aria-label="Large size select"
                    className="h-12 text-lg">
                    <SelectValue placeholder="Large" />
                </SelectTrigger>
                <SelectContent>
                    <SelectItem value="option1">Option 1</SelectItem>
                    <SelectItem value="option2">Option 2</SelectItem>
                </SelectContent>
            </Select>
        </div>
    )
};
