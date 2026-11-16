// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { PopoverButton } from "@/components/ui/popover-button";
import { WarningIcon } from "@/icons";

import { Meta, StoryObj } from "@storybook/react-vite";
import { expect, userEvent, waitFor, within } from "storybook/test";

const meta: Meta<typeof PopoverButton> = {
    title: "UI/PopoverButton",
    component: PopoverButton,

    tags: ["autodocs"],
    argTypes: {}
};

export default meta;
type Story = StoryObj<typeof PopoverButton>;

// Exercises handleClose toggle: open → close via the X button inside popover (line 25)
export const OpenAndClose: Story = {
    render: function OpenAndCloseFunc() {
        return (
            <PopoverButton content={<p>Popover content here</p>}>
                Open
            </PopoverButton>
        );
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        // Open the popover
        await userEvent.click(canvas.getByTestId("popup-trigger"));
        const body = within(document.body);
        await waitFor(() => {
            expect(body.getByText("Popover content here")).toBeVisible();
        });
        // Close via the X button inside the popover
        const closeBtn = document.querySelector(
            "[data-slot='popover-content'] svg"
        ) as HTMLElement;
        if (closeBtn) await userEvent.click(closeBtn);
    }
};

// Exercises default children="Show" and content=null prop fallbacks — render only, no interaction
export const DefaultProps: Story = {
    render: function DefaultPropsFunc() {
        return <PopoverButton />;
    }
};

export const Default: Story = {
    render: function DefaultFunc() {
        return (
            <>
                <span className="text-title-primary me-6 font-bold">
                    {`Click on the 'Warnings' to see popover view`}
                </span>
                <PopoverButton
                    contentClassName="border-amber-500 bg-amber-50 text-title-primary p-6 w-150 rounded-md"
                    content={
                        <div className="flex items-start gap-4">
                            <WarningIcon
                                fill={"true"}
                                className="!fill-amber-500 fill-amber-500 stroke-amber-500"
                                style={{
                                    width: 40
                                }}
                            />
                            <div>
                                <span className="text-title-primary mt-1 font-bold">
                                    {`10 warnings detected`}
                                </span>
                                <ul className="list-disc ps-5">
                                    {Array(5)
                                        .fill("warnings")
                                        .map((w, i) => (
                                            <li
                                                className={
                                                    "text-title-primary my-4 text-base" +
                                                    (i === 9 ? " mb-0" : "")
                                                }
                                                key={`job-warning-${i}`}>
                                                {w}
                                            </li>
                                        ))}
                                </ul>
                            </div>
                        </div>
                    }>
                    <>
                        <WarningIcon className="!fill-amber-500 fill-amber-500" />
                        Warnings
                    </>
                </PopoverButton>
            </>
        );
    }
};
