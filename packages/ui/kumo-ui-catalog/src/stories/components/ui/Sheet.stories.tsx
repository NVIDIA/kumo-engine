// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Button } from "@/components/ui/button";
import {
    Sheet,
    SheetContent,
    SheetDescription,
    SheetFooter,
    SheetHeader,
    SheetTitle
} from "@/components/ui/sheet";
import { CloseSmall } from "@/icons/MaterialIcons/CloseSmall";
import { Meta, StoryObj } from "@storybook/react-vite";
import { useState } from "react";
import { expect, userEvent, waitFor, within } from "storybook/test";

const meta: Meta<typeof Sheet> = {
    title: "UI/Sheet",
    component: Sheet,
    tags: ["autodocs"]
};

export default meta;
type Story = StoryObj<typeof Sheet>;

type ContentProps = {
    handleClose?: (flag: boolean) => void;
};

const Content = ({ handleClose }: ContentProps) => (
    <div>
        {handleClose && (
            <div className="absolute right-4 top-4">
                <Button
                    id="close"
                    variant="light-outline"
                    size="sm"
                    onClick={() => handleClose(false)}>
                    <CloseSmall />
                </Button>
            </div>
        )}
        <SheetHeader>
            <SheetTitle>Title of the sheet</SheetTitle>
            <SheetDescription>Sheet description.</SheetDescription>
            {Array.from({ length: 25 }).map((_, i) => (
                <p className="w-[1400px]" key={i}>
                    This is a repeated sheet description to show
                    vertical/horizontal scroll bar. #{i}
                </p>
            ))}
        </SheetHeader>
        <SheetFooter>Footer</SheetFooter>
    </div>
);

export const SheetFromRight: Story = {
    render: function SheetFromRight() {
        const [open, setOpen] = useState(false);
        const onChange = (change: boolean) => {
            setOpen(change);
        };
        return (
            <>
                <Button onClick={() => onChange(true)}>
                    Click to slide open a Sheet from right
                </Button>

                <Sheet open={open} onOpenChange={onChange}>
                    <SheetContent>
                        <Content handleClose={onChange} />
                    </SheetContent>
                </Sheet>
            </>
        );
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);

        await userEvent.click(
            canvas.getByRole("button", {
                name: /click to slide open a sheet from right/i
            })
        );
        const body = within(document.body);
        await waitFor(() => {
            expect(body.getByText("Title of the sheet")).toBeVisible();
        });

        const closeButton = document.getElementById("close")!;
        await userEvent.click(closeButton);
        await waitFor(() => {
            expect(
                body.queryByText("Title of the sheet")
            ).not.toBeInTheDocument();
        });
    }
};

export const SheetFromLeft: Story = {
    render: function SheetFromLeft() {
        const [open, setOpen] = useState(false);
        const onChange = (change: boolean) => {
            setOpen(change);
        };
        return (
            <>
                <Button onClick={() => onChange(true)}>
                    Click to slide open a Sheet from left
                </Button>

                <Sheet open={open} onOpenChange={onChange}>
                    <SheetContent side="left">
                        <Content handleClose={onChange} />
                    </SheetContent>
                </Sheet>
            </>
        );
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await userEvent.click(
            canvas.getByRole("button", {
                name: /click to slide open a sheet from left/i
            })
        );
        const body = within(document.body);
        await waitFor(() => {
            expect(body.getByText("Title of the sheet")).toBeVisible();
        });
        const closeButton = document.getElementById("close")!;
        await userEvent.click(closeButton);
        await waitFor(() => {
            expect(
                body.queryByText("Title of the sheet")
            ).not.toBeInTheDocument();
        });
    }
};

export const SheetFromTop: Story = {
    render: function SheetFromLeft() {
        const [open, setOpen] = useState(false);
        const onChange = (change: boolean) => {
            setOpen(change);
        };
        return (
            <>
                <Button onClick={() => onChange(true)}>
                    Click to slide open a Sheet from top
                </Button>

                <Sheet open={open} onOpenChange={onChange}>
                    <SheetContent side="top">
                        <Content handleClose={onChange} />
                    </SheetContent>
                </Sheet>
            </>
        );
    }
};

export const SheetFromBottom: Story = {
    render: function SheetFromLeft() {
        const [open, setOpen] = useState(false);
        const onChange = (change: boolean) => {
            setOpen(change);
        };
        return (
            <>
                <Button onClick={() => onChange(true)}>
                    Click to slide open a Sheet from bottom
                </Button>

                <Sheet open={open} onOpenChange={onChange}>
                    <SheetContent side="bottom">
                        <Content handleClose={onChange} />
                    </SheetContent>
                </Sheet>
            </>
        );
    }
};

export const SheetFromTopInteraction: Story = {
    render: function SheetFromTopInteraction() {
        const [open, setOpen] = useState(false);
        const onChange = (change: boolean) => setOpen(change);
        return (
            <>
                <Button onClick={() => onChange(true)}>
                    Open Sheet from top
                </Button>
                <Sheet open={open} onOpenChange={onChange}>
                    <SheetContent side="top">
                        <Content handleClose={onChange} />
                    </SheetContent>
                </Sheet>
            </>
        );
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await userEvent.click(
            canvas.getByRole("button", { name: /open sheet from top/i })
        );
        const body = within(document.body);
        await waitFor(() => {
            expect(body.getByText("Title of the sheet")).toBeVisible();
        });
        await userEvent.click(document.getElementById("close")!);
        await waitFor(() => {
            expect(
                body.queryByText("Title of the sheet")
            ).not.toBeInTheDocument();
        });
    }
};

export const SheetFromBottomInteraction: Story = {
    render: function SheetFromBottomInteraction() {
        const [open, setOpen] = useState(false);
        const onChange = (change: boolean) => setOpen(change);
        return (
            <>
                <Button onClick={() => onChange(true)}>
                    Open Sheet from bottom
                </Button>
                <Sheet open={open} onOpenChange={onChange}>
                    <SheetContent side="bottom">
                        <Content handleClose={onChange} />
                    </SheetContent>
                </Sheet>
            </>
        );
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await userEvent.click(
            canvas.getByRole("button", { name: /open sheet from bottom/i })
        );
        const body = within(document.body);
        await waitFor(() => {
            expect(body.getByText("Title of the sheet")).toBeVisible();
        });
        await userEvent.click(document.getElementById("close")!);
        await waitFor(() => {
            expect(
                body.queryByText("Title of the sheet")
            ).not.toBeInTheDocument();
        });
    }
};

export const WithBuiltInCloseButton: Story = {
    render: function WithBuiltInCloseButton() {
        const [open, setOpen] = useState(false);
        return (
            <>
                <Button onClick={() => setOpen(true)}>
                    Open sheet with close button
                </Button>
                <Sheet open={open} onOpenChange={setOpen}>
                    <SheetContent showCloseButton>
                        <SheetHeader>
                            <SheetTitle>Sheet with close button</SheetTitle>
                        </SheetHeader>
                    </SheetContent>
                </Sheet>
            </>
        );
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await userEvent.click(
            canvas.getByRole("button", {
                name: /open sheet with close button/i
            })
        );
        const body = within(document.body);
        await waitFor(() => {
            expect(body.getByText("Sheet with close button")).toBeVisible();
        });
        const sheetContent = document.querySelector(
            '[data-slot="sheet-content"]'
        ) as HTMLElement;
        await userEvent.click(
            sheetContent.querySelector("button") as HTMLElement
        );
        await waitFor(() => {
            expect(
                body.queryByText("Sheet with close button")
            ).not.toBeInTheDocument();
        });
    }
};
