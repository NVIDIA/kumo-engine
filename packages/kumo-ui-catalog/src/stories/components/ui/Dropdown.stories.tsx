// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Meta, StoryObj } from "@storybook/react-vite";
import { expect, userEvent, waitFor, within } from "storybook/test";
import {
    ChevronRight,
    CreditCard,
    LogOut,
    PlusCircle,
    Settings,
    User
} from "lucide-react";

import { Button } from "@/components/ui/button";
import {
    DropdownMenu,
    DropdownMenuCheckboxItem,
    DropdownMenuContent,
    DropdownMenuGroup,
    DropdownMenuItem,
    DropdownMenuLabel,
    DropdownMenuPortal,
    DropdownMenuRadioGroup,
    DropdownMenuRadioItem,
    DropdownMenuSeparator,
    DropdownMenuShortcut,
    DropdownMenuSub,
    DropdownMenuSubContent,
    DropdownMenuSubTrigger,
    DropdownMenuTrigger
} from "@/components/ui/dropdown-menu";
import { DeleteIcon, Edit, MoreVerticle } from "@/icons";

const meta: Meta<typeof DropdownMenu> = {
    title: "UI/DropdownMenu",
    component: DropdownMenu,
    parameters: {
        layout: "centered"
    },
    tags: ["autodocs"],
    argTypes: {}
};

export default meta;
type Story = StoryObj<typeof DropdownMenu>;

// Basic Dropdown
export const Basic: Story = {
    render: () => (
        <DropdownMenu>
            <DropdownMenuTrigger asChild>
                <button>Open Menu</button>
            </DropdownMenuTrigger>
            <DropdownMenuContent className="w-56">
                <DropdownMenuLabel>My Account</DropdownMenuLabel>
                <DropdownMenuSeparator />
                <DropdownMenuItem>
                    Profile
                    <DropdownMenuShortcut>⇧⌘P</DropdownMenuShortcut>
                </DropdownMenuItem>
                <DropdownMenuItem>
                    Billing
                    <DropdownMenuShortcut>⌘B</DropdownMenuShortcut>
                </DropdownMenuItem>
                <DropdownMenuItem>
                    Settings
                    <DropdownMenuShortcut>⌘S</DropdownMenuShortcut>
                </DropdownMenuItem>
                <DropdownMenuItem>
                    Keyboard shortcuts
                    <DropdownMenuShortcut>⌘K</DropdownMenuShortcut>
                </DropdownMenuItem>
            </DropdownMenuContent>
        </DropdownMenu>
    ),
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await userEvent.click(
            canvas.getByRole("button", { name: /open menu/i })
        );
        const body = within(document.body);
        await waitFor(() => {
            expect(body.getByText("My Account")).toBeVisible();
            expect(body.getByText("Profile")).toBeVisible();
            expect(body.getByText("⇧⌘P")).toBeVisible();
        });
        await userEvent.keyboard("{Escape}");
        await waitFor(() => {
            expect(body.queryByText("Profile")).not.toBeInTheDocument();
        });
    }
};

// With Icons
export const WithIcons: Story = {
    render: () => (
        <DropdownMenu>
            <DropdownMenuTrigger asChild>
                <button>Open Menu</button>
            </DropdownMenuTrigger>
            <DropdownMenuContent className="w-56">
                <DropdownMenuLabel>My Account</DropdownMenuLabel>
                <DropdownMenuSeparator />
                <DropdownMenuGroup>
                    <DropdownMenuItem>
                        <User className="mr-2 h-4 w-4" />
                        <span>Profile</span>
                        <DropdownMenuShortcut>⇧⌘P</DropdownMenuShortcut>
                    </DropdownMenuItem>
                    <DropdownMenuItem>
                        <CreditCard className="mr-2 h-4 w-4" />
                        <span>Billing</span>
                        <DropdownMenuShortcut>⌘B</DropdownMenuShortcut>
                    </DropdownMenuItem>
                    <DropdownMenuItem>
                        <Settings className="mr-2 h-4 w-4" />
                        <span>Settings</span>
                        <DropdownMenuShortcut>⌘S</DropdownMenuShortcut>
                    </DropdownMenuItem>
                </DropdownMenuGroup>
                <DropdownMenuSeparator />
                <DropdownMenuItem>
                    <LogOut className="mr-2 h-4 w-4" />
                    <span>Log out</span>
                    <DropdownMenuShortcut>⇧⌘Q</DropdownMenuShortcut>
                </DropdownMenuItem>
            </DropdownMenuContent>
        </DropdownMenu>
    ),
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await userEvent.click(
            canvas.getByRole("button", { name: /open menu/i })
        );
        const body = within(document.body);
        await waitFor(() => {
            expect(body.getByText("Billing")).toBeVisible();
            expect(body.getByText("Log out")).toBeVisible();
        });
        await userEvent.keyboard("{Escape}");
        await waitFor(() => {
            expect(body.queryByText("Billing")).not.toBeInTheDocument();
        });
    }
};

// With Checkbox Items
export const WithCheckboxes: Story = {
    render: function Render() {
        return (
            <DropdownMenu>
                <DropdownMenuTrigger asChild>
                    <button>Open Menu</button>
                </DropdownMenuTrigger>
                <DropdownMenuContent className="w-56">
                    <DropdownMenuLabel>Appearance</DropdownMenuLabel>
                    <DropdownMenuSeparator />
                    <DropdownMenuCheckboxItem checked>
                        Show status bar
                    </DropdownMenuCheckboxItem>
                    <DropdownMenuCheckboxItem>
                        Show panel
                    </DropdownMenuCheckboxItem>
                    <DropdownMenuCheckboxItem disabled>
                        Show notifications
                    </DropdownMenuCheckboxItem>
                </DropdownMenuContent>
            </DropdownMenu>
        );
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await userEvent.click(
            canvas.getByRole("button", { name: /open menu/i })
        );
        const body = within(document.body);
        await waitFor(() => {
            expect(
                body.getByRole("menuitemcheckbox", { name: /show status bar/i })
            ).toBeChecked();
            expect(
                body.getByRole("menuitemcheckbox", { name: /show panel/i })
            ).not.toBeChecked();
            expect(
                body.getByRole("menuitemcheckbox", {
                    name: /show notifications/i
                })
            ).toHaveAttribute("data-disabled");
        });
        await userEvent.keyboard("{Escape}");
        await waitFor(() => {
            expect(body.queryByText("Show status bar")).not.toBeInTheDocument();
        });
    }
};

// With Radio Items
export const WithRadioItems: Story = {
    render: function Render() {
        return (
            <DropdownMenu>
                <DropdownMenuTrigger asChild>
                    <button>Open Menu</button>
                </DropdownMenuTrigger>
                <DropdownMenuContent className="w-56">
                    <DropdownMenuLabel>Panel Position</DropdownMenuLabel>
                    <DropdownMenuSeparator />
                    <DropdownMenuRadioGroup value="bottom">
                        <DropdownMenuRadioItem value="top">
                            Top
                        </DropdownMenuRadioItem>
                        <DropdownMenuRadioItem value="bottom">
                            Bottom
                        </DropdownMenuRadioItem>
                        <DropdownMenuRadioItem value="right">
                            Right
                        </DropdownMenuRadioItem>
                    </DropdownMenuRadioGroup>
                </DropdownMenuContent>
            </DropdownMenu>
        );
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await userEvent.click(
            canvas.getByRole("button", { name: /open menu/i })
        );
        const body = within(document.body);
        await waitFor(() => {
            expect(
                body.getByRole("menuitemradio", { name: /bottom/i })
            ).toBeChecked();
            expect(
                body.getByRole("menuitemradio", { name: /top/i })
            ).not.toBeChecked();
            expect(
                body.getByRole("menuitemradio", { name: /right/i })
            ).not.toBeChecked();
        });
        await userEvent.keyboard("{Escape}");
        await waitFor(() => {
            expect(body.queryByText("Panel Position")).not.toBeInTheDocument();
        });
    }
};

// With Submenu
export const WithSubmenu: Story = {
    render: () => (
        <DropdownMenu>
            <DropdownMenuTrigger asChild>
                <button>Open Menu</button>
            </DropdownMenuTrigger>
            <DropdownMenuContent className="w-56">
                <DropdownMenuLabel>Actions</DropdownMenuLabel>
                <DropdownMenuSeparator />
                <DropdownMenuGroup>
                    <DropdownMenuItem>
                        <span>New Item</span>
                        <DropdownMenuShortcut>⌘N</DropdownMenuShortcut>
                    </DropdownMenuItem>
                    <DropdownMenuSub>
                        <DropdownMenuSubTrigger>
                            <span>More Options</span>
                            <ChevronRight className="ml-auto" />
                        </DropdownMenuSubTrigger>
                        <DropdownMenuPortal>
                            <DropdownMenuSubContent className="w-48">
                                <DropdownMenuItem>
                                    <span>Save</span>
                                    <DropdownMenuShortcut>
                                        ⌘S
                                    </DropdownMenuShortcut>
                                </DropdownMenuItem>
                                <DropdownMenuItem>
                                    <span>Save as...</span>
                                    <DropdownMenuShortcut>
                                        ⇧⌘S
                                    </DropdownMenuShortcut>
                                </DropdownMenuItem>
                                <DropdownMenuSeparator />
                                <DropdownMenuItem>
                                    <span>Export</span>
                                    <DropdownMenuShortcut>
                                        ⌘E
                                    </DropdownMenuShortcut>
                                </DropdownMenuItem>
                            </DropdownMenuSubContent>
                        </DropdownMenuPortal>
                    </DropdownMenuSub>
                </DropdownMenuGroup>
            </DropdownMenuContent>
        </DropdownMenu>
    ),
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await userEvent.click(
            canvas.getByRole("button", { name: /open menu/i })
        );
        const body = within(document.body);
        await waitFor(() => {
            expect(body.getByText("More Options")).toBeVisible();
        });
        await userEvent.hover(body.getByText("More Options"));
        await waitFor(() => {
            expect(body.getByText("Save")).toBeVisible();
            expect(body.getByText("Export")).toBeVisible();
        });
        await userEvent.keyboard("{Escape}");
        await userEvent.keyboard("{Escape}");
        await waitFor(() => {
            expect(body.queryByText("More Options")).not.toBeInTheDocument();
        });
    }
};

// Dark Mode Variant
export const DarkMode: Story = {
    render: () => (
        <div className="dark">
            <div className="rounded-lg bg-slate-950 p-6">
                <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                        <button>Open Menu</button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent className="w-56">
                        <DropdownMenuLabel>My Account</DropdownMenuLabel>
                        <DropdownMenuSeparator />
                        <DropdownMenuItem>
                            <User className="mr-2 h-4 w-4" />
                            <span>Profile</span>
                        </DropdownMenuItem>
                        <DropdownMenuItem>
                            <CreditCard className="mr-2 h-4 w-4" />
                            <span>Billing</span>
                        </DropdownMenuItem>
                        <DropdownMenuItem>
                            <Settings className="mr-2 h-4 w-4" />
                            <span>Settings</span>
                        </DropdownMenuItem>
                        <DropdownMenuSeparator />
                        <DropdownMenuItem>
                            <LogOut className="mr-2 h-4 w-4" />
                            <span>Log out</span>
                        </DropdownMenuItem>
                    </DropdownMenuContent>
                </DropdownMenu>
            </div>
        </div>
    )
};

// Contextual Menu
export const ContextMenu: Story = {
    render: () => (
        <div className="flex h-40 w-full items-center justify-center rounded-md border text-sm">
            <DropdownMenu>
                <DropdownMenuTrigger className="flex h-full w-full items-center justify-center">
                    Right click me
                </DropdownMenuTrigger>
                <DropdownMenuContent className="w-56">
                    <DropdownMenuItem>
                        <PlusCircle className="mr-2 h-4 w-4" />
                        <span>New</span>
                        <DropdownMenuShortcut>⌘N</DropdownMenuShortcut>
                    </DropdownMenuItem>
                    <DropdownMenuSeparator />
                    <DropdownMenuItem>
                        <span>Cut</span>
                        <DropdownMenuShortcut>⌘X</DropdownMenuShortcut>
                    </DropdownMenuItem>
                    <DropdownMenuItem>
                        <span>Copy</span>
                        <DropdownMenuShortcut>⌘C</DropdownMenuShortcut>
                    </DropdownMenuItem>
                    <DropdownMenuItem>
                        <span>Paste</span>
                        <DropdownMenuShortcut>⌘V</DropdownMenuShortcut>
                    </DropdownMenuItem>
                </DropdownMenuContent>
            </DropdownMenu>
        </div>
    )
};

const actionButtons = [
    {
        content: (
            <>
                <Edit /> {" Edit "}
            </>
        ),
        disabled: false,
        onClick: () => {
            console.log("Edit Button Clicked");
        },
        tooltiptext: ""
    },
    {
        content: (
            <>
                <DeleteIcon /> {" Delete "}
            </>
        ),
        disabled: false,
        onClick: () => {
            console.log("Delete Button Clicked");
        },
        tooltiptext: ""
    }
];

export const Overview_Dropdown: Story = {
    render: () => (
        <div>
            <DropdownMenu>
                <DropdownMenuTrigger asChild>
                    <Button
                        variant="light-outline"
                        className="px-2"
                        aria-label="More actions">
                        <MoreVerticle />
                    </Button>
                </DropdownMenuTrigger>

                <DropdownMenuContent
                    align="end"
                    className="text-sm text-slate-800">
                    {actionButtons?.map((button) => (
                        <DropdownMenuItem
                            tooltipText={button?.tooltiptext ?? ""}
                            onClick={button?.onClick}
                            className={
                                button.disabled
                                    ? "pointer-events-none cursor-default text-slate-500"
                                    : ""
                            }>
                            <span className="flex items-center">
                                {button?.content}
                            </span>
                        </DropdownMenuItem>
                    ))}
                </DropdownMenuContent>
            </DropdownMenu>
        </div>
    ),
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await userEvent.click(
            canvas.getByRole("button", { name: /more actions/i })
        );
        const body = within(document.body);
        await waitFor(() => {
            expect(body.getByText(/edit/i)).toBeVisible();
            expect(body.getByText(/delete/i)).toBeVisible();
        });
        await userEvent.keyboard("{Escape}");
        await waitFor(() => {
            expect(body.queryByText(/delete/i)).not.toBeInTheDocument();
        });
    }
};

// In different positions
export const PositionVariants: Story = {
    render: () => (
        <div className="flex flex-col items-start gap-8">
            <DropdownMenu>
                <DropdownMenuTrigger asChild>
                    <button>Bottom (Default)</button>
                </DropdownMenuTrigger>
                <DropdownMenuContent className="w-56">
                    <DropdownMenuLabel>Position Demo</DropdownMenuLabel>
                    <DropdownMenuSeparator />
                    <DropdownMenuItem>Default position</DropdownMenuItem>
                </DropdownMenuContent>
            </DropdownMenu>

            <DropdownMenu>
                <DropdownMenuTrigger asChild>
                    <button>Top</button>
                </DropdownMenuTrigger>
                <DropdownMenuContent className="w-56" side="top">
                    <DropdownMenuLabel>Position Demo</DropdownMenuLabel>
                    <DropdownMenuSeparator />
                    <DropdownMenuItem>Top position</DropdownMenuItem>
                </DropdownMenuContent>
            </DropdownMenu>

            <DropdownMenu>
                <DropdownMenuTrigger asChild>
                    <button>Right</button>
                </DropdownMenuTrigger>
                <DropdownMenuContent className="w-56" side="right">
                    <DropdownMenuLabel>Position Demo</DropdownMenuLabel>
                    <DropdownMenuSeparator />
                    <DropdownMenuItem>Right position</DropdownMenuItem>
                </DropdownMenuContent>
            </DropdownMenu>

            <DropdownMenu>
                <DropdownMenuTrigger asChild>
                    <button>Left</button>
                </DropdownMenuTrigger>
                <DropdownMenuContent className="w-56" side="left">
                    <DropdownMenuLabel>Position Demo</DropdownMenuLabel>
                    <DropdownMenuSeparator />
                    <DropdownMenuItem>Left position</DropdownMenuItem>
                </DropdownMenuContent>
            </DropdownMenu>
        </div>
    )
};

// With animation variants
export const AnimationVariants: Story = {
    render: () => (
        <div className="flex flex-col items-start gap-8">
            <DropdownMenu>
                <DropdownMenuTrigger asChild>
                    <button>Default Animation</button>
                </DropdownMenuTrigger>
                <DropdownMenuContent className="w-56">
                    <DropdownMenuLabel>Animation Demo</DropdownMenuLabel>
                    <DropdownMenuSeparator />
                    <DropdownMenuItem>Default animation</DropdownMenuItem>
                </DropdownMenuContent>
            </DropdownMenu>

            <DropdownMenu>
                <DropdownMenuTrigger asChild>
                    <button>Custom Animation</button>
                </DropdownMenuTrigger>
                <DropdownMenuContent className="animate-in zoom-in-90 w-56 duration-300">
                    <DropdownMenuLabel>Animation Demo</DropdownMenuLabel>
                    <DropdownMenuSeparator />
                    <DropdownMenuItem>Custom animation</DropdownMenuItem>
                </DropdownMenuContent>
            </DropdownMenu>
        </div>
    )
};
