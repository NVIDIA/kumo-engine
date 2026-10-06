// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Meta, StoryObj } from "@storybook/react-vite";
import { expect, userEvent, waitFor, within } from "storybook/test";
import { useState } from "react";
import {
    Command,
    CommandDialog,
    CommandEmpty,
    CommandGroup,
    CommandInput,
    CommandItem,
    CommandList,
    CommandSeparator,
    CommandShortcut
} from "@/components/ui/command";
import { Button } from "@/components/ui/button";
import {
    Calculator,
    Calendar,
    CreditCard,
    Settings,
    Smile,
    User,
    Mail,
    MessageSquare,
    PlusCircle,
    Github,
    Keyboard,
    LogOut,
    Cloud,
    LifeBuoy,
    Plus
} from "lucide-react";

const meta: Meta<typeof Command> = {
    title: "UI/Command",
    component: Command,
    tags: ["autodocs"],
    parameters: {
        layout: "centered"
    }
};

export default meta;
type Story = StoryObj<typeof Command>;

export const Basic: Story = {
    render: () => (
        <Command className="rounded-lg border shadow-md md:min-w-[450px]">
            <CommandInput placeholder="Type a command or search..." />
            <CommandList>
                <CommandEmpty>No results found.</CommandEmpty>
                <CommandGroup heading="Suggestions">
                    <CommandItem>
                        <Calendar className="mr-2 h-4 w-4" />
                        <span>Calendar</span>
                    </CommandItem>
                    <CommandItem>
                        <Smile className="mr-2 h-4 w-4" />
                        <span>Search Emoji</span>
                    </CommandItem>
                    <CommandItem>
                        <Calculator className="mr-2 h-4 w-4" />
                        <span>Calculator</span>
                    </CommandItem>
                </CommandGroup>
                <CommandSeparator />
                <CommandGroup heading="Settings">
                    <CommandItem>
                        <User className="mr-2 h-4 w-4" />
                        <span>Profile</span>
                        <CommandShortcut>⌘P</CommandShortcut>
                    </CommandItem>
                    <CommandItem>
                        <CreditCard className="mr-2 h-4 w-4" />
                        <span>Billing</span>
                        <CommandShortcut>⌘B</CommandShortcut>
                    </CommandItem>
                    <CommandItem>
                        <Settings className="mr-2 h-4 w-4" />
                        <span>Settings</span>
                        <CommandShortcut>⌘S</CommandShortcut>
                    </CommandItem>
                </CommandGroup>
            </CommandList>
        </Command>
    ),
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await expect(canvas.getByText("Calendar")).toBeVisible();
        await expect(canvas.getByText("Profile")).toBeVisible();
        await expect(
            canvas.getByRole("option", { name: /Settings/ })
        ).toBeInTheDocument();

        const input = canvas.getByPlaceholderText(
            "Type a command or search..."
        );
        await userEvent.type(input, "Cal");
        await expect(canvas.getByText("Calendar")).toBeVisible();
        await expect(
            canvas.queryByRole("option", { name: /Settings/ })
        ).not.toBeInTheDocument();

        await userEvent.clear(input);
        await expect(canvas.getByText("Profile")).toBeVisible();
    }
};

export const WithShortcuts: Story = {
    render: () => (
        <Command className="rounded-lg border shadow-md md:min-w-[450px]">
            <CommandInput placeholder="Search commands..." />
            <CommandList>
                <CommandEmpty>No results found.</CommandEmpty>
                <CommandGroup heading="Actions">
                    <CommandItem>
                        <Mail className="mr-2 h-4 w-4" />
                        <span>New Email</span>
                        <CommandShortcut>⌘N</CommandShortcut>
                    </CommandItem>
                    <CommandItem>
                        <MessageSquare className="mr-2 h-4 w-4" />
                        <span>New Message</span>
                        <CommandShortcut>⌘M</CommandShortcut>
                    </CommandItem>
                    <CommandItem>
                        <PlusCircle className="mr-2 h-4 w-4" />
                        <span>New Project</span>
                        <CommandShortcut>⌘⇧N</CommandShortcut>
                    </CommandItem>
                </CommandGroup>
                <CommandSeparator />
                <CommandGroup heading="Navigation">
                    <CommandItem>
                        <Keyboard className="mr-2 h-4 w-4" />
                        <span>Keyboard Shortcuts</span>
                        <CommandShortcut>⌘K</CommandShortcut>
                    </CommandItem>
                    <CommandItem>
                        <Settings className="mr-2 h-4 w-4" />
                        <span>Preferences</span>
                        <CommandShortcut>⌘,</CommandShortcut>
                    </CommandItem>
                </CommandGroup>
            </CommandList>
        </Command>
    )
};

export const Dialog: Story = {
    render: function DialogExample() {
        const [open, setOpen] = useState(false);

        return (
            <>
                <Button onClick={() => setOpen(true)}>
                    Open Command Palette
                </Button>
                <CommandDialog open={open} onOpenChange={setOpen}>
                    <CommandInput placeholder="Type a command or search..." />
                    <CommandList>
                        <CommandEmpty>No results found.</CommandEmpty>
                        <CommandGroup heading="Suggestions">
                            <CommandItem onSelect={() => setOpen(false)}>
                                <Calendar className="mr-2 h-4 w-4" />
                                <span>Calendar</span>
                            </CommandItem>
                            <CommandItem onSelect={() => setOpen(false)}>
                                <Smile className="mr-2 h-4 w-4" />
                                <span>Search Emoji</span>
                            </CommandItem>
                            <CommandItem onSelect={() => setOpen(false)}>
                                <Calculator className="mr-2 h-4 w-4" />
                                <span>Calculator</span>
                            </CommandItem>
                        </CommandGroup>
                        <CommandSeparator />
                        <CommandGroup heading="Settings">
                            <CommandItem onSelect={() => setOpen(false)}>
                                <User className="mr-2 h-4 w-4" />
                                <span>Profile</span>
                                <CommandShortcut>⌘P</CommandShortcut>
                            </CommandItem>
                            <CommandItem onSelect={() => setOpen(false)}>
                                <Settings className="mr-2 h-4 w-4" />
                                <span>Settings</span>
                                <CommandShortcut>⌘S</CommandShortcut>
                            </CommandItem>
                        </CommandGroup>
                    </CommandList>
                </CommandDialog>
            </>
        );
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await userEvent.click(
            canvas.getByRole("button", { name: /open command palette/i })
        );

        const body = within(document.body);
        await waitFor(() => {
            expect(body.getByText("Calendar")).toBeVisible();
        });
        await expect(body.getByText("Profile")).toBeVisible();

        await userEvent.click(body.getByText("Calendar"));
        await waitFor(() => {
            expect(body.queryByText("Suggestions")).not.toBeInTheDocument();
        });
    }
};

export const MultipleGroups: Story = {
    render: () => (
        <Command className="rounded-lg border shadow-md md:min-w-[450px]">
            <CommandInput placeholder="Search..." />
            <CommandList>
                <CommandEmpty>No results found.</CommandEmpty>
                <CommandGroup heading="My Account">
                    <CommandItem>
                        <User className="mr-2 h-4 w-4" />
                        <span>Profile</span>
                        <CommandShortcut>⌘P</CommandShortcut>
                    </CommandItem>
                    <CommandItem>
                        <CreditCard className="mr-2 h-4 w-4" />
                        <span>Billing</span>
                        <CommandShortcut>⌘B</CommandShortcut>
                    </CommandItem>
                    <CommandItem>
                        <Settings className="mr-2 h-4 w-4" />
                        <span>Settings</span>
                        <CommandShortcut>⌘S</CommandShortcut>
                    </CommandItem>
                    <CommandItem>
                        <Keyboard className="mr-2 h-4 w-4" />
                        <span>Keyboard shortcuts</span>
                    </CommandItem>
                </CommandGroup>
                <CommandSeparator />
                <CommandGroup heading="Team">
                    <CommandItem>
                        <User className="mr-2 h-4 w-4" />
                        <span>Invite users</span>
                    </CommandItem>
                    <CommandItem>
                        <Plus className="mr-2 h-4 w-4" />
                        <span>New Team</span>
                        <CommandShortcut>⌘T</CommandShortcut>
                    </CommandItem>
                </CommandGroup>
                <CommandSeparator />
                <CommandGroup heading="Support">
                    <CommandItem>
                        <Github className="mr-2 h-4 w-4" />
                        <span>GitHub</span>
                    </CommandItem>
                    <CommandItem>
                        <LifeBuoy className="mr-2 h-4 w-4" />
                        <span>Support</span>
                    </CommandItem>
                    <CommandItem>
                        <Cloud className="mr-2 h-4 w-4" />
                        <span>API</span>
                    </CommandItem>
                </CommandGroup>
                <CommandSeparator />
                <CommandGroup heading="">
                    <CommandItem>
                        <LogOut className="mr-2 h-4 w-4" />
                        <span>Log out</span>
                        <CommandShortcut>⇧⌘Q</CommandShortcut>
                    </CommandItem>
                </CommandGroup>
            </CommandList>
        </Command>
    )
};

export const WithDisabledItems: Story = {
    render: () => (
        <Command className="rounded-lg border shadow-md md:min-w-[450px]">
            <CommandInput placeholder="Type a command..." />
            <CommandList>
                <CommandEmpty>No results found.</CommandEmpty>
                <CommandGroup heading="Actions">
                    <CommandItem>
                        <Calendar className="mr-2 h-4 w-4" />
                        <span>Schedule Meeting</span>
                    </CommandItem>
                    <CommandItem disabled>
                        <Mail className="mr-2 h-4 w-4" />
                        <span>Send Email (unavailable)</span>
                    </CommandItem>
                    <CommandItem>
                        <MessageSquare className="mr-2 h-4 w-4" />
                        <span>Send Message</span>
                    </CommandItem>
                    <CommandItem disabled>
                        <CreditCard className="mr-2 h-4 w-4" />
                        <span>Process Payment (requires admin)</span>
                    </CommandItem>
                </CommandGroup>
            </CommandList>
        </Command>
    ),
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const disabled = canvas.getByRole("option", { name: /send email/i });
        await expect(disabled).toHaveAttribute("aria-disabled", "true");
        const enabled = canvas.getByRole("option", {
            name: /schedule meeting/i
        });
        await expect(enabled).not.toHaveAttribute("aria-disabled", "true");
    }
};

export const ControlledSelection: Story = {
    render: function ControlledExample() {
        const [selectedValue, setSelectedValue] = useState<string>("");

        const items = [
            { value: "calendar", label: "Calendar", icon: Calendar },
            { value: "calculator", label: "Calculator", icon: Calculator },
            { value: "settings", label: "Settings", icon: Settings },
            { value: "profile", label: "Profile", icon: User },
            { value: "billing", label: "Billing", icon: CreditCard }
        ];

        return (
            <div className="space-y-4">
                <Command className="rounded-lg border shadow-md md:min-w-[450px]">
                    <CommandInput placeholder="Select an item..." />
                    <CommandList>
                        <CommandEmpty>No results found.</CommandEmpty>
                        <CommandGroup heading="Items">
                            {items.map((item) => (
                                <CommandItem
                                    key={item.value}
                                    value={item.value}
                                    onSelect={(value) =>
                                        setSelectedValue(value)
                                    }
                                    className={
                                        selectedValue === item.value
                                            ? "bg-slate-100"
                                            : ""
                                    }>
                                    <item.icon className="mr-2 h-4 w-4" />
                                    <span>{item.label}</span>
                                </CommandItem>
                            ))}
                        </CommandGroup>
                    </CommandList>
                </Command>
                <div className="rounded-md border p-4">
                    <p className="text-sm">
                        Selected: {selectedValue || "None"}
                    </p>
                    <Button
                        variant="light-outline"
                        size="sm"
                        className="mt-2"
                        onClick={() => setSelectedValue("")}>
                        Clear Selection
                    </Button>
                </div>
            </div>
        );
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await expect(canvas.getByText("Selected: None")).toBeVisible();

        await userEvent.click(canvas.getByText("Calculator"));
        await expect(canvas.getByText("Selected: calculator")).toBeVisible();

        await userEvent.click(canvas.getByText("Profile"));
        await expect(canvas.getByText("Selected: profile")).toBeVisible();

        await userEvent.click(
            canvas.getByRole("button", { name: /clear selection/i })
        );
        await expect(canvas.getByText("Selected: None")).toBeVisible();
    }
};

export const EmptyState: Story = {
    parameters: {
        // Disable specific rules for this story:
        // - aria-required-children: cmdk library sets role="listbox" on the list
        //   and role="presentation" on empty state, creating unavoidable conflict
        // - color-contrast: uses design system muted-foreground color
        a11y: {
            config: {
                rules: [
                    { id: "aria-required-children", enabled: false },
                    { id: "color-contrast", enabled: false }
                ]
            }
        }
    },
    render: () => (
        <Command className="rounded-lg border shadow-md md:min-w-[450px]">
            <CommandInput placeholder="Search for something..." />
            <CommandList>
                <CommandEmpty>
                    <div className="flex flex-col items-center py-4">
                        <Smile className="text-muted-foreground mb-2 h-8 w-8" />
                        <p>No results found.</p>
                        <p className="text-muted-foreground text-xs">
                            Try searching for something else.
                        </p>
                    </div>
                </CommandEmpty>
            </CommandList>
        </Command>
    )
};

export const Compact: Story = {
    render: () => (
        <Command className="w-[250px] rounded-lg border shadow-md">
            <CommandInput placeholder="Search..." />
            <CommandList>
                <CommandEmpty>No results.</CommandEmpty>
                <CommandGroup>
                    <CommandItem>
                        <Calendar className="mr-2 h-4 w-4" />
                        Calendar
                    </CommandItem>
                    <CommandItem>
                        <Settings className="mr-2 h-4 w-4" />
                        Settings
                    </CommandItem>
                    <CommandItem>
                        <User className="mr-2 h-4 w-4" />
                        Profile
                    </CommandItem>
                </CommandGroup>
            </CommandList>
        </Command>
    )
};

export const ApplicationLauncher: Story = {
    render: function LauncherExample() {
        const [open, setOpen] = useState(false);

        const apps = [
            { name: "Calendar", icon: Calendar, shortcut: "⌘1" },
            { name: "Calculator", icon: Calculator, shortcut: "⌘2" },
            { name: "Settings", icon: Settings, shortcut: "⌘3" },
            { name: "Mail", icon: Mail, shortcut: "⌘4" },
            { name: "Messages", icon: MessageSquare, shortcut: "⌘5" },
            { name: "GitHub", icon: Github, shortcut: "⌘6" }
        ];

        return (
            <>
                <Button onClick={() => setOpen(true)} variant="light-outline">
                    <Keyboard className="mr-2 h-4 w-4" />
                    Open Launcher
                </Button>
                <CommandDialog open={open} onOpenChange={setOpen}>
                    <CommandInput placeholder="Search applications..." />
                    <CommandList>
                        <CommandEmpty>No applications found.</CommandEmpty>
                        <CommandGroup heading="Applications">
                            {apps.map((app) => (
                                <CommandItem
                                    key={app.name}
                                    onSelect={() => {
                                        setOpen(false);
                                    }}>
                                    <app.icon className="mr-2 h-4 w-4" />
                                    <span>{app.name}</span>
                                    <CommandShortcut>
                                        {app.shortcut}
                                    </CommandShortcut>
                                </CommandItem>
                            ))}
                        </CommandGroup>
                    </CommandList>
                </CommandDialog>
            </>
        );
    }
};

export const SearchWithCategories: Story = {
    render: () => (
        <Command className="rounded-lg border shadow-md md:min-w-[500px]">
            <CommandInput placeholder="Search everything..." />
            <CommandList>
                <CommandEmpty>No results found.</CommandEmpty>
                <CommandGroup heading="Recent">
                    <CommandItem>
                        <Calendar className="mr-2 h-4 w-4" />
                        <span>Meeting with Team</span>
                        <span className="text-muted-foreground ml-auto text-xs">
                            2 hours ago
                        </span>
                    </CommandItem>
                    <CommandItem>
                        <MessageSquare className="mr-2 h-4 w-4" />
                        <span>Message from Alex</span>
                        <span className="text-muted-foreground ml-auto text-xs">
                            Yesterday
                        </span>
                    </CommandItem>
                </CommandGroup>
                <CommandSeparator />
                <CommandGroup heading="People">
                    <CommandItem>
                        <User className="mr-2 h-4 w-4" />
                        <span>John Doe</span>
                        <span className="text-muted-foreground ml-auto text-xs">
                            Engineering
                        </span>
                    </CommandItem>
                    <CommandItem>
                        <User className="mr-2 h-4 w-4" />
                        <span>Jane Smith</span>
                        <span className="text-muted-foreground ml-auto text-xs">
                            Design
                        </span>
                    </CommandItem>
                </CommandGroup>
                <CommandSeparator />
                <CommandGroup heading="Actions">
                    <CommandItem>
                        <PlusCircle className="mr-2 h-4 w-4" />
                        <span>Create New Project</span>
                        <CommandShortcut>⌘N</CommandShortcut>
                    </CommandItem>
                    <CommandItem>
                        <Settings className="mr-2 h-4 w-4" />
                        <span>Open Settings</span>
                        <CommandShortcut>⌘,</CommandShortcut>
                    </CommandItem>
                </CommandGroup>
            </CommandList>
        </Command>
    )
};
