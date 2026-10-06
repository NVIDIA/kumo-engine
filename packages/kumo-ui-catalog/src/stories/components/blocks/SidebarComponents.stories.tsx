// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Meta, StoryObj } from "@storybook/react-vite";
import { expect, userEvent, waitFor, within } from "storybook/test";
import { useState } from "react";

import { AppLogo } from "@/components/blocks/sidebar/app-logo";
import { NavMain } from "@/components/blocks/sidebar/nav-main";
import { NavNew } from "@/components/blocks/sidebar/nav-new";
import { NavUser } from "@/components/blocks/sidebar/nav-user";
import { SidebarProvider } from "@/components/ui/sidebar";
import { Cube, SoundSampler, DeveloperBoard, Loupe } from "@/icons";

const appLogoMeta: Meta<typeof AppLogo> = {
    title: "Blocks/AppLogo",
    component: AppLogo,
    tags: ["autodocs"],
    decorators: [
        (Story) => (
            <SidebarProvider>
                <Story />
            </SidebarProvider>
        )
    ]
};

export default appLogoMeta;
type AppLogoStory = StoryObj<typeof AppLogo>;

export const AppLogoWithClick: AppLogoStory = {
    args: {
        onLogoClick: () => {},
        onLogoMouseEnter: () => {},
        onLogoMouseLeave: () => {},
        onLogoFocus: () => {}
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const logo = canvas.getByRole("link");

        await userEvent.hover(logo);
        await userEvent.unhover(logo);
        await userEvent.click(logo);
    }
};

export const AppLogoNoCallbacks: AppLogoStory = {
    args: {},
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const logo = canvas.getByRole("link");
        await userEvent.hover(logo);
        await userEvent.unhover(logo);
    }
};

export const AppLogoHoverState: AppLogoStory = {
    args: {
        onLogoClick: () => {}
    }
};

export const NavMainWithItems: StoryObj<typeof NavMain> = {
    render: function DefaultFunc() {
        return (
            <SidebarProvider>
                <NavMain
                    items={[
                        {
                            title: "Models",
                            icon: Cube,
                            show: true,
                            isActive: true,
                            to: "/models",
                            onClick: () => {},
                            onMouseEnter: () => {},
                            onFocus: () => {}
                        },
                        {
                            title: "Predictions",
                            icon: SoundSampler,
                            show: true,
                            isActive: false,
                            to: "/predictions",
                            onClick: () => {},
                            onMouseEnter: () => {},
                            onFocus: () => {}
                        }
                    ]}
                />
            </SidebarProvider>
        );
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const modelsBtn = canvas.getByRole("button", { name: /models/i });

        await userEvent.hover(modelsBtn);
        await userEvent.click(modelsBtn);
        await userEvent.click(
            canvas.getByRole("button", { name: /predictions/i })
        );
    }
};

export const NavMainWithHiddenItem: StoryObj<typeof NavMain> = {
    render: function DefaultFunc() {
        return (
            <SidebarProvider>
                <NavMain
                    items={[
                        {
                            title: "Visible",
                            icon: Cube,
                            show: true,
                            isActive: false,
                            to: "/visible",
                            onClick: () => {}
                        },
                        {
                            title: "Hidden Item",
                            icon: SoundSampler,
                            show: false,
                            isActive: false,
                            to: "/hidden",
                            onClick: () => {}
                        }
                    ]}
                />
            </SidebarProvider>
        );
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await userEvent.click(canvas.getByRole("button", { name: /visible/i }));
        expect(
            canvas.queryByRole("button", { name: /hidden item/i })
        ).toBeNull();
    }
};

export const NavNewWithDropdown: StoryObj<typeof NavNew> = {
    render: function DefaultFunc() {
        return (
            <SidebarProvider>
                <NavNew
                    items={[
                        {
                            title: "New",
                            icon: Loupe,
                            show: true,
                            childs: [
                                {
                                    title: "Model",
                                    description:
                                        "Train a model with a graph schema.",
                                    icon: DeveloperBoard,
                                    onClick: () => {},
                                    onMouseEnter: () => {},
                                    onMouseLeave: () => {},
                                    onFocus: () => {}
                                },
                                {
                                    title: "Prediction",
                                    description:
                                        "Generate predictions using a trained model.",
                                    icon: DeveloperBoard,
                                    onClick: () => {},
                                    onMouseEnter: () => {},
                                    onMouseLeave: () => {},
                                    onFocus: () => {}
                                }
                            ]
                        }
                    ]}
                />
            </SidebarProvider>
        );
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);

        await userEvent.click(canvas.getByRole("button", { name: /new/i }));

        await waitFor(() => {
            expect(within(document.body).getByText("Model")).toBeVisible();
        });

        await userEvent.click(within(document.body).getByText("Model"));

        await waitFor(() => {
            expect(
                within(document.body).queryByText("Model")
            ).not.toBeInTheDocument();
        });
    }
};

export const NavNewHiddenItem: StoryObj<typeof NavNew> = {
    render: function DefaultFunc() {
        return (
            <SidebarProvider>
                <NavNew
                    items={[
                        {
                            title: "Visible Item",
                            icon: Loupe,
                            show: true,
                            childs: [
                                {
                                    title: "Option",
                                    description: "An option",
                                    icon: DeveloperBoard,
                                    onClick: () => {}
                                }
                            ]
                        },
                        {
                            title: "Hidden Item",
                            icon: Loupe,
                            show: false,
                            childs: []
                        }
                    ]}
                />
            </SidebarProvider>
        );
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        expect(
            canvas.queryByRole("button", { name: /hidden item/i })
        ).toBeNull();
        await userEvent.click(
            canvas.getByRole("button", { name: /visible item/i })
        );
        await waitFor(() => {
            expect(within(document.body).getByText("Option")).toBeVisible();
        });
        await userEvent.click(within(document.body).getByText("Option"));

        await waitFor(() => {
            expect(
                within(document.body).queryByText("Option")
            ).not.toBeInTheDocument();
        });
    }
};

export const NavNewWithChildren: StoryObj<typeof NavNew> = {
    render: function DefaultFunc() {
        return (
            <SidebarProvider>
                <NavNew
                    items={[
                        {
                            title: "Custom Children",
                            show: true,
                            children: (
                                <li className="rounded border p-2 text-sm">
                                    Custom child content
                                </li>
                            )
                        }
                    ]}
                />
            </SidebarProvider>
        );
    }
};

export const NavUserWithCallbacks: StoryObj<typeof NavUser> = {
    render: function DefaultFunc() {
        const [log, setLog] = useState<string[]>([]);
        return (
            <SidebarProvider>
                <div className="flex flex-col gap-4">
                    <NavUser
                        user={{
                            name: "Test Account",
                            email: "test@kumo.ai",
                            avatar: "",
                            show: true,
                            onLogoutClick: () =>
                                setLog((l) => [...l, "logout clicked"]),
                            onLogoutMouseEnter: () =>
                                setLog((l) => [...l, "logout mouseenter"]),
                            onLogoutMouseLeave: () =>
                                setLog((l) => [...l, "logout mouseleave"]),
                            onLogoutFocus: () =>
                                setLog((l) => [...l, "logout focus"])
                        }}
                    />
                    {log.length > 0 && (
                        <ul className="text-xs text-slate-600">
                            {log.map((entry, i) => (
                                <li key={i}>{entry}</li>
                            ))}
                        </ul>
                    )}
                </div>
            </SidebarProvider>
        );
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);

        await userEvent.click(
            canvas.getByRole("button", { name: /user menu/i })
        );

        await waitFor(() => {
            expect(within(document.body).getByText("Log Out")).toBeVisible();
        });

        await userEvent.click(within(document.body).getByText("Log Out"));
        await userEvent.keyboard("{Escape}");

        await waitFor(() => {
            expect(
                within(document.body).queryByRole("menu")
            ).not.toBeInTheDocument();
        });
    }
};

export const NavUserMinimal: StoryObj<typeof NavUser> = {
    render: function DefaultFunc() {
        return (
            <SidebarProvider>
                <NavUser
                    user={{
                        name: "Jane Doe",
                        email: "jane@kumo.ai",
                        avatar: "",
                        show: true,
                        onLogoutClick: () => {}
                    }}
                />
            </SidebarProvider>
        );
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await userEvent.click(
            canvas.getByRole("button", { name: /user menu/i })
        );
        await waitFor(() => {
            expect(within(document.body).getByText("Log Out")).toBeVisible();
        });
        await userEvent.click(within(document.body).getByText("Log Out"));
        await userEvent.keyboard("{Escape}");

        await waitFor(() => {
            expect(
                within(document.body).queryByRole("menu")
            ).not.toBeInTheDocument();
        });
    }
};
