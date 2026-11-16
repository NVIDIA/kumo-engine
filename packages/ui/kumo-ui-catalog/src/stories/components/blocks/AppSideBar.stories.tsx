// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Meta, StoryObj } from "@storybook/react-vite";
import { userEvent, waitFor, within } from "storybook/test";
import { useState } from "react";
import { Cube } from "@/icons/MaterialIcons/Cube";

import { AppSidebar } from "@/components/blocks/sidebar/app-sidebar";
import { SideBarData } from "@/types/sidebar";
import { SidebarInset, SidebarProvider } from "@/components/ui/sidebar";
import {
    Cards,
    NetworkNode,
    SoundSampler,
    Power,
    ImportContacts,
    Loupe,
    DeveloperBoard
} from "@/icons";

const meta: Meta<typeof AppSidebar> = {
    title: "Blocks/AppSideBar",
    component: AppSidebar,
    tags: ["autodocs"],
    argTypes: {
        className: { control: "text" },
        sidebarData: { control: "object" }
    }
};

export default meta;

type Story = StoryObj<typeof AppSidebar>;

// Basic App Side Bar
export const Basic: Story = {
    render: function DefaultFunc() {
        const [sidebarOpen, setSidebarOpen] = useState(true);

        return (
            <SidebarProvider open={sidebarOpen} onOpenChange={setSidebarOpen}>
                <AppSidebar className="z-50" sidebarData={sideBarData} />
                <SidebarInset>
                    <div className="p-4">Main content area</div>
                </SidebarInset>
            </SidebarProvider>
        );
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const modelsBtn = await waitFor(() =>
            canvas.getByRole("button", { name: /models/i })
        );
        await userEvent.hover(modelsBtn);
        await userEvent.click(modelsBtn);
    }
};

// App Side Bar Collapsed (icon-only mode)
export const Collapsed: Story = {
    render: function DefaultFunc() {
        const [sidebarOpen, setSidebarOpen] = useState(false);

        return (
            <SidebarProvider open={sidebarOpen} onOpenChange={setSidebarOpen}>
                <AppSidebar className="z-50" sidebarData={sideBarData} />
                <SidebarInset>
                    <div className="p-4">Main content area</div>
                </SidebarInset>
            </SidebarProvider>
        );
    }
};

// App Side Bar With Active Item
export const WithActiveItem: Story = {
    render: function DefaultFunc() {
        const [sidebarOpen, setSidebarOpen] = useState(true);

        const sideBarDataWithActive: SideBarData = {
            ...sideBarData,
            navMainMenuGroup: [
                {
                    items: [
                        {
                            ...group1.items[0],
                            isActive: true
                        },
                        ...group1.items.slice(1)
                    ]
                },
                group2,
                group3
            ]
        };

        return (
            <SidebarProvider open={sidebarOpen} onOpenChange={setSidebarOpen}>
                <AppSidebar
                    className="z-50"
                    sidebarData={sideBarDataWithActive}
                />
                <SidebarInset>
                    <div className="p-4">Main content area</div>
                </SidebarInset>
            </SidebarProvider>
        );
    }
};

const group1 = {
    items: [
        {
            title: "Models",
            icon: Cube,
            show: true,
            isActive: false,
            to: "https://kumo.ai/docs/training/",
            onClick: () => {
                window.open("https://kumo.ai/docs/training/", "_blank");
            },
            onMouseEnter: () => {},
            onFocus: () => {}
        },
        {
            title: "Predictions",
            icon: SoundSampler,
            show: true,
            isActive: false,
            to: `https://kumo.ai/docs/batch-prediction/`,
            onClick: () => {
                window.open("https://kumo.ai/docs/batch-prediction/", "_blank");
            },
            onMouseEnter: () => {},
            onFocus: () => {}
        }
    ]
};
const group2 = {
    items: [
        {
            title: "Connectors",
            icon: Power,
            show: true,
            isActive: false,
            to: `https://kumo.ai/docs/data-connectors/`,
            onClick: () => {
                window.open("https://kumo.ai/docs/data-connectors/", "_blank");
            },
            onMouseEnter: () => {},
            onFocus: () => {}
        },
        {
            title: "Tables",
            icon: Cards,
            show: true,
            isActive: false,
            to: `https://kumo.ai/docs/select-tables/`,
            onClick: () => {
                window.open("https://kumo.ai/docs/select-tables/", "_blank");
            },
            onMouseEnter: () => {},
            onFocus: () => {}
        },
        {
            title: "Graphs",
            icon: NetworkNode,
            show: true,
            isActive: false,
            to: `https://kumo.ai/docs/create-graph/`,
            onClick: () => {
                window.open("https://kumo.ai/docs/create-graph/", "_blank");
            },
            onMouseEnter: () => {},
            onFocus: () => {}
        }
    ]
};
const group3 = {
    items: [
        {
            title: "Documentation",
            icon: ImportContacts,
            show: true,
            to: "https://kumo.ai/docs/",
            onClick: () => {
                window.open("https://kumo.ai/docs/", "_blank");
            }
        }
    ],
    hideSeparator: true
};

const sideBarData: SideBarData = {
    userProfile: {
        name: "Test Account",
        email: "test@kumo.ai",
        avatar: "",
        show: true,
        onLogoutClick: () => {
            window.open("/", "_blank");
        }
    },
    navMainMenuGroup: [group1, group2, group3],

    navFooterMenu: [],
    navQuickStart: [
        {
            title: "New",
            icon: Loupe,
            childs: [
                {
                    title: "Model",
                    description:
                        "Train a model with a graph schema and a predictive query.",
                    icon: DeveloperBoard,
                    onClick: () => {},
                    onMouseEnter: () => {
                        console.log("onNewModelMouseEnter");
                    },
                    onMouseLeave: () => {
                        console.log("onNewModelMouseLeave");
                    },
                    onFocus: () => {
                        console.log("onNewModelFocus");
                    }
                },
                {
                    title: "Prediction",
                    description:
                        "Generate predictions using a trained model and graph.",
                    icon: DeveloperBoard,
                    onClick: () => {},
                    onMouseEnter: () => {
                        console.log("onNewPredictionMouseEnter");
                    },
                    onMouseLeave: () => {
                        console.log("onNewPredictionMouseLeave");
                    },
                    onFocus: () => {
                        console.log("onNewPredictionFocus");
                    }
                }
            ]
        }
    ],
    onLogoClick: () => {},
    onLogoMouseEnter: () => {},
    onLogoFocus: () => {}
};
