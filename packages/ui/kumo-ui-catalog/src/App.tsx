// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AppSidebar } from "@/components/blocks/sidebar/app-sidebar";
import { ThemeProvider } from "@/components/theme-provider";
import {
    SidebarInset,
    SidebarMobileTrigger,
    SidebarProvider
} from "@/components/ui/sidebar";
import { Separator } from "@radix-ui/react-select";
import {
    Breadcrumb,
    BreadcrumbItem,
    BreadcrumbLink,
    BreadcrumbList,
    BreadcrumbSeparator
} from "./components/ui/breadcrumb";
import {
    AdminPanelSettings,
    Cards,
    Cube,
    DeveloperBoard,
    ImportContacts,
    Loupe,
    NetworkNode,
    Power,
    SoundSampler
} from "./icons";
import { SideBarData } from "./types/sidebar";

function App() {
    // This is sample data.
    const data: SideBarData = {
        userProfile: {
            name: "Jane Doe",
            email: "jane.doe@example.com",
            avatar: "",
            onLogoutClick: () => {},
            onLogoutMouseEnter: () => {
                console.log("onLogoutMouseEnter");
            },
            onLogoutMouseLeave: () => {
                console.log("onLogoutMouseLeave");
            },
            onLogoutFocus: () => {
                console.log("onLogoutFocus");
            },
            show: true
        },
        navMainMenuGroup: [
            {
                items: [
                    {
                        title: "Models",
                        icon: Cube,
                        isActive: true,
                        onClick: () => {},
                        show: true,
                        onMouseEnter: () => {
                            console.log("onModelsMouseEnter");
                        },
                        onMouseLeave: () => {
                            console.log("onModelsMouseLeave");
                        },
                        onFocus: () => {
                            console.log("onModelsFocus");
                        }
                    },
                    {
                        title: "Predictions",
                        icon: SoundSampler,
                        onClick: () => {},
                        show: true,
                        onMouseEnter: () => {
                            console.log("onPredictionsMouseEnter");
                        },
                        onMouseLeave: () => {
                            console.log("onPredictionsMouseLeave");
                        },
                        onFocus: () => {
                            console.log("onPredictionsFocus");
                        }
                    }
                ]
            },
            {
                items: [
                    {
                        title: "Tables",
                        icon: Cards,
                        onClick: () => {},
                        show: true,
                        onMouseEnter: () => {
                            console.log("onTablesMouseEnter");
                        },
                        onMouseLeave: () => {
                            console.log("onTablesMouseLeave");
                        },
                        onFocus: () => {
                            console.log("onTablesFocus");
                        }
                    },
                    {
                        title: "Graphs",
                        icon: NetworkNode,
                        onClick: () => {},
                        show: true,
                        onMouseEnter: () => {
                            console.log("onGraphsMouseEnter");
                        },
                        onMouseLeave: () => {
                            console.log("onGraphsMouseLeave");
                        },
                        onFocus: () => {
                            console.log("onGraphsFocus");
                        }
                    },
                    {
                        title: "Documentation",
                        icon: ImportContacts,
                        onClick: () => {},
                        show: true,
                        onMouseEnter: () => {
                            console.log("onDocumentationMouseEnter");
                        },
                        onMouseLeave: () => {
                            console.log("onDocumentationMouseLeave");
                        },
                        onFocus: () => {
                            console.log("onDocumentationFocus");
                        }
                    },
                    {
                        title: "Admin",
                        icon: AdminPanelSettings,
                        onClick: () => {},
                        show: true,
                        onMouseEnter: () => {
                            console.log("onAdminMouseEnter");
                        },
                        onMouseLeave: () => {
                            console.log("onAdminMouseLeave");
                        },
                        onFocus: () => {
                            console.log("onAdminFocus");
                        }
                    }
                ]
            }
        ],
        navFooterMenu: [
            {
                title: "Connector",
                icon: Power,
                onClick: () => {},
                show: true,
                onMouseEnter: () => {
                    console.log("onConnectorMouseEnter");
                },
                onMouseLeave: () => {
                    console.log("onConnectorMouseLeave");
                },
                onFocus: () => {
                    console.log("onConnectorFocus");
                }
            }
        ],
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
        onLogoClick: () => {
            console.log("onLogoClick");
        },
        onLogoMouseEnter: () => {
            console.log("onLogoMouseEnter");
        },
        onLogoMouseLeave: () => {
            console.log("onLogoMouseLeave");
        },
        onLogoFocus: () => {
            console.log("onLogoFocus");
        }
    };

    return (
        <>
            <ThemeProvider defaultTheme="light" storageKey="kumo-ui-theme">
                <SidebarProvider>
                    <AppSidebar sidebarData={data} />
                    <SidebarInset>
                        <SidebarMobileTrigger className="ml-0 mt-0" />
                        <header className="flex h-16 shrink-0 items-center gap-2 transition-[width,height] ease-linear group-has-[[data-collapsible=icon]]/sidebar-wrapper:h-12">
                            <div className="flex items-center gap-2 px-4">
                                <Separator className="mr-2 h-4" />
                                <Breadcrumb>
                                    <BreadcrumbList>
                                        <BreadcrumbItem className="hidden md:block">
                                            <BreadcrumbLink href="#">
                                                Building Your Application
                                            </BreadcrumbLink>
                                        </BreadcrumbItem>
                                        <BreadcrumbSeparator className="hidden md:block" />
                                        <BreadcrumbItem>
                                            <div>Data Fetching</div>
                                        </BreadcrumbItem>
                                    </BreadcrumbList>
                                </Breadcrumb>
                            </div>
                        </header>
                        <div className="flex w-full flex-1 flex-col gap-4 p-4 pt-0">
                            <div className="grid auto-rows-min gap-4 md:grid-cols-3">
                                <div className="bg-muted/50 aspect-video rounded-xl" />
                                <div className="bg-muted/50 aspect-video rounded-xl" />
                                <div className="bg-muted/50 aspect-video rounded-xl" />
                            </div>
                            <div className="bg-muted/50 min-h-[100vh] w-full flex-1 rounded-xl md:min-h-min" />
                        </div>
                    </SidebarInset>
                </SidebarProvider>
            </ThemeProvider>
        </>
    );
}

export default App;
