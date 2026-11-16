// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React, { Suspense, useEffect, useMemo, useRef, useState } from "react";
import { matchPath, Outlet, useLocation } from "react-router";

import {
    AppSidebar,
    Bouncingball,
    Button,
    cn,
    FolderOpen,
    getFromLocalStorage,
    SidebarInset,
    sidebarMenuButtonVariants,
    SidebarMenuItem,
    SidebarMobileTrigger,
    SidebarProvider,
    Skeleton,
    storeToLocalStorage,
    Tooltip,
    TooltipContent,
    TooltipTrigger
} from "@kumo-ai/kumo-ui-catalog";
import { useDebugStore } from "stores/debugStore";

import DebugConsole from "components/Debug/DebugConsole";
import ProjectList from "components/Project/ProjectList";

import { useAdminInfo } from "hooks/auth/admin";
import { useProject, useProjectId } from "hooks/project/projects";
import { useSidebarData } from "hooks/sidebar/useSidebarData";

import { navRoutes, SIDEBAR_VIEW_KEY } from "utils/Constants";

// Routes that operate without project context, retain the global navigation,
// and let projectless users dismiss the project picker.
const workspaceScopedRoutes = [navRoutes.ApiKeys];

const isWorkspaceScopedRoute = (pathname: string) =>
    workspaceScopedRoutes.some((route) => matchPath(route, pathname) !== null);

/**
 *  Main Layout component.
 *
 *  This component contains the header component and outlet.
 *
 *  Outlet is to render all the other pages.
 *
 *  @returns ``` <MainLayout/>```
 *
 */
const MainLayout = () => {
    const { isAdmin, isLoading } = useAdminInfo();
    const { fetchData, configFetched } = useDebugStore();
    const selectedProject = useProjectId();
    const { pathname } = useLocation();
    const [showProjectList, setShowProjectList] = useState(false);
    const hasAutoOpenedProjectList = useRef(false);

    // Auto-open the project picker only on the post-login landing (`/home`),
    // which has no project of its own and is otherwise empty. Workspace-scoped
    // pages such as `/api-keys` and `/admin/*` function without a project, so
    // popping the picker over them is spurious. Matched via `matchPath` so the
    // check mirrors the router (`/home/` reaches this route too).
    useEffect(() => {
        if (
            !hasAutoOpenedProjectList.current &&
            !selectedProject &&
            matchPath(navRoutes.Home, pathname) !== null
        ) {
            hasAutoOpenedProjectList.current = true; // To avoid multiple render
            setShowProjectList(true);
        }
    }, [selectedProject, pathname]);

    const [sidebarOpen, setSidebarOpen] = useState(() => {
        return getFromLocalStorage<boolean>(SIDEBAR_VIEW_KEY) ?? true;
    });
    const hideMenu =
        pathname === navRoutes.NewProject ||
        (!selectedProject && !isWorkspaceScopedRoute(pathname));

    const { projectInfo, isLoading: fetchingProject } = useProject(
        selectedProject as string
    );
    const projectButtonLabel = projectInfo?.display_name ?? "Default";

    // Get sidebar data from custom hook
    const sideBarData = useSidebarData({
        hideMenu,
        projectInfo,
        isAdmin
    });

    const sideBarWithProject = useMemo(() => {
        // Add Project children that requires component state
        if (sideBarData.navQuickStart?.[0]) {
            return {
                ...sideBarData,
                navQuickStart: [
                    {
                        title: "Project",
                        icon: FolderOpen,
                        children: (
                            <SidebarMenuItem key={"Project"} className="pb-6">
                                <Button
                                    data-testid="project-switcher-button"
                                    onClick={() => {
                                        setShowProjectList((prev) => !prev);
                                    }}
                                    id="project"
                                    variant="no-outline"
                                    className={cn(
                                        sidebarMenuButtonVariants({
                                            variant: "secondary"
                                        }),
                                        "hover:bg-kumo-grey-50 hover:border-kumo-grey-300 focus-visible:ring-0 focus-visible:border-inherit rounded-sm justify-start px-4 py-2 h-11 group-data-[collapsible=icon]:border-none"
                                    )}>
                                    <FolderOpen className="h-4 w-4 fill-text-primary" />

                                    <Tooltip id={projectButtonLabel}>
                                        <TooltipTrigger>
                                            <div className="group-data-[collapsible=icon]:hidden">
                                                {fetchingProject ? (
                                                    <Skeleton className="w-20 h-4" />
                                                ) : projectButtonLabel.length >
                                                  11 ? (
                                                    projectButtonLabel.substring(
                                                        0,
                                                        11
                                                    ) + "..."
                                                ) : (
                                                    projectButtonLabel
                                                )}
                                            </div>
                                        </TooltipTrigger>
                                        {!fetchingProject &&
                                            projectButtonLabel.length > 11 && (
                                                <TooltipContent
                                                    arrow={false}
                                                    sideOffset={5}>
                                                    {projectButtonLabel}
                                                </TooltipContent>
                                            )}
                                    </Tooltip>
                                </Button>
                            </SidebarMenuItem>
                        )
                    },
                    ...(!selectedProject ? [] : sideBarData.navQuickStart)
                ]
            };
        }
        return sideBarData;
    }, [fetchingProject, projectButtonLabel, selectedProject, sideBarData]);

    const hideAppSidebar =
        pathname.startsWith(navRoutes.Admin) && pathname !== navRoutes.Admin;

    useEffect(() => {
        fetchData();

        // Expose reopenProjectList globally for NewProject to reopen modal.
        // This is a one-time initialization, placed here alongside fetchData to avoid an additional useEffect.
        if (!window.reopenProjectList) {
            window.reopenProjectList = () => setShowProjectList(true);
        }
    }, [fetchData]);

    useEffect(() => {
        storeToLocalStorage(SIDEBAR_VIEW_KEY, sidebarOpen);
    }, [sidebarOpen]);

    // Show loading untill configs are fetched
    if (configFetched === false || isLoading) {
        return <Bouncingball overlay={true} />;
    }
    return (
        <div data-testid="main-layout">
            <SidebarProvider
                open={sidebarOpen}
                onOpenChange={setSidebarOpen}
                style={
                    hideMenu
                        ? ({
                              "--sidebar-width": "10.6rem"
                          } as React.CSSProperties)
                        : undefined
                }>
                {!hideAppSidebar && (
                    <AppSidebar
                        className={
                            "z-50 [&_*]:leading-5 " +
                            (hideMenu ? "border-none" : "")
                        }
                        sidebarData={sideBarWithProject}
                    />
                )}
                <SidebarInset>
                    <SidebarMobileTrigger />
                    <div
                        className={
                            "flex-1 min-h-0 flex flex-col overflow-hidden " +
                            (hideAppSidebar ? "" : "p-6")
                        }>
                        <Suspense fallback={<Bouncingball overlay={true} />}>
                            <Outlet />
                        </Suspense>
                    </div>
                </SidebarInset>
            </SidebarProvider>
            <ProjectList
                open={showProjectList}
                toggleModal={() => {
                    setShowProjectList((prev) => !prev);
                }}
                enableClose={
                    !!selectedProject || isWorkspaceScopedRoute(pathname)
                }
            />
            <DebugConsole />
        </div>
    );
};

export default MainLayout;
