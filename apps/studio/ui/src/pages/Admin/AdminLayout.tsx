// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React from "react";
import { Outlet, useLocation, useNavigate } from "react-router";

import {
    AdminPanel,
    Button,
    ChevronLeft,
    cn,
    sidebarMenuButtonVariants,
    SidebarMenuItem
} from "@kumo-ai/kumo-ui-catalog";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";

import { navRoutes } from "utils/Constants";

const adminNavItems = [
    { title: "Users", path: navRoutes.AdminMembers },
    { title: "Groups", path: navRoutes.AdminGroups },
    { title: "Identities", path: navRoutes.AdminIdentitySettings }
];

const AdminLayout: React.FC = () => {
    const navigate = useNavigate();
    const location = useLocation();

    return (
        <div className="flex flex-1 ">
            <div className="flex-shrink-0 flex flex-col bg-white px-4 py-6 border-r min-w-56">
                <ul className="m-0 !mb-0 flex w-full min-w-0 flex-col !p-0">
                    <SidebarMenuItem className="pb-6">
                        <Button
                            variant="outline"
                            onClick={() => navigate(navRoutes.Home)}
                            className={cn(
                                sidebarMenuButtonVariants({
                                    variant: "outline"
                                }),
                                "gap-1 justify-start ps-0"
                            )}>
                            <ChevronLeft className="!fill-text-title-secondary !size-8" />
                            <span>Back</span>
                        </Button>
                    </SidebarMenuItem>

                    <div
                        className={
                            "flex w-full items-center px-3 py4 text-base gap-2 mb-4 font-medium text-title-primary"
                        }>
                        <AdminPanel className="fill-text-title-primary " />
                        <span>Admin</span>
                    </div>

                    {adminNavItems.map((item) => (
                        <SidebarMenuItem key={item.title}>
                            <button
                                data-active={
                                    !!item.path &&
                                    location.pathname === item.path
                                }
                                onClick={() => item.path && navigate(item.path)}
                                className={cn(
                                    sidebarMenuButtonVariants({
                                        variant: "outline"
                                    })
                                )}>
                                <span>{item.title}</span>
                            </button>
                        </SidebarMenuItem>
                    ))}
                </ul>
            </div>

            <div className="flex-1 min-w-0 p-6">
                <Outlet />
            </div>
        </div>
    );
};

export default WithErrorBoundary(AdminLayout);
