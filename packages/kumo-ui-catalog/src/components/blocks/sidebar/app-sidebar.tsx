// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import * as React from "react";

import { NavMain } from "@/components/blocks/sidebar/nav-main";
import { NavNew } from "@/components/blocks/sidebar/nav-new";
import { NavUser } from "@/components/blocks/sidebar/nav-user";
import {
    Sidebar,
    SidebarContent,
    SidebarFooter,
    SidebarHeader
} from "@/components/ui/sidebar";
import { AppLogo } from "./app-logo";
import { SideBarData } from "@/types/sidebar";
import { Separator } from "@/components/ui/separator";

export function AppSidebar({
    sidebarData,
    ...props
}: React.ComponentProps<typeof Sidebar> & {
    sidebarData: SideBarData;
}) {
    return (
        <Sidebar collapsible="icon" {...props}>
            <SidebarHeader>
                <AppLogo
                    onLogoClick={sidebarData.onLogoClick}
                    onLogoMouseEnter={sidebarData?.onLogoMouseEnter}
                    onLogoMouseLeave={sidebarData?.onLogoMouseLeave}
                    onLogoFocus={sidebarData?.onLogoFocus}
                    hideTrigger={sidebarData?.hideTrigger}
                />
            </SidebarHeader>
            <SidebarContent>
                <NavNew items={sidebarData.navQuickStart} />
                {sidebarData.navMainMenuGroup.map((group, i) => (
                    <React.Fragment key={`menu-${i}`}>
                        <NavMain items={group.items} className="!py-0" />
                        {!group.hideSeparator && (
                            <div className="px-3 pb-4 group-data-[collapsible=icon]:px-0">
                                <Separator className="my-0" />
                            </div>
                        )}
                    </React.Fragment>
                ))}
            </SidebarContent>
            <SidebarFooter>
                <NavMain items={sidebarData.navFooterMenu} className="!p-0" />
                {sidebarData.userProfile.show && (
                    <NavUser user={sidebarData.userProfile} />
                )}
            </SidebarFooter>
        </Sidebar>
    );
}
