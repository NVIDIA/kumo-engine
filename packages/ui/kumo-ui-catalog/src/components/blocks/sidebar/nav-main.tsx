"use client";
// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0


import {
    SidebarGroup,
    SidebarMenu,
    SidebarMenuButton,
    SidebarMenuItem
} from "@/components/ui/sidebar";
import { SidebarDataNav } from "@/types/sidebar";

export function NavMain({
    className,
    items
}: {
    className?: string;
    items: SidebarDataNav[];
}) {
    return (
        <SidebarGroup className={className}>
            <SidebarMenu>
                {items.map((item) => {
                    return item?.show ? (
                        <SidebarMenuItem
                            key={"item" + item.title}
                            className={item.className}>
                            <SidebarMenuButton
                                variant="outline"
                                tooltip={item.title}
                                data-active={item?.isActive}
                                to={item?.to}
                                onClick={(e) => {
                                    if (item?.onClick) {
                                        const { metaKey, ctrlKey, shiftKey } =
                                            e;

                                        if (metaKey || ctrlKey || shiftKey) {
                                            return;
                                        }
                                        e.preventDefault();
                                        item.onClick();
                                    }
                                }}
                                onMouseEnter={item?.onMouseEnter}
                                onMouseLeave={item?.onMouseLeave}
                                onFocus={item?.onFocus}>
                                <>
                                    {item.icon && (
                                        <item.icon className="h-4 w-4" />
                                    )}
                                    <span className="group-data-[collapsible=icon]:hidden">
                                        {item.title}
                                    </span>
                                </>
                            </SidebarMenuButton>
                        </SidebarMenuItem>
                    ) : null;
                })}
            </SidebarMenu>
        </SidebarGroup>
    );
}
