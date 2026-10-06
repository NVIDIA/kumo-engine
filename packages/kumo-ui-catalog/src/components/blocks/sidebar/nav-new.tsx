// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
    DropdownMenu,
    DropdownMenuContent,
    DropdownMenuItem,
    DropdownMenuTrigger
} from "@/components/ui/dropdown-menu";
import {
    SidebarGroup,
    SidebarMenu,
    SidebarMenuButton,
    sidebarMenuButtonVariants,
    SidebarMenuItem,
    useSidebar
} from "@/components/ui/sidebar";
import { ChevronRight } from "@/icons";
import { cn } from "@/lib/utils";
import { SidebarDataNav } from "@/types/sidebar";
import { Fragment, useState } from "react";

export function NavNew({ items }: { items: SidebarDataNav[] }) {
    const { isMobile } = useSidebar();
    const [open, setOpen] = useState(false);

    return (
        <SidebarGroup>
            <SidebarMenu>
                {items?.map((item) => {
                    if (item.show === false) {
                        return null;
                    }
                    if (item.children) {
                        return (
                            <Fragment key={item.title}>
                                {item.children}
                            </Fragment>
                        );
                    }
                    return (
                        <SidebarMenuItem key={item.title} className="pb-6">
                            <DropdownMenu open={open} onOpenChange={setOpen}>
                                <DropdownMenuTrigger asChild>
                                    <button
                                        aria-label={item.title}
                                        className={cn(
                                            sidebarMenuButtonVariants({
                                                variant: "default"
                                            }),
                                            "h-9"
                                        )}>
                                        {item.icon && <item.icon />}
                                        <span className="group-data-[collapsible=icon]:hidden">
                                            {item.title}
                                        </span>
                                        <ChevronRight className="ml-auto size-5 group-data-[collapsible=icon]:hidden"></ChevronRight>
                                    </button>
                                </DropdownMenuTrigger>
                                <DropdownMenuContent
                                    className="w-51 rounded-sm px-1 py-1"
                                    side={isMobile ? "bottom" : "right"}
                                    align={isMobile ? "end" : "start"}>
                                    {item?.childs?.map((item) => (
                                        <DropdownMenuItem
                                            className="p-0"
                                            key={item.title}
                                            onFocus={item?.onFocus}
                                            onKeyDown={(e) => {
                                                if (e.key === "Enter") {
                                                    e.preventDefault();
                                                    const anchor =
                                                        e.currentTarget.querySelector(
                                                            "a"
                                                        );
                                                    if (anchor) anchor.click();
                                                }
                                            }}>
                                            <SidebarMenuButton
                                                variant="outline"
                                                className={"!h-16 !px-2 !py-1"}
                                                to={item?.to as string}
                                                onClick={(e) => {
                                                    if (item?.onClick) {
                                                        const {
                                                            metaKey,
                                                            ctrlKey,
                                                            shiftKey
                                                        } = e;

                                                        if (
                                                            metaKey ||
                                                            ctrlKey ||
                                                            shiftKey
                                                        ) {
                                                            setOpen(false);
                                                            return;
                                                        }
                                                        e.preventDefault();
                                                        item.onClick();
                                                        setOpen(false);
                                                    }
                                                }}
                                                onMouseEnter={
                                                    item?.onMouseEnter
                                                }
                                                onMouseLeave={
                                                    item?.onMouseLeave
                                                }>
                                                <div className="flex flex-col">
                                                    <div className="inline-flex items-center justify-start gap-1">
                                                        {item.icon && (
                                                            <item.icon className="!size-4" />
                                                        )}
                                                        <div className="justify-start text-xs font-bold leading-none">
                                                            {item.title}
                                                        </div>
                                                    </div>
                                                    <div className="mt-1.5 justify-start self-stretch text-xs font-normal leading-none">
                                                        {item.description}
                                                    </div>
                                                </div>
                                            </SidebarMenuButton>
                                        </DropdownMenuItem>
                                    ))}
                                </DropdownMenuContent>
                            </DropdownMenu>
                        </SidebarMenuItem>
                    );
                })}
            </SidebarMenu>
        </SidebarGroup>
    );
}
