"use client";
// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0


import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import {
    DropdownMenu,
    DropdownMenuContent,
    DropdownMenuLabel,
    DropdownMenuSeparator,
    DropdownMenuTrigger
} from "@/components/ui/dropdown-menu";
import {
    SidebarMenu,
    sidebarMenuButtonVariants,
    SidebarMenuItem,
    useSidebar
} from "@/components/ui/sidebar";
import { cn } from "@/lib/utils";
import { SidebarDataUser } from "@/types/sidebar";

export function NavUser({ user }: { user: SidebarDataUser }) {
    const { isMobile, state } = useSidebar();

    return (
        <SidebarMenu>
            <SidebarMenuItem className="pb-0">
                <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                        <button
                            aria-label={`User menu for ${user?.name || "user"}`}
                            className={cn(
                                sidebarMenuButtonVariants({
                                    variant: "outline"
                                }),
                                ""
                            )}>
                            <Avatar className="size-5">
                                <AvatarImage
                                    src={user?.avatar}
                                    alt={user?.name}
                                />
                                <AvatarFallback className="text-title-primary">
                                    {user?.name?.trim() && (
                                        <span className="">
                                            {user?.name.trim()[0]}
                                        </span>
                                    )}
                                </AvatarFallback>
                            </Avatar>
                            {state === "expanded" && (
                                <div className="grid flex-1 text-left text-base leading-tight">
                                    <span className="truncate font-normal">
                                        {user?.name}
                                    </span>
                                </div>
                            )}
                        </button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent
                        className="w-(--radix-dropdown-menu-trigger-width) min-w-56 rounded-sm px-3 py-2"
                        side={isMobile ? "bottom" : "right"}
                        align="end"
                        sideOffset={4}>
                        <DropdownMenuLabel className="p-0 font-medium">
                            <div className="flex flex-col items-center gap-2 px-1 py-1.5 text-center text-sm">
                                <Avatar className="h-16 w-16 rounded-sm">
                                    <AvatarImage
                                        src={user?.avatar}
                                        alt={user?.name}
                                    />
                                    <AvatarFallback className="text-title-primary text-2xl">
                                        {user?.name?.trim() && (
                                            <span>{user?.name.trim()[0]}</span>
                                        )}
                                    </AvatarFallback>
                                </Avatar>
                                <div className="text-title-primary grid flex-1 text-center leading-tight">
                                    <span className="truncate text-base font-medium">
                                        {user?.name}
                                    </span>
                                    <span className="text-title-secondary truncate text-sm">
                                        {user?.email}
                                    </span>
                                </div>
                            </div>
                        </DropdownMenuLabel>
                        <div className="px-3">
                            <DropdownMenuSeparator className="my-1" />
                        </div>
                        <button
                            className={cn(
                                sidebarMenuButtonVariants({
                                    variant: "outline"
                                }),
                                "flex flex-row !items-center justify-center !text-center !font-medium"
                            )}
                            onClick={() => {
                                user.onLogoutClick();
                            }}
                            onMouseEnter={() => {
                                if (user?.onLogoutMouseEnter) {
                                    user.onLogoutMouseEnter();
                                }
                            }}
                            onMouseLeave={() => {
                                if (user?.onLogoutMouseLeave) {
                                    user.onLogoutMouseLeave();
                                }
                            }}
                            onFocus={() => {
                                if (user?.onLogoutFocus) {
                                    user.onLogoutFocus();
                                }
                            }}>
                            <div>Log Out</div>
                        </button>
                    </DropdownMenuContent>
                </DropdownMenu>
            </SidebarMenuItem>
        </SidebarMenu>
    );
}
