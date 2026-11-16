// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export type SideBarData = {
    userProfile: SidebarDataUser;
    navQuickStart: SidebarDataNav[];
    navMainMenuGroup: SidebarNavMenuGroup[];
    navFooterMenu: SidebarDataNav[];
    hideTrigger?: boolean;
    onLogoClick?: () => void;
    onLogoMouseEnter?: () => void;
    onLogoMouseLeave?: () => void;
    onLogoFocus?: () => void;
};

export type SidebarNavMenuGroup = {
    items: SidebarDataNav[];
    hideSeparator?: boolean;
};

export type SidebarDataUser = {
    name: string;
    email: string;
    show: boolean;
    avatar?: string;
    onLogoutClick: () => void;
    onLogoutMouseEnter?: () => void;
    onLogoutMouseLeave?: () => void;
    onLogoutFocus?: () => void;
};

export type SidebarDataNav = {
    title: string;
    className?: string;
    children?: React.ReactNode;
    isActive?: boolean;
    show?: boolean;
    icon?: ({ className }: { className?: string | undefined }) => JSX.Element;
    childs?: SidebarDataNavChild[];
    onClick?: () => void;
    onMouseEnter?: () => void;
    onMouseLeave?: () => void;
    onFocus?: () => void;
    to?: string;
};

export type SidebarDataNavChild = {
    title: string;
    description: string;
    icon: ({ className }: { className?: string | undefined }) => JSX.Element;
    onClick?: () => void;
    onMouseEnter?: () => void;
    onMouseLeave?: () => void;
    onFocus?: () => void;
    to?: string;
};
