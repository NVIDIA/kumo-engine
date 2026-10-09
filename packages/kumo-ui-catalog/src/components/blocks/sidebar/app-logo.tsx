// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
    SidebarMenu,
    SidebarMenuItem,
    SidebarTrigger,
    useSidebar
} from "@/components/ui/sidebar";

import KumoSvgLogo from "@/images/kumo_logo.svg";

export function AppLogo({
    onLogoClick,
    onLogoMouseEnter,
    onLogoMouseLeave,
    onLogoFocus,
    hideTrigger = false
}: {
    onLogoClick?: () => void;
    onLogoMouseEnter?: () => void;
    onLogoMouseLeave?: () => void;
    onLogoFocus?: () => void;
    hideTrigger?: boolean;
}) {
    const { state } = useSidebar();

    return (
        <SidebarMenu>
            <SidebarMenuItem className="pb-6">
                <div className="inline-flex w-full items-center justify-between transition-[width,height,padding] ease-in">
                    {state === "collapsed" ? (
                        !hideTrigger && <SidebarTrigger className="w-full" />
                    ) : (
                        <>
                            <a
                                className="focus-visible:border-ring w-20 cursor-pointer bg-white focus-visible:ring-[3px]"
                                href="/"
                                onClick={(e) => {
                                    if (onLogoClick) {
                                        e.preventDefault();
                                        onLogoClick();
                                    }
                                }}
                                onMouseEnter={() => {
                                    if (onLogoMouseEnter) {
                                        onLogoMouseEnter();
                                    }
                                }}
                                onMouseLeave={() => {
                                    if (onLogoMouseLeave) {
                                        onLogoMouseLeave();
                                    }
                                }}
                                onFocus={() => {
                                    if (onLogoFocus) {
                                        onLogoFocus();
                                    }
                                }}>
                                <img
                                    src={KumoSvgLogo}
                                    height={35}
                                    width={71.01}
                                    className=""
                                    alt="Kumo logo"
                                />
                            </a>
                            {!hideTrigger && <SidebarTrigger />}
                        </>
                    )}
                </div>
            </SidebarMenuItem>
        </SidebarMenu>
    );
}
