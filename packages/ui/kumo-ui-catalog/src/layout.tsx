// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { SidebarProvider, SidebarTrigger } from "@/components/ui/sidebar";
import { AppSidebar } from "@/components/blocks/sidebar/app-sidebar";

export default function Layout({ children }: { children: React.ReactNode }) {
    return (
        <SidebarProvider>
            <AppSidebar
                sidebarData={{
                    userProfile: {
                        name: "",
                        email: "",
                        avatar: undefined,
                        onLogoutClick: function (): void {
                            throw new Error("Function not implemented.");
                        },
                        show: true
                    },
                    navMainMenuGroup: [],
                    navFooterMenu: [],
                    navQuickStart: []
                }}
            />
            <main>
                <SidebarTrigger />
                {children}
            </main>
        </SidebarProvider>
    );
}
