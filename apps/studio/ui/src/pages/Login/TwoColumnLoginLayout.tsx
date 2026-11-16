// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { cn } from "@kumo-ai/kumo-ui-catalog";
import type { ReactNode } from "react";

import kumoLogo from "images/intro/kumo_logo.svg";

import LoginPanelImage from "pages/Login/LoginPanelImage";

interface TwoColumnLoginLayoutProps {
    children: ReactNode;
}

/*
 * Viewport behavior:
 *   < 1024px        — panels stack (form first, artwork below), page scrolls.
 *   1024px – 1800px — two columns, full-bleed (MacBooks + 1080p).
 *                     Grid is exactly 100vh (h-screen); form content must not exceed it.
 *   > 1800px        — two-column card floats centered on gray bg with border
 *                     + shadow (large external monitors), capped at 1440x900.
 *                     That treatment lives in main.css under the
 *                     [data-login-screen]/[data-login-card] rules with the
 *                     login-scoped tokens.
 */
const TwoColumnLoginLayout = ({ children }: TwoColumnLoginLayoutProps) => (
    <div
        data-login-screen=""
        className="flex min-h-screen items-center justify-center bg-surface-primary">
        <div
            data-login-card=""
            className={cn(
                "grid w-full !grid-cols-1 lg:h-screen lg:!grid-cols-2"
            )}>
            <div className="flex flex-col bg-surface-primary !px-6 !py-8 sm:!px-10 lg:!px-14 lg:!py-10">
                <img
                    src={kumoLogo}
                    alt="Kumo"
                    className="h-7 w-auto self-start"
                />
                <div className="flex flex-1 flex-col items-center justify-center py-10 lg:py-0">
                    {children}
                </div>
            </div>
            <LoginPanelImage />
        </div>
    </div>
);

export default TwoColumnLoginLayout;
