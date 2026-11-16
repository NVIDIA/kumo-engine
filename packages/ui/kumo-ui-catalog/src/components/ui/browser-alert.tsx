// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import * as React from "react";

import { useIsBrowserSupported } from "@/hooks/useBrowserSupport";

import { Alert, AlertDescription } from "./alert";

export interface BrowserAlertProps {
    text: string;
    variant?: React.ComponentProps<typeof Alert>["variant"];
}

export const BrowserAlert: React.FC<BrowserAlertProps> = ({
    text,
    variant = "default"
}) => {
    const isBrowserSupported = useIsBrowserSupported();
    if (isBrowserSupported) return null;
    return (
        <Alert variant={variant}>
            <AlertDescription>{text}</AlertDescription>
        </Alert>
    );
};
