// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useState } from "react";

import { Bouncingball, Button } from "@kumo-ai/kumo-ui-catalog";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";

import { logoutDirectConnect } from "hooks/auth/directConnectAuth";

const DirectConnectLogout = () => {
    const [logoutFailed, setLogoutFailed] = useState(false);
    const [retrying, setRetrying] = useState(false);

    const onLogout = useCallback(async () => {
        setLogoutFailed(false);
        setRetrying(true);
        const loggedOut = await logoutDirectConnect();
        if (!loggedOut) {
            setLogoutFailed(true);
            setRetrying(false);
        }
    }, []);

    useEffect(() => {
        onLogout();
    }, [onLogout]);

    if (logoutFailed) {
        return (
            <div className="flex h-screen w-screen flex-col items-center justify-center gap-4">
                <span className="text-sm text-title-secondary">
                    Sign out failed
                </span>
                <Button size="sm" disabled={retrying} onClick={onLogout}>
                    Try again
                </Button>
            </div>
        );
    }

    return (
        <div className="flex items-center justify-center">
            <Bouncingball overlay={false} className="h-24" />
        </div>
    );
};

export default WithErrorBoundary(DirectConnectLogout);
