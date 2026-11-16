// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useMemo, useState } from "react";

import {
    clearDirectConnectLoggedOut,
    initDirectConnect,
    isDirectConnectLoggedOut,
    loginDirectConnect
} from "hooks/auth/directConnectAuth";

import { navRoutes } from "utils/Constants/routes";
import { isDirectConnectPasswordLogin } from "utils/directConnectAuth";

const LOGIN_PATH = "/login";

interface DirectConnectLoginSession {
    checkingSession: boolean;
    finishLogin: () => Promise<void>;
    loggedOut: boolean;
    onSignInAgainClick: () => void;
    passwordMode: boolean;
}

export const useDirectConnectLoginSession = (): DirectConnectLoginSession => {
    const passwordMode = useMemo(() => isDirectConnectPasswordLogin(), []);
    const loggedOutFromQuery = useMemo(
        () => new URLSearchParams(window.location.search).has("logged_out"),
        []
    );
    const [loggedOut] = useState(
        loggedOutFromQuery || isDirectConnectLoggedOut()
    );
    const [checkingSession, setCheckingSession] = useState(!loggedOut);

    const finishLogin = useCallback(async () => {
        clearDirectConnectLoggedOut();
        await initDirectConnect();
        window.location.assign(navRoutes.Home);
    }, []);

    const onSignInAgainClick = useCallback(() => {
        if (passwordMode) {
            clearDirectConnectLoggedOut();
            window.location.assign(LOGIN_PATH);
            return;
        }
        loginDirectConnect();
    }, [passwordMode]);

    useEffect(() => {
        let cancelled = false;

        const checkSession = async () => {
            const isLoggedIn = await initDirectConnect();
            if (cancelled) return;
            if (isLoggedIn) {
                window.location.assign(navRoutes.Home);
                return;
            }
            if (passwordMode) {
                setCheckingSession(false);
                return;
            }
            loginDirectConnect();
        };

        if (!loggedOut) void checkSession();

        return () => {
            cancelled = true;
        };
    }, [loggedOut, passwordMode]);

    return {
        checkingSession,
        finishLogin,
        loggedOut,
        onSignInAgainClick,
        passwordMode
    };
};
