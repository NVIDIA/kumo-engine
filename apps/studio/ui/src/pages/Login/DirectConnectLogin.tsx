// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Description } from "@kumo-ai/kumo-ui-catalog";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";

import DirectConnectLoggedOut from "pages/Login/DirectConnectLoggedOut";
import DirectConnectPasswordForm from "pages/Login/DirectConnectPasswordForm";
import TwoColumnLoginLayout from "pages/Login/TwoColumnLoginLayout";
import { useDirectConnectLoginSession } from "pages/Login/useDirectConnectLoginSession";

const DirectConnectLogin = () => {
    const {
        checkingSession,
        finishLogin,
        loggedOut,
        onSignInAgainClick,
        passwordMode
    } = useDirectConnectLoginSession();

    if (loggedOut) {
        return (
            <DirectConnectLoggedOut
                passwordMode={passwordMode}
                onSignInAgainClick={onSignInAgainClick}
            />
        );
    }

    if (passwordMode) {
        return (
            <TwoColumnLoginLayout>
                <DirectConnectPasswordForm
                    checkingSession={checkingSession}
                    onLoginComplete={finishLogin}
                />
            </TwoColumnLoginLayout>
        );
    }

    return (
        <div className="flex h-screen w-screen items-center justify-center">
            <Description className="text-sm text-title-secondary">
                Redirecting to login...
            </Description>
        </div>
    );
};

export default WithErrorBoundary(DirectConnectLogin);
