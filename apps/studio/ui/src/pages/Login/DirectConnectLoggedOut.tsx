// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Button } from "@kumo-ai/kumo-ui-catalog";

import TwoColumnLoginLayout from "pages/Login/TwoColumnLoginLayout";

interface DirectConnectLoggedOutProps {
    passwordMode: boolean;
    onSignInAgainClick: () => void;
}

const DirectConnectLoggedOut = ({
    passwordMode,
    onSignInAgainClick
}: DirectConnectLoggedOutProps) => (
    <TwoColumnLoginLayout>
        <div className="mx-auto flex w-full max-w-sm flex-col items-center gap-6 text-center">
            <div className="flex flex-col gap-3">
                <h1 className="text-xl font-semibold text-title-primary">
                    You&apos;ve signed out
                </h1>
                <p className="text-sm text-title-secondary">
                    Your Kumo session has ended.
                    {!passwordMode &&
                        " You may still be signed in to your organization's identity provider."}
                </p>
            </div>
            <Button className="min-w-40" onClick={onSignInAgainClick}>
                Sign in again
            </Button>
        </div>
    </TwoColumnLoginLayout>
);

export default DirectConnectLoggedOut;
