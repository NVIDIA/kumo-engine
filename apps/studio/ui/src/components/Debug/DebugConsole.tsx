// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useState } from "react";

import {
    Dialog,
    DialogContent,
    DialogFooter,
    DialogTitle,
    storeToLocalStorage,
    useCustomKeyPress
} from "@kumo-ai/kumo-ui-catalog";
import { useDebugStore } from "stores/debugStore";
import { useUserStore } from "stores/userStore";

import { useDebugInfo } from "hooks/misc/constants";

import { getBaseURL } from "services/axios_client";
import tokenService from "services/token_service";

import CacheConfiguration from "./CacheConfiguration";
import DebugConsoleFooter from "./DebugConsoleFooter";
import DebugInfoKeyPairs from "./DebugInfoKeyPairs";

const DebugConsole = () => {
    const [open, setOpen] = useState(false);
    const [message, setMessage] = useState("");

    const { data } = useDebugStore();
    const userId = useUserStore((state) => state.userDetails.user_id);

    const handleClose = () => {
        setOpen(false);
    };

    const { debugInfo, isLoading: globalLoading } = useDebugInfo();

    const hideINProd = process.env.NODE_ENV === "production";

    const toggleModal = () => {
        setOpen(!open);
        // Enables some hidden UI features (must refresh to see)
        storeToLocalStorage("DEBUG", "TRUE");
    };

    // Run `toggleModal` when CMD + D is clicked
    useCustomKeyPress(
        /* key */ "d",
        /* withMeta */ true,
        /* callback */ toggleModal
    );
    // Also support CTRL + M for windows users
    useCustomKeyPress(
        /* key */ "m",
        /* withMeta */ false,
        /* callback */ toggleModal
    );

    const handleCopyAll = () => {
        const pairs: string[] = [];
        if (debugInfo?.release_version)
            pairs.push(`Kumo Version: ${debugInfo?.release_version}`);
        if (userId) pairs.push(`User ID: ${userId}`);
        const accessToken = tokenService.getAccessToken();
        if (accessToken) pairs.push(`Access Token: ${accessToken}`);
        const baseUrl = getBaseURL();
        if (baseUrl) pairs.push(`REST API URL: ${baseUrl}`);
        if (debugInfo?.temporal_url)
            pairs.push(`Temporal URL: ${debugInfo.temporal_url}`);
        if (debugInfo?.job_queue_status)
            pairs.push(`Job Queue Info: ${debugInfo.job_queue_status}`);
        if (!hideINProd && debugInfo?.db_path)
            pairs.push(`DB Path: ${debugInfo.db_path}`);

        const allInfo = pairs.join("\n");
        navigator.clipboard.writeText(allInfo).then(() => {
            setMessage("Copied all debug info!");
            setTimeout(() => setMessage(""), 2000);
        });
    };

    return (
        <Dialog open={open} onOpenChange={() => handleClose()}>
            <DialogContent
                data-testid={"sdk-content"}
                className="!max-w-220 p-4 top-1/2">
                <DialogTitle className="!m-3 !mb-1 !font-bold">
                    {"Debug Console"}
                </DialogTitle>
                <div className="mx-3 max-h-[70vh] overflow-auto">
                    <DebugInfoKeyPairs
                        data={data}
                        debugInfo={debugInfo}
                        loading={globalLoading}
                        hideINProd={hideINProd}
                        userId={userId as string}
                        accessToken={tokenService.getAccessToken()}
                    />
                    <hr />
                    <CacheConfiguration />
                </div>
                <DialogFooter className="pe-2 me-2">
                    <DebugConsoleFooter
                        message={message}
                        handleCopyAll={handleCopyAll}
                        loading={globalLoading}
                        handleClose={handleClose}
                    />
                </DialogFooter>
            </DialogContent>
        </Dialog>
    );
};

export default DebugConsole;
