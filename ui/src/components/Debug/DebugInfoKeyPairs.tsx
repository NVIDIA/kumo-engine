// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { KeyPair } from "@kumo-ai/kumo-ui-catalog";
import { DebugInfo } from "gen/v0/types";
import { DebugAppInfo } from "stores/debugStore";

import { getBaseURL } from "services/axios_client";

export interface DebugInfoKeyPairsProps {
    data: DebugAppInfo;
    debugInfo: DebugInfo | null;
    loading: boolean;
    hideINProd: boolean;
    userId: string;
    accessToken: string;
}

const DebugInfoKeyPairs = ({
    debugInfo,
    data,
    loading,
    hideINProd,
    userId,
    accessToken
}: DebugInfoKeyPairsProps) => {
    const baseURL = getBaseURL() || "";
    return (
        <>
            <KeyPair
                keyLabel="Kumo Version"
                value={
                    debugInfo?.release_version ||
                    data.release_version ||
                    (loading ? "Loading..." : "N/A")
                }
                showCopy={true}
            />

            <KeyPair
                keyLabel="User ID"
                value={userId as string}
                showCopy={true}
            />

            <KeyPair
                keyLabel="Access Token"
                value={accessToken}
                showCopy={true}
            />

            <KeyPair
                keyLabel="REST API URL"
                value={baseURL || ""}
                showCopy={true}
            />

            <KeyPair
                keyLabel="Temporal URL"
                value={
                    debugInfo?.temporal_url || (loading ? "Loading..." : "N/A")
                }
                showCopy={true}
            />

            <KeyPair
                keyLabel="Job Queue Info"
                value={
                    debugInfo?.job_queue_status ||
                    (loading ? "Loading..." : "N/A")
                }
                showCopy={true}
            />

            <KeyPair
                keyLabel="DB Path"
                value={
                    (!hideINProd && debugInfo?.db_path) ||
                    (loading ? "Loading..." : "N/A")
                }
                showCopy={true}
            />
        </>
    );
};

export default DebugInfoKeyPairs;
