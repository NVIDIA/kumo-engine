// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useMemo, useState } from "react";

import { Button } from "@kumo-ai/kumo-ui-catalog";
import { ReactQueryDevtoolsPanel } from "@tanstack/react-query-devtools";

import { getReactQueryDevToolMode } from "utils/storage";

const ReactQueryDevTool = () => {
    const [isReactQueryDevToolOpen, setIsReactQueryDevToolOpen] =
        useState(false);
    const showReactQueryDevTool = useMemo(() => {
        return getReactQueryDevToolMode();
    }, []);

    return showReactQueryDevTool === "ENABLED" ? (
        <>
            <Button
                className="fixed bottom-0 left-0 z-[2000]"
                onClick={() =>
                    setIsReactQueryDevToolOpen(!isReactQueryDevToolOpen)
                }>{`${isReactQueryDevToolOpen ? "Close" : "Show"} react query devtools`}</Button>
            {isReactQueryDevToolOpen && (
                <div className="fixed bottom-0 right-0 z-[2000] w-125">
                    <ReactQueryDevtoolsPanel
                        onClose={() => setIsReactQueryDevToolOpen(false)}
                    />
                </div>
            )}
        </>
    ) : null;
};

export default ReactQueryDevTool;
