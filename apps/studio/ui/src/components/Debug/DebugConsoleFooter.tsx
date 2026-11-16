// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Button } from "@kumo-ai/kumo-ui-catalog";

export interface DebugConsoleFooterProps {
    message: string;
    handleCopyAll: () => void;
    loading: boolean;
    handleClose: () => void;
}

const DebugConsoleFooter = ({
    message,
    handleCopyAll,
    loading,
    handleClose
}: DebugConsoleFooterProps) => {
    return (
        <>
            <div className="mt-1">{message}</div>

            <Button
                id="CopyAll"
                onClick={handleCopyAll}
                disabled={loading}
                variant="outline">
                {"Copy All"}
            </Button>
            <Button id={"Close"} onClick={handleClose}>
                {"Close"}
            </Button>
        </>
    );
};

export default DebugConsoleFooter;
