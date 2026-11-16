// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
    Download,
    Downloading,
    Tooltip,
    TooltipContent,
    TooltipTrigger
} from "@kumo-ai/kumo-ui-catalog";

import useFeatureFlags from "hooks/utils/featureFlags";

export const HoldoutDownload = ({
    downloadClicked,
    onDownloadClick
}: {
    downloadClicked?: boolean;
    onDownloadClick?: () => void;
}) => {
    const featureFlags = useFeatureFlags();
    if (featureFlags.IS_DATABRICKS_DEPLOY || !onDownloadClick) return null;

    return (
        <div
            className="m-0 inline-flex items-center p-0 leading-none"
            onClick={onDownloadClick}>
            {downloadClicked ? (
                <Tooltip showDuration={3000} delayDuration={0}>
                    <TooltipTrigger>
                        <div>
                            <Downloading className="cursor-pointer !size-4" />
                        </div>
                    </TooltipTrigger>
                    <TooltipContent
                        arrow={false}
                        side={"bottom"}
                        sideOffset={10}>
                        {"Downloading..."}
                    </TooltipContent>
                </Tooltip>
            ) : (
                <Tooltip delayDuration={0}>
                    <TooltipTrigger>
                        <div>
                            <Download className="cursor-pointer !size-4" />
                        </div>
                    </TooltipTrigger>
                    <TooltipContent
                        arrow={false}
                        side={"bottom"}
                        sideOffset={10}>
                        {
                            "Download the Holdout value set used in this evaluation"
                        }
                    </TooltipContent>
                </Tooltip>
            )}
        </div>
    );
};
