// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useLocation } from "react-router";

import {
    AlertCtaItem,
    AlertCtaLevel,
    AlertCtaList,
    cn
} from "@kumo-ai/kumo-ui-catalog";
import type { ErrWarnDetails } from "gen/v0/types";

import { ctaLevels } from "types/constants";
import { CTALevel } from "types/warnings";

import { warningsToShow } from "utils/Constants";
import { adjustError } from "utils/errorHandler";
import { useCustomNavigation } from "utils/navigator";

export interface WarningErrorCTAProps {
    id?: string;
    data: Array<ErrWarnDetails>;
    ctaLevel: CTALevel;
    handleClose?: () => void;
    title?: string;
    hideTitle?: boolean;
    className?: string;
}

function levelToCatalog(value: string): AlertCtaLevel {
    if (value === ctaLevels.Info.value) return "info";
    if (value === ctaLevels.Warning.value) return "warning";
    return "error";
}

const WarningErrorCTA = ({
    id,
    data,
    ctaLevel,
    handleClose,
    title,
    hideTitle = false,
    className = ""
}: WarningErrorCTAProps) => {
    const navigate = useCustomNavigation();
    const location = useLocation();

    if (!data || data.length === 0) return null;

    const catalogLevel = levelToCatalog(ctaLevel.value);

    return (
        <AlertCtaList
            id={id}
            data={data as AlertCtaItem[]}
            level={catalogLevel}
            title={title}
            hideTitle={hideTitle}
            previewCount={warningsToShow}
            onClose={handleClose}
            shouldShowCta={(url) => url !== location.pathname}
            onCtaClick={(url, e) =>
                navigate(
                    url,
                    undefined,
                    e as unknown as React.MouseEvent<HTMLAnchorElement>
                )
            }
            formatDescription={(text) => adjustError(text ?? "")}
            className={cn(
                // Standardize the error banner to a fixed height that fits a
                // typical message (~3 lines) and scrolls beyond it, so longer
                // messages are no longer cramped in a too-short banner.
                catalogLevel === "error" && "max-h-36 overflow-y-auto",
                className
            )}
        />
    );
};

export default WarningErrorCTA;
