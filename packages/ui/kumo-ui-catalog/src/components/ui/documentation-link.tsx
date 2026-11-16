// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import * as React from "react";

import { TextLink } from "@/components/ui/text-link";
import { ArrowOutward } from "@/icons/MaterialIcons/ArrowOutward";

interface DocumentationLinkProps {
    id: string;
    pageLink?: string;
    linkText: string;
    className?: string;
    url?: string;
    isNewDocs?: boolean;
    showExternalIcon?: boolean;
}

export const DocumentationLink: React.FC<DocumentationLinkProps> = ({
    id,
    pageLink,
    linkText,
    className = "",
    url,
    isNewDocs = false,
    showExternalIcon = false
}) => {
    const docsBaseUrl: string = "https://kumo.ai/docs/";
    const newDocsURL = isNewDocs ? "https://kumo.ai/docs/" : docsBaseUrl;
    const linkTitle = showExternalIcon ? (
        <span className="inline-flex items-center gap-0.5">
            {linkText}
            <ArrowOutward
                className="inline-block size-3.5 fill-current align-middle"
                aria-hidden="true"
            />
        </span>
    ) : (
        linkText
    );

    return (
        <TextLink
            id={id}
            data_testid={`testid-${id}`}
            pageLink={url ? url : newDocsURL + pageLink}
            className={className}
            newTab={true}
            onClick={(e) => {
                e.preventDefault();
                window.open(url ? url : newDocsURL + pageLink, "_blank");
            }}
            title={linkTitle}
        />
    );
};

export type { DocumentationLinkProps };
