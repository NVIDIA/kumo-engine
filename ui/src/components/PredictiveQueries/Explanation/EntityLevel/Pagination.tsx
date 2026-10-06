// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React from "react";

import {
    Button,
    ChevronLeft,
    ChevronRight,
    Tooltip,
    TooltipContent,
    TooltipTrigger
} from "@kumo-ai/kumo-ui-catalog";

export type PaginationComponentProps = {
    currentSelection: number;
    totalOptions: number;
    onSelectChange: (selection: number) => void;
    siblingCount?: number;
    boundaryCount?: number;
    labelText?: string;
    entityIdentifier?: string;
};

const LABEL_TRUNCATE_LENGTH = 28;

const PaginationComponent: React.FC<PaginationComponentProps> = ({
    currentSelection,
    totalOptions,
    onSelectChange,
    labelText,
    entityIdentifier
}) => {
    const isTruncated = !!labelText && labelText.length > LABEL_TRUNCATE_LENGTH;

    return (
        <div className="flex items-center gap-4">
            <Button
                id={"Previous"}
                variant="light-outline"
                className="size-9 p-0"
                aria-label="Previous"
                disabled={currentSelection === 1}
                onClick={() =>
                    onSelectChange(Math.max(1, currentSelection - 1))
                }>
                <ChevronLeft className="size-4" />
            </Button>

            <div className="flex items-center gap-2">
                {entityIdentifier && (
                    <span className="text-base font-bold text-title-primary">
                        {entityIdentifier}
                    </span>
                )}
                {labelText &&
                    (isTruncated ? (
                        <Tooltip id={"entity-tooltip"}>
                            <TooltipTrigger>
                                <span className="text-base font-bold text-title-primary">
                                    {labelText.slice(
                                        0,
                                        LABEL_TRUNCATE_LENGTH - 1
                                    ) + "..."}
                                </span>
                            </TooltipTrigger>
                            <TooltipContent
                                arrow={false}
                                side={"bottom"}
                                sideOffset={5}>
                                {labelText}
                            </TooltipContent>
                        </Tooltip>
                    ) : (
                        <span className="text-base font-bold text-title-primary">
                            {labelText}
                        </span>
                    ))}
            </div>

            <Button
                id={"next"}
                variant="light-outline"
                className="size-9 p-0"
                aria-label="Next"
                disabled={currentSelection === totalOptions}
                onClick={() =>
                    onSelectChange(Math.min(totalOptions, currentSelection + 1))
                }>
                <ChevronRight className="size-4" />
            </Button>
        </div>
    );
};

export default PaginationComponent;
