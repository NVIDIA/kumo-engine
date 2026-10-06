// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
    ColumnDef,
    Row,
    Tooltip,
    TooltipContent,
    TooltipTrigger
} from "@kumo-ai/kumo-ui-catalog";

import { SubgraphSummary } from "types/training_job";

export const SubgraphColumns = (): ColumnDef<SubgraphSummary>[] => {
    return [
        {
            id: "property",
            header: "Property",
            accessorKey: "property",
            meta: { className: "text-title-primary" }
        },
        {
            id: "current",
            header: "Current Entity",
            accessorKey: "current",
            meta: { className: "text-title-primary" },
            cell: ({ row }: { row: Row<SubgraphSummary> }) => {
                const value = row?.original?.current;
                if (value === "<MISSING>") {
                    return (
                        <div className="flex items-center justify-between">
                            <Tooltip
                                id={
                                    "tooltip-missing-current-" +
                                    row?.original.global
                                }>
                                <TooltipTrigger>
                                    <span
                                        data-tip
                                        data-for={
                                            "tooltip-missing-current-" +
                                            row?.original.global
                                        }
                                        data-tip-disable={false}>
                                        <i>{"<MISSING>"}</i>
                                    </span>
                                </TooltipTrigger>
                                <TooltipContent side="top" className="w-80">
                                    Cannot extract corresponding value due to
                                    the limitation of graph structure or the
                                    value is unsupported for the current task.
                                </TooltipContent>
                            </Tooltip>
                        </div>
                    );
                }

                return (value ?? null) === null ? value : String(value);
            }
        },
        {
            id: "global",
            header: "Global",
            accessorKey: "global",
            meta: { className: "text-title-primary" },
            cell: ({ row }: { row: Row<SubgraphSummary> }) => {
                const value = row?.original?.global;
                if (value === "<MISSING>") {
                    return (
                        <div className="flex items-center justify-between">
                            <Tooltip
                                id={
                                    "tooltip-missing-global-" +
                                    row?.original.global
                                }>
                                <TooltipTrigger>
                                    <span
                                        data-tip
                                        data-for={
                                            "tooltip-missing-global-" +
                                            row?.original.global
                                        }
                                        data-tip-disable={false}>
                                        <i>{"<MISSING>"}</i>
                                    </span>
                                </TooltipTrigger>
                                <TooltipContent side="top" className="w-80">
                                    Cannot extract corresponding value due to
                                    the limitation of graph structure or the
                                    value is unsupported for the current task.
                                </TooltipContent>
                            </Tooltip>
                        </div>
                    );
                }
                return (value ?? null) === null ? value : String(value);
            }
        }
    ];
};
