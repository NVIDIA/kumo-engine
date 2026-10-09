// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
    cn,
    ColumnDef,
    Row,
    Tooltip,
    TooltipContent,
    TooltipTrigger
} from "@kumo-ai/kumo-ui-catalog";

import { EdgeStats } from "types/graphs";

import { graphEdgeThreshold } from "utils/Constants";
import { customSort } from "utils/Helpers/sortHelpers";

export const GraphLinkHealthColumns = (): ColumnDef<EdgeStats>[] => {
    return [
        {
            header: () => <span className="ml-2.5">Primary Key (A)</span>,
            accessorKey: "col_a",
            cell: ({ row }: { row: Row<EdgeStats> }) => {
                return (
                    <>
                        <label className="text-title-primary">
                            {(row?.original?.col_a_table_name as string) ?? ""}
                        </label>
                        <label className="text-title-secondary">
                            {"." + row?.original?.col_a_column_name}
                        </label>
                    </>
                );
            }
        },
        {
            header: () => <span className="ml-2.5">Foreign Key (B)</span>,
            accessorKey: "col_b",
            cell: ({ row }: { row: Row<EdgeStats> }) => {
                return (
                    <>
                        <label className="text-title-primary">
                            {(row?.original?.col_b_table_name as string) ?? ""}
                        </label>
                        <label className="text-title-secondary">
                            {"." + row?.original?.col_b_column_name}
                        </label>
                    </>
                );
            }
        },
        {
            header: () => <span className="ml-2.5">Link Type</span>,
            accessorKey: "link_type"
        },
        {
            header: () => (
                <Tooltip id={"Matches-A-in-B"}>
                    <TooltipTrigger>
                        <span className="ml-2.5">Matches (A in B)</span>
                    </TooltipTrigger>
                    <TooltipContent arrow={false} sideOffset={5}>
                        {"Percent of rows in A that have a matching key in B."}
                    </TooltipContent>
                </Tooltip>
            ),
            accessorKey: "matches_a_in_b",
            sortingFn: (a: Row<EdgeStats>, b: Row<EdgeStats>) =>
                customSort(a, b, "matches_a_in_b_original"),
            cell: ({ row }: { row: Row<EdgeStats> }) => {
                const isBothWeak =
                    row?.original?.weak_a_in_b && row?.original?.weak_b_in_a;
                const isDirectionWeak = row?.original?.weak_a_in_b;

                if (isBothWeak) {
                    return (
                        <Tooltip id={row?.original?.percent_a_in_b as string}>
                            <TooltipTrigger>
                                <span>
                                    <span className="text-title-primary">
                                        {
                                            row?.original
                                                ?.matches_a_in_b as string
                                        }
                                    </span>
                                    <br />
                                    <div className="text-orange-500">
                                        ({row?.original?.percent_a_in_b})
                                    </div>
                                </span>
                            </TooltipTrigger>
                            <TooltipContent
                                className="max-w-xs text-justify"
                                arrow={false}
                                side={"bottom"}
                                sideOffset={5}>
                                {`This edge has a match rate lower than ` +
                                    `our recommended percentage of ` +
                                    `${graphEdgeThreshold}%. Please make sure ` +
                                    `this is expected, as it could cause a ` +
                                    `degradation in model performance.`}
                            </TooltipContent>
                        </Tooltip>
                    );
                }

                return (
                    <div className="text-title-secondary">
                        <span className="text-title-primary">
                            {row?.original?.matches_a_in_b as string}
                        </span>
                        <br />
                        <div
                            className={cn(
                                isDirectionWeak && "text-orange-500"
                            )}>
                            ({row?.original?.percent_a_in_b})
                        </div>
                    </div>
                );
            }
        },
        {
            header: () => (
                <Tooltip id={"Matches-B-in-A"}>
                    <TooltipTrigger>
                        <span className="ml-2.5">Matches (B in A)</span>
                    </TooltipTrigger>
                    <TooltipContent arrow={false} sideOffset={5}>
                        {"Percent of rows in B that have a matching key in A."}
                    </TooltipContent>
                </Tooltip>
            ),
            accessorKey: "matches_b_in_a",
            cell: ({ row }: { row: Row<EdgeStats> }) => {
                const isBothWeak =
                    row?.original?.weak_a_in_b && row?.original?.weak_b_in_a;
                const isDirectionWeak = row?.original?.weak_b_in_a;

                if (isBothWeak) {
                    return (
                        <Tooltip id={row?.original?.percent_b_in_a as string}>
                            <TooltipTrigger>
                                <span>
                                    <span className="text-title-primary">
                                        {
                                            row?.original
                                                ?.matches_b_in_a as string
                                        }
                                    </span>
                                    <br />
                                    <div className="text-orange-500">
                                        ({row?.original?.percent_b_in_a})
                                    </div>
                                </span>
                            </TooltipTrigger>
                            <TooltipContent
                                className="max-w-xs text-justify"
                                arrow={false}
                                side={"left"}
                                sideOffset={5}>
                                {`This edge has a match rate lower than ` +
                                    `our recommended percentage of ` +
                                    `${graphEdgeThreshold}%. Please make sure ` +
                                    `this is expected, as it could cause a ` +
                                    `degradation in model performance.`}
                            </TooltipContent>
                        </Tooltip>
                    );
                }

                return (
                    <div className="text-title-secondary">
                        <span className="text-title-primary">
                            {row?.original?.matches_b_in_a as string}
                        </span>
                        <br />
                        <div
                            className={cn(
                                isDirectionWeak && "text-orange-500"
                            )}>
                            ({row?.original?.percent_b_in_a})
                        </div>
                    </div>
                );
            }
        }
    ];
};
