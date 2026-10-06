// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
    Button,
    CellContext,
    ColumnDef,
    Info,
    Row,
    Symbol,
    Tooltip,
    TooltipContent,
    TooltipTrigger
} from "@kumo-ai/kumo-ui-catalog";

import { TimeFrameStats } from "types/stats";

import { parseNumbers } from "utils/Helpers/numberHelpers";

import { CommonTrainingTableStatsColumns } from "./CommonTrainingTableStats";

export const BinaryClassificationStatsColumns = (
    isSplitStats: boolean = false,
    query_type: string,
    distributionAvailable: boolean = false
) => {
    const header: ColumnDef<TimeFrameStats, number>[] =
        CommonTrainingTableStatsColumns(
            query_type,
            isSplitStats,
            `Anchor Date`
        );

    header.push(
        {
            id: "num_entities",
            header: () => (
                <span className="flex flex-row">
                    {"Unique entities"}
                    <Tooltip id={"tooltip-bc-entities"}>
                        <TooltipTrigger className="overflow-visible">
                            {
                                <span className="ml-1 flex items-center">
                                    <Info className="size-3.5" />
                                </span>
                            }
                        </TooltipTrigger>
                        <TooltipContent
                            className="max-w-75 text-justify"
                            arrow={false}
                            side={"top"}
                            sideOffset={5}>
                            {
                                "Number of distinct entities that appear in this split"
                            }
                        </TooltipContent>
                    </Tooltip>
                </span>
            ),
            accessorKey: "num_entities",
            cell: ({ row }: { row: Row<TimeFrameStats> }) => (
                <div>{parseNumbers(row?.original?.num_entities)}</div>
            )
        },
        {
            id: "target_distribution.count.True",
            header: `Positive samples`,
            accessorKey: "target_distribution.count.True",
            cell: ({ row, getValue }: CellContext<TimeFrameStats, number>) => {
                const value = getValue();
                return (
                    <div>
                        {!value ? (
                            <Symbol type="emdash" />
                        ) : (
                            value +
                            " (" +
                            parseNumbers(
                                ((value as number) /
                                    ((isSplitStats
                                        ? row?.original?.num_train_rows
                                        : row?.original
                                              ?.num_entities) as number)) *
                                    100,
                                { decimals: 1 }
                            ) +
                            "%)"
                        )}
                    </div>
                );
            }
        },
        {
            id: "target_distribution.count.False",
            header: `Negative samples`,
            accessorKey: "target_distribution.count.False",
            cell: ({ row, getValue }: CellContext<TimeFrameStats, number>) => {
                const value = getValue();
                return (
                    <div>
                        {!value ? (
                            <Symbol type="emdash" />
                        ) : (
                            value +
                            " (" +
                            parseNumbers(
                                ((value as number) /
                                    ((isSplitStats
                                        ? row?.original?.num_train_rows
                                        : row?.original
                                              ?.num_entities) as number)) *
                                    100,
                                { decimals: 1 }
                            ) +
                            "%)"
                        )}
                    </div>
                );
            }
        }
    );
    if (distributionAvailable)
        header.push({
            id: "expand",
            header: "Target Distribution",
            accessorKey: "target_distrib",
            size: 150,
            cell: ({ row }: { row: Row<TimeFrameStats> }) => (
                <Button
                    id={"more" + row.original.id}
                    onClick={(e: React.MouseEvent<HTMLButtonElement>) => {
                        e.stopPropagation();
                        row.toggleExpanded();
                    }}
                    size="sm">
                    {row.getIsExpanded() ? "Close Details" : "See Details"}
                </Button>
            )
        });
    return header;
};
