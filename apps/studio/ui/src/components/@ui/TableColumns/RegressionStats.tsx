// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
    Button,
    ColumnDef,
    Info,
    Row,
    Tooltip,
    TooltipContent,
    TooltipTrigger
} from "@kumo-ai/kumo-ui-catalog";

import { RegressionTargetDistribution, TimeFrameStats } from "types/stats";

import { parseNumbers } from "utils/Helpers/numberHelpers";

import { CommonTrainingTableStatsColumns } from "./CommonTrainingTableStats";

export const RegressionStatsColumns = (
    isSplitStats: boolean = false,
    query_type: string,
    distributionAvailable: boolean = false
) => {
    const header: ColumnDef<TimeFrameStats>[] = CommonTrainingTableStatsColumns(
        query_type,
        isSplitStats,
        `Anchor Time`
    );

    header.push(
        {
            id: "num_entities",
            header: () => (
                <span className="flex flex-row">
                    {"Unique entities"}
                    <Tooltip id={"tooltip-reg-entities"}>
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
                <div>{parseNumbers(row?.original?.num_entities as number)}</div>
            )
        },
        {
            id: "target_distribution.target_mean",
            header: "Mean",
            size: 50,
            accessorKey: "target_distribution.target_mean",
            cell: ({ row }: { row: Row<TimeFrameStats> }) => (
                <div>
                    {parseNumbers(
                        (
                            row?.original
                                ?.target_distribution as RegressionTargetDistribution
                        )?.target_mean,
                        { decimals: 1 }
                    )}
                </div>
            )
        },
        {
            id: "target_distribution.target_mode",
            header: "Mode",
            size: 75,
            accessorKey: "target_distribution.target_mode",
            cell: ({ row }: { row: Row<TimeFrameStats> }) => (
                <div>
                    {parseNumbers(
                        (
                            row?.original
                                ?.target_distribution as RegressionTargetDistribution
                        )?.target_mode
                    ) +
                        " (" +
                        parseNumbers(
                            (
                                row?.original
                                    ?.target_distribution as RegressionTargetDistribution
                            )?.target_mode_percentage * 100,
                            { decimals: 1 }
                        ) +
                        "%)"}
                </div>
            )
        },
        {
            id: "target_distribution.target_min",
            header: "Min",
            size: 50,
            accessorKey: "target_distribution.target_min",
            cell: ({ row }: { row: Row<TimeFrameStats> }) => (
                <div>
                    {parseNumbers(
                        (
                            row?.original
                                ?.target_distribution as RegressionTargetDistribution
                        )?.target_min
                    )}
                </div>
            )
        },
        {
            id: "target_distribution.target_percentile_25",
            header: "25%",
            size: 50,
            accessorKey: "target_distribution.target_percentile_25",
            cell: ({ row }: { row: Row<TimeFrameStats> }) => (
                <div>
                    {parseNumbers(
                        (
                            row?.original
                                ?.target_distribution as RegressionTargetDistribution
                        )?.target_percentile_25
                    )}
                </div>
            )
        },
        {
            id: "target_distribution.target_median",
            header: "Median",
            size: 50,
            accessorKey: "target_distribution.target_median",
            cell: ({ row }: { row: Row<TimeFrameStats> }) => (
                <div>
                    {parseNumbers(
                        (
                            row?.original
                                ?.target_distribution as RegressionTargetDistribution
                        )?.target_median
                    )}
                </div>
            )
        },
        {
            id: "target_distribution.target_percentile_75",
            header: "75%",
            size: 50,
            accessorKey: "target_distribution.target_percentile_75",
            cell: ({ row }: { row: Row<TimeFrameStats> }) => (
                <div>
                    {parseNumbers(
                        (
                            row?.original
                                ?.target_distribution as RegressionTargetDistribution
                        )?.target_percentile_75
                    )}
                </div>
            )
        },
        {
            id: "target_distribution.target_max",
            header: "Max",
            size: 50,
            accessorKey: "target_distribution.target_max",
            cell: ({ row }: { row: Row<TimeFrameStats> }) => (
                <div>
                    {parseNumbers(
                        (
                            row?.original
                                ?.target_distribution as RegressionTargetDistribution
                        )?.target_max
                    )}
                </div>
            )
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
