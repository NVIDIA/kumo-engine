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

import { TimeFrameStats } from "types/stats";

import { timeZone } from "utils/Constants";
import { parseNumbers } from "utils/Helpers/numberHelpers";

import { CommonTrainingTableStatsColumns } from "./CommonTrainingTableStats";

export const PQStatsColumns = (
    showDistribution: boolean,
    isSplitStats = false,
    query_type: string
) => {
    const header: ColumnDef<TimeFrameStats>[] = CommonTrainingTableStatsColumns(
        query_type,
        isSplitStats,
        `Anchor Time (${timeZone})`
    );

    header.push({
        id: "num_entities",
        header: () => (
            <span className="flex flex-row">
                {"Unique entities"}
                <Tooltip id={"tooltip-stats-entities"}>
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
    });
    header.push({
        id: "expand",
        header: "Target Distribution",
        accessorKey: "target_distrib",
        size: 150,
        cell: ({ row }: { row: Row<TimeFrameStats> }) => (
            <div className="query-custom-row">
                <Tooltip id={"tooltip-more" + row.original.id}>
                    <TooltipTrigger>
                        <span>
                            <Button
                                id={"more" + row.original.id}
                                disabled={!showDistribution}
                                onClick={(
                                    e: React.MouseEvent<HTMLButtonElement>
                                ) => {
                                    e.stopPropagation();
                                    row.toggleExpanded();
                                }}
                                size="sm">
                                {row.getIsExpanded()
                                    ? "Close Details"
                                    : "See Details"}
                            </Button>
                        </span>
                    </TooltipTrigger>
                    <TooltipContent
                        className="max-w-100 text-justify"
                        arrow={false}
                        side={"top"}
                        sideOffset={5}>
                        {!showDistribution
                            ? `${isSplitStats ? "Split" : "Timeframe"} stats are unavailable for Link Prediction problems`
                            : ""}
                    </TooltipContent>
                </Tooltip>
            </div>
        )
    });

    return header;
};
