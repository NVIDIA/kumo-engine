// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import Spacer from "react-spacer";

import {
    ColumnDef,
    Info,
    Row,
    Symbol,
    Tooltip,
    TooltipContent,
    TooltipTrigger
} from "@kumo-ai/kumo-ui-catalog";

import { QueryType } from "types/predictiveQueries";
import { TimeFrameStats } from "types/stats";

import {
    anchorDateMessage,
    predictionWindowNonSplitPopoverMessage,
    predictionWindowSplitPopoverMessage
} from "utils/Constants";
import { parseNumbers } from "utils/Helpers/numberHelpers";
import { customSort } from "utils/Helpers/sortHelpers";

export function CommonTrainingTableStatsColumns(
    query_type: string,
    isSplitStats: boolean,
    anchorLabel: string
) {
    const isTemporal = query_type === QueryType.TEMPORAL;
    const header: ColumnDef<TimeFrameStats>[] = [];
    try {
        if (isSplitStats) {
            header.push({
                id: "data_split",
                header: "Split",
                accessorKey: "data_split",
                cell: ({ row }: { row: Row<TimeFrameStats> }) => (
                    <div>{row?.original?.data_split as string}</div>
                )
            });
        }

        if (!isSplitStats) {
            header.push({
                id: "anchor_date",
                header: () => (
                    <span className="flex flex-row items-center">
                        {anchorLabel}
                        <Tooltip id={"tooltip-anchor"}>
                            <TooltipTrigger>
                                {
                                    <span className="ml-1 flex items-center">
                                        <Info className="size-3.5" />
                                    </span>
                                }
                            </TooltipTrigger>
                            <TooltipContent
                                className="max-w-100 text-justify"
                                arrow={false}
                                side={"top"}
                                sideOffset={5}>
                                {anchorDateMessage}
                            </TooltipContent>
                        </Tooltip>
                    </span>
                ),
                accessorKey: "anchor_date",
                cell: ({ row }: { row: Row<TimeFrameStats> }) => (
                    <div>{row?.original?.anchor_date}</div>
                ),
                sortingFn: (a: Row<TimeFrameStats>, b: Row<TimeFrameStats>) =>
                    customSort(a, b, "anchor_date_toSort")
            });
        }

        if (isTemporal) {
            const prediction_window_popover_message = isSplitStats
                ? predictionWindowNonSplitPopoverMessage
                : predictionWindowSplitPopoverMessage;
            header.push({
                id: "min_date",
                header: () => (
                    <span className="flex flex-row">
                        Prediction window
                        <Tooltip id={"tooltip-Horizon"}>
                            <TooltipTrigger className="overflow-visible">
                                {
                                    <span className="ml-1 flex items-center">
                                        <Info className="size-3.5" />
                                    </span>
                                }
                            </TooltipTrigger>
                            <TooltipContent
                                className="max-w-100 text-justify"
                                arrow={false}
                                side={"top"}
                                sideOffset={5}>
                                {prediction_window_popover_message}
                            </TooltipContent>
                        </Tooltip>
                    </span>
                ),
                accessorKey: "min_date",
                cell: ({ row }: { row: Row<TimeFrameStats> }) => (
                    <div className="flex items-center">
                        {row?.original?.min_date}
                        <Spacer width={10} />
                        {" - "}
                        <Spacer width={10} />
                        {row?.original?.max_date}
                    </div>
                ),
                sortingFn: (a: Row<TimeFrameStats>, b: Row<TimeFrameStats>) =>
                    customSort(a, b, "min_date_toSort")
            });
            if (isSplitStats) {
                header.push({
                    id: "num_train_rows",
                    accessorKey: "num_train_rows",
                    header: () => (
                        <span className="flex flex-row">
                            Total rows
                            <Tooltip id={"tooltip-total-rows"}>
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
                                        "Total number of data points (samples) used in this split"
                                    }
                                </TooltipContent>
                            </Tooltip>
                        </span>
                    ),
                    cell: ({ row }: { row: Row<TimeFrameStats> }) => {
                        if (row?.original?.num_train_rows)
                            return (
                                <div>
                                    {parseNumbers(
                                        row?.original?.num_train_rows as number
                                    )}
                                </div>
                            );
                        else return <Symbol type="emdash" />;
                    }
                });
            }
        }
    } catch (error) {
        console.warn("createCommonHeaders error", error);
    }

    return header;
}
