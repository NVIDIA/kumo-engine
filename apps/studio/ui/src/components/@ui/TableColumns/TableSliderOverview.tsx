// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
    Row,
    Skeleton,
    Symbol,
    Tooltip,
    TooltipContent,
    TooltipTrigger
} from "@kumo-ai/kumo-ui-catalog";
import type { ColStatsResponse } from "gen/v0/types";

import { ColumnArgs, TableColData, TableDetails } from "types/tables";

import { SampleData } from "../Chart/ColumnStatsChart";

export const TableSliderOverviewColumns = (
    table: TableDetails,
    fetchingSamples: boolean,
    columnStats: ColStatsResponse | null
) => {
    return [
        {
            id: "name",
            header: "Column Name",
            accessorKey: "name",
            size: 25
        },
        {
            id: "dtype",
            header: "Data type",
            accessorKey: "dtype",
            size: 25
        },
        {
            id: "stype",
            header: "Semantic Type",
            accessorKey: "stype",
            size: 25
        },
        {
            id: "props",
            header: "Properties",
            accessorKey: "props",
            size: 25,
            cell: ({ row }: { row: Row<TableColData> }) => {
                const props =
                    table?.pkey === row?.original?.name
                        ? "Primary key"
                        : table?.time_col === row?.original?.name
                          ? "Time Column"
                          : "-";
                return props === "-" ? (
                    <Symbol type="emdash" />
                ) : (
                    <p className="mb-0">{props}</p>
                );
            }
        },
        {
            id: "sampleRows",
            header: "Preview",
            accessorKey: "sampleRows",
            size: 200,
            cell: ({ row }: { row: Row<TableColData> }) => {
                if (fetchingSamples)
                    return (
                        <div className="min-w-175 w-175">
                            <Skeleton className="h-4 w-40" />
                        </div>
                    );

                const name: string = row?.original?.name as string;
                const column: ColumnArgs = columnStats?.columns?.find(
                    (col) => col.name === name
                ) as ColumnArgs;
                const samples: Array<string> = column?.sample_rows
                    ? (column?.sample_rows as string[])
                    : [];
                //Replace nulls with N/A if any,
                samples?.forEach((value: string, i: number) => {
                    if (value == null || value == "") {
                        samples[i] = "N/A";
                    }
                });

                if (!samples) {
                    return <Symbol type="emdash" />;
                } else {
                    const images = samples?.filter((a) =>
                        /\.(jpeg|jpg|gif|png)$/.test(a as string)
                    );
                    if (images?.length > 0) {
                        return (
                            <>
                                {images?.slice(0, 4).map((image) => (
                                    <SampleData value={image} />
                                ))}
                                ...
                            </>
                        );
                    }
                    const textString = samples ? samples?.join(", ") : "";
                    return (
                        <>
                            {textString?.length <= 75 ? (
                                (textString ?? "")
                            ) : (
                                <Tooltip id={row?.id as string}>
                                    <TooltipTrigger>
                                        <span>
                                            {textString?.slice(0, 75) + "..."}
                                        </span>
                                    </TooltipTrigger>
                                    <TooltipContent
                                        arrow={false}
                                        className="max-w-100 text-justify"
                                        side={"top"}
                                        sideOffset={5}>
                                        {textString ?? ""}
                                    </TooltipContent>
                                </Tooltip>
                            )}
                        </>
                    );
                }
            }
        }
    ];
};
