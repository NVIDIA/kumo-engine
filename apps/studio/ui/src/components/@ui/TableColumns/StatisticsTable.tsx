// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
    CalendarAdd,
    ColumnDef,
    Key,
    Row,
    StopCircle,
    Symbol,
    Tooltip,
    TooltipContent,
    TooltipTrigger,
    WarningIcon
} from "@kumo-ai/kumo-ui-catalog";

import { ColumnArgs } from "types/tables";

import { parseNumbers } from "utils/Helpers/numberHelpers";
import { customSort, dynamicCustomSort } from "utils/Helpers/sortHelpers";
import { isValid } from "utils/Helpers/typeGuards";

export const StatisticsTableColumns = (
    statistics_available: boolean = false
): ColumnDef<ColumnArgs & { component?: React.ReactNode }>[] => {
    return [
        {
            id: "name",
            header: () => <span className="ml-2.5">Column</span>,
            accessorKey: "name",
            cell: ({
                row
            }: {
                row: Row<ColumnArgs & { component?: React.ReactNode }>;
            }) => {
                if (row?.original?.isPrimary)
                    return (
                        <div className="flex flex-row items-center gap-1 text-black ml-2.5">
                            <Key className="size-5.5 fill-pink-600" />{" "}
                            {row?.original?.name}
                        </div>
                    );
                else if (row?.original?.isTimeColumn)
                    return (
                        <div className="flex flex-row items-center gap-1 text-black ml-2.5">
                            <CalendarAdd className="size-5.5 fill-pink-600" />{" "}
                            <label className="ps-1">
                                {row?.original?.name}
                            </label>
                        </div>
                    );
                if (row?.original?.isEndTimeColumn)
                    return (
                        <div className="flex flex-row items-center gap-1 text-black ml-2.5">
                            <StopCircle className="size-5.5 fill-pink-600" />
                            <label className="ps-1">
                                {row?.original?.name}
                            </label>
                        </div>
                    );
                else
                    return (
                        <div className="text-black ml-2.5">
                            {row?.original?.name}
                        </div>
                    );
            }
        },
        {
            id: "data_type",
            header: () => (
                <Tooltip id={"Data-Type"}>
                    <TooltipTrigger>
                        <span>Data Type</span>
                    </TooltipTrigger>
                    <TooltipContent arrow={false} sideOffset={5}>
                        {
                            "The Data Type represents how the data is physically stored"
                        }
                    </TooltipContent>
                </Tooltip>
            ),
            accessorKey: "dtype",
            sortingFn: (a, b) => customSort(a, b, "dtype"),
            cell: ({ getValue }) => {
                const value = getValue() as string;
                return (
                    <div className="whitespace-nowrap">
                        {value === "Sequence" ? "Embedding" : value}
                    </div>
                );
            }
        },
        {
            id: "stype",
            header: () => (
                <Tooltip id={"Semantic-Type"}>
                    <TooltipTrigger>
                        <span>Semantic Type</span>
                    </TooltipTrigger>
                    <TooltipContent arrow={false} sideOffset={5}>
                        {
                            "The Semantic Type determines how data is processed and encoded"
                        }
                    </TooltipContent>
                </Tooltip>
            ),
            accessorKey: "stype",
            sortingFn: (a, b) => customSort(a, b, "stype"),
            cell: ({ getValue }) => {
                const value = getValue() as string;
                return (
                    <div className="whitespace-nowrap">
                        {value === "Sequence" ? "Embedding" : value}
                    </div>
                );
            }
        },
        {
            id: "statistics.counts.missing_percentage",
            header: () => (
                <Tooltip id={"Missing-Percentage"}>
                    <TooltipTrigger>
                        <span>Missing</span>
                    </TooltipTrigger>
                    <TooltipContent
                        arrow={false}
                        sideOffset={5}
                        className="w-150">
                        {
                            "Missing values (NULL or empty) in the column. These may be handled differently depending on the column type and usage."
                        }
                    </TooltipContent>
                </Tooltip>
            ),
            accessorKey: "statistics.counts.missing_percentage",
            sortingFn: (
                a: Row<ColumnArgs & { component?: React.ReactNode }>,
                b: Row<ColumnArgs & { component?: React.ReactNode }>
            ) =>
                dynamicCustomSort(a, b, "statistics.counts.missing_percentage"),
            cell: ({
                row
            }: {
                row: Row<ColumnArgs & { component?: React.ReactNode }>;
            }) => {
                const missingPercentage = Number(
                    row?.original?.statistics?.counts?.missing_percentage
                );

                return (
                    <div className="flex items-center justify-between">
                        {statistics_available &&
                        isValid(
                            row?.original?.statistics?.counts
                                ?.missing_percentage
                        ) ? (
                            <Tooltip id={`missing-${row?.original?.name}`}>
                                <TooltipTrigger>
                                    <span>
                                        <DataPercentageCell
                                            value={missingPercentage}
                                        />
                                    </span>
                                </TooltipTrigger>
                                <TooltipContent
                                    arrow={false}
                                    sideOffset={5}
                                    className="max-w-100">
                                    {missingPercentage >= 51
                                        ? `High missing rate (${parseNumbers(missingPercentage)}%): More than half of the values in this column are missing. This may impact model performance.`
                                        : missingPercentage >= 10
                                          ? `${parseNumbers(missingPercentage)}% of values in this column are missing (NULL or empty).`
                                          : `${parseNumbers(missingPercentage)}% of values in this column are missing.`}
                                </TooltipContent>
                            </Tooltip>
                        ) : (
                            <Symbol type="emdash" />
                        )}
                    </div>
                );
            }
        },
        {
            id: "statistics.counts.invalid_percentage",
            header: () => (
                <Tooltip id={"Invalid-Percentage"}>
                    <TooltipTrigger>
                        <span>Invalid</span>
                    </TooltipTrigger>
                    <TooltipContent
                        arrow={false}
                        sideOffset={5}
                        className="w-150">
                        {
                            "Invalid rows may be dropped due to: NULL values in time columns, dangling foreign keys (IDs not present in related tables), invalid timestamps, or data type mismatches. These rows will not be included in training."
                        }
                    </TooltipContent>
                </Tooltip>
            ),
            accessorKey: "statistics.counts.invalid_percentage",
            sortingFn: (
                a: Row<ColumnArgs & { component?: React.ReactNode }>,
                b: Row<ColumnArgs & { component?: React.ReactNode }>
            ) =>
                dynamicCustomSort(a, b, "statistics.counts.invalid_percentage"),
            cell: ({
                row
            }: {
                row: Row<ColumnArgs & { component?: React.ReactNode }>;
            }) => {
                const invalidPercentage = row?.original?.statistics?.counts
                    ?.invalid_percentage as number;

                return (
                    <div className="flex items-center justify-between">
                        {statistics_available ? (
                            isValid(invalidPercentage) ? (
                                <Tooltip id={`invalid-${row?.original?.name}`}>
                                    <TooltipTrigger>
                                        <span>
                                            <DataPercentageCell
                                                value={invalidPercentage}
                                            />
                                        </span>
                                    </TooltipTrigger>
                                    <TooltipContent
                                        arrow={false}
                                        sideOffset={5}
                                        className="max-w-150">
                                        {invalidPercentage >= 51
                                            ? `High invalid rate (${parseNumbers(invalidPercentage)}%): More than half of the values in this column are invalid. This may indicate issues with NULL values, dangling foreign keys, or data quality problems. These rows will be excluded from training.`
                                            : invalidPercentage >= 10
                                              ? `${parseNumbers(invalidPercentage)}% of values in this column are invalid and will be excluded from training. Common causes: NULL values in time columns, dangling foreign keys, or invalid timestamps.`
                                              : `${parseNumbers(invalidPercentage)}% of values in this column are invalid.`}
                                    </TooltipContent>
                                </Tooltip>
                            ) : (
                                <Symbol type="emdash" />
                            )
                        ) : (
                            <Symbol type="emdash" />
                        )}
                    </div>
                );
            }
        },
        {
            id: "statistics.counts.total",
            header: "Total Count",
            accessorKey: "statistics.counts.total",
            enableHiding: true,
            cell: ({
                row
            }: {
                row: Row<ColumnArgs & { component?: React.ReactNode }>;
            }) => (
                <div className="flex items-center justify-between">
                    {statistics_available ? (
                        isValid(row?.original?.statistics?.counts.total) ? (
                            <label>
                                {row?.original?.statistics?.counts.total}
                            </label>
                        ) : (
                            <Symbol type="emdash" />
                        )
                    ) : (
                        <Symbol type="emdash" />
                    )}
                </div>
            )
        }
    ];
};

function getColor(value: number) {
    return value < 10 ? "#059669" : value < 51 ? "#D97706" : "#DC2626";
}

// function getBgColor(isHighWarning: boolean, showWarning: boolean) {
//     return isHighWarning
//         ? "rgba(130, 8, 12, 0.1)"
//         : showWarning
//           ? "rgba(148, 146, 13, 0.1)"
//           : "transparent";
// }

const DataPercentageCell = ({ value }: { value: number }) => {
    const showWarning = value >= 10;
    // const isHighWarning = value >= 51;
    const color = getColor(value);
    // const bgColor = getBgColor(isHighWarning, showWarning);

    return (
        <div
            className="flex flex-row items-center gap-1 rounded-sm"
            style={{
                color: color,
                // backgroundColor: bgColor,
                padding: "2px 6px"
            }}>
            {showWarning && (
                <WarningIcon className="size-4" style={{ fill: color }} />
            )}
            {parseNumbers(value) + "%"}
        </div>
    );
};
