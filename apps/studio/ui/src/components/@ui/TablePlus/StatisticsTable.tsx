// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React, { useCallback, useMemo } from "react";

import {
    ChevronDown,
    ChevronRight,
    ColumnDef,
    DataTable,
    Description,
    Row
} from "@kumo-ai/kumo-ui-catalog";
import type { ColumnStats } from "gen/v0/types";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";

import { ExtendedColStatsResponse } from "types/snapshots";
import {
    DistributionCategory,
    DistributionCounts,
    DistributionDate,
    DistributionPoints,
    Statistics,
    SummaryStats
} from "types/stats";
import { ColumnArgs } from "types/tables";

import { isValid } from "utils/Helpers/typeGuards";

import ColumnStatsChart from "../Chart/ColumnStatsChart";
import { StatisticsTableColumns } from "../TableColumns/StatisticsTable";

export interface StatisticsTableProps {
    columnStats?: ExtendedColStatsResponse | null;
    loading: boolean;
    search?: boolean;
    gotError?: boolean;
    hiddenColumns?: string[];
    primaryKey?: string | null;
    timeCol?: string | null;
    endTimeCol?: string | null;
    isSnapShot?: boolean;
}

const StatisticsTable = ({
    columnStats,
    loading,
    gotError,
    search = true,
    hiddenColumns = [],
    primaryKey = null,
    timeCol = null,
    endTimeCol = null,
    isSnapShot = false
}: StatisticsTableProps) => {
    const columns = useMemo(
        () => StatisticsTableColumns(columnStats?.statistics_available),
        [columnStats?.statistics_available]
    );

    const mapRestConfigToStatistics = useCallback(
        (stats?: ColumnStats | null): Statistics | undefined => {
            if (!stats) return undefined;
            // We are transforming RestConfigTableColumnStats (raw API shape with unknown[][])
            // into a Statistics object for safe UI.
            return {
                counts: stats.counts as DistributionCounts,
                summary_stats: stats.summary_stats as SummaryStats[],
                distribution_points: stats.distribution_points as
                    | DistributionPoints[]
                    | null,
                distribution_category: stats.distribution_category as
                    | DistributionCategory[]
                    | null,
                distribution_date: stats.distribution_date as
                    | DistributionDate[]
                    | null,
                bp_stats: stats.bp_stats as {
                    [key: string]: { [key: string]: Array<number> | number };
                },
                approx: stats.approx
            };
        },
        []
    );

    const stats = useMemo(() => {
        const columns: (ColumnArgs & { component?: React.ReactNode })[] = (
            columnStats?.columns ?? []
        ).map((col) => ({
            ...col,
            statistics: mapRestConfigToStatistics(col.statistics)
        }));

        //Creating sample rows values if it is available but not used
        if (columns && columns.length > 0) {
            columns?.forEach((element) => {
                if (element?.sample_rows) {
                    let samples = element.sample_rows as Array<string | null>;
                    let nonEmpty: string | undefined | null = samples?.find(
                        (d) => isValid(d)
                    );
                    if (nonEmpty === undefined) {
                        //extra secure
                        nonEmpty = null;
                    }

                    samples = samples?.map((val) =>
                        isValid(val) ? val : "N/A"
                    );

                    element.sample_rows = (
                        nonEmpty === null ? nonEmpty : samples
                    ) as string | string[];
                    element.nonEmpty = nonEmpty;
                }
                element.isPrimary = element.name === primaryKey;
                element.isTimeColumn = element.name === timeCol;
                element.isEndTimeColumn = element.name === endTimeCol;
                element.component = (
                    <ColumnStatsChart data={element} isSnapShot={isSnapShot} />
                );
            });
        }
        return columns;
    }, [
        columnStats?.columns,
        primaryKey,
        timeCol,
        endTimeCol,
        isSnapShot,
        mapRestConfigToStatistics
    ]);
    const approximateStatsMessage =
        columnStats?.approx === true
            ? `Statistics are approximate${
                  columnStats.rows_used > 0
                      ? ` and based on ${columnStats.rows_used.toLocaleString()} sampled rows`
                      : ""
              }.`
            : null;

    return (
        <>
            {gotError ? (
                <>
                    <Description
                        variant="secondary"
                        size="sm"
                        className="mb-2 leading-5"
                        id={"col_stats_approx_desc"}>
                        {"Statistics for this table are not available. To fix this, " +
                            "try running a new set of predictions. If this doesn't work, " +
                            "you may contact your POC at Kumo to investigate further."}
                    </Description>
                </>
            ) : null}
            {approximateStatsMessage ? (
                <Description
                    variant="secondary"
                    size="sm"
                    className="mb-2 leading-5 text-title-secondary"
                    id={"col_stats_sampled_desc"}>
                    {approximateStatsMessage}
                </Description>
            ) : null}
            <DataTable
                id="column-stats-table"
                columns={
                    [
                        ...columns,
                        {
                            id: "expand",
                            header: "",
                            size: 25,
                            cell: ({
                                row
                            }: {
                                row: Row<
                                    ColumnArgs & { component?: React.ReactNode }
                                >;
                            }) => {
                                return (
                                    <span>
                                        {row?.getIsExpanded() ? (
                                            <ChevronDown className="size-5.5" />
                                        ) : (
                                            <ChevronRight className="size-5.5" />
                                        )}
                                    </span>
                                );
                            }
                        }
                    ] as ColumnDef<
                        ColumnArgs & {
                            component?: React.ReactNode;
                        },
                        unknown
                    >[]
                }
                data={stats}
                isLoading={loading}
                pageCount={
                    stats?.length > 0 ? Math.ceil(stats?.length / 50) : 1
                }
                defaultPageSize={50}
                enableMultiSort={true}
                enableGlobalFilter={search}
                rowCanExpand={true}
                columnsVisibility={
                    hiddenColumns?.length > 0
                        ? Object.fromEntries(
                              hiddenColumns.flatMap((l) => [[l, false]])
                          )
                        : {}
                }
            />
        </>
    );
};

export default WithErrorBoundary(StatisticsTable);
