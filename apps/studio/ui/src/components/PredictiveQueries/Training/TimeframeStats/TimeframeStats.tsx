// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React from "react";

import { DataTable, GridCol, GridRow } from "@kumo-ai/kumo-ui-catalog";

import { TrainingTimeFrameStatsData } from "types/stats";

import { jobStatus } from "utils/Constants";

/**
 * Component to show Predictive Query training table timeframe stats
 * in table format *
 * @param queryStatus Status of the Query *
 * @param data state information from parent component which hold stats data
 * @param show show/hide this stats
 * @returns ```<TimeframeStats />```
 */

export interface TimeframeStatsProps {
    queryStatus: string;
    data: TrainingTimeFrameStatsData;
    show: boolean;
    expandRow?: boolean;
}

const TimeframeStats = ({
    queryStatus,
    data,
    show,
    expandRow
}: TimeframeStatsProps) => {
    return !show ? null : (
        <GridRow className="flex flex-col">
            <GridCol>
                <DataTable
                    id="pquery_table_stats"
                    columns={data?.tableData?.columns}
                    rowCanExpand={expandRow}
                    data={data?.tableData?.data}
                    isLoading={
                        queryStatus !== jobStatus.Running &&
                        queryStatus !== jobStatus.Queued &&
                        data?.receivedResponse
                            ? false
                            : data?.allStats === null
                    }
                    pageCount={
                        data?.tableData?.data?.length > 0
                            ? Math.ceil(data?.tableData?.data?.length / 50)
                            : 1
                    }
                    defaultPageSize={50}
                    enableMultiSort={true}
                    enableGlobalFilter={true}
                    columnsVisibility={{
                        "node_degree_stats": false,
                        "target_distribution": false
                    }}
                />
            </GridCol>
        </GridRow>
    );
};

export default TimeframeStats;
