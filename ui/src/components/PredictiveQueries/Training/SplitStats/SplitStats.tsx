// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
    DataTable,
    EmptyTable,
    GridCol,
    GridRow,
    useWindowDimensions
} from "@kumo-ai/kumo-ui-catalog";
import { QueryType, SplitTrainTableStats } from "gen/v0/types";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";
import { SplitStatsColumns } from "components/PredictiveQueries/Training/SplitStats/SplitStatsColumns";
import { useRestructureStats } from "components/PredictiveQueries/Training/SplitStats/trainingDataStats";

import { TaskType } from "types/predictiveQueries";

import { jobStatus } from "utils/Constants";
import { isLinkPredictionType } from "utils/Helpers/typeGuards";

/**
 * Component to show Predictive Query training table split stats in transposed format *
 * @param queryStatus Status of the Query *
 * @param taskType Type of Query (BC, regression, etc) *
 * @param show show/hide this stats
 * @param splitStats Split statistics data
 * @param isTrainingEvaluation If true, only shows Prediction Window, Split Distribution, Total Rows
 * @returns ```<SplitStats />```
 */

export interface SplitStatsProps {
    queryStatus: jobStatus;
    taskType: TaskType;
    queryType: QueryType;
    show: boolean;
    splitStats: SplitTrainTableStats | null;
    isTrainingEvaluation?: boolean;
}

const SplitStats = ({
    queryStatus,
    taskType,
    show,
    splitStats,
    isTrainingEvaluation = false
}: SplitStatsProps) => {
    const { tableData, allStats } = useRestructureStats(
        taskType,
        splitStats,
        isTrainingEvaluation
    );

    const { width } = useWindowDimensions();
    const columns = SplitStatsColumns({ windowWidth: width });

    return !show ? null : (
        <GridRow className="flex flex-col">
            <GridCol xs={12} sm={12} md={12}>
                {tableData?.length > 0 ? (
                    <DataTable
                        id="split_stats"
                        columns={columns}
                        disableRowSelectionHover={true}
                        data={tableData}
                        isLoading={
                            isLinkPredictionType(taskType) ||
                            (queryStatus !== jobStatus.Running &&
                                queryStatus !== jobStatus.Queued)
                                ? false
                                : allStats === null
                        }
                        pageCount={
                            tableData?.length > 0
                                ? Math.ceil(tableData?.length / 50)
                                : 1
                        }
                        defaultPageSize={50}
                        enableMultiSort={false}
                        enableGlobalFilter={false}
                        columnsVisibility={{
                            "node_degree_stats": false,
                            "target_distribution": false
                        }}
                    />
                ) : (
                    <EmptyTable
                        id={"no-split-stats"}
                        label={"No split stats found"}
                        showLoading={
                            isLinkPredictionType(taskType) ||
                            (queryStatus !== jobStatus.Running &&
                                queryStatus !== jobStatus.Queued)
                                ? false
                                : allStats === null
                        }
                    />
                )}
            </GridCol>
        </GridRow>
    );
};
export default WithErrorBoundary(SplitStats);
