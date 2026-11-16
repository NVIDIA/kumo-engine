// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import StatisticsTable from "components/@ui/TablePlus/StatisticsTable";

import { useGetBpColumnStatsAndSampleRows } from "hooks/jobs/prediction/prediction";

import { UnlinkedBatchPredictionJobData } from "types/batchPredictions";

export interface BPOutputStatisticsProps {
    jobInfo: UnlinkedBatchPredictionJobData | null;
    show: boolean;
    isJobRunning: boolean;
    handleError: (err: Error) => void;
}

const BPOutputStatistics = ({
    jobInfo,
    show,
    isJobRunning,
    handleError
}: BPOutputStatisticsProps) => {
    const { columnStatsAndSampleRows } = useGetBpColumnStatsAndSampleRows(
        jobInfo?.id as string,
        handleError
    );
    return (
        <div className={show ? "block" : "hidden"}>
            {isJobRunning && jobInfo && (
                <>
                    <StatisticsTable
                        columnStats={columnStatsAndSampleRows?.columnStats}
                        loading={
                            columnStatsAndSampleRows?.loadingStats ?? false
                        }
                        gotError={columnStatsAndSampleRows?.gotError}
                        search={false}
                    />
                </>
            )}
        </div>
    );
};

export default BPOutputStatistics;
