// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
    Card,
    CardContent,
    ColumnDef,
    DataTable
} from "@kumo-ai/kumo-ui-catalog";
import { TrainingJobResults } from "gen/v0/types";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";
import { useEvalMetricsAsTableStructure } from "components/PredictiveQueries/Evaluation/Metrics/evalMetricsTableStructure";

import { PredictiveQueryInfo } from "types/predictiveQueries";

/**
 * UI component for EvalMetricsTable Result Component with
 * Kumo Model Metrics*
 */

export interface EvalMetricsProps {
    queryInfo: PredictiveQueryInfo;
    results?: TrainingJobResults | undefined;
    tuneMetric: string;
    downloadClicked?: boolean;
    onDownloadClick?: () => void;
}

const EvalMetricsTable = ({
    queryInfo,
    results,
    tuneMetric,
    downloadClicked,
    onDownloadClick
}: EvalMetricsProps) => {
    const { rows, headers } = useEvalMetricsAsTableStructure({
        trainingResults: results,
        tuneMetric,
        taskType: queryInfo?.taskType,
        downloadClicked,
        onDownloadClick
    });

    /** UI Rendering */
    return (
        <Card className="rounded-none py-0 border-0">
            <CardContent className="px-0">
                <DataTable
                    id={"evaluation_metrics"}
                    disableRowSelectionHover={true}
                    columns={
                        headers as ColumnDef<
                            { [key: string]: unknown },
                            unknown
                        >[]
                    }
                    data={rows}
                    defaultPageSize={50}
                    enableGlobalFilter={false}
                />
            </CardContent>
        </Card>
    );
};

export default WithErrorBoundary(EvalMetricsTable);
