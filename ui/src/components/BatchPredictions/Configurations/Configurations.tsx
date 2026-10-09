// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useMemo } from "react";

import { DataTable, PropertiesOverviewCard } from "@kumo-ai/kumo-ui-catalog";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";
import { PredictionOutputTableColumns } from "components/@ui/TableColumns/PredictionOutputTable";
import { usePredictionJobSettingProperties } from "components/Overview/JobProperties";

import { useBatchPredictionDownloadBp } from "hooks/jobs/prediction/prediction";

import { UnlinkedBatchPredictionJobData } from "types/batchPredictions";
import { ColumnArgs } from "types/tables";
import { HandleErrorFunc } from "types/validation";

import { isObjectStoreOutputStorage } from "utils/objectStoreTablePath";

export interface ConfigurationsProps {
    anchorTime?: string | null;
    predictionInfo?: UnlinkedBatchPredictionJobData | null;
    handleError: HandleErrorFunc;
}

const Configurations = ({
    anchorTime,
    predictionInfo,
    handleError
}: ConfigurationsProps) => {
    const data = useMemo(() => {
        const configs: ColumnArgs[] = [];
        const outputArtifactStats =
            predictionInfo?.execution?.results?.output_artifact_stats;
        if (predictionInfo?.outputs || outputArtifactStats) {
            const maxOutput = Math.max(
                predictionInfo?.outputs?.length || 0,
                outputArtifactStats?.length || 0
            );
            for (let i = 0; i < maxOutput; i++) {
                const output = predictionInfo?.outputs?.[i];
                const resultOutput = outputArtifactStats?.[i];
                configs.push({
                    artifact_type:
                        output?.artifact_type ?? resultOutput?.artifact_type,
                    num_cols: resultOutput?.num_cols,
                    num_rows: resultOutput?.num_rows,
                    size_bytes: resultOutput?.size_bytes,
                    output_path:
                        (output &&
                        isObjectStoreOutputStorage(output.storage_type) &&
                        "file_path" in output
                            ? output.file_path
                            : undefined) ?? resultOutput?.output_path,
                    max_categories:
                        output?.artifact_type?.toLocaleUpperCase() ===
                            "PREDICTIONS" ||
                        resultOutput?.artifact_type?.toLocaleUpperCase() ===
                            "PREDICTIONS"
                            ? predictionInfo?.classes
                            : null,
                    table_name:
                        (output as { table_name?: string })?.table_name ?? ""
                } as unknown as ColumnArgs);
            }
        }
        return configs;
    }, [predictionInfo]);

    const { s3PresignedUrl, isError } = useBatchPredictionDownloadBp(
        predictionInfo?.id as string,
        handleError,
        predictionInfo?.execution?.status,
        predictionInfo?.downloadSupported === true
    );
    const properties = usePredictionJobSettingProperties(
        predictionInfo ?? null,
        anchorTime,
        s3PresignedUrl,
        isError,
        predictionInfo?.downloadSupported === true
    );

    const columns = useMemo(() => {
        return PredictionOutputTableColumns(
            predictionInfo?.outputs?.[0]?.storage_type
        );
    }, [predictionInfo]);

    const hidenColumns = useMemo(() => {
        if (data.length > 0) {
            const keys = [
                "artifact_type",
                "num_cols",
                "num_rows",
                "size_bytes",
                "output_path",
                "max_categories",
                "table_name"
            ];
            const tmp: { [key: string]: boolean } = {};
            keys.forEach((key) => {
                tmp[key] = data.some(
                    (d) =>
                        d[key as keyof ColumnArgs] !== "-" &&
                        d[key as keyof ColumnArgs] !== "" &&
                        d[key as keyof ColumnArgs] !== null &&
                        d[key as keyof ColumnArgs] !== undefined
                );
            });
            return tmp;
        }
        return {};
    }, [data]);

    return (
        <>
            <PropertiesOverviewCard
                properties={properties}
                hideTitle
                cardClassName={"p-0 pb-6"}
                titleSize="md"
            />
            {data.length > 0 && (
                <DataTable
                    id="output-settings"
                    disableRowSelectionHover={true}
                    columns={columns}
                    data={data || []}
                    isLoading={false}
                    defaultPageSize={50}
                    enableGlobalFilter={false}
                    manualFiltering={false}
                    columnsVisibility={hidenColumns}
                />
            )}
        </>
    );
};

export default WithErrorBoundary(Configurations);
