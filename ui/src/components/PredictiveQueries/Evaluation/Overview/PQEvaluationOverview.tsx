// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useMemo, useState } from "react";

import {
    Alert,
    cn,
    CopyToClipBoard,
    GridCol,
    GridRow,
    MetricCard,
    useWindowDimensions
} from "@kumo-ai/kumo-ui-catalog";
import { SplitTableStats, SplitTrainTableStats } from "gen/v0/types";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";
import WarningErrorCTA from "components/@ui/Warning/WarningErrorCTA";
import BinaryClassification from "components/PredictiveQueries/Evaluation/BinaryClassification/BinaryClassification";
import Histogram from "components/PredictiveQueries/Evaluation/Histogram/Histogram";
import RegressionHeatMap from "components/PredictiveQueries/Evaluation/RegressionHeatMap/RegressionHeatMap";
import SplitStats from "components/PredictiveQueries/Training/SplitStats/SplitStats";

import { useDownloadHoldout } from "hooks/jobs/training/downloadHoldout";
import useFeatureFlags from "hooks/utils/featureFlags";

import { ctaLevels } from "types/constants";
import {
    DataSplit,
    PredictiveQueryInfo,
    QueryType,
    TaskType,
    TimePeriod,
    TimePeriods,
    TrainingJobResults
} from "types/predictiveQueries";

import {
    dataSplitKeys,
    jobStatus,
    SUPPORT_CONTACT_MESSAGE
} from "utils/Constants";
import { parseNumbers } from "utils/Helpers/numberHelpers";
import { getValues } from "utils/Helpers/objectHelpers";
import { isValid } from "utils/Helpers/typeGuards";
import { parsePresignedURLasDatabricks } from "utils/Helpers/urlHelpers";
import { getTuneMetric } from "utils/predictiveQueries";

import EvalMetricsTable from "../Metrics/EvalMetricsTable";

type TestMetric = { name?: string; val?: number | null };

const findMetricValue = (
    metrics: TestMetric[] | null | undefined,
    metricKey: string,
    decimals: number,
    caseInsensitive = false
): string | null => {
    if (!metrics?.length || !metricKey) return null;
    const metric = metrics.find((e) =>
        caseInsensitive
            ? e?.name?.toLowerCase() === metricKey.toLowerCase()
            : e.name === metricKey
    );
    return metric
        ? `${parseNumbers(metric.val as number, { decimals })}`
        : null;
};

/**
 * Evaluation Output Component
 *
 */
export interface PQEvaluationOverviewProps {
    queryInfo: PredictiveQueryInfo;
    show: boolean;
    handleError: (err: Error, title?: string, isWarning?: boolean) => void;
}

const PQEvaluationOverview = ({
    queryInfo,
    show,
    handleError
}: PQEvaluationOverviewProps) => {
    const featureFlags = useFeatureFlags();
    const { width } = useWindowDimensions();
    const [dataSplitOptions, setDataSplitOptions] = useState<Array<DataSplit>>(
        []
    );
    const [trainTableSplitStats, setTrainTableSplitStats] =
        useState<SplitTrainTableStats | null>(null);
    const [downloadClicked, setDownloadClicked] = useState<boolean>(false);
    const [evaluationSplit, setEvaluationSplit] = useState<string>(
        dataSplitKeys.HoldOut
    );
    const [downloadError, setDownloadError] = useState<string>("");

    const [fileReadCommand, setFileReadCommand] = useState<string>("N/A");
    const [presignedUrlFetchError, setPresignedUrlFetchError] =
        useState<boolean>(false);

    const id = queryInfo?.trainingJobId
        ? queryInfo.trainingJobId
        : queryInfo.id;
    const isTrainingJob = !!queryInfo?.trainingJobId;

    const {
        presignedUrl,
        isError: isDownloadHoldoutError,
        refetch: refetchDownloadHoldout,
        error: downloadHoldoutError
    } = useDownloadHoldout({
        id,
        isTrainingJob,
        enabled:
            dataSplitOptions[0]?.value === dataSplitKeys.HoldOut &&
            downloadClicked
    });

    useEffect(() => {
        if (!isValid(presignedUrl)) {
            if (isDownloadHoldoutError) {
                console.error(
                    "Backend did not return a valid URL",
                    presignedUrl
                );
                setPresignedUrlFetchError(true);
            }
            return;
        }

        const parser = featureFlags.IS_DATABRICKS_DEPLOY
            ? parsePresignedURLasDatabricks
            : null;

        if (parser) {
            const parsedUrl = parser(presignedUrl);
            if (parsedUrl) {
                setFileReadCommand(parsedUrl);
            } else {
                setPresignedUrlFetchError(true);
            }
        }
    }, [
        featureFlags.IS_DATABRICKS_DEPLOY,
        isDownloadHoldoutError,
        presignedUrl
    ]);

    const parseMetricsDate = useCallback(
        (timestamp: {
            start: { toString: () => string };
            end: { toString: () => string };
        }) => {
            return (
                timestamp?.start?.toString().split("T")[0] +
                " - " +
                timestamp?.end?.toString().split("T")[0]
            );
        },
        []
    );

    const parseDataSplits = useCallback(
        (data: TrainingJobResults, isRefit = false) => {
            const dataSplitOptions: DataSplit[] = [];
            const splitKeys = [
                { "key": "train_table_stats", value_key: "train" },
                { "key": "val_table_stats", value_key: "val" },
                { "key": "test_table_stats", value_key: "test" }
            ];
            const trainTableSplitStats: SplitTrainTableStats = {
                train_table_stats: {} as SplitTableStats
            };
            type SplitKeys =
                | "train_table_stats"
                | "val_table_stats"
                | "test_table_stats";

            splitKeys.forEach((split) => {
                const key: SplitKeys = split.key as SplitKeys;
                const periodKey = split.value_key as keyof TimePeriods;
                const timePeriod = data.time_periods?.[periodKey];
                const maxDate = (timePeriod?.end as string) ?? "";
                const minDate = (timePeriod?.start as string) ?? "";
                const numTrainRows = data.split_sizes?.[periodKey] ?? 0;

                trainTableSplitStats[key] = {
                    max_date: maxDate,
                    min_date: minDate,
                    num_entities: 0,
                    num_targets: 0,
                    num_train_rows: numTrainRows,
                    target_distribution: {
                        count: { test: 0 },
                        percentage: { test: 0 }
                    },
                    anchor_date: ""
                };
            });

            const dataSplits = [
                {
                    key: dataSplitKeys.HoldOut,
                    value: "Holdout Data" + (isRefit ? " (Refitted)" : "")
                },
                { key: dataSplitKeys.Validation, value: "Validation Data" },
                { key: dataSplitKeys.Training, value: "Training Data" }
            ];
            let splitSizeSum = 0;

            if (data?.split_sizes?.["test"] == null) {
                setEvaluationSplit(dataSplitKeys.Validation);
            }

            if (data?.split_sizes) {
                getValues(data.split_sizes)?.forEach((size) => {
                    if (size) {
                        splitSizeSum += parseInt(size);
                    }
                });
            }

            dataSplits.forEach((split) => {
                let prediction_horizon_repr = null;
                let allData = "";
                const timePeriodKey = split.key as keyof TimePeriods;
                const timestamp = data.time_periods?.[timePeriodKey];
                if (data?.prediction_horizons) {
                    const prediction_horizon =
                        data.prediction_horizons[timePeriodKey];
                    if (
                        data.time_periods &&
                        timestamp &&
                        !timestamp?.is_pseudo
                    ) {
                        prediction_horizon_repr = parseMetricsDate(
                            prediction_horizon as TimePeriod
                        );
                    }
                }

                let percentage: string | number = 0;
                if (
                    data?.split_sizes &&
                    data?.split_sizes[timePeriodKey] &&
                    splitSizeSum > 0
                ) {
                    const val =
                        (data.split_sizes[timePeriodKey] / splitSizeSum) * 100;

                    percentage = val.toFixed(2);
                }
                allData = ` (${percentage.toString()}% of samples)`;
                dataSplitOptions.push({
                    label: split?.value + allData,
                    value: split?.key,
                    samples: data?.split_sizes?.[timePeriodKey] ?? "N/A",
                    prediction_horizon: prediction_horizon_repr ?? "N/A"
                });
            });
            return { dataSplitOptions, trainTableSplitStats };
        },
        [parseMetricsDate]
    );
    const onDownloadClick = useCallback(() => {
        if (!downloadClicked) {
            setDownloadClicked(true);
        }
    }, [downloadClicked]);

    // Handle download when presignedUrl is available
    useEffect(() => {
        if (downloadClicked && presignedUrl) {
            window.location.href = presignedUrl;
            setDownloadClicked(false);
        }
    }, [downloadClicked, presignedUrl]);

    // Handle download error
    useEffect(() => {
        if (isDownloadHoldoutError) {
            setDownloadClicked(false);
            if (downloadHoldoutError) {
                setDownloadError(downloadHoldoutError);
            }
        }
    }, [downloadClicked, downloadHoldoutError, isDownloadHoldoutError]);

    const updateJobInfo = useCallback(() => {
        if (queryInfo?.trainingStatus === jobStatus.Done) {
            const results = queryInfo?.results;
            const isRefit = queryInfo?.advancedOptionsYaml
                ?.toLowerCase()
                ?.includes("refit_trainval: true");
            const { dataSplitOptions, trainTableSplitStats } = parseDataSplits(
                results as TrainingJobResults,
                isRefit
            );
            const selectedSplit = dataSplitOptions[0] ?? null;
            if (
                selectedSplit.value === dataSplitKeys.HoldOut &&
                featureFlags.IS_DATABRICKS_DEPLOY
            ) {
                refetchDownloadHoldout();
            }
            setDataSplitOptions(dataSplitOptions);
            setTrainTableSplitStats(trainTableSplitStats);
        }
    }, [
        queryInfo?.trainingStatus,
        queryInfo?.results,
        queryInfo?.advancedOptionsYaml,
        parseDataSplits,
        featureFlags.IS_DATABRICKS_DEPLOY,
        refetchDownloadHoldout
    ]);

    useEffect(() => {
        updateJobInfo();
    }, [updateJobInfo]);

    const tuneMetric = useMemo(() => {
        const metricKey: string = getTuneMetric(queryInfo.modelPlannerYaml);
        const metricVal = findMetricValue(
            queryInfo.results?.test_metrics,
            metricKey,
            3
        );
        if (metricKey && metricVal) {
            return `${metricKey.toUpperCase()} ${metricVal}`;
        }
        return `-`;
    }, [queryInfo?.modelPlannerYaml, queryInfo?.results?.test_metrics]);

    const localityMetric = useMemo(() => {
        return findMetricValue(
            queryInfo.results?.test_metrics,
            "locality_score",
            2,
            true
        );
    }, [queryInfo?.results?.test_metrics]);

    const showFileReadCommand =
        featureFlags.IS_DATABRICKS_DEPLOY &&
        ((!presignedUrlFetchError && fileReadCommand !== "N/A") ||
            presignedUrlFetchError);

    return (
        <div
            data-testid="pq-eval-overview-container"
            className={cn(
                "flex flex-col gap-6 font-inter",
                show ? "" : "hidden"
            )}>
            {showFileReadCommand && (
                <Alert
                    variant="default"
                    enableClose={false}
                    className="mb-6 flex [&>div]:w-full break-words w-full">
                    {presignedUrlFetchError &&
                        "Holdout set location can't be displayed. " +
                            SUPPORT_CONTACT_MESSAGE +
                            " for more " +
                            "information if you want to access the " +
                            "holdout set."}
                    {!presignedUrlFetchError && (
                        <GridRow className="flex justify-between">
                            <GridCol>
                                <p className="break-words whitespace-pre-line">
                                    {fileReadCommand}
                                </p>
                            </GridCol>
                            <GridCol>
                                <CopyToClipBoard
                                    copiedText={fileReadCommand}
                                    iconSize={16}
                                    tooltipText="Copied"
                                    className="align-middle ml-1"
                                />
                            </GridCol>
                        </GridRow>
                    )}
                </Alert>
            )}
            {downloadError && (
                <WarningErrorCTA
                    id={"error-download"}
                    ctaLevel={ctaLevels.Critical}
                    title="Download failed!"
                    data={[
                        {
                            title: "",
                            description: downloadError,
                            is_error: true,
                            is_dismissible: false
                        }
                    ]}
                    handleClose={() => setDownloadError("")}
                />
            )}
            <div className="flex gap-4">
                {localityMetric && (
                    <MetricCard
                        id={"tune-metric-card"}
                        metricList={[
                            {
                                name: "Locality_score",
                                val: localityMetric
                            }
                        ]}
                    />
                )}
                <MetricCard
                    id={"tune-metric-card"}
                    metricList={[
                        {
                            name: "Tune Metrics",
                            val: tuneMetric
                        }
                    ]}
                />
            </div>
            <EvalMetricsTable
                queryInfo={queryInfo}
                results={queryInfo?.results}
                tuneMetric={tuneMetric?.split(" ")[0]}
                downloadClicked={downloadClicked}
                onDownloadClick={onDownloadClick}
            />
            <SplitStats
                queryType={queryInfo?.queryType as QueryType}
                queryStatus={queryInfo.trainingStatus as jobStatus}
                taskType={queryInfo.taskType}
                splitStats={trainTableSplitStats as SplitTrainTableStats}
                show={true}
                isTrainingEvaluation={true}
            />
            {queryInfo?.taskType === TaskType.REGRESSION &&
                dataSplitOptions[0] && (
                    <>
                        <RegressionHeatMap
                            queryId={queryInfo.id}
                            trainingJobId={queryInfo.trainingJobId as string}
                            dataSplit={dataSplitOptions[0]?.value as string}
                            handleError={handleError}
                        />
                        <Histogram
                            queryId={queryInfo.id}
                            trainingJobId={queryInfo.trainingJobId as string}
                            handleError={handleError}
                            queryInfo={queryInfo}
                            samples={`${dataSplitOptions?.find((d) => d.value === (dataSplitKeys.HoldOut as string))?.samples}`}
                        />
                    </>
                )}
            {queryInfo?.graphName !== "N/A" &&
                dataSplitOptions[0]?.value === evaluationSplit &&
                queryInfo?.taskType === TaskType.BINARY_CLASSIFICATION && (
                    <BinaryClassification
                        queryId={queryInfo.id}
                        trainingJobId={queryInfo.trainingJobId as string}
                        taskType={queryInfo?.taskType}
                        width={width * 0.7}
                        handleError={handleError}
                    />
                )}
        </div>
    );
};

export default WithErrorBoundary(PQEvaluationOverview);
