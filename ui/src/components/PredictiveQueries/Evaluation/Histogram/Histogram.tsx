// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useMemo, useState } from "react";
import Spacer from "react-spacer";

import {
    Alert,
    AlertDescription,
    Card,
    Description,
    GridCol,
    GridRow,
    HistogramChart,
    Info,
    Select,
    Skeleton,
    Symbol,
    TargetHistogram,
    Title,
    Tooltip,
    TooltipContent,
    TooltipTrigger
} from "@kumo-ai/kumo-ui-catalog";
import {
    AllTargetRegressionStats,
    HistogramAndStatsGroup,
    TargetHistogramAndStats
} from "gen/v0/types";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";

import { useRegressionHistogram } from "hooks/jobs/training/eval/regression";

import {
    PredictiveQueryInfo,
    RegressionHistogramDistribution
} from "types/predictiveQueries";

import { dataSplitKeys } from "utils/Constants";
import { parseNumbers } from "utils/Helpers/numberHelpers";
import { getKeys } from "utils/Helpers/objectHelpers";
import { isValid } from "utils/Helpers/typeGuards";

/**
 * Stats component to parse the given value and set it 2 decimal places
 *
 * @param {string|number} value Stats value
 * @returns `<Stats/>`
 */
const Stat = ({ value }: { value: string | number }) => {
    return (
        <label className="text-xl">
            {!value && value?.toString() !== "0" ? (
                <Symbol type="emdash" />
            ) : (
                parseNumbers(value, { decimals: 2 })
            )}
        </label>
    );
};

interface StatsTableProps {
    headers: string[];
    values: (string | number | null | undefined)[];
}

const StatsTable = ({ headers, values }: StatsTableProps) => (
    <table className="text-sm leading-none bg-transparent [&_tbody_td]:pt-2 [&_tbody_td]:pr-10 [&_tbody_td]:pb-1.25 [&_tbody_td]:pl-0">
        <tbody>
            <tr>
                {headers.map((header) => (
                    <td key={header}>{header}</td>
                ))}
            </tr>
            <tr>
                {values.map((value, index) => (
                    <td key={index}>
                        <Stat value={value as number} />
                    </td>
                ))}
            </tr>
        </tbody>
    </table>
);

export interface HistogramProps {
    queryId: string;
    trainingJobId: string;
    handleError: (err: Error, title?: string) => void;
    queryInfo: PredictiveQueryInfo;
    samples?: string;
}

const Histogram = ({
    queryId,
    trainingJobId,
    handleError,
    queryInfo,
    samples
}: HistogramProps) => {
    const [isLoading, setLoading] = useState<boolean>(true);
    const [isEmptyStats, setEmptyStats] = useState<boolean>(true);
    const [currentStats, setCurrentStats] = useState<TargetHistogramAndStats>(
        {}
    );
    const [stats, setStats] = useState<AllTargetRegressionStats>({
        label_distribution: {
            train_table_stats: {},
            val_table_stats: {},
            test_table_stats: {}
        },
        prediction_distribution: {
            train_table_stats: {},
            val_table_stats: {},
            test_table_stats: {}
        }
    });

    const [shouldRenderChart, setShouldRenderChart] = useState<boolean>(false);
    const [statskey, setStatsKey] = useState(dataSplitKeys.HoldOut as string);

    const distributions: Array<RegressionHistogramDistribution> =
        useMemo(() => {
            return [
                {
                    value: "label_distribution",
                    label: "Actuals"
                },
                {
                    value: "prediction_distribution",
                    label: "Predictions"
                }
            ];
        }, []);

    const [selectedDistribution, setSelectedDistribution] =
        useState<RegressionHistogramDistribution>(distributions[0]);

    const splitText = useMemo(() => {
        return statskey === dataSplitKeys.HoldOut
            ? "holdout"
            : statskey === dataSplitKeys.Training
              ? "training"
              : statskey === dataSplitKeys.Validation
                ? "evaluation"
                : "";
    }, [statskey]);

    const getDataSplit = useCallback(() => {
        let split = "";
        if (statskey) {
            split = statskey + "_table_stats";
        }
        return split;
    }, [statskey]);

    /**
     * Checking any of the Distribution has stats for data split values.
     *
     * If none of the Distibution has stats value then show error message
     * on UI
     *
     * If any of the stats exists then show summary stats and histogram stats
     * based on selected Data split and Distribution.
     *
     * @param {AllTargetRegressionStats} stats Statistics
     * @returns bool
     */
    const isEmptyStatsFunc = useCallback(
        (stats: AllTargetRegressionStats) => {
            let emptyStats = true;
            const split = getDataSplit() as keyof HistogramAndStatsGroup;
            if (stats && isValid(split)) {
                const labelKey = distributions[0]
                    .value as keyof AllTargetRegressionStats;
                const predicitionKey = distributions[1]
                    .value as keyof AllTargetRegressionStats;
                if (stats[labelKey]) {
                    if (stats[labelKey][split]) {
                        emptyStats = false;
                    }
                }
                if (stats[predicitionKey]) {
                    if (stats[predicitionKey][split]) {
                        emptyStats = false;
                    }
                }
            }
            return emptyStats;
        },
        [distributions, getDataSplit]
    );

    /**
     * Get current stats to show in UI based on
     * selected DataSplits and Distribution
     *
     * @param {AllTargetRegressionStats} data stats object
     * @param {RegressionHistogramDistribution} selectedDistribution Selected distribution. actual|prediction
     * @returns stats
     */
    const getCurrentStats = useCallback(
        (
            data: AllTargetRegressionStats,
            selectedDistribution: RegressionHistogramDistribution
        ) => {
            let selectedStats: TargetHistogramAndStats = {};
            if (getDataSplit()) {
                if (selectedDistribution?.value) {
                    const distributionKey =
                        selectedDistribution?.value as keyof AllTargetRegressionStats;
                    const statsKey =
                        getDataSplit() as keyof HistogramAndStatsGroup;
                    selectedStats = data[distributionKey]
                        ? (data[distributionKey][statsKey] ?? {})
                        : {};
                }
            }
            return selectedStats;
        },
        [getDataSplit]
    );

    /**
     * Invoking REST API end point to get histogram stats for regression
     * type queries
     */
    const { data: regressionHistogramResponse, isLoading: isDataLoading } =
        useRegressionHistogram({
            queryId,
            trainingJobId,
            handleError
        });

    useEffect(() => {
        if (regressionHistogramResponse?.data) {
            setEmptyStats(isEmptyStatsFunc(regressionHistogramResponse?.data));
            setStats(regressionHistogramResponse?.data);
            setCurrentStats(
                getCurrentStats(
                    regressionHistogramResponse?.data,
                    selectedDistribution
                )
            );
            setLoading(false);
        }
    }, [
        regressionHistogramResponse,
        getCurrentStats,
        isEmptyStatsFunc,
        selectedDistribution
    ]);

    useEffect(() => {
        setShouldRenderChart(false);
        const timer = setTimeout(() => {
            setShouldRenderChart(true);
        }, 100);
        return () => clearTimeout(timer);
    }, [currentStats, selectedDistribution]);

    /**
     * Method to invoke from Parent class to update the stats whenever user
     * selects specific Data Split from drop-down
     *
     */
    const refreshStats = useCallback(
        (
            stats: AllTargetRegressionStats,
            selectedDistribution: RegressionHistogramDistribution
        ) => {
            setTimeout(() => {
                setEmptyStats(isEmptyStatsFunc(stats));
                setCurrentStats(getCurrentStats(stats, selectedDistribution));
            }, 1);
        },
        [getCurrentStats, isEmptyStatsFunc]
    );

    const onChangeDistribution = useCallback(
        (e: RegressionHistogramDistribution) => {
            setSelectedDistribution(e);
            refreshStats(stats, e);
        },
        [refreshStats, stats]
    );

    const isForecast = queryInfo?.taskType === "forecasting";

    if (isForecast) {
        return (
            <Card className="rounded-sm mb-1 shadow-none px-6 py-4">
                <Title
                    variant="tertiary-heading"
                    children={"Distribution of Predictions"}
                />
                <Spacer height={8} />
                <Alert variant="default" enableClose={false}>
                    <AlertDescription>
                        {"Distribution stats are not available for forecasting"}
                    </AlertDescription>
                </Alert>
            </Card>
        );
    }
    const actual = "label_distribution" as keyof AllTargetRegressionStats;
    const predict = "prediction_distribution" as keyof AllTargetRegressionStats;

    const dataSplitKey = getDataSplit() as keyof HistogramAndStatsGroup;

    return (
        <Card className="rounded-sm mb-1 shadow-none p-6">
            {isLoading || isDataLoading ? (
                <Skeleton className="mt-2.5 h-12.5" />
            ) : (
                <>
                    <GridRow className="flex justify-between">
                        <GridCol>
                            <Title
                                variant="tertiary-heading"
                                children={"Distribution of Predictions"}
                            />
                        </GridCol>
                        <GridCol>
                            <div className="flex justify-end">
                                <Select
                                    id={"select-split"}
                                    selectedValues={[statskey]}
                                    popoverClassName="w-32"
                                    options={[
                                        {
                                            value: dataSplitKeys.HoldOut,
                                            label: "Holdout"
                                        },
                                        {
                                            value: dataSplitKeys.Validation,
                                            label: "Validation"
                                        },
                                        {
                                            value: dataSplitKeys.Training,
                                            label: "Training"
                                        }
                                    ]}
                                    onValuesChange={(newValue) => {
                                        setStatsKey(newValue[0] as string);
                                    }}
                                    search={false}
                                    mode="single"
                                    placeholder={"Select"}
                                    variant="secondary"
                                    size="sm"
                                    isInValid={false}
                                />
                            </div>
                        </GridCol>
                    </GridRow>
                    {currentStats && getKeys(currentStats).length > 0 ? (
                        <Description
                            variant="secondary"
                            size="md"
                            className="leading-5">
                            {"Histogram of the predicted value alongside the actual" +
                                " labels (when available) for all entities in your " +
                                "Predictive Query, in the " +
                                splitText +
                                " set of the training table."}
                            <Tooltip id={"histogram-tootip"}>
                                <TooltipTrigger
                                    className="ml-1"
                                    data-testid="tooltip-trigger">
                                    {
                                        <span className="inline-flex">
                                            <Info className="size-3.5" />
                                        </span>
                                    }
                                </TooltipTrigger>
                                <TooltipContent
                                    className="max-w-100 text-justify"
                                    arrow={false}
                                    side={"right"}
                                    sideOffset={5}>
                                    {"This plot can be used to spot-check the quality" +
                                        " of the model that was trained. In a perfect" +
                                        " model, the shape of the two distributions" +
                                        " should match. In most real situations," +
                                        " some divergence is expected, even for good" +
                                        " models. The predictions were generated by " +
                                        "applying the model to every single row in the " +
                                        "training table that was generated by your " +
                                        "predictive query. The actuals are the labels " +
                                        "from the training table."}
                                </TooltipContent>
                            </Tooltip>
                        </Description>
                    ) : (
                        <Description
                            variant="secondary"
                            size="md"
                            className="leading-5">
                            {statskey?.includes("Holdout")
                                ? samples === "N/A"
                                    ? "The distribution histogram for this query is not available" +
                                      " for the selected data split."
                                    : "The distribution histogram for this query is not available." +
                                      " To fix this, try retraining the query. If this doesn't work," +
                                      " you may contact your POC at Kumo to investigate further"
                                : "The distribution histogram for this query is not available" +
                                  " for the selected data split."}
                        </Description>
                    )}
                    <Spacer height={8} />
                    {!isEmptyStats && (
                        <>
                            <GridRow className="flex justify-between">
                                <GridCol
                                    xs={6}
                                    sm={6}
                                    md={6}
                                    lg={6}
                                    className="w-full">
                                    {shouldRenderChart &&
                                        currentStats &&
                                        stats &&
                                        stats[predict]?.[dataSplitKey] &&
                                        stats[actual]?.[dataSplitKey] && (
                                            <HistogramChart
                                                key={`${statskey}-${selectedDistribution.value}`}
                                                id={"histogram-preds-actuals"}
                                                className="flex h-[50vh] w-160"
                                                predData={
                                                    stats[predict][dataSplitKey]
                                                        .histogram as TargetHistogram
                                                }
                                                labelData={
                                                    stats[actual][dataSplitKey]
                                                        .histogram as TargetHistogram
                                                }
                                                xAxisLabel="Actual or Predicted Value"
                                                yAxisLabel="Number of Occurrences"
                                            />
                                        )}
                                </GridCol>
                                <GridCol
                                    xs={5}
                                    sm={5}
                                    md={5}
                                    lg={5}
                                    className="ml-12.5">
                                    <div className="text-title-secondary mb-1">
                                        {"Summary stats for: "}
                                        <div className="max-w-50 !z-[1046]">
                                            <Select
                                                placeholder={"Select option"}
                                                search={false}
                                                mode="single"
                                                variant="secondary"
                                                popoverClassName="w-50"
                                                selectedValues={
                                                    selectedDistribution?.value
                                                        ? Array.of(
                                                              selectedDistribution?.value
                                                          )
                                                        : undefined
                                                }
                                                options={[
                                                    {
                                                        value: "label_distribution",
                                                        label: "Actuals"
                                                    },
                                                    {
                                                        value: "prediction_distribution",
                                                        label: "Predictions"
                                                    }
                                                ]}
                                                onValuesChange={(newValue) => {
                                                    onChangeDistribution(
                                                        distributions?.find(
                                                            (item) =>
                                                                newValue.includes(
                                                                    item.value
                                                                )
                                                        ) as RegressionHistogramDistribution
                                                    );
                                                }}
                                            />
                                        </div>
                                    </div>
                                    <StatsTable
                                        headers={["Min", "Mean", "Max"]}
                                        values={[
                                            currentStats?.target_min,
                                            currentStats?.target_mean,
                                            currentStats?.target_max
                                        ]}
                                    />
                                    <Spacer height={20} />
                                    <span className="text-title-secondary">
                                        {"Percentiles"}
                                    </span>
                                    <Spacer height={5} />
                                    <StatsTable
                                        headers={["25%", "50%", "75%"]}
                                        values={[
                                            currentStats?.target_percentile_25,
                                            currentStats?.target_median,
                                            currentStats?.target_percentile_75
                                        ]}
                                    />
                                </GridCol>
                            </GridRow>
                        </>
                    )}
                </>
            )}
        </Card>
    );
};

export default WithErrorBoundary(Histogram);
