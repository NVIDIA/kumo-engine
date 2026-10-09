// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useState } from "react";

import {
    Card,
    cn,
    GridCol,
    GridRow,
    PerformanceChart,
    Select,
    Title
} from "@kumo-ai/kumo-ui-catalog";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";
import TrainingSetting from "components/PredictiveQueries/Overview/TrainingSetting";
import ModelPlanner from "components/PredictiveQueries/Training/ModelPlanner/ModelPlanner";

import { DropdownOption, SelectOption } from "types/inputs";
import { PredictiveQueryInfo } from "types/predictiveQueries";
import { CTAInfo } from "types/warnings";

import { jobStatus } from "utils/Constants";
import { getTuneMetric } from "utils/predictiveQueries";

import ExperimentMonitoringMetrics from "../ExperimentMonitoringMetrics/ExperimentMonitoringMetrics";

/**
 * Train Query Component *
 */

function parseModelPlan(queryInfo: PredictiveQueryInfo): string {
    if (queryInfo.isV2 !== true || queryInfo.jobSource === "SDK") {
        const trainingInfoSplit =
            queryInfo?.modelPlannerYaml?.split(/(.*?training_job)/g);
        return trainingInfoSplit?.slice(1).join("") ?? "";
    }
    return queryInfo?.modelPlannerYaml ?? "";
}

export interface PQTrainingOverviewProps {
    queryInfo: PredictiveQueryInfo;
    show: boolean;
    ctaMsgs: CTAInfo;
}

const PQTrainingOverview = ({
    queryInfo,
    show,
    ctaMsgs
}: PQTrainingOverviewProps) => {
    const experimentResults =
        queryInfo?.queryProgressStats?.datasets?.[0]?.value || [];
    const bestExperimentIndex =
        queryInfo?.queryProgressStats?.mode === "max"
            ? experimentResults.indexOf(Math.max(...experimentResults))
            : experimentResults.indexOf(Math.min(...experimentResults));

    const [selectedOption, setSelectedOption] = useState<DropdownOption>({
        value: "",
        label: `Loading...`
    });
    const [options, setOptions] = useState<DropdownOption[]>([]);

    // Extract tune_metric from modelPlannerYaml
    const tuneMetric = getTuneMetric(queryInfo?.modelPlannerYaml);

    const status = queryInfo?.trainingStatus;
    const advancedOptions =
        status === jobStatus.Running ||
        queryInfo?.trainingStatus === jobStatus.Done
            ? ""
            : queryInfo.advancedOptionsYaml || "";

    const generationPlan =
        status === jobStatus.Running || status === jobStatus.Done
            ? ""
            : queryInfo?.generationPlan || "";

    return (
        <div className={cn("flex flex-col gap-6", !show && "hidden")}>
            {queryInfo?.queryProgressStats && (
                <Card className="p-6">
                    <GridRow className="flex items-center justify-between">
                        <GridCol>
                            <Title
                                variant="tertiary-heading"
                                children={"Experiments"}
                            />
                        </GridCol>
                        <GridCol>
                            <div className="max-w-62.5">
                                <Select
                                    id={"experiment-dropdown"}
                                    placeholder="Experiment 0"
                                    search={false}
                                    mode="single"
                                    variant="secondary"
                                    popoverClassName="w-50"
                                    selectedValues={
                                        (selectedOption as DropdownOption).value
                                            ? (Array.of(
                                                  (
                                                      selectedOption as DropdownOption
                                                  )?.value
                                              ) as string[])
                                            : undefined
                                    }
                                    options={
                                        options as unknown as SelectOption[]
                                    }
                                    onValuesChange={(newValue: string[]) => {
                                        setSelectedOption(
                                            (options as SelectOption[])?.find(
                                                (item) =>
                                                    newValue.includes(
                                                        item.value
                                                    )
                                            ) as DropdownOption
                                        );
                                    }}
                                />
                            </div>
                        </GridCol>
                    </GridRow>
                    <GridRow className="flex flex-wrap justify-between gap-1 mt-2 px-6">
                        <GridCol xs={5} sm={5} md={5} lg={5}>
                            <PerformanceChart
                                progressData={queryInfo.queryProgressStats}
                                jobstatus={
                                    queryInfo.trainingStatus as jobStatus
                                }
                                showTitle={false}
                                width={window.innerWidth < 1300 ? 380 : 480}
                            />
                        </GridCol>

                        {bestExperimentIndex !== undefined &&
                            bestExperimentIndex !== -1 && (
                                <GridCol
                                    xs={5}
                                    sm={5}
                                    md={5}
                                    lg={5}
                                    className="w-[45%] max-w-[45%] shrink-0">
                                    <ModelPlanner
                                        trainingJobId={
                                            queryInfo?.trainingJobId as string
                                        }
                                        bestExperimentIndex={
                                            bestExperimentIndex
                                        }
                                        jobStatus={
                                            queryInfo?.trainingStatus as jobStatus
                                        }
                                        setOptions={setOptions}
                                        selectedOption={selectedOption}
                                        setSelectedOption={setSelectedOption}
                                        fallbackPlan={
                                            queryInfo?.modelPlannerYaml
                                        }
                                    />

                                    <Title
                                        variant="section-header"
                                        children={"Model Plan"}
                                        className="mb-1 mt-2 w-full text-center leading-none"
                                    />
                                </GridCol>
                            )}
                    </GridRow>
                </Card>
            )}
            {queryInfo?.id &&
                queryInfo?.trainingStatus !== jobStatus.Queued && (
                    <>
                        <ExperimentMonitoringMetrics
                            jobId={
                                (queryInfo?.trainingJobId ??
                                    queryInfo?.latestTrainingJobId) as string
                            }
                            trainingStatus={
                                queryInfo?.trainingStatus as jobStatus
                            }
                            tuneMetric={tuneMetric}
                        />
                        <TrainingSetting
                            trainingJobId={
                                queryInfo?.trainingJobId ??
                                (queryInfo?.latestTrainingJobId as string)
                            }
                            trainingstatus={queryInfo?.trainingStatus}
                            queryId={queryInfo?.id as string}
                            advancedOptions={advancedOptions}
                            trainingPlan={queryInfo?.trainingPlan as string}
                            modelPlan={parseModelPlan(queryInfo)}
                            generationPlan={generationPlan}
                            ctaMsgs={ctaMsgs}
                            loadingStatus={false}
                        />
                    </>
                )}
        </div>
    );
};

export default WithErrorBoundary(PQTrainingOverview);
