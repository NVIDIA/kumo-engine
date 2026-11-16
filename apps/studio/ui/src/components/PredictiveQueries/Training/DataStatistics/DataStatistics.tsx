// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useMemo, useState } from "react";

import {
    Card,
    CardContent,
    CardHeader,
    Description,
    Tabs,
    TabsContent,
    TabsList,
    TabsTrigger,
    Title,
    useWindowDimensions
} from "@kumo-ai/kumo-ui-catalog";
import {
    JobEventLogEntry,
    QueryType,
    SplitTrainTableStats
} from "gen/v0/types";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";
import StatsChart from "components/@ui/Chart/StatsChart";
import { BinaryClassificationStatsColumns } from "components/@ui/TableColumns/BinaryClassificationStats";
import { PQStatsColumns } from "components/@ui/TableColumns/PQStats";
import { RegressionStatsColumns } from "components/@ui/TableColumns/RegressionStats";
import SplitStats from "components/PredictiveQueries/Training/SplitStats/SplitStats";
import TimeframeChart from "components/PredictiveQueries/Training/TimeframeChart/TimeframeChart";

import {
    useSplitStats,
    useTrainTableGenerationStats
} from "hooks/jobs/trainTableGen/trainTableGen";

import { TaskType } from "types/predictiveQueries";
import {
    TimeFrameStats,
    TrainingTimeFrameStatsData,
    TrainTableGenStats
} from "types/stats";

import { jobStatus } from "utils/Constants";
import { isLinkPredictionType } from "utils/Helpers/typeGuards";

import TimeframeStats from "../TimeframeStats/TimeframeStats";

/**
 * Wrapper component for Training table Stats View with switch button.
 *
 * Can switch between Training timeframe stats table vierw,
 * Training split stats table view and timeframe stats chart view
 *
 * @param queryId query ID
 * @param queryStatus query training status
 * @param queryStage query training stage
 * @param queryType Type of query
 * @returns <TrainTableStatsPanel />
 */

export interface DataStatisticsProps {
    id: string;
    queryStatus: jobStatus;
    queryStage: JobEventLogEntry | undefined;
    taskType: TaskType;
    queryType: QueryType;
    isSourceSDK?: boolean;
    jobId?: string | null;
    isTrainingPage?: boolean;
}

const DataStatistics = ({
    id,
    queryStatus,
    queryStage,
    taskType,
    queryType,
    isSourceSDK = false,
    jobId = null,
    isTrainingPage
}: DataStatisticsProps) => {
    const [allStats, setAllStats] = useState<TrainTableGenStats | null>(null);
    const [tableData, setTableData] = useState<TimeFrameStats[]>([]);
    const [selectedStats, setSelectedStats] = useState<TimeFrameStats | null>(
        null
    );
    const [finishedIterations, setFinishedIterations] = useState<number>(-1);
    const [expectedIterations, setExpectedIterations] = useState<number>(-1);

    const [expandRow, setExpandRow] = useState<boolean>(false);

    const isTemporal = queryType === "temporal";

    const { width } = useWindowDimensions();

    const columns = useMemo(() => {
        let header = [];
        if (taskType === TaskType.BINARY_CLASSIFICATION) {
            header = BinaryClassificationStatsColumns(false, queryType);
            setExpandRow(false);
        } else if (
            taskType === TaskType.REGRESSION ||
            taskType === TaskType.FORECASTING
        ) {
            header = RegressionStatsColumns(false, queryType);
            setExpandRow(false);
        } else {
            const showDistribution = !isLinkPredictionType(taskType);
            header = PQStatsColumns(showDistribution, false, queryType);
            setExpandRow(true);
        }
        return header;
    }, [queryType, taskType]);

    const { stats, receivedResponse } = useTrainTableGenerationStats(
        id,
        queryStatus
    );

    useEffect(() => {
        if (stats && stats?.iteration_data) {
            setSelectedStats(selectedStats);
            setTableData(stats?.iteration_data as TimeFrameStats[]);
            setFinishedIterations(stats?.num_finished_iterations);
            setExpectedIterations(stats?.num_expected_iterations);
            setAllStats(stats);
        }
    }, [selectedStats, stats]);

    let progress =
        allStats === null && isTemporal
            ? ""
            : `Generated ${finishedIterations} out of ` +
              `${expectedIterations} timeframes`;

    if (finishedIterations === -1 || expectedIterations === -1) {
        progress = "";
    }
    if (progress && queryStage?.stage_name !== "Generating Training Data") {
        if (expectedIterations > finishedIterations) {
            progress += `   (${
                expectedIterations - finishedIterations
            } were empty after applying filters)`;
        }
    }

    let tabs = [{ label: "Split stats", value: 2 }];
    if (isTemporal) {
        tabs = [
            { label: "Split stats", value: 2 },
            { label: "Timeframe chart", value: 0 },
            { label: "Timeframe stats", value: 1 }
        ];
    }
    const [activeTab, setActiveTab] = useState(
        tabs.length > 0 ? `${tabs[0].value}` : ""
    );

    const rowsWithComponents = tableData.map((row: TimeFrameStats) => ({
        ...row,
        component: (
            <StatsChart
                data={row}
                title={"Timeframe Statistics"}
                width={width / 2.5}
                height={"300px"}
            />
        )
    }));

    const data = { data: rowsWithComponents, columns: columns };
    const stateProps: TrainingTimeFrameStatsData = {
        allStats,
        tableData: data,
        selectedStats,
        finishedIterations,
        expectedIterations,
        receivedResponse,
        isTemporal,
        queryStatus
    };
    const effectiveId =
        isSourceSDK && jobId ? jobId : isSourceSDK ? id : (jobId as string);
    const effectiveIsSDK = isSourceSDK && !jobId;
    const splitStats = useSplitStats(effectiveId, effectiveIsSDK, queryStatus);
    return (
        <Card className="p-6 rounded-sm gap-6">
            <CardHeader className="px-0">
                {isTrainingPage ? (
                    <Title
                        variant="tertiary-heading"
                        children={"Training Data Statistics"}
                    />
                ) : (
                    <Title
                        variant="secondary-heading"
                        children={"Training Data Statistics"}
                    />
                )}
            </CardHeader>
            <CardContent className="px-0">
                <Tabs
                    value={activeTab}
                    onValueChange={setActiveTab}
                    defaultValue={activeTab}>
                    <TabsList>
                        {tabs?.map(({ label, value }) => (
                            <TabsTrigger
                                key={`tab-${value}`}
                                value={`${value}`}
                                className={"p-1 pl-0 pr-0 leading-5"}>
                                {label}
                            </TabsTrigger>
                        ))}
                    </TabsList>
                    <hr />
                    <TabsContent className="p-0 pt-6" value={activeTab}>
                        {activeTab === `2` ? (
                            <Description
                                variant="secondary"
                                size="md"
                                className="!mb-1 pt-0 leading-5">
                                Statistics about the machine learning data split
                                are useful if you want to analyze your model's
                                performance based on the quantity and
                                distribution of training examples the model was
                                trained on (X% of the training examples had a
                                target value of Y in data split Z)
                            </Description>
                        ) : (
                            <div>
                                <Description
                                    variant="secondary"
                                    size="md"
                                    className="!mb-1 pt-0 leading-5">
                                    In order to create your pQuery, Kumo
                                    generated training examples from prior
                                    historical timeframes to help your pQuery
                                    learn to extrapolate historic patterns into
                                    the future. These training examples are of
                                    the form [Timeframe, Entity, Target value],
                                    where only Entities that meet your pQuery
                                    filter criteria are included, and the target
                                    value is the ground truth value obtained by
                                    running your pQuery at that historic point
                                    in time.
                                </Description>
                                <Description
                                    variant="secondary"
                                    size="md"
                                    className="!mb-1 pt-0 leading-5">
                                    These charts and tables show the
                                    distribution of those target values over
                                    different timeframes, and can be useful to
                                    spot anomalous trends (potentially meaning
                                    you need to fix your input data tables or
                                    change your pQuery).
                                </Description>
                            </div>
                        )}
                        {progress !== "" && (
                            <Description
                                variant="secondary"
                                size="md"
                                className="!mb-1 pt-0 leading-5">
                                {progress}
                            </Description>
                        )}
                        <div>
                            {tableData?.length > 0 && (
                                <TimeframeChart
                                    queryStatus={queryStatus}
                                    taskType={taskType}
                                    timeFrameStats={tableData}
                                    noData={
                                        !receivedResponse || allStats === null
                                    }
                                    show={activeTab === `0`}
                                />
                            )}
                            {tableData?.length > 0 && (
                                <TimeframeStats
                                    queryStatus={queryStatus}
                                    data={stateProps}
                                    show={activeTab === `1`}
                                    expandRow={
                                        expandRow &&
                                        !isLinkPredictionType(taskType)
                                    }
                                />
                            )}
                            <SplitStats
                                queryType={queryType}
                                queryStatus={queryStatus}
                                taskType={taskType}
                                show={activeTab === `2`}
                                splitStats={splitStats as SplitTrainTableStats}
                            />
                        </div>
                    </TabsContent>
                </Tabs>
            </CardContent>
        </Card>
    );
};

export default WithErrorBoundary(DataStatistics);
