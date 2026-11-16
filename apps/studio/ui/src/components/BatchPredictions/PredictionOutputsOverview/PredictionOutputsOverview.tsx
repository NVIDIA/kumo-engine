// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useState } from "react";

import {
    Card,
    CardContent,
    CardHeader,
    Tabs,
    TabsContent,
    TabsList,
    TabsTrigger,
    Title
} from "@kumo-ai/kumo-ui-catalog";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";

import { UnlinkedBatchPredictionJobData } from "types/batchPredictions";

import { jobStatus } from "utils/Constants";

import Configurations from "../Configurations/Configurations";
import BPExplanationOverview from "../Explanation/BPExplanationOverview";
import BPOutputStatistics from "../OutputStatistics/OutputStatistics";

interface PredictionOutputsOverviewProps {
    predictionInfo?: UnlinkedBatchPredictionJobData | null;
    handleError: (err: Error) => void;
    queryYaml: string;
    isJobRunning: boolean;
    anchorTime?: string | null;
}

const PredictionOutputsOverview = ({
    predictionInfo,
    handleError,
    queryYaml,
    isJobRunning,
    anchorTime
}: PredictionOutputsOverviewProps) => {
    const [activeTab, setActiveTab] = useState("");

    const [tabs, setTabs] = useState<{ title: string }[]>([]);

    const hasOutputArtifacts =
        predictionInfo?.execution?.results?.output_artifact_stats &&
        predictionInfo?.execution?.results?.output_artifact_stats.length > 0;

    // Add tabs based on job status and available data
    useEffect(() => {
        if (predictionInfo?.status === jobStatus.Done) {
            const tabList: { title: string }[] = [];

            if (hasOutputArtifacts) {
                tabList.push({ title: "Configurations" });
            }

            tabList.push({ title: "Output statistics" });

            if (predictionInfo?.explanations === true) {
                tabList.push({ title: "Explanation" });
            }

            setTabs(tabList);
            setActiveTab(
                hasOutputArtifacts ? "Configurations" : "Output statistics"
            );
        } else {
            setTabs([]);
            setActiveTab("");
        }
    }, [
        predictionInfo?.status,
        predictionInfo?.explanations,
        hasOutputArtifacts
    ]);

    if (!tabs || tabs.length < 1) {
        return null;
    }

    return (
        <Card
            data-testid="prediction-outputs-overview"
            className="p-6 mb-4 border-none rounded-lg gap-6">
            <CardHeader className="p-0">
                <Title
                    data-testid="prediction-outputs-heading"
                    variant="tertiary-heading"
                    className="text-xl"
                    children={"Prediction Outputs"}
                />
            </CardHeader>
            <CardContent className="p-0">
                <Tabs
                    value={activeTab}
                    onValueChange={setActiveTab}
                    defaultValue={activeTab}>
                    <TabsList className="gap-8">
                        {tabs?.map(({ title }) => (
                            <TabsTrigger
                                key={`tab-${title}`}
                                value={title}
                                className={"p-1 pl-0 pr-0 leading-5"}>
                                {title}
                            </TabsTrigger>
                        ))}
                    </TabsList>
                    <hr />

                    {activeTab !== "" && (
                        <TabsContent className="pt-6 pb-0" value={activeTab}>
                            {predictionInfo?.id &&
                                predictionInfo?.taskType &&
                                predictionInfo?.status === jobStatus.Done && (
                                    <>
                                        {hasOutputArtifacts &&
                                            activeTab === "Configurations" && (
                                                <Configurations
                                                    anchorTime={anchorTime}
                                                    predictionInfo={
                                                        predictionInfo
                                                    }
                                                    handleError={handleError}
                                                />
                                            )}
                                        <BPOutputStatistics
                                            jobInfo={predictionInfo}
                                            show={
                                                activeTab ===
                                                "Output statistics"
                                            }
                                            isJobRunning={isJobRunning}
                                            handleError={handleError}
                                        />
                                        {predictionInfo?.explanations && (
                                            <BPExplanationOverview
                                                queryYaml={queryYaml}
                                                predictiveQueryType={
                                                    predictionInfo?.taskType
                                                }
                                                jobInfo={predictionInfo}
                                                show={
                                                    activeTab === "Explanation"
                                                }
                                                showGroundTruth={false}
                                                jobId={
                                                    predictionInfo?.trainingJobID
                                                }
                                            />
                                        )}
                                    </>
                                )}
                        </TabsContent>
                    )}
                </Tabs>
            </CardContent>
        </Card>
    );
};

export default WithErrorBoundary(PredictionOutputsOverview);
