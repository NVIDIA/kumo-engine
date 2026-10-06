// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useMemo, useState } from "react";
import Spacer from "react-spacer";

import {
    Card,
    CardContent,
    CardHeader,
    Description,
    Tabs,
    TabsContent,
    TabsList,
    TabsTrigger,
    Title
} from "@kumo-ai/kumo-ui-catalog";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";
import PQEvaluationOverview from "components/PredictiveQueries/Evaluation/Overview/PQEvaluationOverview";
import PQExplanationOverview from "components/PredictiveQueries/Explanation/Overview/PQExplanationOverview";
import PQTrainingOverview from "components/PredictiveQueries/Training/Overview/PQTrainingOverview";

import { PredictiveQueryInfo } from "types/predictiveQueries";
import { CTAInfo } from "types/warnings";

import { jobStatus, JobStatusType, queryTabType } from "utils/Constants";

interface ModelPerformanceOverviewProps {
    queryInfo: PredictiveQueryInfo | null;
    handleError: (err: Error) => void;
    ctaMsgs: CTAInfo;
    status?: JobStatusType;
}

const ModelPerformanceOverview = ({
    queryInfo,
    handleError,
    ctaMsgs,
    status
}: ModelPerformanceOverviewProps) => {
    const [activeTab, setActiveTab] = useState<string>(queryTabType.Evaluation);

    useEffect(() => {
        if (status === jobStatus.Cancelled || status === jobStatus.Failed) {
            setActiveTab(queryTabType.Training);
        }
    }, [status]);

    const tabs = useMemo(() => {
        if (status === jobStatus.Cancelled || status === jobStatus.Failed) {
            return [{ title: queryTabType.Training }];
        }
        return [
            { title: queryTabType.Evaluation },
            { title: queryTabType.Training },
            { title: queryTabType.Explanation }
        ];
    }, [status]);

    return (
        <Card className="p-6 border-none rounded-lg gap-6">
            <CardHeader className="p-0">
                <Title
                    variant="tertiary-heading"
                    className="text-xl"
                    children={"Model Performance"}
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
                                className={"p-1 pl-0 pr-0"}>
                                {title}
                            </TabsTrigger>
                        ))}
                    </TabsList>
                    <hr />

                    {activeTab !== "" && (
                        <TabsContent className="py-0" value={activeTab}>
                            {queryInfo?.id && (
                                <>
                                    <Spacer height={24} />
                                    {status === jobStatus.Done ? (
                                        <PQEvaluationOverview
                                            queryInfo={queryInfo}
                                            show={
                                                activeTab ===
                                                queryTabType.Evaluation
                                            }
                                            handleError={handleError}
                                        />
                                    ) : (
                                        activeTab ===
                                            queryTabType.Evaluation && (
                                            <Description
                                                variant="secondary"
                                                size="md"
                                                className="leading-5 font-medium">
                                                Evaluation metrics are on their
                                                way! In the meantime, you can
                                                explore the training details in
                                                the Training tab.
                                            </Description>
                                        )
                                    )}
                                    <PQTrainingOverview
                                        queryInfo={queryInfo}
                                        show={
                                            activeTab === queryTabType.Training
                                        }
                                        ctaMsgs={ctaMsgs}
                                    />
                                    {status === jobStatus.Done ? (
                                        <PQExplanationOverview
                                            queryInfo={queryInfo}
                                            show={
                                                activeTab ===
                                                queryTabType.Explanation
                                            }
                                        />
                                    ) : (
                                        activeTab ===
                                            queryTabType.Explanation && (
                                            <Description
                                                variant="secondary"
                                                size="md"
                                                className="leading-5">
                                                Explanation are on their way! In
                                                the meantime, you can explore
                                                the training details in the
                                                Training tab.
                                            </Description>
                                        )
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

export default WithErrorBoundary(ModelPerformanceOverview);
