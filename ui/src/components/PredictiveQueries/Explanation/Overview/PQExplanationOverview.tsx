// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React, { useEffect, useMemo, useState } from "react";
import Spacer from "react-spacer";

import {
    Alert,
    AlertDescription,
    Card,
    cn,
    Skeleton,
    TabCard,
    TabSwitcher,
    Title
} from "@kumo-ai/kumo-ui-catalog";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";
import EntityLevel from "components/PredictiveQueries/Explanation/EntityLevel/EntityLevel";
import { SubgraphSumaryTable } from "components/PredictiveQueries/Explanation/SubGraph/SubGraph";
import { SubgraphTreeView } from "components/PredictiveQueries/Explanation/SubGraph/SubgraphTreeView";
import { Top3Features } from "components/PredictiveQueries/Top3Features/Top3Features";

import {
    useEntityCategories,
    useEntityData
} from "hooks/jobs/training/eval/explorer";
import { useTrainingJobNodePredRfm } from "hooks/jobs/training/trainingJobNodePredRfm";
import useFeatureFlags from "hooks/utils/featureFlags";

import {
    GraphNodeData,
    GraphXaiScores,
    LinkPredictionEntityOutput
} from "types/explorer";
import { PredictiveQueryInfo, TaskType } from "types/predictiveQueries";

import { jobStatus, SUPPORT_CONTACT_MESSAGE } from "utils/Constants";
import { getKeys } from "utils/Helpers/objectHelpers";
import { isLinkPredictionType, isValid } from "utils/Helpers/typeGuards";
import { parseDateTime } from "utils/timeConverter";

import GraphColumnLevelOverview from "../GraphColumnLevelOverview/GraphColumnLevelOverview";
import Prediction from "../Prediction/Prediction";

export interface PQExplanationProps {
    queryInfo: PredictiveQueryInfo;
    show: boolean;
}

const PQExplanationOverview: React.FC<PQExplanationProps> = ({
    queryInfo,
    show
}) => {
    const [entityIdCategory, setEntityIdCategory] = useState<string>("");
    const [category, setCategory] = useState<string>("");
    const [entityId, setEntityId] = useState<string>("");
    const [originalEntityId, setOriginalEntityId] = useState<string>("");
    const [selectedEntityIndex, setSelectedEntityIndex] = useState<number>(0);
    const [xaiData, setXaiData] = useState<GraphXaiScores | null>(null);
    const [activeSubGraph, setActiveSubGraph] = useState<GraphNodeData | null>(
        null
    );
    const [selectedPredictionColumns, setSelectedPredictionColumns] = useState<
        Array<string>
    >([]);

    const [activeTab, setActiveTab] = useState<"global" | "local">("global");

    const handleToggle = (value: string) => {
        setActiveTab(value as "global" | "local");
    };

    const isTlpTask = queryInfo?.taskType === TaskType.TEMPORAL_LINK_PREDICTION;

    const {
        filteredEntityIdCategories: entityIdCategories,
        filteredCategories: categories,
        isLoading: isLoadingCategories
    } = useEntityCategories(
        queryInfo?.id,
        queryInfo?.trainingStatus as jobStatus,
        (err: Error) => console.error(err),
        queryInfo?.trainingJobId as string
    );

    const {
        entityOutput,
        historicalItems,
        subgraph,
        isLoading: isDataLoading
    } = useEntityData({
        queryId: queryInfo?.id,
        selectedEntity: entityId,
        taskType: queryInfo?.taskType as TaskType,
        status: queryInfo?.trainingStatus as jobStatus,
        trainingJobId: queryInfo?.trainingJobId as string
    });

    const { isNodePredRfmLoading, subgraphSummary } = useTrainingJobNodePredRfm(
        queryInfo?.trainingJobId as string,
        entityId
    );

    const [anchorTime, setAnchorTime] = useState<string>("");
    const [predictionAnchorTime, setPredictionAnchorTime] =
        useState<string>("");

    const featureFlags = useFeatureFlags();

    useEffect(() => {
        if (
            category === "" ||
            (categories && categories[category] === undefined)
        ) {
            if (getKeys(categories).length > 0) {
                setCategory(getKeys(categories)[0]);
            }
        }

        if (
            entityIdCategory === "" ||
            (entityIdCategories && entityIdCategories[category] === undefined)
        ) {
            if (getKeys(entityIdCategories).length > 0) {
                setEntityIdCategory(getKeys(entityIdCategories)[0]);
            }
        }
    }, [entityIdCategories, entityIdCategory, categories, category]);

    useEffect(() => {
        if (category && categories[category]) {
            setEntityId(categories[category][selectedEntityIndex]);
            if (
                Array.isArray(entityIdCategories[category]) &&
                entityIdCategories[category]?.length > 0
            ) {
                setOriginalEntityId(
                    entityIdCategories[category][selectedEntityIndex]
                );
            }
            setAnchorTime(
                parseDateTime(subgraph?.subgraph_store?.seed_time as string)
            );
            setPredictionAnchorTime(
                subgraph?.subgraph_store?.seed_time as string
            );
        }
    }, [
        category,
        selectedEntityIndex,
        categories,
        entityIdCategories,
        subgraph?.subgraph_store?.seed_time
    ]);

    const summaryTabs = useMemo(() => {
        const hasSummaryData = !subgraphSummary.every(
            (row) => !isValid(row?.current) && !isValid(row?.global)
        );

        if (isTlpTask) {
            const tabs: Array<{ label: string; value: number }> = [
                { label: "Sample evaluation", value: 0 }
            ];
            if (hasSummaryData) {
                tabs.push({ label: "Subgraph summary", value: 1 });
                tabs.push({ label: "Top 3 columns", value: 2 });
            } else {
                tabs.push({ label: "Top 3 columns", value: 1 });
            }
            return tabs;
        }

        if (hasSummaryData) {
            return [
                { label: "Subgraph summary", value: 0 },
                { label: "Top 3 columns", value: 1 }
            ];
        }
        return [{ label: "Top 3 columns", value: 0 }];
    }, [isTlpTask, subgraphSummary]);

    const xaiDisabledForTaskType =
        isLinkPredictionType(queryInfo?.taskType) ||
        queryInfo?.taskType === TaskType.MULTILABEL_RANKING;

    const maybeColumnAnalysis = (
        <>
            {featureFlags.ENABLE_COLUMN_ANALYSIS && !xaiDisabledForTaskType && (
                <GraphColumnLevelOverview
                    queryId={queryInfo?.id}
                    show={show && activeTab === "global"}
                    trainingJobId={queryInfo?.trainingJobId as string}
                    setXaiScore={setXaiData}
                    subGraphInfo={activeSubGraph}
                    predictionAnchorTime={predictionAnchorTime}
                />
            )}
            {!featureFlags.ENABLE_COLUMN_ANALYSIS &&
                show &&
                activeTab === "global" && (
                    <Alert variant="default" enableClose={false}>
                        <AlertDescription>
                            Column Analysis is not available in this version of
                            Kumo. For more information,{" "}
                            {SUPPORT_CONTACT_MESSAGE}.
                        </AlertDescription>
                    </Alert>
                )}
            {xaiDisabledForTaskType && show && activeTab === "global" && (
                <Alert variant="default" enableClose={false}>
                    <AlertDescription>
                        {`Column Analysis is not available for ` +
                            `${queryInfo?.taskType.replace("_", " ").toUpperCase()} tasks. ` +
                            `For more information, ${SUPPORT_CONTACT_MESSAGE.toLocaleLowerCase()}.`}
                    </AlertDescription>
                </Alert>
            )}
        </>
    );

    const msg = `There are no entities to explore available for this query.
                    This may be due to an unsupported task type or specific
                    model configuration, for which we do not provide any
                    entities to explore yet. If you believe this is an error,${" "}
                    ${SUPPORT_CONTACT_MESSAGE}.`;

    // For TLP, local content is always shown (no tab toggle needed)
    const showLocalContent = isTlpTask || activeTab === "local";

    const tlpTabCardContent = (currentTab: number) => (
        <div>
            {currentTab === 0 && (
                <Prediction
                    entityOutput={entityOutput}
                    taskType={queryInfo?.taskType}
                    historicalItems={historicalItems}
                    isDataLoading={isDataLoading}
                    query={queryInfo?.queryYaml}
                    anchorTime={anchorTime}
                    showGroundTruth={true}
                    selectedColumns={selectedPredictionColumns}
                    setSelectedColumns={setSelectedPredictionColumns}
                />
            )}
            {currentTab !== 0 &&
                (summaryTabs.length > 2 && currentTab === 1 ? (
                    <SubgraphSumaryTable subgraphSummary={subgraphSummary} />
                ) : (
                    <Top3Features
                        entityId={entityId}
                        jobId={queryInfo?.trainingJobId as string}
                        category={"random"}
                    />
                ))}
        </div>
    );

    const nonTlpTabCardContent = (currentTab: number) => (
        <div>
            {summaryTabs.length > 1 && currentTab === 0 ? (
                <SubgraphSumaryTable subgraphSummary={subgraphSummary} />
            ) : (
                <Top3Features
                    entityId={entityId}
                    jobId={queryInfo?.trainingJobId as string}
                    category={"random"}
                />
            )}
        </div>
    );

    return (
        <div className={cn("min-h-220", show ? "block" : "hidden")}>
            {!isTlpTask && (
                <>
                    <TabSwitcher
                        tabs={[
                            {
                                label: "Global column explanations",
                                value: "global"
                            },
                            {
                                label: "Local subgraph explanation",
                                value: "local"
                            }
                        ]}
                        value={activeTab}
                        onValueChange={handleToggle}
                    />
                    <Spacer height={24} />
                </>
            )}
            {isTlpTask && (
                <Title
                    variant="section-header"
                    className="font-bold text-title-primary mb-6">
                    Local Subgraph Explanation
                </Title>
            )}
            {isLoadingCategories && <Skeleton className="w-full h-40" />}
            {!isLoadingCategories &&
                !isTlpTask &&
                activeTab === "global" &&
                maybeColumnAnalysis}
            {!isLoadingCategories &&
                getKeys(categories).length === 0 &&
                showLocalContent && (
                    <div className={show ? "block" : "hidden"}>
                        <Alert variant="default" enableClose={false}>
                            <AlertDescription>{msg}</AlertDescription>
                        </Alert>
                    </div>
                )}
            {!isLoadingCategories &&
                getKeys(categories).length > 0 &&
                showLocalContent && (
                    <div>
                        <EntityLevel
                            categories={categories}
                            category={category}
                            setCategory={setCategory}
                            selectedEntityIndex={selectedEntityIndex}
                            setSelectedEntityIndex={setSelectedEntityIndex}
                            entityId={entityId}
                            originalEntityId={originalEntityId}
                            entityIdentifier={
                                subgraph?.subgraph_store?.nodes?.[
                                    String(subgraph?.subgraph_store?.root_id)
                                ]?.pkey_name ?? undefined
                            }
                        />
                        <Spacer height={24} />
                        {isTlpTask ? (
                            <>
                                <Card className="p-6 gap-6 mb-6">
                                    <Title
                                        variant="section-header"
                                        children={"Sample details"}
                                        className="font-bold text-title-primary"
                                    />
                                    {isNodePredRfmLoading ? (
                                        <Skeleton className="w-full h-20" />
                                    ) : (
                                        <TabCard
                                            title=""
                                            tabs={summaryTabs}
                                            className={"border-none p-0"}
                                            contentClassName={"px-0 py-0 mt-6"}
                                            defaultSelected={0}>
                                            {tlpTabCardContent}
                                        </TabCard>
                                    )}
                                </Card>
                            </>
                        ) : (
                            <>
                                <Card className="p-6 gap-6 mb-6">
                                    <Title
                                        variant="section-header"
                                        children={"Sample details"}
                                        className="font-bold text-title-primary"
                                    />
                                    <Prediction
                                        entityOutput={entityOutput}
                                        taskType={queryInfo?.taskType}
                                        historicalItems={historicalItems}
                                        isDataLoading={isDataLoading}
                                        query={queryInfo?.queryYaml}
                                        anchorTime={anchorTime}
                                        showGroundTruth={true}
                                        selectedColumns={
                                            selectedPredictionColumns
                                        }
                                        setSelectedColumns={
                                            setSelectedPredictionColumns
                                        }
                                    />
                                    {isNodePredRfmLoading ? (
                                        <Skeleton className="w-full h-20" />
                                    ) : (
                                        <TabCard
                                            title=""
                                            tabs={summaryTabs}
                                            className={"border-none p-0"}
                                            contentClassName={"px-0 py-0 mt-6"}
                                            defaultSelected={0}>
                                            {nonTlpTabCardContent}
                                        </TabCard>
                                    )}
                                </Card>
                            </>
                        )}
                        <SubgraphTreeView
                            key={entityId}
                            subgraph={subgraph}
                            xaiData={xaiData as GraphXaiScores}
                            anchorTime={anchorTime}
                            setActiveSubGraph={setActiveSubGraph}
                            entityId={entityId}
                            taskType={queryInfo?.taskType}
                            // Cast is safe: useSubgraphTreeView internally
                            // gates entityOutput usage on isLinkPredictionType.
                            entityOutput={
                                entityOutput as LinkPredictionEntityOutput | null
                            }
                            historicalItems={historicalItems}
                            isLoading={isDataLoading}
                        />
                    </div>
                )}
        </div>
    );
};

export default WithErrorBoundary(PQExplanationOverview);
