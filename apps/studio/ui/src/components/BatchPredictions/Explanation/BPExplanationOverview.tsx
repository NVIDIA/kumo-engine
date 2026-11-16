// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React, { useEffect, useMemo, useState } from "react";
import Spacer from "react-spacer";

import {
    Alert,
    AlertDescription,
    Card,
    CardContent,
    CardHeader,
    cn,
    Skeleton,
    TabSwitcher,
    Title
} from "@kumo-ai/kumo-ui-catalog";
import { EntityOutput, HistoricalItems } from "gen/v0/types";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";
import EntityLevel from "components/PredictiveQueries/Explanation/EntityLevel/EntityLevel";
import GraphColumnLevelOverview from "components/PredictiveQueries/Explanation/GraphColumnLevelOverview/GraphColumnLevelOverview";
import Prediction from "components/PredictiveQueries/Explanation/Prediction/Prediction";
import { SubgraphTreeView } from "components/PredictiveQueries/Explanation/SubGraph/SubgraphTreeView";

import {
    useBpEntityCategories,
    useBpEntityData
} from "hooks/jobs/training/eval/explorer";
import useFeatureFlags from "hooks/utils/featureFlags";

import { UnlinkedBatchPredictionJobData } from "types/batchPredictions";
import {
    GraphNodeData,
    GraphXaiScores,
    LinkPredictionEntityOutput
} from "types/explorer";
import { TaskType } from "types/predictiveQueries";

import { SUPPORT_CONTACT_MESSAGE } from "utils/Constants";
import { getKeys } from "utils/Helpers/objectHelpers";
import { isLinkPredictionType } from "utils/Helpers/typeGuards";
import { parseDateTime } from "utils/timeConverter";

export interface BPExplanationProps {
    jobInfo: UnlinkedBatchPredictionJobData;
    show: boolean;
    predictiveQueryType: TaskType;
    queryYaml: string;
    showGroundTruth?: boolean;
    jobId?: string;
}

interface ColumnAnalysisSectionProps {
    featureFlagEnabled: boolean;
    xaiDisabledForTaskType: boolean;
    show: boolean;
    predictiveQueryType: TaskType;
    jobId: string;
    setXaiData: (data: GraphXaiScores | null) => void;
    activeSubGraph: GraphNodeData | null;
    predictionAnchorTime: string;
}

const ColumnAnalysisSection: React.FC<ColumnAnalysisSectionProps> = ({
    featureFlagEnabled,
    xaiDisabledForTaskType,
    show,
    predictiveQueryType,
    jobId,
    setXaiData,
    activeSubGraph,
    predictionAnchorTime
}) => {
    if (featureFlagEnabled && !xaiDisabledForTaskType) {
        return (
            <>
                <Spacer height={24} />
                <GraphColumnLevelOverview
                    bpjobId={jobId}
                    show={true}
                    setXaiScore={setXaiData}
                    subGraphInfo={activeSubGraph}
                    predictionAnchorTime={predictionAnchorTime}
                />
            </>
        );
    }

    if (!featureFlagEnabled) {
        return (
            <Alert variant="default" enableClose={false} className="my-6">
                <AlertDescription>
                    Column Analysis is not available in this version of Kumo.
                    For more information, {SUPPORT_CONTACT_MESSAGE}.
                </AlertDescription>
            </Alert>
        );
    }

    if (xaiDisabledForTaskType && show) {
        return (
            <Alert variant="default" enableClose={false} className="my-6">
                <AlertDescription>
                    {`Column Analysis is not available for ` +
                        `${predictiveQueryType.replace("_", " ").toUpperCase()} tasks. ` +
                        `For more information, ${SUPPORT_CONTACT_MESSAGE.toLocaleLowerCase()}.`}
                </AlertDescription>
            </Alert>
        );
    }

    return null;
};

/**
 * Renders the Explorer component, which allows users to view and compare a set
 * of entity's predicted items against historical data and ground truth.
 *
 * Users can dive into the details to gain insights and validate predictions.
 *
 * @component
 * @param {PQExplanationProps} props - The component props.
 * @param {QueryInfo} props.queryInfo - The query information.
 * @param {boolean} props.show - Indicates whether to show the Explorer component.
 * @returns {JSX.Element|null} The rendered Explorer component or null if `show` is false.
 */
const BPExplanationOverview: React.FC<BPExplanationProps> = ({
    jobInfo,
    show,
    predictiveQueryType,
    queryYaml,
    showGroundTruth
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
    const {
        filteredEntityIdCategories: entityIdCategories,
        filteredCategories: categories,
        isLoading: isLoadingCategories
    } = useBpEntityCategories(jobInfo?.id, jobInfo?.status);

    const categoryKeys = useMemo(() => getKeys(categories), [categories]);
    const entityIdCategoryKeys = useMemo(
        () => getKeys(entityIdCategories),
        [entityIdCategories]
    );

    const {
        entityOutput,
        historicalItems,
        subgraph,
        isLoading: isDataLoading
    } = useBpEntityData(
        jobInfo?.id,
        entityId,
        predictiveQueryType,
        jobInfo?.status
    );
    const [anchorTime, setAnchorTime] = useState<string>("");
    const [predictionAnchorTime, setPredictionAnchorTime] =
        useState<string>("");

    const featureFlags = useFeatureFlags();

    // Set the category to the first category in the list of categories
    useEffect(() => {
        if (
            category === "" ||
            (categories && categories[category] === undefined)
        ) {
            if (categoryKeys.length > 0) {
                setCategory(categoryKeys[0]);
            }
        }

        if (
            entityIdCategory === "" ||
            (entityIdCategories && entityIdCategories[category] === undefined)
        ) {
            if (entityIdCategoryKeys.length > 0) {
                setEntityIdCategory(entityIdCategoryKeys[0]);
            }
        }
    }, [
        entityIdCategories,
        entityIdCategory,
        categories,
        category,
        categoryKeys,
        entityIdCategoryKeys
    ]);

    // Set the entityId to the first entity in the list of entities
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

    const xaiDisabledForTaskType =
        isLinkPredictionType(predictiveQueryType) ||
        predictiveQueryType === TaskType.MULTILABEL_RANKING;

    const msg = `There are no entities to explore available for this query.
                    This may be due to an unsupported task type or specific
                    model configuration, for which we do not provide any
                    entities to explore yet. If you believe this is an error,${" "}
                    ${SUPPORT_CONTACT_MESSAGE}.`;
    return (
        <div className={cn("min-h-220", show ? "block" : "hidden")}>
            <TabSwitcher
                tabs={[
                    { label: "Global column explanations", value: "global" },
                    { label: "Local subgraph explanation", value: "local" }
                ]}
                value={activeTab}
                onValueChange={handleToggle}
            />
            {isLoadingCategories && <Skeleton className="w-120 h-20 my-6" />}
            {!isLoadingCategories && categoryKeys.length === 0 && (
                <>
                    {activeTab === "global" && (
                        <ColumnAnalysisSection
                            featureFlagEnabled={
                                featureFlags.ENABLE_COLUMN_ANALYSIS ?? false
                            }
                            xaiDisabledForTaskType={xaiDisabledForTaskType}
                            show={show}
                            predictiveQueryType={predictiveQueryType}
                            jobId={jobInfo?.id}
                            setXaiData={setXaiData}
                            activeSubGraph={activeSubGraph}
                            predictionAnchorTime={predictionAnchorTime}
                        />
                    )}
                    {activeTab === "local" && (
                        <div className={cn("my-6", show ? "block" : "hidden")}>
                            <Alert variant="default" enableClose={false}>
                                <AlertDescription>{msg}</AlertDescription>
                            </Alert>
                        </div>
                    )}
                </>
            )}

            {!isLoadingCategories && categoryKeys.length > 0 && (
                <>
                    {activeTab === "global" && (
                        <ColumnAnalysisSection
                            featureFlagEnabled={
                                featureFlags.ENABLE_COLUMN_ANALYSIS ?? false
                            }
                            xaiDisabledForTaskType={xaiDisabledForTaskType}
                            show={show}
                            predictiveQueryType={predictiveQueryType}
                            jobId={jobInfo?.id}
                            setXaiData={setXaiData}
                            activeSubGraph={activeSubGraph}
                            predictionAnchorTime={predictionAnchorTime}
                        />
                    )}
                    {activeTab === "local" && (
                        <div className={"mt-6"}>
                            <EntityLevel
                                categories={categories}
                                category={category}
                                selectedEntityIndex={selectedEntityIndex}
                                setSelectedEntityIndex={setSelectedEntityIndex}
                                entityId={entityId}
                                originalEntityId={originalEntityId}
                                showCategories={false}
                                entityIdentifier={
                                    subgraph?.subgraph_store?.nodes?.[
                                        String(
                                            subgraph?.subgraph_store?.root_id
                                        )
                                    ]?.pkey_name ?? undefined
                                }
                            />
                            <Spacer height={24} />
                            <Card className="p-0 gap-0 mb-6">
                                <CardHeader
                                    className="px-3 py-3 h-12 rounded-none justify-center"
                                    bottomSeparator={true}>
                                    <Title
                                        variant="section-header"
                                        children={"Sample details"}
                                        className="font-bold text-title-primary"
                                    />
                                </CardHeader>
                                <CardContent className="px-3 py-3">
                                    <Prediction
                                        entityOutput={
                                            entityOutput as EntityOutput
                                        }
                                        taskType={predictiveQueryType}
                                        historicalItems={
                                            historicalItems as HistoricalItems
                                        }
                                        isDataLoading={isDataLoading}
                                        query={queryYaml}
                                        anchorTime={anchorTime}
                                        showGroundTruth={showGroundTruth}
                                        selectedColumns={
                                            selectedPredictionColumns
                                        }
                                        setSelectedColumns={
                                            setSelectedPredictionColumns
                                        }
                                    />
                                </CardContent>
                            </Card>
                            <SubgraphTreeView
                                key={entityId}
                                subgraph={subgraph}
                                xaiData={xaiData as GraphXaiScores}
                                anchorTime={anchorTime}
                                setActiveSubGraph={setActiveSubGraph}
                                entityId={entityId}
                                taskType={predictiveQueryType}
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
                </>
            )}
        </div>
    );
};

export default WithErrorBoundary(BPExplanationOverview);
