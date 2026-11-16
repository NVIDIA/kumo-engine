// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useState } from "react";
import Spacer from "react-spacer";

import {
    Card,
    CardContent,
    Description,
    GridCol,
    GridRow,
    QueryCard
} from "@kumo-ai/kumo-ui-catalog";
import { GenericJob } from "gen/v0/types";
import { ErrorDetail, QueryType } from "gen/v1/types";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";
import PQCodeEditor from "components/CodeEditor/PQCodeEditor";
import JobOverview from "components/Jobs/Overview/JobOverview";

import useFluidCardHeight from "hooks/utils/useFluidCardHeight";

import { HighLightText } from "types/common";
import { InputState } from "types/inputs";
import { JobOverviewType } from "types/jobs";
import { CreatePredictionJobArgs } from "types/predictionJob";
import { PredictiveQueryInfo, TaskType } from "types/predictiveQueries";

import { formatQueryForTooltip, getIdHash } from "utils/Helpers/stringHelpers";

import PredictionSchema from "../Schema/PredictionSchema";
import PredictionSettings from "../Settings/PredictionSettings";

export interface PredictionConfigProps {
    selectedTrainingJob: (GenericJob & PredictiveQueryInfo) | null;
    setPredictionJobArgs: React.Dispatch<
        React.SetStateAction<CreatePredictionJobArgs | null>
    >;
    setError: React.Dispatch<React.SetStateAction<ErrorDetail[]>>;
    setFilterErrors: React.Dispatch<React.SetStateAction<ErrorDetail[]>>;
    setValidationState: React.Dispatch<React.SetStateAction<InputState>>;
    setFilterValidState: React.Dispatch<React.SetStateAction<InputState>>;
    error: ErrorDetail[];
}

const PredictionConfig: React.FC<PredictionConfigProps> = ({
    selectedTrainingJob,
    setPredictionJobArgs,
    setError,
    setFilterErrors,
    setValidationState,
    setFilterValidState,
    error
}) => {
    const { headerRef, cardHeight } = useFluidCardHeight<HTMLDivElement>();

    const [showTrainingJobOverview, setShowTrainingJobOverview] =
        useState(false);
    const [highLightQuery, setHighLightQuery] = useState<HighLightText>({
        text: selectedTrainingJob?.queryYaml as string,
        position: [
            { start: 0, end: 0 },
            { start: 0, end: 0 }
        ]
    });

    const handleViewTrainingJob = () => {
        setShowTrainingJobOverview(true);
    };

    const handleCloseViewTrainingJob = () => {
        setShowTrainingJobOverview(false);
    };

    return (
        <>
            <Card ref={headerRef} className="border-border-secondary p-4">
                <CardContent className="p-0">
                    <GridRow className="items-center">
                        <GridCol xs={12} sm={12} md={6}>
                            <span className="text-base font-bold text-text-primary leading-4">
                                Training Job{" "}
                                <span
                                    onClick={handleViewTrainingJob}
                                    className="underline decoration-solid cursor-pointer">
                                    {getIdHash(selectedTrainingJob?.id)}
                                </span>
                            </span>
                        </GridCol>
                    </GridRow>
                    <Spacer height={8} />
                    <Description
                        variant="secondary"
                        className="font-medium text-sm leading-4">
                        {"Query"}
                    </Description>
                    <Spacer height={4} />
                    <QueryCard
                        id="query-box"
                        data-testid={"pquery-card"}
                        footerClassName={"border-t-0 flex:grow"}
                        contentClassName="p-0 m-0"
                        className="py-4 px-2 border-border-secondary rounded-sm"
                        content={
                            highLightQuery?.text ? (
                                <PQCodeEditor
                                    language="PQLANG_V2"
                                    value={formatQueryForTooltip(
                                        highLightQuery?.text
                                    )}
                                    onChange={() => {}}
                                    editable={false}
                                    enableHighlight={true}
                                    highlightPositions={
                                        highLightQuery?.position
                                    }
                                    hideBorder={true}
                                    height="fit-content"
                                    lineNumbers={false}
                                    highlightActiveLineGutter={false}
                                    highlightActiveLine={false}
                                />
                            ) : null
                        }
                        copyEnabled={false}
                    />
                </CardContent>
            </Card>
            <Spacer height={16} />
            <Card
                data-screenshot-ready="new-prediction-config-layout"
                data-testid="new-prediction-config-layout"
                className="border-border-secondary py-0 overflow-hidden"
                style={cardHeight ? { height: cardHeight } : undefined}>
                <CardContent className="p-0 h-full">
                    <GridRow
                        data-testid="new-prediction-settings-schema-layout"
                        className="px-0 gap-0 h-full">
                        <PredictionSettings
                            selectedTrainingJob={selectedTrainingJob}
                            setPredictionJobArgs={setPredictionJobArgs}
                            setError={setError}
                            setFilterErrors={setFilterErrors}
                            setValidationState={setValidationState}
                            setFilterValidState={setFilterValidState}
                            setHighLightQuery={setHighLightQuery}
                            error={error}
                        />
                        <PredictionSchema
                            queryId={selectedTrainingJob?.queryId as string}
                            taskType={selectedTrainingJob?.taskType as TaskType}
                            queryType={
                                (selectedTrainingJob?.queryType as QueryType) ??
                                undefined
                            }
                            trainingTableId={
                                selectedTrainingJob?.trainTableJobId
                            }
                            graphSnapshotId={
                                selectedTrainingJob?.graphSnapshotId as string
                            }
                        />
                    </GridRow>
                </CardContent>
            </Card>
            {showTrainingJobOverview && (
                <JobOverview
                    jobId={selectedTrainingJob?.id as string}
                    jobOverviewType={JobOverviewType.TRAINING_JOB}
                    onClose={() => {
                        handleCloseViewTrainingJob();
                    }}
                />
            )}
        </>
    );
};

export default WithErrorBoundary(PredictionConfig);
