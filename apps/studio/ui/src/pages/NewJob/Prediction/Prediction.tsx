// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useState } from "react";
import { useLocation } from "react-router";
import Spacer from "react-spacer";

import {
    Button,
    ChevronRight,
    Description,
    GridCol,
    GridRow,
    StartJobPopover,
    Title
} from "@kumo-ai/kumo-ui-catalog";
import { GenericJob } from "gen/v0/types";
import { ErrorDetail } from "gen/v1/types";

import CancelAlert from "components/NewJob/CancelAlert";
import PredictionConfig from "components/NewJob/Prediction/Config/PredictionConfig";
import TrainingJobSelectList from "components/NewJob/Prediction/Select/TrainingJobSelectList";
import Validation from "components/Validation/Validation";
import ValidationStatus from "components/Validation/ValidationStatus";

import { useCreateBatchPredictionJob } from "hooks/jobs/prediction/prediction";
import { useTrainingJobWorkflow } from "hooks/jobs/training/trainingJob";
import { useProjectId } from "hooks/project/projects";
import { useValidationState } from "hooks/validation/validation";

import { InputState } from "types/inputs";
import { CreatePredictionJobArgs } from "types/predictionJob";
import { PredictiveQueryInfo } from "types/predictiveQueries";

import { getProjectPath } from "utils/Constants";
import { isObjectEmpty } from "utils/Helpers/objectHelpers";
import { getIdHash } from "utils/Helpers/stringHelpers";
import { useCustomNavigation } from "utils/navigator";

const NewPredictionJob = () => {
    const navigate = useCustomNavigation();
    const [loading, setLoading] = useState<boolean>(false);
    const projectId = useProjectId();

    const [banner, alert, handleError, clearErrors, dialogRef] =
        useValidationState(setLoading);

    const [hasErrors, setError] = useState<ErrorDetail[]>([]);
    //Entity-Target filter errors
    const [hasFilterErrors, setFilterErrors] = useState<ErrorDetail[]>([]);

    const [validationState, setValidationState] = useState<InputState>(
        InputState.None
    );
    const [filterValidState, setFilterValidState] = useState<InputState>(
        InputState.None
    );
    const location = useLocation();

    const [createPredictionJobArgs, setPredictionJobArgs] =
        useState<CreatePredictionJobArgs | null>(null);

    const queryInfoFromTraining = location?.state?.queryInfo;

    const [trainingJob, setTrainingJob] = useState<
        (GenericJob & PredictiveQueryInfo) | null
    >(queryInfoFromTraining ?? null);

    const [openPopover, setOpenPopover] = useState(false);
    const [isEmptyList, setIsEmptyList] = useState(false);
    const projectPaths = getProjectPath(projectId);
    const [showCancelDialog, setShowCancelDialog] = useState(false);

    const isDirty = trainingJob !== null;

    const onCancelClick = useCallback(() => {
        if (isDirty) {
            setShowCancelDialog(true);
        } else {
            navigate(
                `${projectPaths.PredictionJobsList}?page=1&job_type=BATCH_PREDICTION_JOB`,
                { replace: true }
            );
        }
    }, [isDirty, navigate, projectPaths.PredictionJobsList]);

    const { mutateAsync: createBatchPredictionJob } =
        useCreateBatchPredictionJob(
            handleError,
            setError,
            setLoading,
            setValidationState,
            projectId
        );
    const {
        queryInfo: workflowQueryInfo,
        isLoading: isWorkflowLoading,
        isError: isWorkflowError,
        error: workflowError
    } = useTrainingJobWorkflow({
        id: trainingJob?.job_id ?? "",
        handleError,
        setPageLoading: setLoading
    });

    useEffect(() => {
        if (isWorkflowLoading) {
            setLoading(true);
        }
        if (isWorkflowError) {
            console.warn(
                "Failed to fetch full training job details",
                workflowError
            );
        }
    }, [isWorkflowError, isWorkflowLoading, workflowError]);

    const startPredicting = useCallback(
        async (tags: Record<string, string>) => {
            if (hasErrors?.length > 0) {
                setValidationState(InputState.ValidationError);
                return;
            }
            if (!projectId) {
                handleError(
                    new Error("A project must be selected."),
                    "Unable to create prediction"
                );
                return;
            }
            setLoading(true);
            setError([]);
            setValidationState(InputState.Valid);

            const requestPayload: CreatePredictionJobArgs = {
                ...createPredictionJobArgs,
                model_training_job_id:
                    createPredictionJobArgs?.model_training_job_id ?? "",
                predict_options: createPredictionJobArgs?.predict_options ?? {},
                job_tags: {
                    ...(createPredictionJobArgs?.job_tags ?? {}),
                    ...tags
                },
                project_id: projectId
            };

            try {
                await createBatchPredictionJob({ data: requestPayload });
            } catch (error) {
                // Error is already handled by onError callback
                // This catch prevents unhandled rejection
                console.error("Batch prediction job creation failed:", error);
            } finally {
                setLoading(false);
            }
        },
        [
            hasErrors?.length,
            createPredictionJobArgs,
            createBatchPredictionJob,
            handleError,
            projectId
        ]
    );

    return (
        <>
            <CancelAlert
                show={showCancelDialog}
                title="Discard this prediction?"
                description={`Your prediction settings for "${getIdHash(trainingJob?.job_id ? trainingJob?.job_id : trainingJob?.id)}" will not be saved.`}
                handleClose={() => setShowCancelDialog(false)}
                handleConfirm={() => {
                    navigate(
                        `${projectPaths.PredictionJobsList}?page=1&job_type=BATCH_PREDICTION_JOB`,
                        { replace: true }
                    );
                }}
            />
            <GridRow
                className="flex justify-between items-center"
                data-screenshot-ready={
                    loading || isWorkflowLoading
                        ? undefined
                        : "new-prediction-page"
                }>
                <GridCol xs={12} sm={12} md={6}>
                    <Title
                        data-testid="new-prediction-page-heading"
                        variant="page-heading"
                        className="text-3xl">
                        New Prediction
                    </Title>
                    {!isEmptyList && (
                        <Description
                            size="md"
                            className="leading-5 mt-2 text-title-secondary">
                            {!trainingJob
                                ? "Select a training job to generate predictions on the latest data"
                                : undefined}
                        </Description>
                    )}
                </GridCol>

                {!isEmptyList && (
                    <GridCol xs={12} sm={12} md={6}>
                        <GridRow
                            xs={12}
                            sm={12}
                            md="auto"
                            className="flex flex-wrap justify-end items-center gap-1">
                            {trainingJob &&
                                (filterValidState === InputState.Validating ||
                                filterValidState === InputState.Warning ||
                                filterValidState === InputState.Valid ||
                                filterValidState === InputState.InValid ? (
                                    <GridCol
                                        xs={12}
                                        sm={12}
                                        md="auto"
                                        className="p-1">
                                        <ValidationStatus
                                            state={filterValidState}
                                            errorInfo={
                                                hasFilterErrors as ErrorDetail[]
                                            }
                                        />
                                    </GridCol>
                                ) : (
                                    (validationState === InputState.InValid ||
                                        validationState ===
                                            InputState.Valid) && (
                                        <GridCol
                                            xs={12}
                                            sm={12}
                                            md="auto"
                                            className="p-1">
                                            <ValidationStatus
                                                state={validationState}
                                                errorInfo={
                                                    hasErrors as ErrorDetail[]
                                                }
                                            />
                                        </GridCol>
                                    )
                                ))}
                            <GridCol xs={12} sm={12} md="auto">
                                <div className="flex items-center gap-2">
                                    {trainingJob && (
                                        <Button
                                            id="back-job"
                                            variant="outline"
                                            onClick={() => {
                                                setTrainingJob(null);
                                            }}>
                                            Back
                                        </Button>
                                    )}
                                    <Button
                                        id="cancel-job"
                                        variant="outline"
                                        onClick={onCancelClick}>
                                        Cancel
                                    </Button>
                                    {trainingJob && (
                                        <>
                                            <StartJobPopover
                                                showPopover={openPopover}
                                                setShowPopover={setOpenPopover}
                                                startButtonTitle={
                                                    "Start predicting "
                                                }
                                                jobsSectionTitle="Prediction jobs"
                                                headerLabel={
                                                    "The following jobs will be created when you start predicting."
                                                }
                                                tagHeader="Tags"
                                                tagDescription={
                                                    "Add key-value tags to organize and filter this prediction job. Tags will be applied to both jobs."
                                                }
                                                jobs={[
                                                    "Prediction Job",
                                                    "Prediction Table Generation"
                                                ]}
                                                startTraining={startPredicting}
                                                disablePopover={
                                                    !(
                                                        validationState ===
                                                            InputState.Valid &&
                                                        (filterValidState ===
                                                            InputState.None ||
                                                            filterValidState ===
                                                                InputState.Warning ||
                                                            filterValidState ===
                                                                InputState.Valid)
                                                    ) ||
                                                    !createPredictionJobArgs ||
                                                    isObjectEmpty(
                                                        createPredictionJobArgs
                                                    )
                                                }>
                                                <div role="alert">
                                                    <Button
                                                        id="start-predicting"
                                                        onClick={() => {
                                                            setOpenPopover(
                                                                !openPopover
                                                            );
                                                        }}
                                                        disabled={
                                                            !(
                                                                validationState ===
                                                                    InputState.Valid &&
                                                                (filterValidState ===
                                                                    InputState.None ||
                                                                    filterValidState ===
                                                                        InputState.Warning ||
                                                                    filterValidState ===
                                                                        InputState.Valid)
                                                            ) ||
                                                            !createPredictionJobArgs ||
                                                            isObjectEmpty(
                                                                createPredictionJobArgs
                                                            )
                                                        }>
                                                        {"Start predicting "}
                                                        <ChevronRight className="fill-white" />
                                                    </Button>
                                                </div>
                                            </StartJobPopover>
                                        </>
                                    )}
                                </div>
                            </GridCol>
                        </GridRow>
                    </GridCol>
                )}
            </GridRow>
            <Spacer height={trainingJob ? 16 : 24} />
            <Validation
                pageLoading={loading}
                banner={banner}
                alert={alert}
                ctaMsgs={null}
                dialogRef={dialogRef}
                hideWarnings={() => {}}
                clearErrors={clearErrors}
            />
            {trainingJob ? (
                <PredictionConfig
                    selectedTrainingJob={{
                        ...trainingJob,
                        ...(workflowQueryInfo ?? {})
                    }}
                    setPredictionJobArgs={setPredictionJobArgs}
                    setError={setError}
                    setFilterErrors={setFilterErrors}
                    error={hasErrors}
                    setValidationState={setValidationState}
                    setFilterValidState={setFilterValidState}
                />
            ) : (
                <TrainingJobSelectList
                    setNext={setTrainingJob}
                    projectId={projectId ?? undefined}
                    onEmptyChange={setIsEmptyList}
                />
            )}
        </>
    );
};

export default NewPredictionJob;
