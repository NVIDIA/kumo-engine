// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useState } from "react";

import { Block, Button } from "@kumo-ai/kumo-ui-catalog";
import { useQueryClient } from "@tanstack/react-query";
import { ErrWarnDetails, JobType, QueryType } from "gen/v0/types";

import { TrainingJobOverviewDetail } from "components/Jobs/TrainingTable/TrainingTableGenOverview";
import Validation from "components/Validation/Validation";

import { useManageJob } from "hooks/jobs/list/jobList";
import {
    useGetQueryWarnings,
    useQueryDefinition,
    useQueryMetaData
} from "hooks/jobs/training/query";
import { useTrainingJobWorkflow } from "hooks/jobs/training/trainingJob";
import {
    useCancelTrainTableJob,
    useTrainingTableDetails
} from "hooks/jobs/trainTableGen/trainTableGen";
import { useProjectId } from "hooks/project/projects";
import { useValidationState } from "hooks/validation/validation";

import { CustomError } from "types/common";
import { PredictiveQueryInfo, TaskType } from "types/predictiveQueries";
import { TrainTableGeneration } from "types/training_job";
import { CTAInfo } from "types/warnings";

import { jobStatus, JobStatusType } from "utils/Constants";
import { parseTrainTableGenResponse } from "utils/Converters/jobsConverter";
import { isValid } from "utils/Helpers/typeGuards";
import {
    invalidateActiveJobs,
    invalidateJobQueries,
    invalidateJobsList,
    invalidateSnapshotsGraph
} from "utils/queryInvalidation";

export interface TrainingTableGenerationProps {
    id?: string;
    onClose: () => void;
    onArchive?: () => void;
    onRecover?: () => void;
    hidePreTitle?: boolean;
    showRelatedJobs?: boolean;
    onTableStatusUpdate?: (status: jobStatus) => void;
    modelPlan?: string | null;
    trainingPlan?: string | null;
    isTrainingPage?: boolean;
    queryInfo?: PredictiveQueryInfo;
    setWarnings?: (warning: ErrWarnDetails[]) => void;
}

const TrainingTableGeneration = ({
    id,
    onClose,
    onArchive,
    onRecover,
    hidePreTitle = false,
    showRelatedJobs = true,
    onTableStatusUpdate,
    modelPlan,
    trainingPlan,
    isTrainingPage = false,
    queryInfo,
    setWarnings
}: TrainingTableGenerationProps) => {
    // Page State
    const [pageLoading, setPageLoading] = useState(true);
    const [trainTableDetails, setTrainTableDetails] =
        useState<TrainTableGeneration | null>(null);
    const projectId = useProjectId();
    const hideMutationControls = !projectId;

    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    const [banner, alert, handleError, clearErrors, dialogRef, openDialog] =
        useValidationState(setPageLoading);

    const {
        onClickArchive,
        onClickRecover,
        onConfirm,
        setIsArchived,
        isManagingJob,
        isArchived
    } = useManageJob({
        jobId: id as string,
        jobType: JobType.GENERATE_TRAIN_TABLE_JOB,
        jobName: id,
        onArchive: () => onArchive?.(),
        onRecover: () => onRecover?.(),
        handleError,
        clearErrors,
        dialogRef,
        setPageLoading
    });

    const [queryId, setQueryId] = useState<string | null>(null);
    const [trainingJobId, setTrainingJobId] = useState<string | null>(null);
    // If `trainingJobId` or `queryInfo` then no need to invoke `resource` API,
    // set queryYaml from `workflow` API or `queryInfo`

    const { queryYaml, setQueryYaml } = useQueryDefinition(
        trainingJobId || queryInfo ? null : queryId
    );

    const [taskType, setTaskType] = useState<TaskType>(TaskType.NONE);
    const [queryType, setQueryType] = useState<QueryType>(QueryType.STATIC);
    // If `trainingJobId` or `queryInfo` then no need to invoke `metadata` API,
    // set taskType &  queryType from `workflow` API or `queryInfo`
    const { taskType: taskTypeFromMetadata, queryType: queryTypeFromMetadata } =
        useQueryMetaData(trainingJobId || queryInfo ? null : queryId);

    // Needs to be here due to dependencies.
    // Warnings live on the parent training job (`trainingJobId`, resolved
    // from the train-table job's `parent_job` tag); the page's own
    // gen-traintable `id` is rejected with a 404 by the training-job
    // warnings route (kumo-ai/kumo#30119).
    const [ctaMsgs, hideWarnings] = useGetQueryWarnings(
        trainingJobId,
        handleError,
        trainTableDetails?.jobStatusReport?.status
    );
    const [graphId, setGraphId] = useState<string | null>(null);
    const [userMetadata, setUserMetadata] = useState<string | null>(null);
    const [prevStatus, setPrevStatus] = useState<JobStatusType>(jobStatus.None);

    const queryClient = useQueryClient();

    // Invoking Query details API to get Predictive query information
    const {
        data: trainTableGenerationResponse,
        isLoading,
        isError,
        error
    } = useTrainingTableDetails(id);

    useEffect(() => {
        if (isLoading) {
            setPageLoading(true);
            return;
        }
        if (isError) {
            invalidateJobsList(queryClient);
            setPageLoading(false);
            handleError(error as CustomError);
            return;
        }

        if (trainTableGenerationResponse?.data) {
            const parsed = parseTrainTableGenResponse(
                trainTableGenerationResponse.data
            );
            setTrainTableDetails(parsed);
            setIsArchived(parsed.archived);

            const tags = trainTableGenerationResponse.data.resource?.tags;
            if (tags?.["pquery_id"]) {
                setQueryId(tags["pquery_id"]);
            }
            if (tags?.["parent_job"]) {
                setTrainingJobId(tags["parent_job"]);
            }

            const trainingStatus =
                trainTableGenerationResponse.data.resource?.job_status_report
                    ?.status;
            if (
                prevStatus !== jobStatus.None &&
                prevStatus !== trainingStatus
            ) {
                invalidateJobsList(queryClient);
                const id = queryInfo?.graphSnapshotId || parsed.graphSnapshotId;
                if (isValid(id)) {
                    invalidateSnapshotsGraph(queryClient, id as string);
                }
            }
            if (
                prevStatus === jobStatus.None &&
                (trainingStatus === jobStatus.Queued ||
                    trainingStatus === jobStatus.Running)
            ) {
                // Invalidate specific job cache on status change
                // fresh data is fetched and UI updates immediately
                // for all transitions: Queued->Running, Running->Done, etc.
                invalidateJobQueries(
                    queryClient,
                    JobType.GENERATE_TRAIN_TABLE_JOB,
                    id as string
                );
            }

            setPrevStatus(trainingStatus as jobStatus);
            if (onTableStatusUpdate && trainingStatus) {
                onTableStatusUpdate(trainingStatus as jobStatus);
            }
            if (parsed.user) {
                setUserMetadata(parsed?.user);
            }

            setPageLoading(false);
        }
    }, [
        isLoading,
        isError,
        error,
        setQueryId,
        setTrainTableDetails,
        setTrainingJobId,
        handleError,
        setPageLoading,
        trainTableGenerationResponse,
        onTableStatusUpdate,
        setIsArchived,
        prevStatus,
        queryClient,
        queryInfo?.graphSnapshotId,
        id
    ]);

    const { queryInfo: queryResponse } = useTrainingJobWorkflow({
        id: trainingJobId ? (trainingJobId as string) : "",
        setPageLoading,
        handleError
    });

    useEffect(() => {
        if (queryResponse) {
            setGraphId(queryResponse.graphName as string);
            setTaskType((queryResponse.taskType as TaskType) ?? TaskType.NONE);
            setQueryType(
                (queryResponse.queryType as QueryType) ?? QueryType.STATIC
            );
            setQueryYaml(queryResponse.queryYaml);
        } else if (queryTypeFromMetadata && taskTypeFromMetadata) {
            setTaskType(taskTypeFromMetadata ?? TaskType.NONE);
            setQueryType(queryTypeFromMetadata ?? QueryType.STATIC);
        } else if (queryInfo) {
            setQueryYaml(queryInfo.queryYaml);
            setTaskType(queryInfo.taskType ?? TaskType.NONE);
            setQueryType(queryInfo.queryType ?? QueryType.STATIC);
        }
    }, [
        queryResponse,
        setQueryYaml,
        queryTypeFromMetadata,
        taskTypeFromMetadata,
        queryInfo
    ]);

    const onAbortTrainClick = () => {
        dialogRef?.current?.toggleDialog(
            "Cancel Training Job?",
            "Stopping now will discard all progress on this job. This cannot be undone.",
            undefined,
            false,
            "Cancel job",
            "destructive"
        );
    };

    const onAbortTrainConfirm = () => {
        dialogRef?.current?.toggleDialog();
        abortTraining();
    };

    const { cancelTrainTable } = useCancelTrainTableJob({
        jobId: id as string,
        trainingJobId,
        onSuccess: () => {
            invalidateActiveJobs(queryClient);
        },
        handleError
    });

    /**
     * cancel an ongoing train table generation hook
     */
    const abortTraining = () => {
        setPageLoading(true);
        cancelTrainTable({ id: id as string });
    };

    const buttons = [];
    if (
        !hideMutationControls &&
        (trainTableDetails?.status === jobStatus.Running ||
            trainTableDetails?.status === jobStatus.Queued)
    ) {
        buttons.push(
            <Button
                id={"cancel_train_table"}
                onClick={onAbortTrainClick}
                key="cancel_train_table">
                <>
                    <Block className="size-5" />
                    {" Cancel this job"}
                </>
            </Button>
        );
    }

    return (
        <>
            <TrainingJobOverviewDetail
                title={id as string}
                buttons={buttons}
                trainTableDetails={trainTableDetails}
                ctaMsgs={ctaMsgs as CTAInfo}
                loadingStatus={pageLoading}
                onClose={onClose}
                queryYaml={queryYaml}
                taskType={taskType}
                queryType={queryType}
                hidePreTitle={hidePreTitle}
                showRelatedJobs={showRelatedJobs}
                graphId={graphId}
                modelPlan={modelPlan as string}
                trainingPlan={trainingPlan as string}
                isTrainingPage={isTrainingPage}
                onClickArchive={onClickArchive}
                onClickRecover={onClickRecover}
                isArchived={isArchived}
                user={userMetadata}
                setWarnings={setWarnings}
            />
            <Validation
                pageLoading={false}
                banner={banner}
                alert={alert}
                ctaMsgs={null}
                dialogRef={dialogRef}
                hideWarnings={hideWarnings}
                onDialogConfirm={
                    isManagingJob ? onConfirm : onAbortTrainConfirm
                }
                clearErrors={clearErrors}
            />
        </>
    );
};
export default TrainingTableGeneration;
