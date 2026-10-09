// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useState } from "react";

import { Block, Button } from "@kumo-ai/kumo-ui-catalog";
import { useQueryClient } from "@tanstack/react-query";
import { ErrWarnDetails, JobType } from "gen/v0/types";

import Validation from "components/Validation/Validation";

import { useManageJob } from "hooks/jobs/list/jobList";
import { useBatchPredictionSummary } from "hooks/jobs/prediction/prediction";
import {
    useCancelGenpredtableJobs,
    useGenpredtableAnchorTime,
    useGenPredTableSummary
} from "hooks/jobs/predictionTableGen/predictionTableGen";
import {
    useQueryDefinition,
    useQueryMetaData
} from "hooks/jobs/training/query";
import { useProjectId } from "hooks/project/projects";
import { useValidationState } from "hooks/validation/validation";

import { jobStatus } from "utils/Constants";
import { invalidateActiveJobs } from "utils/queryInvalidation";

import PredictionTableGenOverview from "./PredictionTableGenOverview";

export interface PredictionTableOverviewProps {
    id?: string;
    onClose: () => void;
    onArchive?: () => void;
    onRecover?: () => void;
    hidePreTitle?: boolean;
    showRelatedJobs?: boolean;
    setStatus?: (status: jobStatus) => void;
    jobSource?: string;
    queryYaml?: string | null;
    setWarnings?: (warning: ErrWarnDetails[]) => void;
}

const PredictionTableGeneration = ({
    id,
    onClose,
    onArchive,
    onRecover,
    hidePreTitle = false,
    showRelatedJobs = true,
    setStatus,
    jobSource,
    queryYaml: queryYamlFromProps,
    setWarnings
}: PredictionTableOverviewProps) => {
    const [pageLoading, setPageLoading] = useState(true);
    const queryClient = useQueryClient();
    const projectId = useProjectId();
    const hideMutationControls = !projectId;

    const [banner, , handleError, clearErrors, dialogRef] =
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
        jobType: JobType.GENERATE_PREDICTION_TABLE_JOB,
        jobName: id,
        onArchive: () => onArchive?.(),
        onRecover: () => onRecover?.(),
        handleError,
        clearErrors,
        dialogRef,
        setPageLoading
    });

    const { predictionTableDetails, bpJobId, queryId, isDataLoading } =
        useGenPredTableSummary(
            id as string,
            handleError,
            setPageLoading,
            setStatus,
            setIsArchived
        );

    const { queryYaml } = useQueryDefinition(queryId, queryYamlFromProps);

    const { taskType } = useQueryMetaData(queryId);

    const { anchorTime } = useGenpredtableAnchorTime(
        id as string,
        predictionTableDetails,
        handleError,
        setPageLoading
    );
    const { graphId } = useBatchPredictionSummary(
        bpJobId as string,
        handleError,
        setPageLoading
    );

    const onAbortPredTableGenClick = () => {
        dialogRef?.current?.toggleDialog(
            "Cancel Prediction Job?",
            "Stopping now will discard all progress on this job. This cannot be undone.",
            undefined,
            false,
            "Cancel job",
            "destructive"
        );
    };

    const onAbortPredTableGenConfirm = () => {
        dialogRef?.current?.toggleDialog();
        abortPredTableGen();
    };

    /**
     *  This method invokes REST API call to Cancel an ongoing Prediction Table Generation.
     */

    const { cancelGenpredtableJobs } = useCancelGenpredtableJobs({
        bpId: bpJobId as string,
        id: id as string,
        onSuccess: () => {
            invalidateActiveJobs(queryClient);
        },
        handleError
    });

    const abortPredTableGen = () => {
        setPageLoading(true);
        cancelGenpredtableJobs({ id: id as string });
    };

    const buttons = [];
    if (
        !hideMutationControls &&
        (predictionTableDetails?.status === jobStatus.Running ||
            predictionTableDetails?.status === jobStatus.Queued)
    ) {
        buttons.push(
            <Button
                id={"cancel_bp_table"}
                onClick={onAbortPredTableGenClick}
                key="cancel_bp_table">
                <>
                    <Block className="size-5" />
                    {" Cancel this job"}
                </>
            </Button>
        );
    }

    return (
        <>
            <PredictionTableGenOverview
                title={id as string}
                buttons={buttons}
                predictionTableDetails={predictionTableDetails}
                loadingStatus={isDataLoading || pageLoading}
                onClose={onClose}
                queryYaml={queryYaml || (queryYamlFromProps ?? "")}
                taskType={taskType}
                anchorTime={anchorTime}
                hidePreTitle={hidePreTitle}
                showRelatedJobs={showRelatedJobs}
                graphId={graphId}
                parentJobSource={jobSource}
                onClickArchive={onClickArchive}
                onClickRecover={onClickRecover}
                isArchived={isArchived}
                user={predictionTableDetails?.user}
                setWarnings={setWarnings}
            />
            <Validation
                pageLoading={false}
                banner={banner}
                alert={null}
                ctaMsgs={null}
                dialogRef={dialogRef}
                onDialogConfirm={
                    isManagingJob ? onConfirm : onAbortPredTableGenConfirm
                }
                clearErrors={clearErrors}
            />
        </>
    );
};
export default PredictionTableGeneration;
