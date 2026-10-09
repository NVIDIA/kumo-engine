// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useState } from "react";

import { Block, Button } from "@kumo-ai/kumo-ui-catalog";
import { useQueryClient } from "@tanstack/react-query";
import { JobType } from "gen/v0/types";

import TrainingJobOverviewDetail from "components/Jobs/Training/TrainingJobOverview";
import Validation from "components/Validation/Validation";

import { useManageJob } from "hooks/jobs/list/jobList";
import { useGetQueryWarnings } from "hooks/jobs/training/query";
import {
    useCancelTrainingJob,
    useTrainingJobWorkflow
} from "hooks/jobs/training/trainingJob";
import { useProjectId } from "hooks/project/projects";
import { useValidationState } from "hooks/validation/validation";

import { PredictiveQueryInfo } from "types/predictiveQueries";
import { CTAInfo } from "types/warnings";

import { jobStatus } from "utils/Constants";
import { invalidateActiveJobs } from "utils/queryInvalidation";

export interface TrainingJobOverviewProps {
    id?: string;
    hidePreTitle?: boolean;
    showRelatedJobs?: boolean;
    onClose?: () => void;
    setStatus?: (status: jobStatus) => void;
    onArchive?: () => void;
    onRecover?: () => void;
}

const TrainingJob = ({
    id,
    hidePreTitle = false,
    showRelatedJobs = true,
    onClose,
    setStatus,
    onArchive,
    onRecover
}: TrainingJobOverviewProps) => {
    // Page State
    const [pageLoading, setPageLoading] = useState(true);
    const [queryInfo, setQueryInfo] = useState<PredictiveQueryInfo | null>(
        null
    );
    const queryClient = useQueryClient();
    const projectId = useProjectId();
    const hideMutationControls = !projectId;

    const [banner, alert, handleError, clearErrors, dialogRef] =
        useValidationState(setPageLoading);

    // Needs to be here due to dependencies
    const [ctaMsgs, hideWarnings, warnings] = useGetQueryWarnings(
        id as string,
        handleError,
        queryInfo?.trainingStatus
    );

    const {
        onClickArchive,
        onClickRecover,
        onConfirm,
        setIsArchived,
        isManagingJob,
        isArchived
    } = useManageJob({
        jobId: id as string,
        jobType: JobType.TRAINING_JOB,
        jobName: queryInfo?.id,
        onArchive: () => {
            onArchive?.();
        },
        onRecover: () => {
            onRecover?.();
        },
        handleError,
        clearErrors,
        dialogRef,
        setPageLoading
    });

    // Invoking Query details API to get Predictive query information
    const { queryInfo: queryResponse, isLoading } = useTrainingJobWorkflow({
        id: id ? (id as string) : "",
        setStatus,
        setPageLoading,
        handleError,
        setIsArchived
    });

    useEffect(() => {
        if (queryResponse) setQueryInfo(queryResponse);
    }, [queryResponse]);

    const isDataLoading = isLoading || pageLoading;

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

    // cancel training job hook
    const { cancelTraining } = useCancelTrainingJob({
        jobId: id as string,
        onSuccess: () => {
            invalidateActiveJobs(queryClient);
        },
        handleError
    });

    const abortTraining = () => {
        setPageLoading(true);
        cancelTraining({ id: id as string });
    };

    const buttons = [];
    if (
        !hideMutationControls &&
        (queryInfo?.trainingStatus === jobStatus.Running ||
            queryInfo?.trainingStatus === jobStatus.Queued)
    ) {
        buttons.push(
            <Button
                id={"cancel_train"}
                onClick={onAbortTrainClick}
                key="cancel_train">
                <>
                    <Block className="size-5" />
                    {" Cancel this job"}
                </>
            </Button>
        );
    }

    const validationComponent = (
        <Validation
            pageLoading={false}
            banner={banner}
            alert={alert}
            ctaMsgs={null}
            dialogRef={dialogRef}
            hideWarnings={hideWarnings}
            onDialogConfirm={isManagingJob ? onConfirm : onAbortTrainConfirm}
            clearErrors={clearErrors}
        />
    );

    return (
        <TrainingJobOverviewDetail
            title={id as string}
            buttons={buttons}
            queryInfo={queryInfo}
            validationComponent={validationComponent}
            handleError={handleError}
            ctaMsgs={ctaMsgs as CTAInfo}
            loadingStatus={isDataLoading}
            hidePreTitle={hidePreTitle}
            showRelatedJobs={showRelatedJobs}
            onClose={onClose}
            onClickArchive={onClickArchive}
            onClickRecover={onClickRecover}
            isArchived={isArchived}
            warnings={warnings}
        />
    );
};
export default TrainingJob;
