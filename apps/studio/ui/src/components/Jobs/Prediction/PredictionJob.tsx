// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useState } from "react";

import { Block, Button } from "@kumo-ai/kumo-ui-catalog";
import { useQueryClient } from "@tanstack/react-query";
import { JobType } from "gen/v0/types";

import { SDKCodeModal } from "components/SDKCodeModal/SDKCodeModal";
import Validation from "components/Validation/Validation";

import { useManageJob } from "hooks/jobs/list/jobList";
import {
    useBatchPredictionAnchorTime,
    useBatchPredictionSummary,
    useCancelPredictionJob
} from "hooks/jobs/prediction/prediction";
import { useQueryDefinition } from "hooks/jobs/training/query";
import { useProjectId } from "hooks/project/projects";
import { useValidationState } from "hooks/validation/validation";

import { BatchPredictionJobID } from "types/batchPredictions";
import { SDKObjectType } from "types/constants";

import { getJobSource } from "utils/jobs";
import { invalidateActiveJobs } from "utils/queryInvalidation";

import PredictionJobOverview from "./PredictionJobOverview";

/**
 * PredictionJob for Jobs details page
 *
 */
export interface PredictionJobProps {
    bpId?: BatchPredictionJobID;
    onClose: () => void;
    onArchive?: () => void;
    onRecover?: () => void;
}

const PredictionJob = ({
    bpId,
    onClose,
    onArchive,
    onRecover
}: PredictionJobProps) => {
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [isLoading, setLoading] = useState<boolean>(true);
    const queryClient = useQueryClient();
    const [refetchTrigger, setRefetchTrigger] = useState(0);
    const projectId = useProjectId();
    const hideMutationControls = !projectId;
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    const [banner, alert, handleError, clearErrors, dialogRef, _] =
        useValidationState(setLoading);

    const {
        onClickArchive,
        onClickRecover,
        onConfirm,
        setIsArchived,
        isManagingJob,
        isArchived
    } = useManageJob({
        jobId: bpId as string,
        jobType: JobType.BATCH_PREDICTION_JOB,
        jobName: bpId,
        onArchive: () => {
            setRefetchTrigger((prev) => prev + 1);
            onArchive?.();
        },
        onRecover: () => {
            setRefetchTrigger((prev) => prev + 1);
            onRecover?.();
        },
        handleError,
        clearErrors,
        dialogRef,
        setPageLoading: setLoading
    });

    const {
        isDataLoading,
        batchPredictionInfo: predictionInfo,
        queryId,
        bpJobStatus
    } = useBatchPredictionSummary(
        bpId as string,
        handleError,
        setLoading,
        refetchTrigger,
        setIsArchived
    );

    const { queryYaml } = useQueryDefinition(queryId);

    const source = getJobSource(predictionInfo?.source);

    const { anchorTime } = useBatchPredictionAnchorTime(
        bpId as string,
        source,
        handleError,
        setLoading,
        predictionInfo?.generatePredTableJobId as string
    );
    const onAbortBatchPredictionConfirm = () => {
        dialogRef.current?.toggleDialog();
        abortBatchPrediction();
    };

    const onAbortBatchPrediction = useCallback(() => {
        dialogRef?.current?.toggleDialog(
            "Cancel Prediction Job?",
            "Stopping now will discard all progress on this job. This cannot be undone.",
            undefined,
            false,
            "Cancel job",
            "destructive"
        );
    }, [dialogRef]);

    /**
     *  This method invokes REST API call to Cancel an ongoing batch prediction.
     */

    const { cancelPrediction } = useCancelPredictionJob({
        bpId: bpId as string,
        onSuccess: () => {
            invalidateActiveJobs(queryClient);
        },
        handleError
    });
    const abortBatchPrediction = useCallback(async () => {
        setLoading(true);
        cancelPrediction({ id: bpId as string });
    }, [bpId, cancelPrediction]);

    const buttons = hideMutationControls
        ? []
        : [
              <Button
                  id={"cancel_bp_job"}
                  onClick={onAbortBatchPrediction}
                  key="cancel_bp_job">
                  <>
                      <Block className="size-5 fill-white" />
                      {" Cancel this job"}
                  </>
              </Button>
          ];

    return (
        <>
            <PredictionJobOverview
                predictionInfo={predictionInfo}
                anchorTime={anchorTime}
                queryYaml={queryYaml}
                bpJobStatus={bpJobStatus}
                loading={isDataLoading || isLoading}
                onClose={onClose}
                toggleSdkModel={setIsModalOpen}
                onClickArchive={onClickArchive}
                onClickRecover={onClickRecover}
                handleError={handleError}
                isArchived={isArchived}
                bpId={bpId}
                buttons={buttons}
            />
            <Validation
                pageLoading={false}
                banner={null}
                alert={alert}
                ctaMsgs={null}
                dialogRef={dialogRef}
                clearErrors={clearErrors}
                onDialogConfirm={
                    isManagingJob ? onConfirm : onAbortBatchPredictionConfirm
                }
            />
            {isModalOpen && (
                <SDKCodeModal
                    id={predictionInfo?.id as string}
                    open={isModalOpen}
                    toggleModel={() => setIsModalOpen(!isModalOpen)}
                    objectTypeToGenScript={SDKObjectType.PredictionJob}
                />
            )}
        </>
    );
};

export default PredictionJob;
