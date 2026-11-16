// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React, { useCallback, useMemo, useState } from "react";
import Spacer from "react-spacer";

import {
    Archive,
    ArchiveRestore,
    AvgTime,
    Button,
    CodeIcon,
    OverviewDropdown,
    PageHeader,
    ProgressBar,
    ProgressInfo,
    PropertiesOverviewCard,
    Skeleton,
    StatusCard,
    StatusIcon,
    Tooltip,
    TooltipContent,
    TooltipTrigger,
    WarningsPopover
} from "@kumo-ai/kumo-ui-catalog";
import { useQueryClient } from "@tanstack/react-query";
import { ErrWarnDetails, JobType, KumoapiCommonJobStatus } from "gen/v0/types";

import RelatedJobsCard from "components/@ui/Card/RelatedJobsCard";
import PredictionOutputsOverview from "components/BatchPredictions/PredictionOutputsOverview/PredictionOutputsOverview";
import { JobQueryPreview } from "components/CodeEditor/JobQueryPreview";
import GraphOverview from "components/Graphs/Overview/GraphOverview";
import GraphSnapshotOverview from "components/Jobs/Graph/GraphSnapshotOverview";
import { jobStatusIconMap } from "components/Jobs/jobStatusIcons";
import TrainingJobOverview from "components/Jobs/Training/TrainingJob";
import { usePredictionJobsProperties } from "components/Overview/JobProperties";

import { usePredictionWarnings } from "hooks/jobs/prediction/prediction";
import { useJobTagMutations, useTags } from "hooks/jobs/tags/tags";
import { useProjectId } from "hooks/project/projects";

import {
    BatchPredictionJobID,
    UnlinkedBatchPredictionJobData
} from "types/batchPredictions";
import { HandleErrorFunc } from "types/validation";

import { jobErrTitle, jobStatus, JobStatusType } from "utils/Constants";
import { PROTECTED_TAG_KEYS } from "utils/Constants/config";
import { formatQueryForTooltip, getIdHash } from "utils/Helpers/stringHelpers";
import { isValid } from "utils/Helpers/typeGuards";
import { estimateJobProgressPercent } from "utils/jobProgress";
import {
    getColorVariant,
    getJobSource,
    PredictionStatusTitle,
    StatusColors
} from "utils/jobs";
import { invalidateJobQueries } from "utils/queryInvalidation";
import { jobRunTimeFromJobResponse } from "utils/timeConverter";

import PredictionTableOverview from "../PredictionTable/PredictionTableGeneration";

/**
 * PredictionJobOverview for Jobs details page
 *
 */
export interface PredictionJobOverviewProps {
    bpId?: BatchPredictionJobID;
    predictionInfo: UnlinkedBatchPredictionJobData | null;
    anchorTime: string | null;
    bpJobStatus: JobStatusType;
    queryYaml: string;
    loading: boolean;
    buttons: React.ReactNode[];
    isArchived: boolean;
    onClose: () => void;
    onClickArchive: () => void;
    onClickRecover: () => void;
    handleError: HandleErrorFunc;
    toggleSdkModel: (value: boolean) => void;
}

const PredictionJobOverview = ({
    bpId,
    predictionInfo,
    anchorTime,
    bpJobStatus,
    queryYaml,
    loading,
    buttons,
    isArchived,
    onClose,
    onClickArchive,
    onClickRecover,
    toggleSdkModel,
    handleError
}: PredictionJobOverviewProps) => {
    const projectId = useProjectId();
    const [graphSnapshotStatus, setGraphSnapshotStatus] = useState(
        jobStatus.None
    );
    const [predictionGenStatus, setPredictionGenStatus] = useState(
        jobStatus.None
    );
    const [trainingJobStatus, setTrainingJobStatus] = useState(jobStatus.None);

    const [graphWarnings, setGraphWarnings] = useState<ErrWarnDetails[]>([]);
    const [predTableJobWarnings, setPredTableJobWarnings] = useState<
        ErrWarnDetails[]
    >([]);

    const { warnings } = usePredictionWarnings(
        bpId as string,
        predictionInfo?.status as KumoapiCommonJobStatus
    );

    const { tags, updateTag, removeTag } = useTags(
        predictionInfo?.tags ?? undefined
    );
    const { addTag, deleteTag } = useJobTagMutations();
    const queryClient = useQueryClient();

    const source = getJobSource(predictionInfo?.source);

    const runTime = useMemo(() => {
        if (
            predictionInfo?.execution?.start_time &&
            predictionInfo?.execution.last_updated_at
        ) {
            try {
                const duration = jobRunTimeFromJobResponse(
                    predictionInfo?.execution?.start_time,
                    predictionInfo?.execution.last_updated_at
                );
                return duration;
            } catch (e) {
                console.warn(e);
            }
        }
        return "";
    }, [
        predictionInfo?.execution?.start_time,
        predictionInfo?.execution?.last_updated_at
    ]);

    const status = predictionInfo?.status || jobStatus.None;

    const properties = usePredictionJobsProperties(
        predictionInfo,
        status,
        runTime,
        source
    );

    const progressInfoChild = useMemo(() => {
        const items: {
            row1: React.JSX.Element;
            row2: string | null | undefined;
        }[] = [];
        if (anchorTime != null) {
            items.push({
                row1: (
                    <>
                        <AvgTime className="size-4.5 fill-black" />
                        {" Prediction anchor time"}
                    </>
                ),
                row2: anchorTime
            });
        }

        function roughPercentProgress(): number {
            return estimateJobProgressPercent(
                predictionInfo?.execution?.progress
            );
        }
        if (
            bpJobStatus === jobStatus.Queued ||
            bpJobStatus === jobStatus.Running
        ) {
            return (
                <>
                    {bpJobStatus === jobStatus.Running &&
                        predictionInfo?.log?.stage_name === "Predicting" && (
                            <>
                                <ProgressBar
                                    progress={roughPercentProgress()}
                                    color={StatusColors[bpJobStatus]}
                                />
                                <Spacer height={16} />
                            </>
                        )}
                    {items?.length > 0 && (
                        <>
                            <ProgressInfo info={items} />
                            <Spacer height={16} />
                        </>
                    )}
                    {buttons}
                </>
            );
        }

        if (
            bpJobStatus === jobStatus.Failed &&
            isValid(predictionInfo?.execution?.failure_info?.message)
        ) {
            return (
                <div>{predictionInfo?.execution?.failure_info?.message}</div>
            );
        }

        return null;
    }, [
        buttons,
        anchorTime,
        bpJobStatus,
        predictionInfo?.log?.stage_name,
        predictionInfo?.execution?.progress,
        predictionInfo?.execution?.failure_info?.message
    ]);

    const jobIsRunning =
        predictionInfo?.status === jobStatus.Running ||
        predictionInfo?.status === jobStatus.Done;

    // Display appropriate title based on job status
    const progressTitle = useMemo(() => {
        let title: string = PredictionStatusTitle[bpJobStatus];

        // When running, show current stage if available
        if (predictionInfo?.status === jobStatus.Running) {
            if (predictionInfo?.log?.stage_name != null) {
                title = predictionInfo?.log?.stage_name;
            }
        }

        // When done, show completion message
        if (predictionInfo?.status === jobStatus.Done) {
            title = "Prediction complete";
        }
        // When failed, use generic title and let failure_info.message (error body) provide details
        // The Related Jobs cards show which phase failed via status indicators
        else if (predictionInfo?.status === jobStatus.Failed) {
            title = isValid(predictionInfo?.execution?.failure_info?.message)
                ? PredictionStatusTitle[jobStatus.Failed] // "Prediction failed with the following errors"
                : jobErrTitle; // Fallback if no error message
        }

        return title;
    }, [
        bpJobStatus,
        predictionInfo?.status,
        predictionInfo?.log?.stage_name,
        predictionInfo?.execution?.failure_info?.message
    ]);

    const handleTagAdd = useCallback(
        async (key: string, value: string | undefined) => {
            if (!bpId) return;

            // Update local state optimistically
            updateTag(key, value as string);

            try {
                // Call API to add tag on the server
                await addTag(bpId, key, value);
                invalidateJobQueries(
                    queryClient,
                    JobType.BATCH_PREDICTION_JOB,
                    bpId
                );
            } catch (error) {
                console.error("PredictionJob ADD TAG ERROR", error);
                // Revert optimistic update on error
                removeTag(key);
            }
        },
        [bpId, updateTag, addTag, queryClient, removeTag]
    );

    const handleTagRemove = useCallback(
        async (key: string) => {
            if (!bpId) return;

            // Store the old value for reverting on error
            const oldValue = tags[key];

            // Update local state optimistically
            removeTag(key);

            try {
                // Call API to remove tag on the server
                await deleteTag(bpId, key);
                invalidateJobQueries(
                    queryClient,
                    JobType.BATCH_PREDICTION_JOB,
                    bpId
                );
            } catch (error) {
                console.error("PredictionJob REMOVE TAG ERROR", error);
                // Revert optimistic update on error
                if (oldValue !== undefined) {
                    updateTag(key, oldValue);
                }
            }
        },
        [bpId, tags, removeTag, deleteTag, queryClient, updateTag]
    );

    const allWarnings = useMemo(() => {
        const compare = (
            warning: ErrWarnDetails,
            newWarning: ErrWarnDetails
        ) => {
            return (
                warning.description.trim() === newWarning.description.trim() ||
                warning.description.trim() ===
                    `${newWarning.title}: ${newWarning.description.trim()}`
            );
        };
        const withGraphWarnings = [
            ...warnings,
            ...graphWarnings.filter(
                (gw) => !warnings.some((w) => compare(w, gw))
            )
        ];
        const withPredTableJobWarnings = [
            ...withGraphWarnings,
            ...predTableJobWarnings.filter(
                (tw) => !withGraphWarnings.some((w) => compare(w, tw))
            )
        ];
        return withPredTableJobWarnings;
    }, [graphWarnings, predTableJobWarnings, warnings]);

    return (
        <>
            <PageHeader
                screenshotTestId="prediction-detail-page"
                preTitle="Batch Prediction"
                titleCopyText={bpId}
                title={
                    <Tooltip id={"bpid-tooltip"}>
                        <TooltipTrigger>
                            <span>{getIdHash(bpId)}</span>
                        </TooltipTrigger>
                        <TooltipContent
                            arrow={false}
                            side={"bottom"}
                            sideOffset={5}>
                            {bpId as string}
                        </TooltipContent>
                    </Tooltip>
                }
                onCloseDialog={onClose}
                loadingStatus={loading}
                headerButton={
                    !projectId ? (
                        <></>
                    ) : (
                        <React.Fragment key="header-buttons">
                            {allWarnings.length > 0 && (
                                <WarningsPopover warnings={allWarnings} />
                            )}
                            {isArchived ? (
                                <Button
                                    data-testid="restore-button"
                                    variant="outline"
                                    className="hover:bg-kui-bg-hover border-border-primary"
                                    onClick={onClickRecover}>
                                    <ArchiveRestore className="size-4" />
                                    Restore archive
                                </Button>
                            ) : (
                                <OverviewDropdown
                                    id="tarining-overview-dropdown"
                                    buttonClassName="h-9 w-9 px-2 py-2 hover:bg-kui-bg-hover border-border-primary"
                                    actionItems={[
                                        {
                                            content: (
                                                <>
                                                    <CodeIcon className="size-4" />{" "}
                                                    {" SDK Code "}
                                                </>
                                            ),
                                            disabled: false,
                                            onSelect: () =>
                                                toggleSdkModel(true),
                                            tooltipText: ""
                                        },
                                        {
                                            content: (
                                                <>
                                                    <Archive className="size-4" />{" "}
                                                    {" Archive"}
                                                </>
                                            ),
                                            disabled: false,
                                            onSelect: onClickArchive,
                                            tooltipText: ""
                                        }
                                    ]}
                                />
                            )}
                        </React.Fragment>
                    )
                }
            />

            <Spacer height={16} />
            <PropertiesOverviewCard
                properties={properties}
                queryContent={
                    queryYaml ? (
                        <JobQueryPreview
                            value={
                                formatQueryForTooltip(queryYaml.trim()) || ""
                            }
                        />
                    ) : undefined
                }
                tags={tags}
                loading={loading}
                onTagAdd={handleTagAdd}
                onTagRemove={handleTagRemove}
                protectedKeys={PROTECTED_TAG_KEYS}
            />

            <Spacer height={16} />
            {((graphSnapshotStatus === jobStatus.None &&
                predictionInfo?.graphSnapshotId) ||
                (predictionGenStatus === jobStatus.None &&
                    predictionInfo?.generatePredTableJobId)) &&
            predictionInfo?.status === jobStatus.Failed ? (
                <Skeleton className="w-full h-20 mb-4" />
            ) : (
                (predictionInfo?.status === jobStatus.Queued ||
                    predictionInfo?.status === jobStatus.Running ||
                    predictionInfo?.status === jobStatus.Failed) && (
                    <StatusCard
                        title={progressTitle}
                        titleIcon={
                            <StatusIcon
                                status={bpJobStatus}
                                iconMap={jobStatusIconMap}
                            />
                        }
                        id={"prediction-job"}
                        className="mb-4 rounded-lg"
                        variant={getColorVariant(bpJobStatus)}
                        enableDivider={bpJobStatus !== jobStatus.Failed}
                        contentChild={progressInfoChild}></StatusCard>
                )
            )}

            <PredictionOutputsOverview
                queryYaml={queryYaml}
                predictionInfo={predictionInfo}
                handleError={handleError}
                isJobRunning={jobIsRunning}
                anchorTime={anchorTime}
            />

            {(predictionInfo?.graphSnapshotId ||
                predictionInfo?.generatePredTableJobId ||
                predictionInfo?.graphId) && (
                <>
                    {predictionInfo?.graphSnapshotId ? (
                        <RelatedJobsCard
                            jobName="Graph Snapshot"
                            jobId={getIdHash(predictionInfo?.graphSnapshotId)}
                            loadingStatus={
                                graphSnapshotStatus === jobStatus.None &&
                                predictionInfo?.graphSnapshotId
                                    ? true
                                    : false
                            }
                            status={graphSnapshotStatus}>
                            <GraphSnapshotOverview
                                id={predictionInfo?.graphSnapshotId}
                                hidePreTitle={true}
                                setStatus={setGraphSnapshotStatus}
                                parentJobSource={source}
                                setWarnings={setGraphWarnings}
                            />
                        </RelatedJobsCard>
                    ) : predictionInfo?.graphId ? (
                        <RelatedJobsCard
                            jobName="Graph"
                            jobId={predictionInfo?.graphId}>
                            <GraphOverview
                                graphId={predictionInfo?.graphId}
                                onCopyClick={() => {}}
                                onDeleteClick={() => {}}
                                onEditClick={() => {}}
                                isRelatedJob={true}
                            />
                        </RelatedJobsCard>
                    ) : (
                        <></>
                    )}
                    {predictionInfo?.generatePredTableJobId && (
                        <RelatedJobsCard
                            jobName="Prediction Table Generation"
                            jobId={getIdHash(
                                predictionInfo?.generatePredTableJobId
                            )}
                            loadingStatus={
                                predictionGenStatus === jobStatus.None
                            }
                            status={predictionGenStatus}>
                            <PredictionTableOverview
                                id={predictionInfo?.generatePredTableJobId}
                                hidePreTitle={true}
                                onClose={() => {}}
                                showRelatedJobs={false}
                                setStatus={setPredictionGenStatus}
                                jobSource={source}
                                queryYaml={queryYaml}
                                setWarnings={setPredTableJobWarnings}
                            />
                        </RelatedJobsCard>
                    )}
                </>
            )}
            {predictionInfo?.trainingJobID && (
                <RelatedJobsCard
                    jobName="Training Job"
                    jobId={getIdHash(predictionInfo?.trainingJobID)}
                    loadingStatus={trainingJobStatus === jobStatus.None}
                    status={trainingJobStatus}>
                    <TrainingJobOverview
                        id={predictionInfo?.trainingJobID}
                        hidePreTitle={true}
                        onClose={() => {}}
                        showRelatedJobs={false}
                        setStatus={setTrainingJobStatus}
                    />
                </RelatedJobsCard>
            )}
        </>
    );
};

export default PredictionJobOverview;
