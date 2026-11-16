// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React, { useCallback, useEffect, useMemo, useState } from "react";
import Spacer from "react-spacer";

import {
    Archive,
    ArchiveRestore,
    AvgTime,
    Button,
    OverviewDropdown,
    PageHeader,
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
import { ErrWarnDetails, JobType } from "gen/v0/types";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";
import RelatedJobsCard from "components/@ui/Card/RelatedJobsCard";
import JobsError from "components/@ui/Jobs/JobsError";
import { JobQueryPreview } from "components/CodeEditor/JobQueryPreview";
import GraphOverview from "components/Graphs/Overview/GraphOverview";
import { jobStatusIconMap } from "components/Jobs/jobStatusIcons";
import { usePredictionTableGenProperties } from "components/Overview/JobProperties";

import { useGenPredErrorsById } from "hooks/jobs/prediction/prediction";
import { usePredictionTableNumEntities } from "hooks/jobs/predictionTableGen/predictionTableGen";
import { useJobTagMutations, useTags } from "hooks/jobs/tags/tags";
import { useProjectId } from "hooks/project/projects";

import { PredictionTableGeneration } from "types/batchPredictions";
import { TaskType } from "types/predictiveQueries";

import { jobErrTitle, jobStatus, JobStatusType } from "utils/Constants";
import { PROTECTED_TAG_KEYS } from "utils/Constants/config";
import { formatQueryForTooltip, getIdHash } from "utils/Helpers/stringHelpers";
import { isValid } from "utils/Helpers/typeGuards";
import {
    formatTableGenCancellationMessage,
    formatTableGenFailureMessage
} from "utils/jobFailureHandling";
import {
    getColorVariant,
    getJobSource,
    PredictionTableStatusTitle
} from "utils/jobs";
import { invalidateJobQueries } from "utils/queryInvalidation";

import GraphSnapshotOverview from "../Graph/GraphSnapshotOverview";

export interface PredictionTableGenOverviewProps {
    title: string;
    buttons: Array<React.ReactNode>;
    predictionTableDetails: PredictionTableGeneration | null;
    loadingStatus?: boolean;
    onClose: () => void;
    onClickArchive: () => void;
    onClickRecover: () => void;
    queryYaml?: string;
    taskType: TaskType;
    anchorTime?: string | null;
    hidePreTitle?: boolean;
    showRelatedJobs?: boolean;
    graphId: string | null;
    parentJobSource?: string;
    isArchived?: boolean;
    user?: string | null;
    setWarnings?: (warning: ErrWarnDetails[]) => void;
}

const PredictionTableGenOverview = ({
    title,
    buttons,
    predictionTableDetails,
    loadingStatus,
    onClose,
    onClickArchive,
    onClickRecover,
    queryYaml,
    taskType,
    anchorTime,
    hidePreTitle = false,
    showRelatedJobs = true,
    graphId,
    parentJobSource,
    isArchived,
    user = null,
    setWarnings
}: PredictionTableGenOverviewProps) => {
    const projectId = useProjectId();
    const [graphSnapshotStatus, setGraphSnapshotStatus] = useState(
        jobStatus.None
    );

    const [graphWarnings, setGraphWarnings] = useState<ErrWarnDetails[]>([]);
    const { tags, updateTag, removeTag } = useTags(
        predictionTableDetails?.tags ?? undefined
    );
    const { addTag, deleteTag } = useJobTagMutations();
    const queryClient = useQueryClient();
    // Check if there's a valid graph snapshot child job
    const hasValidGraphSnapshot = useMemo(
        () =>
            predictionTableDetails &&
            isValid(predictionTableDetails.graphSnapshotId),
        [predictionTableDetails]
    );

    // Track child job status (only one possible child for table gen jobs)
    const hasFailedChild =
        hasValidGraphSnapshot && graphSnapshotStatus === jobStatus.Failed;
    const hasCancelledChild =
        hasValidGraphSnapshot && graphSnapshotStatus === jobStatus.Cancelled;

    // Invoking get_errors API to get Prediction table job errors

    const { fetchingErrors, errors, warnings } = useGenPredErrorsById(
        predictionTableDetails?.jobId as string,
        predictionTableDetails?.status
    );

    useEffect(() => {
        if (warnings && warnings.length > 0 && !showRelatedJobs) {
            setWarnings?.(warnings);
        }
    }, [setWarnings, showRelatedJobs, warnings]);

    const source = getJobSource(
        parentJobSource ?? predictionTableDetails?.source
    );

    const status = useMemo(() => {
        return predictionTableDetails?.status || jobStatus.None;
    }, [predictionTableDetails?.status]);

    const numEntities = usePredictionTableNumEntities({
        id: predictionTableDetails?.jobId as string,
        status: status as jobStatus
    });

    const properties = usePredictionTableGenProperties({
        id: showRelatedJobs ? null : title,
        info: predictionTableDetails,
        source,
        status: status as JobStatusType,
        taskType,
        user,
        anchorTime,
        numEntities
    });

    // Handle job failure with no message.
    const progressTitle = useMemo(() => {
        let title: string = PredictionTableStatusTitle[status];
        if (predictionTableDetails?.status === jobStatus.Failed) {
            // Priority 1: Show cancellation context (backend doesn't provide this)
            if (hasCancelledChild) {
                title = formatTableGenCancellationMessage("prediction-table");
            }
            // Priority 2: Show child failure context (more specific than generic message)
            else if (hasFailedChild) {
                title = formatTableGenFailureMessage("prediction-table");
            }
            // Priority 3: Still fetching errors
            else if (fetchingErrors) {
                title = "Fetching errors...";
            }
            // Priority 4: Failed but no errors available
            else if (errors?.length < 1) {
                title = jobErrTitle;
            }
            // else: keep generic header, let error body display the errors
        }
        return title;
    }, [
        hasCancelledChild,
        hasFailedChild,
        predictionTableDetails?.status,
        fetchingErrors,
        errors?.length,
        status
    ]);

    const progressInfoChild = useMemo(() => {
        const items: {
            row1: React.JSX.Element;
            row2: string | null | undefined;
        }[] = [];
        if (anchorTime != null) {
            items.push({
                row1: (
                    <div className="flex items-center">
                        <AvgTime className="size-4.5 fill-black" />
                        <label className="ml-1">
                            {" Prediction anchor time"}
                        </label>
                    </div>
                ),
                row2: anchorTime
            });
        }
        if (status === jobStatus.Queued || status === jobStatus.Running) {
            return (
                <>
                    <ProgressInfo info={items} />
                    <Spacer height={16} />
                    {buttons}
                </>
            );
        }
        if (status === jobStatus.Failed && !hasCancelledChild) {
            if (errors?.length > 0) {
                return (
                    <JobsError
                        errors={errors}
                        fetchingErrors={fetchingErrors}
                    />
                );
            }
        }
        return null;
    }, [
        anchorTime,
        buttons,
        status,
        errors,
        fetchingErrors,
        hasCancelledChild
    ]);

    const handleTagAdd = useCallback(
        async (key: string, value: string | undefined) => {
            // Update local state optimistically
            updateTag(key, value as string);

            try {
                // Call API to add tag on the server
                await addTag(title, key, value);
                invalidateJobQueries(
                    queryClient,
                    JobType.GENERATE_PREDICTION_TABLE_JOB,
                    title
                );
            } catch (error) {
                console.error("PredictionTable ADD TAG ERROR", error);
                // Revert optimistic update on error
                removeTag(key);
            }
        },
        [updateTag, addTag, title, queryClient, removeTag]
    );

    const handleTagRemove = useCallback(
        async (key: string) => {
            // Store the old value for reverting on error
            const oldValue = tags[key];

            // Update local state optimistically
            removeTag(key);

            try {
                // Call API to remove tag on the server
                await deleteTag(title, key);
                invalidateJobQueries(
                    queryClient,
                    JobType.GENERATE_PREDICTION_TABLE_JOB,
                    title
                );
            } catch (error) {
                console.error("PredictionTable REMOVE TAG ERROR", error);
                // Revert optimistic update on error
                if (oldValue !== undefined) {
                    updateTag(key, oldValue);
                }
            }
        },
        [tags, removeTag, deleteTag, title, queryClient, updateTag]
    );

    const allWarnings = useMemo(() => {
        if (!showRelatedJobs) {
            return [];
        }
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

        return withGraphWarnings;
    }, [graphWarnings, showRelatedJobs, warnings]);

    return (
        <>
            <PageHeader
                loadingStatus={loadingStatus}
                screenshotTestId="prediction-table-detail-page"
                preTitle={"Prediction Table Generation"}
                title={
                    hidePreTitle ? null : (
                        <Tooltip id={"bpid-tooltip"}>
                            <TooltipTrigger>
                                <span>{getIdHash(title)}</span>
                            </TooltipTrigger>
                            <TooltipContent
                                arrow={false}
                                side={"bottom"}
                                sideOffset={5}>
                                {title}
                            </TooltipContent>
                        </Tooltip>
                    )
                }
                titleCopyText={title}
                onCloseDialog={onClose}
                hidePreTitle={hidePreTitle}
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
                                    id="prediction-overview-dropdown"
                                    buttonClassName="h-9 w-9 px-2 py-2 hover:bg-kui-bg-hover border-border-primary"
                                    actionItems={[
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
            {!hidePreTitle && <Spacer height={16} />}

            <PropertiesOverviewCard
                hideTitle={hidePreTitle}
                properties={properties}
                queryContent={
                    queryYaml ? (
                        <JobQueryPreview
                            value={formatQueryForTooltip(queryYaml)}
                        />
                    ) : undefined
                }
                tags={tags}
                loading={loadingStatus}
                onTagAdd={handleTagAdd}
                onTagRemove={handleTagRemove}
                protectedKeys={PROTECTED_TAG_KEYS}
            />

            {showRelatedJobs && <Spacer height={16} />}

            {predictionTableDetails &&
                (status === jobStatus.Queued ||
                    status === jobStatus.Running ||
                    status === jobStatus.Failed) && (
                    <div className={!showRelatedJobs ? " px-6" : ""}>
                        <StatusCard
                            className="mb-4 rounded-lg"
                            title={progressTitle}
                            titleIcon={
                                <StatusIcon
                                    status={status}
                                    iconMap={jobStatusIconMap}
                                />
                            }
                            id={"prediction-table-generation"}
                            variant={getColorVariant(status)}
                            enableDivider={status !== jobStatus.Failed}
                            contentChild={progressInfoChild}
                        />
                    </div>
                )}
            {showRelatedJobs &&
                (predictionTableDetails?.graphSnapshotId || graphId) && (
                    <>
                        {predictionTableDetails?.graphSnapshotId ? (
                            <RelatedJobsCard
                                status={graphSnapshotStatus}
                                loadingStatus={loadingStatus}
                                jobName={"Graph Snapshot"}
                                jobId={getIdHash(
                                    predictionTableDetails?.graphSnapshotId
                                )}>
                                <GraphSnapshotOverview
                                    id={predictionTableDetails?.graphSnapshotId}
                                    hidePreTitle={true}
                                    setStatus={setGraphSnapshotStatus}
                                    parentJobSource={source}
                                    setWarnings={setGraphWarnings}
                                />
                            </RelatedJobsCard>
                        ) : graphId ? (
                            <RelatedJobsCard jobName="Graph" jobId={graphId}>
                                <GraphOverview
                                    graphId={graphId}
                                    onCopyClick={() => {}}
                                    onDeleteClick={() => {}}
                                    onEditClick={() => {}}
                                    isRelatedJob={true}
                                />
                            </RelatedJobsCard>
                        ) : (
                            <Skeleton className="w-full h-20" />
                        )}
                    </>
                )}
        </>
    );
};

export default WithErrorBoundary(PredictionTableGenOverview);
