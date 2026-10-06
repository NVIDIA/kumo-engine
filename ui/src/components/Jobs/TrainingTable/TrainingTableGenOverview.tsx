// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React, { useCallback, useEffect, useMemo, useState } from "react";
import Spacer from "react-spacer";

import {
    Archive,
    ArchiveRestore,
    Button,
    cn,
    OverviewDropdown,
    PageHeader,
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
import { ErrWarnDetails, JobType, QueryType } from "gen/v0/types";
import yaml from "js-yaml";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";
import RelatedJobsCard from "components/@ui/Card/RelatedJobsCard";
import JobsError from "components/@ui/Jobs/JobsError";
import { JobQueryPreview } from "components/CodeEditor/JobQueryPreview";
import GraphOverview from "components/Graphs/Overview/GraphOverview";
import { jobStatusIconMap } from "components/Jobs/jobStatusIcons";
import { useTrainingTableGenProperties } from "components/Overview/JobProperties";
import TrainingSetting from "components/PredictiveQueries/Overview/TrainingSetting";
import DataStatistics from "components/PredictiveQueries/Training/DataStatistics/DataStatistics";

import { useJobTagMutations, useTags } from "hooks/jobs/tags/tags";
import { useTraintableJobErrors } from "hooks/jobs/trainTableGen/trainTableGen";
import { useProjectId } from "hooks/project/projects";

import { JobSource } from "types/jobs";
import { PredictiveQueryInfo, TaskType } from "types/predictiveQueries";
import { TrainTableGeneration } from "types/training_job";
import { CTAInfo } from "types/warnings";

import { jobErrTitle, jobStatus, JobStatusType } from "utils/Constants";
import { PROTECTED_TAG_KEYS } from "utils/Constants/config";
import { getIdHash } from "utils/Helpers/stringHelpers";
import { isValid } from "utils/Helpers/typeGuards";
import {
    formatTableGenCancellationMessage,
    formatTableGenFailureMessage
} from "utils/jobFailureHandling";
import {
    getColorVariant,
    getJobSource,
    TrainingTableStatusTitle
} from "utils/jobs";
import { invalidateJobQueries } from "utils/queryInvalidation";

import GraphSnapshotOverview from "../Graph/GraphSnapshotOverview";

export interface TrainingTableGenOverviewProps {
    title: string;
    buttons: Array<React.ReactNode>;
    trainTableDetails: TrainTableGeneration | null;
    ctaMsgs: CTAInfo;
    loadingStatus?: boolean;
    onClose: () => void;
    onClickArchive: () => void;
    onClickRecover: () => void;
    queryYaml: string;
    taskType: TaskType;
    queryType: QueryType;
    hidePreTitle?: boolean;
    showRelatedJobs?: boolean;
    graphId?: string | null;
    modelPlan?: string | null;
    trainingPlan?: string | null;
    isTrainingPage?: boolean;
    queryInfo?: PredictiveQueryInfo;
    isArchived?: boolean;
    user?: string | null;
    setWarnings?: (warning: ErrWarnDetails[]) => void;
}

const TrainingTableGenOverview = ({
    title,
    buttons,
    trainTableDetails,
    ctaMsgs,
    loadingStatus,
    onClose,
    onClickArchive,
    onClickRecover,
    queryYaml,
    taskType,
    queryType,
    hidePreTitle = false,
    showRelatedJobs = true,
    graphId,
    modelPlan,
    trainingPlan,
    isTrainingPage = false,
    queryInfo,
    isArchived,
    user = null,
    setWarnings
}: TrainingTableGenOverviewProps) => {
    const projectId = useProjectId();
    const [graphSnapshotStatus, setGraphSnapshotStatus] = useState(
        jobStatus.None
    );
    const [graphWarnings, setGraphWarnings] = useState<ErrWarnDetails[]>([]);
    const { tags, updateTag, removeTag } = useTags(
        trainTableDetails?.tags ?? undefined
    );
    const { addTag, deleteTag } = useJobTagMutations();
    const queryClient = useQueryClient();

    // Check if there's a valid graph snapshot child job
    const hasValidGraphSnapshot = useMemo(
        () => trainTableDetails && isValid(trainTableDetails.graphSnapshotId),
        [trainTableDetails]
    );

    // Track child job status (only one possible child for table gen jobs)
    const hasFailedChild =
        hasValidGraphSnapshot && graphSnapshotStatus === jobStatus.Failed;
    const hasCancelledChild =
        hasValidGraphSnapshot && graphSnapshotStatus === jobStatus.Cancelled;

    const { errors, warnings, fetchingErrors } = useTraintableJobErrors({
        jobId: trainTableDetails?.jobId,
        trainTableDetails
    });

    useEffect(() => {
        if (warnings && warnings.length > 0 && !showRelatedJobs) {
            setWarnings?.(warnings);
        }
    }, [setWarnings, showRelatedJobs, warnings]);

    const source = getJobSource(trainTableDetails?.source);
    const status = useMemo(() => {
        return trainTableDetails?.status || jobStatus.None;
    }, [trainTableDetails?.status]);

    const properties = useTrainingTableGenProperties({
        info: trainTableDetails,
        jobId: isTrainingPage ? title : null,
        source,
        status: status as JobStatusType,
        taskType,
        user
    });

    // Handle job failure with no message.
    const progressTitle = useMemo(() => {
        let title: string = TrainingTableStatusTitle[status];
        if (trainTableDetails?.status === jobStatus.Failed) {
            // Priority 1: Show cancellation context (backend doesn't provide this)
            if (hasCancelledChild) {
                title = formatTableGenCancellationMessage("training-table");
            }
            // Priority 2: Show child failure context (more specific than generic message)
            else if (hasFailedChild) {
                title = formatTableGenFailureMessage("training-table");
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
        trainTableDetails?.status,
        fetchingErrors,
        errors?.length,
        status,
        hasCancelledChild,
        hasFailedChild
    ]);

    const progressInfoChild = useMemo(() => {
        if (status === jobStatus.Failed && !hasCancelledChild) {
            if (!errors?.length) return null;
            return (
                <JobsError errors={errors} fetchingErrors={fetchingErrors} />
            );
        }
        if (buttons?.length > 0) {
            return <>{buttons}</>;
        }
        return null;
    }, [status, hasCancelledChild, errors, fetchingErrors, buttons]);

    const handleTagAdd = useCallback(
        async (key: string, value: string | undefined) => {
            // Update local state optimistically
            updateTag(key, value as string);

            try {
                // Call API to add tag on the server
                await addTag(title, key, value);
                invalidateJobQueries(
                    queryClient,
                    JobType.GENERATE_TRAIN_TABLE_JOB,
                    title
                );
            } catch (error) {
                console.error("TrainingTable ADD TAG ERROR", error);
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
                    JobType.GENERATE_TRAIN_TABLE_JOB,
                    title
                );
            } catch (error) {
                console.error("TrainingTable REMOVE TAG ERROR", error);
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
                hidePreTitle={hidePreTitle}
                loadingStatus={loadingStatus}
                screenshotTestId="training-table-detail-page"
                preTitle={"Training Table"}
                title={
                    !hidePreTitle && (
                        <Tooltip id={"id-tooltip"}>
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
                properties={properties}
                queryContent={
                    queryYaml ? (
                        <JobQueryPreview value={queryYaml.trim() || ""} />
                    ) : undefined
                }
                hideTitle={hidePreTitle}
                tags={tags}
                loading={loadingStatus}
                onTagAdd={handleTagAdd}
                onTagRemove={handleTagRemove}
                protectedKeys={PROTECTED_TAG_KEYS}
            />

            {trainTableDetails &&
                (status === jobStatus.Queued ||
                    status === jobStatus.Running ||
                    status === jobStatus.Failed) && (
                    <div
                        className={
                            !showRelatedJobs
                                ? "px-6 pb-4 rounded-lg"
                                : "my-4 rounded-lg"
                        }>
                        <StatusCard
                            title={progressTitle}
                            titleIcon={
                                <StatusIcon
                                    status={status}
                                    iconMap={jobStatusIconMap}
                                />
                            }
                            id={"trai-table-genreration"}
                            variant={getColorVariant(status)}
                            enableDivider={status !== jobStatus.Failed}
                            contentChild={progressInfoChild}></StatusCard>
                    </div>
                )}

            {trainTableDetails?.jobId && (
                <>
                    {(trainTableDetails?.status === jobStatus.Queued ||
                        trainTableDetails?.status === jobStatus.Running ||
                        trainTableDetails?.status === jobStatus.Failed ||
                        trainTableDetails?.status === jobStatus.Cancelled) &&
                        !isTrainingPage && (
                            <>
                                {trainTableDetails?.status ===
                                    jobStatus.Cancelled && (
                                    <Spacer height={16} />
                                )}
                                <TrainingSetting
                                    trainingJobId={trainTableDetails?.jobId}
                                    trainingstatus={trainTableDetails.status}
                                    queryId={
                                        trainTableDetails?.config
                                            .pquery_id as string
                                    }
                                    advancedOptions=""
                                    modelPlan={modelPlan as string}
                                    trainingPlan={trainingPlan as string}
                                    generationPlan={
                                        trainTableDetails?.config?.plan
                                            ? yaml.dump(
                                                  trainTableDetails?.config
                                                      ?.plan
                                              )
                                            : ""
                                    }
                                    ctaMsgs={ctaMsgs}
                                    loadingStatus={false}
                                />
                            </>
                        )}

                    {trainTableDetails?.status === jobStatus.Done && (
                        <div
                            className={cn(
                                showRelatedJobs ? "mt-4" : "mx-6 mb-6"
                            )}>
                            <DataStatistics
                                id={trainTableDetails?.jobId}
                                queryStatus={status as jobStatus}
                                queryStage={
                                    trainTableDetails?.jobStatusReport
                                        ?.event_log?.[0]
                                }
                                taskType={taskType}
                                queryType={queryType}
                                isSourceSDK={
                                    trainTableDetails?.source === JobSource.SDK
                                }
                                isTrainingPage={isTrainingPage}
                                jobId={
                                    trainTableDetails?.tags
                                        ? trainTableDetails?.tags["parent_job"]
                                        : null
                                }
                            />
                        </div>
                    )}
                </>
            )}
            {showRelatedJobs && (
                <>
                    <Spacer height={16} />
                    {trainTableDetails?.graphSnapshotId ? (
                        <RelatedJobsCard
                            status={graphSnapshotStatus}
                            loadingStatus={loadingStatus}
                            jobName={"Graph Snapshot"}
                            jobId={getIdHash(
                                trainTableDetails?.graphSnapshotId
                            )}>
                            <GraphSnapshotOverview
                                id={trainTableDetails?.graphSnapshotId}
                                hidePreTitle
                                setStatus={setGraphSnapshotStatus}
                                parentJobSource={source}
                                queryInfo={queryInfo}
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

export const TrainingJobOverviewDetail = WithErrorBoundary(
    TrainingTableGenOverview
);
