// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React, { useCallback, useMemo, useState } from "react";
import Spacer from "react-spacer";

import {
    Archive,
    ArchiveRestore,
    Button,
    CodeIcon,
    Duplicate,
    OverviewDropdown,
    PageHeader,
    ProgressBar,
    PropertiesOverviewCard,
    Skeleton,
    SoundSampler,
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
import { JobQueryPreview } from "components/CodeEditor/JobQueryPreview";
import GraphOverview from "components/Graphs/Overview/GraphOverview";
import { jobStatusIconMap } from "components/Jobs/jobStatusIcons";
import TableGenerationOverview from "components/Jobs/TrainingTable/TrainingTableGeneration";
import { useTrainingJobProperties } from "components/Overview/JobProperties";
import ModelPerformanceOverview from "components/PredictiveQueries/Overview/ModelPerformanceOverview";
import { SDKCodeModal } from "components/SDKCodeModal/SDKCodeModal";

import { useJobTagMutations, useTags } from "hooks/jobs/tags/tags";
import { useProjectId } from "hooks/project/projects";

import { SDKObjectType } from "types/constants";
import { JobSource } from "types/jobs";
import { PredictiveQueryInfo } from "types/predictiveQueries";
import { CTAInfo } from "types/warnings";

import { getProjectPath, jobErrTitle, jobStatus } from "utils/Constants";
import { PROTECTED_TAG_KEYS } from "utils/Constants/config";
import { getIdHash } from "utils/Helpers/stringHelpers";
import { isValid } from "utils/Helpers/typeGuards";
import { estimateJobProgressPercent } from "utils/jobProgress";
import {
    assembleCopyTrainingJobArgs,
    getColorVariant,
    getJobSource,
    StatusColors,
    TrainingStatusTitle
} from "utils/jobs";
import { useCustomNavigation } from "utils/navigator";
import { invalidateJobQueries } from "utils/queryInvalidation";

import GraphSnapshotOverview from "../Graph/GraphSnapshotOverview";

export interface TrainingJobOverviewDetailProps {
    title: string;
    buttons: Array<React.ReactNode>;
    queryInfo: PredictiveQueryInfo | null;
    validationComponent: React.ReactNode;
    handleError: (err: Error) => void;
    ctaMsgs: CTAInfo;
    loadingStatus?: boolean;
    hidePreTitle?: boolean;
    showRelatedJobs?: boolean;
    onClose: (() => void) | undefined;
    onClickArchive: () => void;
    onClickRecover: () => void;
    isArchived?: boolean;
    warnings: ErrWarnDetails[];
}

const TrainingJobOverviewDetail = ({
    title,
    buttons,
    queryInfo,
    validationComponent,
    handleError,
    ctaMsgs,
    loadingStatus,
    hidePreTitle = false,
    showRelatedJobs = true,
    onClose,
    onClickArchive,
    onClickRecover,
    isArchived,
    warnings = []
}: TrainingJobOverviewDetailProps) => {
    const [isModalOpen, setIsModalOpen] = useState(false);
    const projectId = useProjectId();
    const projectPaths = getProjectPath(projectId);
    const [graphSnapshotStatus, setGraphSnapshotStatus] = useState(
        jobStatus.None
    );
    const [tableStatus, setTableStatus] = useState<jobStatus>(jobStatus.None);
    const { tags, updateTag, removeTag } = useTags(
        queryInfo?.tags ?? undefined
    );
    const [graphWarnings, setGraphWarnings] = useState<ErrWarnDetails[]>([]);
    const [trainTableWarnings, setTrainTableWarnings] = useState<
        ErrWarnDetails[]
    >([]);
    const { addTag, deleteTag } = useJobTagMutations();
    const queryClient = useQueryClient();

    const navigate = useCustomNavigation();

    const source = getJobSource(queryInfo?.jobSource);
    const status =
        graphSnapshotStatus === jobStatus.Queued
            ? jobStatus.Queued
            : queryInfo?.trainingStatus;

    const properties = useTrainingJobProperties({
        queryInfo,
        jobId: showRelatedJobs ? null : title,
        status,
        source
    });

    const roughPercentProgress = useCallback(
        (): number => estimateJobProgressPercent(queryInfo?.trialProgress),
        [queryInfo?.trialProgress]
    );

    const headerBpbutton =
        queryInfo?.trainingStatus === jobStatus.Done ? (
            <Button
                id="create-prediction"
                className="px-2"
                variant="primary"
                onClick={() =>
                    navigate(projectPaths.NewPredictionJob, {
                        state: { queryInfo }
                    })
                }>
                <SoundSampler />
                Generate predictions
            </Button>
        ) : null;

    const progressInfoChild = useMemo(() => {
        if (queryInfo?.trainingStatus === jobStatus.Failed) {
            if (!isValid(queryInfo?.trainingStatusMessage)) return null;
            return <div>{queryInfo.trainingStatusMessage}</div>;
        }
        if (buttons?.length > 0) {
            return (
                <>
                    {queryInfo?.trainingStatus === jobStatus.Running &&
                        queryInfo?.log?.stage_name === "Training" && (
                            <>
                                <ProgressBar
                                    progress={roughPercentProgress()}
                                    color={
                                        StatusColors[queryInfo?.trainingStatus]
                                    }
                                />
                                <Spacer height={16} />
                            </>
                        )}
                    {buttons}
                </>
            );
        }
        return null;
    }, [
        queryInfo?.trainingStatus,
        queryInfo?.trainingStatusMessage,
        queryInfo?.log?.stage_name,
        buttons,
        roughPercentProgress
    ]);

    // Display appropriate title based on job status
    const progressTitle = useMemo(() => {
        let title: string =
            TrainingStatusTitle[queryInfo?.trainingStatus as jobStatus];

        // When running, show current stage if not "Training"
        if (queryInfo?.trainingStatus === jobStatus.Running) {
            if (queryInfo?.log?.stage_name !== "Training") {
                title = queryInfo.log.stage_name;
            }
        }

        // When failed, use generic title and let failure_info.message (error body) provide details
        // The Related Jobs cards show which phase failed via status indicators
        if (queryInfo?.trainingStatus === jobStatus.Failed) {
            title = isValid(queryInfo?.trainingStatusMessage)
                ? TrainingStatusTitle[jobStatus.Failed] // "Training job failed"
                : jobErrTitle; // Fallback if no error message
        }

        return title;
    }, [
        queryInfo?.trainingStatus,
        queryInfo?.trainingStatusMessage,
        queryInfo?.log
    ]);

    const onCopyJobClick = () => {
        const copyTrainingJobArgs = assembleCopyTrainingJobArgs(queryInfo);
        const copyTrainingJobTags = Object.fromEntries(
            Object.entries(queryInfo?.tags ?? {}).filter(
                ([key]) => !PROTECTED_TAG_KEYS.has(key)
            )
        );

        navigate(projectPaths.NewTrainingJob, {
            state: {
                copytrainingargs: copyTrainingJobArgs,
                copytrainingtags: copyTrainingJobTags,
                isSDKJob: queryInfo?.jobSource === JobSource.SDK,
                taskType: queryInfo?.taskType ?? ""
            }
        });
    };

    const showStatusCard =
        status === jobStatus.Failed ||
        status === jobStatus.Queued ||
        status === jobStatus.Running;

    const isFetchingStatus =
        ((graphSnapshotStatus === jobStatus.None &&
            queryInfo?.graphSnapshotId) ||
            tableStatus === jobStatus.None) &&
        status === jobStatus.Failed;

    const handleTagAdd = useCallback(
        async (key: string, value: string | undefined) => {
            // Update local state optimistically
            updateTag(key, value as string);

            try {
                // Call API to add tag on the server
                await addTag(title, key, value);
                invalidateJobQueries(queryClient, JobType.TRAINING_JOB, title);
            } catch (error) {
                console.error("TrainingJob ADD TAG ERROR", error);
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
                invalidateJobQueries(queryClient, JobType.TRAINING_JOB, title);
            } catch (error) {
                console.error("----TrainingJob REMOVE TAG ERROR", error);
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
        const withAllWarnings = [
            ...withGraphWarnings,
            ...trainTableWarnings.filter(
                (tw) => !withGraphWarnings.some((w) => compare(w, tw))
            )
        ];

        return withAllWarnings;
    }, [graphWarnings, showRelatedJobs, trainTableWarnings, warnings]);

    return (
        <>
            <PageHeader
                screenshotTestId="model-detail-page"
                loadingStatus={loadingStatus}
                hidePreTitle={hidePreTitle}
                preTitle={"Training Job"}
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
                    ) : isArchived ? (
                        <React.Fragment key="header-buttons">
                            {allWarnings.length > 0 && (
                                <WarningsPopover warnings={allWarnings} />
                            )}
                            <Button
                                data-testid="restore-button"
                                variant="outline"
                                className="hover:bg-kui-bg-hover border-border-primary"
                                onClick={onClickRecover}>
                                <ArchiveRestore className="size-4" />
                                Restore archive
                            </Button>
                        </React.Fragment>
                    ) : (
                        <React.Fragment key="header-buttons">
                            {headerBpbutton}
                            {allWarnings.length > 0 && (
                                <WarningsPopover warnings={allWarnings} />
                            )}
                            <OverviewDropdown
                                key="overview-dropdown"
                                id="tarining-overview-dropdown"
                                buttonClassName="h-9 w-9 px-2 py-2 hover:bg-kui-bg-hover border-border-primary"
                                actionItems={[
                                    ...(status !== jobStatus.Running &&
                                    status !== jobStatus.Queued
                                        ? [
                                              {
                                                  content: (
                                                      <>
                                                          <Duplicate className="size-4 fill-icon-secondary" />{" "}
                                                          Duplicate
                                                      </>
                                                  ),
                                                  disabled: false,
                                                  onSelect: onCopyJobClick,
                                                  tooltipText: ""
                                              }
                                          ]
                                        : []),
                                    {
                                        content: (
                                            <>
                                                <CodeIcon className="size-4" />{" "}
                                                {" SDK Code "}
                                            </>
                                        ),
                                        disabled: false,
                                        onSelect: () => setIsModalOpen(true),
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
                        </React.Fragment>
                    )
                }
            />

            {!hidePreTitle && <Spacer height={16} />}
            <PropertiesOverviewCard
                properties={properties}
                hideTitle={!showRelatedJobs}
                queryContent={
                    queryInfo?.queryYaml ? (
                        <JobQueryPreview
                            value={queryInfo?.queryYaml?.trim() || ""}
                        />
                    ) : undefined
                }
                tags={tags}
                loading={loadingStatus}
                onTagAdd={handleTagAdd}
                onTagRemove={handleTagRemove}
                cardClassName={showRelatedJobs ? "" : "pb-0"}
                protectedKeys={PROTECTED_TAG_KEYS}
            />

            {!hidePreTitle && <Spacer height={16} />}
            {isFetchingStatus ? (
                <Skeleton className="w-full h-20 mb-4" />
            ) : (
                <>
                    {queryInfo && showStatusCard && (
                        <StatusCard
                            title={progressTitle}
                            titleIcon={
                                <StatusIcon
                                    status={status as jobStatus}
                                    iconMap={jobStatusIconMap}
                                />
                            }
                            className={"mb-4 rounded-lg"}
                            id={"test"}
                            variant={getColorVariant(status)}
                            enableDivider={status !== jobStatus.Failed}
                            contentChild={progressInfoChild}></StatusCard>
                    )}
                </>
            )}
            {queryInfo && (
                <ModelPerformanceOverview
                    queryInfo={queryInfo}
                    ctaMsgs={ctaMsgs}
                    handleError={handleError}
                    status={status}
                />
            )}

            {showRelatedJobs === true && (
                <>
                    <Spacer height={16} />

                    {queryInfo?.graphSnapshotId ? (
                        <RelatedJobsCard
                            jobName="Graph Snapshot"
                            jobId={getIdHash(queryInfo?.graphSnapshotId)}
                            loadingStatus={loadingStatus}
                            status={graphSnapshotStatus}>
                            <GraphSnapshotOverview
                                id={queryInfo?.graphSnapshotId}
                                queryInfo={queryInfo}
                                hidePreTitle
                                setStatus={setGraphSnapshotStatus}
                                parentJobSource={source}
                                setWarnings={setGraphWarnings}
                            />
                        </RelatedJobsCard>
                    ) : queryInfo?.graphName ? (
                        <RelatedJobsCard
                            jobName="Graph"
                            jobId={queryInfo?.graphName}
                            loadingStatus={loadingStatus}>
                            <GraphOverview
                                graphId={queryInfo?.graphName}
                                onCopyClick={() => {}}
                                onDeleteClick={() => {}}
                                onEditClick={() => {}}
                                isRelatedJob={true}
                            />
                        </RelatedJobsCard>
                    ) : (
                        <></>
                    )}
                    {queryInfo?.trainTableJobId && (
                        <RelatedJobsCard
                            jobName="Training Table Generation"
                            jobId={getIdHash(queryInfo?.trainTableJobId)}
                            loadingStatus={loadingStatus}
                            status={tableStatus}>
                            <TableGenerationOverview
                                id={queryInfo?.trainTableJobId}
                                hidePreTitle
                                onClose={() => {}}
                                showRelatedJobs={false}
                                onTableStatusUpdate={setTableStatus}
                                modelPlan={
                                    queryInfo?.modelPlannerYaml as string
                                }
                                trainingPlan={queryInfo?.trainingPlan as string}
                                isTrainingPage={true}
                                queryInfo={queryInfo}
                                setWarnings={setTrainTableWarnings}
                            />
                        </RelatedJobsCard>
                    )}
                </>
            )}
            {validationComponent}
            {isModalOpen && (
                <SDKCodeModal
                    id={queryInfo?.trainingJobId as string}
                    open={isModalOpen}
                    toggleModel={() => setIsModalOpen(!isModalOpen)}
                    objectTypeToGenScript={SDKObjectType.TrainingJob}
                />
            )}
        </>
    );
};

export default WithErrorBoundary(TrainingJobOverviewDetail);
