// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
    Badge,
    Card,
    CellContext,
    ColumnDef,
    Description,
    HoverCard,
    HoverCardContent,
    HoverCardTrigger,
    Row,
    Skeleton,
    Symbol,
    Tag,
    Tooltip,
    TooltipContent,
    TooltipTrigger
} from "@kumo-ai/kumo-ui-catalog";
import { GenericJob } from "gen/v0/types";

import PQCodeEditor from "components/CodeEditor/PQCodeEditor";

import { TaskType } from "types/predictiveQueries";

import { jobStatus } from "utils/Constants";
import { getKeys } from "utils/Helpers/objectHelpers";
import {
    capitalizeFirstLetter,
    formatQueryForTooltip,
    getIdHash,
    toPascalCaseWithSpace
} from "utils/Helpers/stringHelpers";
import { getColorVariant, isPredictionsJobsPage } from "utils/jobs";
import {
    formatToLongLocalDateString,
    jobRunTimeFromJobResponse,
    timeDifference
} from "utils/timeConverter";

export const JobListColumns = (
    queryYamlResponse: { [key: string]: string },
    taskType: { [key: string]: TaskType },
    activeJobsResponse: { [job_id: string]: GenericJob | jobStatus }
) => {
    const SIDE_BAR_WIDTH = 234 + 44;
    const COLUMN_WIDTH = 150;
    const columnList: ColumnDef<GenericJob>[] = [
        {
            id: "job_id",
            header: () => <span className="ml-0.5">Job ID</span>,
            accessorKey: "job_id",
            size: 150,
            cell: ({ row }: { row: Row<GenericJob> }) => {
                if (row?.original?.job_id) {
                    const jobId = row?.original?.job_id;

                    return (
                        <HoverCard
                            id={`${jobId}-tooltip`}
                            openDelay={100}
                            closeDelay={0}>
                            <HoverCardTrigger>
                                <div id={`${jobId}`}>{getIdHash(jobId)}</div>
                            </HoverCardTrigger>

                            <HoverCardContent
                                arrow={false}
                                className="p-4 max-w-120">
                                <div className="text-title-primary font-medium leading-4">
                                    {!taskType ||
                                    getKeys(taskType ?? {}).length < 1 ? (
                                        <Skeleton className="w-40 h-5" />
                                    ) : !taskType[jobId] ? (
                                        <Symbol type={"emdash"}></Symbol>
                                    ) : (
                                        toPascalCaseWithSpace(taskType[jobId])
                                    )}
                                </div>
                                <Description
                                    size="sm"
                                    variant="secondary"
                                    className="font-medium">
                                    {"Query"}
                                </Description>
                                {queryYamlResponse &&
                                queryYamlResponse[jobId] ? (
                                    <Card className="px-2">
                                        <PQCodeEditor
                                            value={formatQueryForTooltip(
                                                queryYamlResponse[jobId]
                                            )}
                                            language={"PQLANG_V2"}
                                            editable={false}
                                            isInvalid={false}
                                            errorMsg={""}
                                            onChange={() => {}}
                                            graphTables={[]}
                                            lineNumbers={false}
                                            highlightActiveLineGutter={false}
                                            highlightActiveLine={false}
                                            showCopyButton={false}
                                            hideBorder
                                            fontSize={14}
                                        />
                                    </Card>
                                ) : (
                                    <Skeleton className="w-100 h-16" />
                                )}
                            </HoverCardContent>
                        </HoverCard>
                    );
                } else return <></>;
            },
            meta: {
                sticky: { type: "always" as const, left: true, zIndex: 20 }
            }
        },
        {
            id: "metadata.status",
            header: "Status",
            size: 150,
            accessorKey: "metadata.status",
            cell: ({ row }: { row: Row<GenericJob> }) => {
                const jobId = row?.original?.job_id;
                const status =
                    activeJobsResponse && jobId in activeJobsResponse
                        ? typeof activeJobsResponse[jobId] === "string"
                            ? activeJobsResponse[jobId]
                            : activeJobsResponse[jobId]?.metadata?.status
                        : row?.original?.metadata?.status;
                if (status)
                    return (
                        <Badge
                            size="md"
                            variant={getColorVariant(status as jobStatus)}>
                            {capitalizeFirstLetter(status?.toLowerCase())}
                        </Badge>
                    );
                else return <></>;
            }
        },
        {
            id: "task_type",
            header: "Task Type",
            accessorKey: "task_type",
            size: 150,
            enableSorting: false,
            cell: ({ row }: { row: Row<GenericJob> }) => {
                const jobId = row?.original?.job_id;
                if (!taskType || !taskType[jobId]) {
                    return <Symbol type="emdash" />;
                }
                return (
                    <div className="text-title-secondary">
                        {toPascalCaseWithSpace(taskType[jobId])}
                    </div>
                );
            }
        },
        {
            id: "source",
            header: "Source",
            accessorKey: "source",
            size: 120,
            enableSorting: false,
            cell: ({ row }: { row: Row<GenericJob> }) => {
                const source = row?.original?.source;
                if (!source) return <Symbol type="emdash" />;
                return <div className="text-title-secondary">{source}</div>;
            }
        },
        {
            id: "test_metric",
            header: () => (
                <Tooltip id={"Column-Name"}>
                    <TooltipTrigger>
                        <span>Tune Metric</span>
                    </TooltipTrigger>
                    <TooltipContent arrow={false} sideOffset={5}>
                        {
                            "The value of a tune metric is the score from the best experiment on the holdout set"
                        }
                    </TooltipContent>
                </Tooltip>
            ),
            accessorKey: "test_metric",
            size: 150,
            sortDescFirst: false,
            cell: ({ row }: { row: Row<GenericJob> }) => {
                if (row?.original?.test_metric) {
                    const tuneMetricName = row?.original?.test_metric?.name;
                    const tuneMetricValue =
                        row?.original?.test_metric?.val?.toFixed(2);
                    return (
                        <Tooltip
                            id={`${tuneMetricName}-tooltip`}
                            delayDuration={100}
                            skipDelayDuration={0}>
                            <TooltipTrigger>
                                <span className="m-2 text-title-secondary">
                                    {tuneMetricValue}
                                </span>
                            </TooltipTrigger>
                            <TooltipContent arrow={false}>
                                {tuneMetricName}
                            </TooltipContent>
                        </Tooltip>
                    );
                } else
                    return <Symbol className="px-2" type={"emdash"}></Symbol>;
            }
        },
        {
            id: "metadata.start_time",
            header: "Started",
            size: 150,
            accessorKey: "metadata.start_time",
            cell: ({ row }: { row: Row<GenericJob> }) => {
                if (row?.original?.metadata?.start_time) {
                    return (
                        <Tooltip
                            id={
                                row.original.job_id +
                                row?.original?.metadata?.start_time?.toString()
                            }
                            delayDuration={100}
                            skipDelayDuration={0}>
                            <TooltipTrigger>
                                <div className="text-title-secondary">
                                    {timeDifference(
                                        row.original.metadata.start_time
                                    )}
                                </div>
                            </TooltipTrigger>
                            <TooltipContent arrow={false}>
                                {formatToLongLocalDateString(
                                    row.original.metadata.start_time
                                )}
                            </TooltipContent>
                        </Tooltip>
                    );
                } else return <></>;
            }
        },
        {
            id: "user",
            header: "User",
            accessorKey: "user",
            enableSorting: false,
            size: 150,
            cell: ({ getValue }: CellContext<GenericJob, unknown>) => {
                const value = getValue() as string;
                if (value?.trim()) {
                    return (
                        <div className="text-title-secondary overflow-hidden text-ellipsis whitespace-nowrap max-w-30">
                            {value}
                        </div>
                    );
                }
                return <span className="text-title-secondary">Admin user</span>;
            }
        },
        {
            id: "duration",
            header: "Duration",
            accessorKey: "duration",
            size: 100,
            sortDescFirst: false,
            cell: ({ row }: { row: Row<GenericJob> }) => {
                const job = row?.original;
                const metadata = job?.metadata;
                // Until the queue admits a job there is no execution
                // start_time, so no runtime can be reported yet.
                const hasStartedExecution =
                    metadata?.status !== jobStatus.Queued &&
                    metadata?.status !== jobStatus.Running &&
                    metadata?.status !== jobStatus.Untrained;
                const duration =
                    hasStartedExecution &&
                    metadata?.start_time &&
                    metadata?.end_time &&
                    job?.job_id
                        ? jobRunTimeFromJobResponse(
                              metadata.start_time,
                              metadata.end_time
                          )
                        : "";
                return duration ? (
                    <div className="text-title-secondary">{duration}</div>
                ) : (
                    <Symbol type="emdash" />
                );
            }
        }
    ];
    const ALL_COLUMNS =
        COLUMN_WIDTH *
        (isPredictionsJobsPage(
            window?.location?.pathname ?? "",
            window?.location?.search ?? ""
        )
            ? columnList.length - 1
            : columnList.length);

    columnList.push({
        id: "metadata.tags",
        header: "Tags",
        accessorKey: "metadata.tags",
        sortDescFirst: false,
        size: window.innerWidth - SIDE_BAR_WIDTH - ALL_COLUMNS,
        cell: ({ row }: { row: Row<GenericJob> }) => (
            <div className="flex gap-2">
                {Object.entries(row?.original?.metadata?.tags ?? {})?.map(
                    ([tagKey, tag], index) => {
                        if (tagKey.startsWith("pquery_id")) return null;
                        let newTagKey = tagKey;
                        let newTag = tag;
                        if (tag.startsWith("trainingjob-")) {
                            newTagKey = "TRAININGJOBID";
                            newTag = getIdHash(tag);
                        }
                        return (
                            <Tag
                                key={`${tagKey}-${tag}-${index}`}
                                tagKey={newTagKey}
                                value={newTag}
                            />
                        );
                    }
                )}
            </div>
        )
    });
    return columnList;
};
