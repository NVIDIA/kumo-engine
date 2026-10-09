// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
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

import { PredictiveQueryInfo } from "types/predictiveQueries";

import { getIdHash, toPascalCaseWithSpace } from "utils/Helpers/stringHelpers";
import {
    formatToLongLocalDateString,
    timeDifference
} from "utils/timeConverter";

export const TrainingJobSelectListColumns = (): ColumnDef<
    GenericJob & PredictiveQueryInfo
>[] => {
    const SIDE_BAR_WIDTH = 300;
    const FIXED_COLS_COUNT = 3; // job_id, user, started
    const TAGS_COL_RATIO = 3; // tags column gets 3x the width of each fixed column
    const availableWidth = window.innerWidth - SIDE_BAR_WIDTH;
    const FIXED_COL_WIDTH = Math.floor(
        availableWidth / (FIXED_COLS_COUNT + TAGS_COL_RATIO)
    );
    const ALL_COLUMNS = FIXED_COL_WIDTH * FIXED_COLS_COUNT;

    return [
        {
            id: "job_id",
            header: "Job ID",
            accessorKey: "job_id",
            size: FIXED_COL_WIDTH,
            meta: {
                sticky: { type: "always", left: true, zIndex: 20 }
            },
            cell: ({ row }: { row: Row<GenericJob & PredictiveQueryInfo> }) => {
                if (row?.original?.job_id) {
                    const jobId = row?.original?.job_id;
                    const taskType = row?.original?.taskType;

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
                                    {!taskType ? (
                                        <Symbol type={"emdash"}></Symbol>
                                    ) : (
                                        toPascalCaseWithSpace(taskType)
                                    )}
                                </div>
                                <Description
                                    size="sm"
                                    variant="secondary"
                                    className="font-medium">
                                    {"Query"}
                                </Description>
                                {row?.original?.queryYaml ? (
                                    <Card className="px-2">
                                        <PQCodeEditor
                                            value={row?.original?.queryYaml?.trim()}
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
            }
        },
        {
            id: "user",
            header: "User",
            accessorKey: "user",
            enableSorting: false,
            size: FIXED_COL_WIDTH,
            cell: ({
                getValue
            }: CellContext<GenericJob & PredictiveQueryInfo, unknown>) => {
                const value = getValue() as string;
                if (value?.trim()) {
                    return (
                        <span className="text-title-secondary overflow-hidden text-ellipsis whitespace-nowrap max-w-30">
                            {value}
                        </span>
                    );
                }
                return <span className="text-title-secondary">Admin user</span>;
            }
        },
        {
            id: "metadata.start_time",
            header: "Started",
            accessorKey: "metadata.start_time",
            size: FIXED_COL_WIDTH,
            cell: ({ row }: { row: Row<GenericJob & PredictiveQueryInfo> }) => {
                if (row?.original?.metadata?.start_time) {
                    return (
                        <Tooltip
                            id={`updated-${row?.original?.job_id}`}
                            delayDuration={100}
                            skipDelayDuration={0}>
                            <TooltipTrigger>
                                <span className="text-title-secondary">
                                    {timeDifference(
                                        row.original.metadata.start_time
                                    )}
                                </span>
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
            id: "metadata.tags",
            header: "Tags",
            accessorKey: "metadata.tags",
            size: window.innerWidth - (SIDE_BAR_WIDTH + ALL_COLUMNS),
            cell: ({ row }: { row: Row<GenericJob & PredictiveQueryInfo> }) => (
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
        }
    ];
};
