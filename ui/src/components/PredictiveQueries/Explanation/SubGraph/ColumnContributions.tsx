// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useMemo, useState } from "react";

import {
    type ColumnDef,
    DataTable,
    Info,
    InkSelection,
    Key,
    PREDICTED_NODE_TYPE_INFO,
    type SortingState,
    Tooltip,
    TooltipContent,
    TooltipTrigger,
    TruncatedTooltip
} from "@kumo-ai/kumo-ui-catalog";

import { SubgraphContributionColumns } from "components/@ui/TableColumns/SubgraphContributions";

import type { GraphNodeData, XaiColumnArgs } from "types/explorer";
import type { ColumnArgs } from "types/tables";

import { isValid } from "utils/Helpers/typeGuards";

interface ColumnContributionsProps {
    selectedNode: GraphNodeData | null;
    tableColor: string;
}

export function ColumnContributions({
    selectedNode,
    tableColor
}: ColumnContributionsProps) {
    const [sortingState, setSortingState] = useState<SortingState>([
        { id: "grad_score", desc: true }
    ]);

    const scoreIncluded = useMemo(
        () => selectedNode?.cols?.some((c) => c.grad_score != null),
        [selectedNode]
    );

    const columns = useMemo(
        () => SubgraphContributionColumns(scoreIncluded),
        [scoreIncluded]
    );

    const data = useMemo(() => {
        if (!selectedNode?.cols) return [];
        return [...selectedNode.cols].filter(isValid);
    }, [selectedNode]);

    if (!selectedNode) {
        return (
            <div className="flex h-full flex-col items-center justify-center gap-3 text-center text-sm">
                <InkSelection className="size-5 fill-kumo-grey-400" />
                <p className="self-stretch text-center text-sm font-medium leading-6 text-title-secondary">
                    Click a node to see each
                    <br />
                    column&apos;s gradient score.
                </p>
            </div>
        );
    }

    const tableName = selectedNode.tableName?.includes(".")
        ? selectedNode.tableName.split(".").pop()
        : selectedNode.tableName;

    return (
        <div className="flex flex-col overflow-hidden gap-4">
            <div className="flex flex-col gap-2 pt-0">
                <div className="flex items-center gap-2">
                    <span
                        className="size-3 rounded-full flex-shrink-0"
                        style={{ backgroundColor: tableColor }}
                    />
                    <span className="text-base font-bold leading-4 text-title-primary">
                        {tableName}
                    </span>
                </div>
                <div className="flex min-w-0 items-center gap-4 text-sm font-medium leading-6 text-title-secondary">
                    {selectedNode.idName ? (
                        <div className="flex min-w-0 shrink items-center gap-2">
                            <Key className="size-3.5 shrink-0 fill-pink-600" />
                            <span className="shrink-0">
                                {selectedNode.idName}
                            </span>
                            <TruncatedTooltip
                                className="truncate"
                                tooltipContent={selectedNode.value}>
                                {selectedNode.value}
                            </TruncatedTooltip>
                        </div>
                    ) : (
                        <TruncatedTooltip
                            className="truncate"
                            tooltipContent={selectedNode.value}>
                            {selectedNode.value}
                        </TruncatedTooltip>
                    )}
                    {selectedNode.hopLevel !== undefined && (
                        <span className="shrink-0">
                            {selectedNode.hopLevel === 0
                                ? "Entity"
                                : `Hop ${selectedNode.hopLevel}`}
                        </span>
                    )}
                    {selectedNode.nodeType ? (
                        // TLP: type label with info-icon tooltip replaces the
                        // aggregate "Importance score" — gradient scores aren't
                        // computed for predicted nodes.
                        <span className="flex shrink-0 items-center gap-1">
                            {
                                PREDICTED_NODE_TYPE_INFO[selectedNode.nodeType]
                                    .label
                            }
                            <Tooltip delayDuration={0}>
                                <TooltipTrigger asChild>
                                    <Info className="text-title-secondary size-3.5 shrink-0" />
                                </TooltipTrigger>
                                <TooltipContent
                                    side="bottom"
                                    variant="dark"
                                    className="max-w-60">
                                    {
                                        PREDICTED_NODE_TYPE_INFO[
                                            selectedNode.nodeType
                                        ].description
                                    }
                                </TooltipContent>
                            </Tooltip>
                        </span>
                    ) : (
                        <span className="shrink-0">
                            Importance score{" "}
                            {selectedNode.totalScore.toFixed(2)}
                        </span>
                    )}
                </div>
            </div>
            <div className="flex-1 overflow-y-auto overflow-x-hidden [&_table]:table-fixed [&_td]:text-title-primary">
                <DataTable
                    id="subgraph-node-contributions"
                    disableRowSelectionHover
                    columns={
                        columns as ColumnDef<
                            ColumnArgs & XaiColumnArgs,
                            unknown
                        >[]
                    }
                    data={data}
                    enableGlobalFilter={false}
                    defaultPageSize={10}
                    pageCount={Math.ceil(data.length / 10) || 1}
                    showTopRoundedBorder={false}
                    manualSortingState={
                        scoreIncluded
                            ? sortingState
                            : sortingState.filter((s) => s.id !== "grad_score")
                    }
                    setManualSortingState={setSortingState}
                />
            </div>
        </div>
    );
}
