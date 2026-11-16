// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Predictions panel: a visual-only background node + one interactive node per
// item. Layout helpers live in @/utils/visual-graph (kept separate so this
// file only exports components — required by react-refresh/only-export-components).

import { memo } from "react";

import { Handle, type Node, type NodeProps, Position } from "@xyflow/react";

import {
    Tooltip,
    TooltipContent,
    TooltipTrigger
} from "@/components/ui/tooltip";
import {
    type PredictionItemNodeData,
    type PredictionsPanelBackgroundData
} from "@/types/visual-graph";
import {
    getShapeForNodeType,
    PREDICTED_NODE_TYPE_INFO
} from "@/utils/xai-subgraph-shapes";

import { NodeShape } from "./node-shape";

/** Default size for items without an LHS counterpart (exploratory / hit). */
const DEFAULT_PREDICTION_ITEM_SIZE = 10;

export const PredictionsPanelBackgroundNode = memo(
    function PredictionsPanelBackgroundNode({
        data
    }: NodeProps<Node<PredictionsPanelBackgroundData>>) {
        return (
            <div
                data-slot="predictions-panel-background"
                style={{ width: data.width, height: data.height }}
                className="border-color-border-secondary pointer-events-none rounded-md border border-dashed bg-white px-6 py-4">
                <Handle
                    type="target"
                    position={Position.Left}
                    className="h-0! min-h-0! w-0! min-w-0! border-0! opacity-0!"
                />
                <div className="text-title-primary text-xs font-medium leading-4">
                    {data.title}
                </div>
            </div>
        );
    }
);

export const PredictionItemNode = memo(function PredictionItemNode({
    data
}: NodeProps<Node<PredictionItemNodeData>>) {
    const { item, isHovered, isSelected, isDimmed } = data;
    const { variant, hasRing } = getShapeForNodeType(item.nodeType);
    const typeLabel = PREDICTED_NODE_TYPE_INFO[item.nodeType].label;
    const hasGraphPresence = item.hopLevel !== undefined;
    return (
        <Tooltip delayDuration={0}>
            <TooltipTrigger asChild>
                <div
                    data-slot="prediction-item"
                    className="overflow-visible! relative flex cursor-pointer flex-col items-center gap-0.5">
                    <span
                        className="text-title-secondary text-[10px] font-medium leading-4 transition-opacity"
                        style={{ opacity: isDimmed ? 0.3 : 1 }}>
                        #{item.rank}
                    </span>
                    <NodeShape
                        variant={variant}
                        size={
                            item.halfSize !== undefined
                                ? item.halfSize * 2
                                : DEFAULT_PREDICTION_ITEM_SIZE
                        }
                        fillColor={item.color}
                        hasRing={hasRing}
                        isHovered={isHovered}
                        isSelected={isSelected}
                        opacity={isDimmed ? 0.3 : 1}
                    />
                </div>
            </TooltipTrigger>
            <TooltipContent side="bottom" variant="dark">
                {hasGraphPresence ? (
                    <div className="flex flex-col gap-2 py-1">
                        <div>{typeLabel}</div>
                        <div className="flex justify-between gap-4">
                            <span>{item.tableName}</span>
                            <span>
                                {item.hopLevel === 0
                                    ? "Entity"
                                    : `Hop ${item.hopLevel}`}
                            </span>
                        </div>
                    </div>
                ) : (
                    <div className="flex justify-between gap-4 py-1">
                        <span>{item.tableName}</span>
                        <span>{typeLabel}</span>
                    </div>
                )}
            </TooltipContent>
        </Tooltip>
    );
});
