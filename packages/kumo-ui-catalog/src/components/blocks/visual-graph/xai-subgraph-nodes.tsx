// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { memo } from "react";

import { Handle, type Node, type NodeProps, Position } from "@xyflow/react";

import {
    Tooltip,
    TooltipContent,
    TooltipTrigger
} from "@/components/ui/tooltip";
import { type ChipNodeData, type SubgraphNodeData } from "@/types/visual-graph";
import { ENTITY_PREDICTIONS_HANDLE_ID } from "@/utils/xai-subgraph-panel-layout";
import {
    getNodeFootprint,
    getShapeForNodeType,
    PREDICTED_NODE_TYPE_INFO
} from "@/utils/xai-subgraph-shapes";

import { NodeShape } from "./node-shape";

export const SubgraphNode = memo(function SubgraphNode({
    data
}: NodeProps<Node<SubgraphNodeData>>) {
    const { variant, hasRing } = getShapeForNodeType(data.nodeType);
    const isDimmed =
        data.isDimmedByHover ||
        data.status === "path-only" ||
        data.status === "table-excluded";
    const boxSize = getNodeFootprint(data.halfSize, data.nodeType);
    const cursor = data.status === "active" ? "pointer" : "default";
    const isHovered = data.isHovered && data.status === "active";
    const opacity = isDimmed ? 0.3 : 1;
    // overflow-visible! overrides TooltipTrigger's overflow-hidden so hover/select rings don't clip.
    return (
        <Tooltip delayDuration={0}>
            <TooltipTrigger asChild>
                <div
                    data-slot="subgraph-node"
                    style={{ width: boxSize, height: boxSize, cursor }}
                    className="overflow-visible! relative transition-opacity duration-150">
                    <NodeShape
                        variant={variant}
                        size={data.halfSize * 2}
                        fillColor={data.color}
                        hasRing={hasRing}
                        isHovered={isHovered}
                        isSelected={data.isSelected}
                        opacity={opacity}
                    />
                    <Handle
                        type="target"
                        position={Position.Top}
                        className="h-0! min-h-0! w-0! min-w-0! border-0! opacity-0!"
                    />
                    <Handle
                        type="source"
                        position={Position.Bottom}
                        className="h-0! min-h-0! w-0! min-w-0! border-0! opacity-0!"
                    />
                    {/* Right-side source handle used only by the entity→panel dashed edge. */}
                    <Handle
                        type="source"
                        id={ENTITY_PREDICTIONS_HANDLE_ID}
                        position={Position.Right}
                        className="h-0! min-h-0! w-0! min-w-0! border-0! opacity-0!"
                    />
                </div>
            </TooltipTrigger>
            {data.status === "active" && (
                <TooltipContent side="bottom" variant="dark">
                    <div className="flex flex-col gap-2 py-1">
                        {data.nodeType && (
                            <div>
                                {PREDICTED_NODE_TYPE_INFO[data.nodeType].label}
                            </div>
                        )}
                        <div className="flex justify-between gap-4">
                            <span>{data.graphData.tableName}</span>
                            <span>
                                {data.graphData.hopLevel === 0
                                    ? "Entity"
                                    : `Hop ${data.graphData.hopLevel}`}
                            </span>
                        </div>
                        {/* Predicted-type nodes have no importance score (gradients not computed). */}
                        {!data.nodeType && (
                            <div className="flex justify-between gap-4">
                                <span>Importance score</span>
                                <span>
                                    {data.graphData.totalScore.toFixed(2)}
                                </span>
                            </div>
                        )}
                    </div>
                </TooltipContent>
            )}
        </Tooltip>
    );
});

export const CollapseChipNode = memo(function CollapseChipNode({
    data
}: NodeProps<Node<ChipNodeData>>) {
    return (
        <div
            data-slot="collapse-chip"
            className="border-(--color-border-primary) relative inline-flex cursor-pointer items-center justify-center rounded-full border border-dashed bg-white px-1.5 py-px">
            <span className="text-title-primary text-[10px] font-medium leading-none">
                + {data.count}
            </span>
            <Handle
                type="target"
                position={Position.Top}
                style={{ top: "50%" }}
                className="h-0! min-h-0! w-0! min-w-0! border-0! opacity-0!"
            />
        </div>
    );
});

export const CollapseExpandChipNode = memo(function CollapseExpandChipNode() {
    return (
        <div
            data-slot="collapse-expand-chip"
            className="border-(--color-border-primary) inline-flex h-3 cursor-pointer items-center justify-center rounded-full border border-dashed bg-white px-1.5">
            <span className="text-title-primary text-xs font-medium leading-none">
                {/* U+2212 MINUS SIGN */}−
            </span>
        </div>
    );
});
