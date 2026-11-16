// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ChevronDown, ChevronUp } from "@/icons";
import {
    getShapeForNodeType,
    PREDICTED_NODE_TYPE_INFO,
    PREDICTED_TYPE_LEGEND_ORDER
} from "@/utils/xai-subgraph-shapes";

import { NodeShape } from "./node-shape";

const LEGEND_PANEL_ID = "xai-subgraph-legend-panel";

export function LegendPopover({
    isOpen,
    setIsOpen,
    items,
    zoom,
    tierLabels,
    predictedTypesSampleColor
}: {
    isOpen: boolean;
    setIsOpen: (v: boolean) => void;
    items: { color: string; label: string; isEntity: boolean }[];
    zoom: number;
    tierLabels: string[];
    /** If set, render the "Predicted node types" section (TLP tasks only). */
    predictedTypesSampleColor?: string;
}) {
    // Radii derived from tier count: bottom tier = 3.5px, +0.5px per tier.
    const tiersDesc = [...tierLabels].reverse().map((label, i) => ({
        label,
        halfSize: 3.5 + (tierLabels.length - 1 - i) * 0.5
    }));

    return (
        <div className="relative">
            {isOpen && (
                <div
                    id={LEGEND_PANEL_ID}
                    className="absolute bottom-full left-0 mb-2 w-56 rounded-sm border border-gray-200 bg-white p-3 shadow-md">
                    <div className="flex flex-col gap-3">
                        {/* Tables */}
                        <div className="flex flex-col gap-1.5">
                            <span className="text-title-secondary text-xs font-medium">
                                Tables
                            </span>
                            {items.map((item) => (
                                <div
                                    key={item.label}
                                    className="flex items-center gap-2 text-xs">
                                    <span
                                        className="size-2.5 shrink-0 rounded-full"
                                        style={{ backgroundColor: item.color }}
                                    />
                                    <span className="text-title-secondary truncate">
                                        {item.label}
                                        {item.isEntity && " (entity)"}
                                    </span>
                                </div>
                            ))}
                        </div>
                        {/* Predicted node types (TLP only) */}
                        {predictedTypesSampleColor && (
                            <div className="flex flex-col gap-1.5">
                                <span className="text-title-secondary text-xs font-medium">
                                    Predicted node types
                                </span>
                                {PREDICTED_TYPE_LEGEND_ORDER.map((type) => {
                                    const { variant, hasRing } =
                                        getShapeForNodeType(type);
                                    return (
                                        <div
                                            key={type}
                                            className="flex items-center gap-2 text-xs">
                                            <span className="flex size-3.5 shrink-0 items-center justify-center">
                                                <NodeShape
                                                    variant={variant}
                                                    size={10}
                                                    fillColor={
                                                        predictedTypesSampleColor
                                                    }
                                                    hasRing={hasRing}
                                                />
                                            </span>
                                            <span className="text-title-secondary">
                                                {
                                                    PREDICTED_NODE_TYPE_INFO[
                                                        type
                                                    ].label
                                                }
                                            </span>
                                        </div>
                                    );
                                })}
                            </div>
                        )}
                        {/* Node size (only when consumer provides tier labels) */}
                        {tiersDesc.length > 0 && (
                            <div className="flex flex-col gap-1.5">
                                <span className="text-title-secondary text-xs font-medium">
                                    Node size = rank by total score
                                </span>
                                {tiersDesc.map(({ label, halfSize }) => {
                                    const size = halfSize * 2 * zoom;
                                    return (
                                        <div
                                            key={label}
                                            className="text-title-secondary flex items-center gap-2 text-xs">
                                            <span
                                                className="shrink-0 rounded-full bg-slate-400"
                                                style={{
                                                    width: size,
                                                    height: size
                                                }}
                                            />
                                            <span>{label}</span>
                                        </div>
                                    );
                                })}
                            </div>
                        )}
                    </div>
                </div>
            )}
            <button
                onClick={() => setIsOpen(!isOpen)}
                aria-label="Toggle legend"
                aria-expanded={isOpen}
                aria-controls={LEGEND_PANEL_ID}
                className="rounded-4xl outline-(--color-border-secondary) focus-visible:ring-(--color-border-secondary) inline-flex h-6 items-center justify-start gap-2 bg-white px-4 py-2 outline-1 -outline-offset-1 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-1">
                <div className="flex items-center justify-start gap-1 overflow-hidden">
                    <div className="flex items-center justify-start gap-0.5">
                        {items.slice(0, 5).map((item) => (
                            <span
                                key={item.label}
                                className="size-2.5 shrink-0 rounded-full"
                                style={{ backgroundColor: item.color }}
                            />
                        ))}
                    </div>
                    <span className="text-title-secondary text-sm font-medium leading-5">
                        Legend
                    </span>
                </div>
                {isOpen ? (
                    <ChevronDown className="text-title-secondary size-4 shrink-0" />
                ) : (
                    <ChevronUp className="text-title-secondary size-4 shrink-0" />
                )}
            </button>
        </div>
    );
}
