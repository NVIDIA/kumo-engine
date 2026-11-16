// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useState } from "react";

import type { Meta, StoryObj } from "@storybook/react-vite";
import { expect, userEvent, within } from "storybook/test";
import type { Edge, Node } from "@xyflow/react";

import { XaiSubgraph } from "@/components/blocks/visual-graph/xai-subgraph";
import { Sheet, SheetContent, SheetTitle } from "@/components/ui/sheet";
import {
    type NodeStatus,
    type SubgraphNodeData,
    type XaiSubgraphPredictions
} from "@/types/visual-graph";
import {
    buildTableColorMap,
    xaiSubgraphApplyLayout
} from "@/utils/visual-graph";
import {
    type PredictedNodeType,
    type PredictionItem
} from "@/utils/xai-subgraph-shapes";

const meta: Meta<typeof XaiSubgraph> = {
    title: "Blocks/XaiSubgraph",
    component: XaiSubgraph,
    tags: ["autodocs"],
    parameters: {
        layout: "padded",
        docs: { extractComponentDescription: () => "" }
    }
};

export default meta;
type Story = StoryObj<typeof XaiSubgraph>;

// ─── Sample data ──────────────────────────────────────────────────────────────
// Five-node e-commerce subgraph: customer entity at hop 0,
// orders + sessions at hop 1, products + reviews at hop 2.
//
// Score quartile tiers, radii derived at 3.5px + 0.5px per tier (bottom→top):
//   tier 3 (≥ q75 0.65): customer (0.85), orders (0.65)  → halfSize 5
//   tier 2 (≥ q50 0.45): products (0.45)                 → halfSize 4.5
//   tier 1 (≥ q25 0.30): reviews (0.30)                  → halfSize 4
//   tier 0 (< q25):      sessions (0.10)                 → halfSize 3.5

const TIER_LABELS = ["Bottom 25%", "50–75%", "25–50%", "Top 25%"];

type RawNodeDef = {
    id: string;
    tableName: string;
    hopLevel: number;
    totalScore: number;
    halfSize: number;
};

const ROOT_TABLE = "customers";

const RAW_NODES: RawNodeDef[] = [
    {
        id: "node_customer",
        tableName: "customers",
        hopLevel: 0,
        totalScore: 0.85,
        halfSize: 5
    },
    {
        id: "node_orders",
        tableName: "orders",
        hopLevel: 1,
        totalScore: 0.65,
        halfSize: 5
    },
    {
        id: "node_sessions",
        tableName: "sessions",
        hopLevel: 1,
        totalScore: 0.1,
        halfSize: 3.5
    },
    {
        id: "node_products",
        tableName: "products",
        hopLevel: 2,
        totalScore: 0.45,
        halfSize: 4.5
    },
    {
        id: "node_reviews",
        tableName: "reviews",
        hopLevel: 2,
        totalScore: 0.3,
        halfSize: 4
    }
];

const COLOR_MAP = buildTableColorMap(
    RAW_NODES.map((n) => n.tableName),
    ROOT_TABLE
);

const LEGEND_ITEMS = Object.entries(COLOR_MAP).map(([label, color]) => ({
    label,
    color,
    isEntity: label === ROOT_TABLE
}));

const BASE_EDGES: Edge[] = [
    { id: "e1", source: "node_customer", target: "node_orders" },
    { id: "e2", source: "node_customer", target: "node_sessions" },
    { id: "e3", source: "node_orders", target: "node_products" },
    { id: "e4", source: "node_orders", target: "node_reviews" }
];

// status overrides for the WithDimmedNodes story
const DIMMED_STATUS: Record<string, NodeStatus> = {
    node_sessions: "path-only",
    node_reviews: "path-only"
};

// ─── Link-prediction sample data ─────────────────────────────────────────────
// Same data shape covers both TLP and SLP — the catalog renders whatever inputs
// it's given; the TLP-vs-SLP distinction lives in the consumer (kumo/ui's
// useSubgraphTreeView). Same 5-node graph but with 3 product nodes that
// demonstrate the LHS link-prediction node types (organic / recurring /
// recurring_hit). Predictions panel adds 5 ranked items showing all 5 shape
// variants. Items #1 and #2 share ids with LHS recurring / recurring_hit nodes
// so the cross-highlight wiring is exercised.

type LinkPredRawNodeDef = RawNodeDef & { nodeType?: PredictedNodeType };

const LINK_PRED_RAW_NODES: LinkPredRawNodeDef[] = [
    {
        id: "node_customer",
        tableName: "customers",
        hopLevel: 0,
        totalScore: 0.85,
        halfSize: 5
    },
    {
        id: "node_orders",
        tableName: "orders",
        hopLevel: 1,
        totalScore: 0.65,
        halfSize: 5
    },
    {
        // Organic: in history, not predicted
        id: "node_product_organic",
        tableName: "products",
        hopLevel: 2,
        totalScore: 0.4,
        halfSize: 4,
        nodeType: "organic"
    },
    {
        // Recurring: in history + predicted (matches #1 in panel)
        id: "node_product_recurring",
        tableName: "products",
        hopLevel: 2,
        totalScore: 0.45,
        halfSize: 4.5,
        nodeType: "recurring"
    },
    {
        // Recurring hit: in history + predicted + GT (matches #2 in panel)
        id: "node_product_recurring_hit",
        tableName: "products",
        hopLevel: 2,
        totalScore: 0.5,
        halfSize: 4.5,
        nodeType: "recurring_hit"
    }
];

const LINK_PRED_BASE_EDGES: Edge[] = [
    { id: "te1", source: "node_customer", target: "node_orders" },
    { id: "te2", source: "node_orders", target: "node_product_organic" },
    { id: "te3", source: "node_orders", target: "node_product_recurring" },
    { id: "te4", source: "node_orders", target: "node_product_recurring_hit" }
];

const LINK_PRED_COLOR_MAP = buildTableColorMap(
    LINK_PRED_RAW_NODES.map((n) => n.tableName),
    ROOT_TABLE
);

const LINK_PRED_LEGEND_ITEMS = Object.entries(LINK_PRED_COLOR_MAP).map(
    ([label, color]) => ({
        label,
        color,
        isEntity: label === ROOT_TABLE
    })
);

// Predictions panel items. The first two share ids with LHS nodes so the same
// hoveredId string drives cross-highlight in both directions; the rest are
// RHS-only (exploratory / hit) and have no LHS counterpart.
const LINK_PRED_PREDICTION_ITEMS: PredictionItem[] = [
    {
        id: "node_product_recurring",
        rank: 1,
        tableName: "products",
        nodeType: "recurring",
        color: LINK_PRED_COLOR_MAP["products"] ?? "",
        hopLevel: 2
    },
    {
        id: "node_product_recurring_hit",
        rank: 2,
        tableName: "products",
        nodeType: "recurring_hit",
        color: LINK_PRED_COLOR_MAP["products"] ?? "",
        hopLevel: 2
    },
    {
        id: "pred_exploratory_1",
        rank: 3,
        tableName: "products",
        nodeType: "exploratory",
        color: LINK_PRED_COLOR_MAP["products"] ?? ""
    },
    {
        id: "pred_hit_1",
        rank: 4,
        tableName: "products",
        nodeType: "hit",
        color: LINK_PRED_COLOR_MAP["products"] ?? ""
    },
    {
        id: "pred_exploratory_2",
        rank: 5,
        tableName: "products",
        nodeType: "exploratory",
        color: LINK_PRED_COLOR_MAP["products"] ?? ""
    }
];

function buildLinkPredNodes(
    hoveredId: string | null,
    selectedId: string | null
): Node[] {
    const raw = LINK_PRED_RAW_NODES.map((n) => {
        const data: SubgraphNodeData = {
            graphData: {
                tableName: n.tableName,
                hopLevel: n.hopLevel,
                totalScore: n.totalScore
            },
            color: LINK_PRED_COLOR_MAP[n.tableName] ?? "",
            halfSize: n.halfSize,
            status: "active",
            isHovered: n.id === hoveredId,
            isSelected: n.id === selectedId,
            isDimmedByHover: hoveredId !== null && n.id !== hoveredId,
            nodeType: n.nodeType
        };
        return {
            id: n.id,
            type: "subgraphNode",
            position: { x: 0, y: 0 },
            data
        };
    });
    return xaiSubgraphApplyLayout(raw as Node[], LINK_PRED_BASE_EDGES);
}

function buildNodes(
    hoveredId: string | null,
    selectedId: string | null,
    statusOverrides?: Record<string, NodeStatus>
): Node[] {
    const raw = RAW_NODES.map((n) => {
        const data: SubgraphNodeData = {
            graphData: {
                tableName: n.tableName,
                hopLevel: n.hopLevel,
                totalScore: n.totalScore
            },
            color: COLOR_MAP[n.tableName] ?? "",
            halfSize: n.halfSize,
            status: statusOverrides?.[n.id] ?? "active",
            isHovered: n.id === hoveredId,
            isSelected: n.id === selectedId,
            isDimmedByHover: hoveredId !== null && n.id !== hoveredId
        };
        return {
            id: n.id,
            type: "subgraphNode",
            position: { x: 0, y: 0 },
            data
        };
    });
    return xaiSubgraphApplyLayout(raw as Node[], BASE_EDGES);
}

// ─── Stories ──────────────────────────────────────────────────────────────────

export const Default: Story = {
    render: function Render() {
        const [hoveredId, setHoveredId] = useState<string | null>(null);
        const [selectedId, setSelectedId] = useState<string | null>(null);
        const [isLegendOpen, setIsLegendOpen] = useState(false);

        return (
            <div
                className="w-full overflow-hidden rounded border"
                style={{ height: 400 }}>
                <XaiSubgraph
                    displayNodes={buildNodes(hoveredId, selectedId)}
                    displayEdges={BASE_EDGES}
                    isLegendOpen={isLegendOpen}
                    legendItems={LEGEND_ITEMS}
                    tierLabels={TIER_LABELS}
                    onNodeMouseEnter={(_, node) => setHoveredId(node.id)}
                    onNodeMouseLeave={() => setHoveredId(null)}
                    onNodeClick={(_, node) =>
                        setSelectedId((p) => (p === node.id ? null : node.id))
                    }
                    onPaneClick={() => setSelectedId(null)}
                    onLegendOpenChange={setIsLegendOpen}
                    onOpenFullscreen={() => {}}
                />
            </div>
        );
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const btn = canvas.getByRole("button", { name: "Toggle legend" });
        await expect(btn).toHaveAttribute("aria-expanded", "false");
        await userEvent.click(btn);
        await expect(btn).toHaveAttribute("aria-expanded", "true");
        await userEvent.click(btn);
        await expect(btn).toHaveAttribute("aria-expanded", "false");
    }
};

export const WithDimmedNodes: Story = {
    render: function Render() {
        const [hoveredId, setHoveredId] = useState<string | null>(null);
        const [selectedId, setSelectedId] = useState<string | null>(null);
        const [isLegendOpen, setIsLegendOpen] = useState(false);

        return (
            <div
                className="w-full overflow-hidden rounded border"
                style={{ height: 400 }}>
                <XaiSubgraph
                    displayNodes={buildNodes(
                        hoveredId,
                        selectedId,
                        DIMMED_STATUS
                    )}
                    displayEdges={BASE_EDGES}
                    isLegendOpen={isLegendOpen}
                    legendItems={LEGEND_ITEMS}
                    tierLabels={TIER_LABELS}
                    onNodeMouseEnter={(_, node) => setHoveredId(node.id)}
                    onNodeMouseLeave={() => setHoveredId(null)}
                    onNodeClick={(_, node) =>
                        setSelectedId((p) => (p === node.id ? null : node.id))
                    }
                    onPaneClick={() => setSelectedId(null)}
                    onLegendOpenChange={setIsLegendOpen}
                    onOpenFullscreen={() => {}}
                />
            </div>
        );
    }
};

export const WithFullscreen: Story = {
    render: function Render() {
        const [hoveredId, setHoveredId] = useState<string | null>(null);
        const [selectedId, setSelectedId] = useState<string | null>(null);
        const [isLegendOpen, setIsLegendOpen] = useState(false);
        const [open, setOpen] = useState(false);

        const nodes = buildNodes(hoveredId, selectedId);

        return (
            <>
                <div
                    className="w-full overflow-hidden rounded border"
                    style={{ height: 400 }}>
                    <XaiSubgraph
                        displayNodes={nodes}
                        displayEdges={BASE_EDGES}
                        isLegendOpen={isLegendOpen}
                        legendItems={LEGEND_ITEMS}
                        tierLabels={TIER_LABELS}
                        onNodeMouseEnter={(_, node) => setHoveredId(node.id)}
                        onNodeMouseLeave={() => setHoveredId(null)}
                        onNodeClick={(_, node) =>
                            setSelectedId((p) =>
                                p === node.id ? null : node.id
                            )
                        }
                        onPaneClick={() => setSelectedId(null)}
                        onLegendOpenChange={setIsLegendOpen}
                        onOpenFullscreen={() => setOpen(true)}
                    />
                </div>
                <Sheet open={open} onOpenChange={setOpen}>
                    <SheetContent
                        className="w-[99%] rounded"
                        style={{ height: "98vh" }}
                        side="top">
                        <SheetTitle className="sr-only">
                            Graph fullscreen view
                        </SheetTitle>
                        <div className="flex h-full w-full flex-col">
                            <XaiSubgraph
                                displayNodes={nodes}
                                displayEdges={BASE_EDGES}
                                isLegendOpen={isLegendOpen}
                                legendItems={LEGEND_ITEMS}
                                tierLabels={TIER_LABELS}
                                onNodeMouseEnter={(_, node) =>
                                    setHoveredId(node.id)
                                }
                                onNodeMouseLeave={() => setHoveredId(null)}
                                onNodeClick={(_, node) =>
                                    setSelectedId((p) =>
                                        p === node.id ? null : node.id
                                    )
                                }
                                onPaneClick={() => setSelectedId(null)}
                                onLegendOpenChange={setIsLegendOpen}
                                onOpenFullscreen={() => {}}
                                onCloseFullscreen={() => setOpen(false)}
                            />
                        </div>
                    </SheetContent>
                </Sheet>
            </>
        );
    }
};

// Link-prediction variant (covers both TLP and SLP — the catalog renders
// identically; the task-type distinction lives in the consumer). LHS shows
// organic / recurring / recurring_hit (3 product instances under orders);
// RHS Predictions panel shows all 5 shape types. Items #1, #2 share ids with
// their LHS counterparts so hovering on either side cross-highlights the other.
export const LinkPrediction: Story = {
    render: function Render() {
        const [hoveredId, setHoveredId] = useState<string | null>(null);
        // Shared selection state across LHS + RHS — clicking either side
        // updates this single id, which drives both selection rings.
        const [selectedId, setSelectedId] = useState<string | null>(null);
        const [isLegendOpen, setIsLegendOpen] = useState(false);
        const [isFullscreen, setIsFullscreen] = useState(false);

        const predictions: XaiSubgraphPredictions = {
            items: LINK_PRED_PREDICTION_ITEMS,
            entityNodeId: "node_customer",
            hoveredId,
            selectedId
        };

        // Prediction items have prefixed ReactFlow ids; data.hoverKey is the
        // matching LHS node id (or pkey if no LHS counterpart). Mapping both
        // hover and click through hoverKey keeps a single shared id driving
        // both sides' hover ring and selection ring.
        const resolveKey = (node: Node): string | null => {
            if (node.type === "predictionItem") {
                return (node.data as { hoverKey?: string }).hoverKey ?? null;
            }
            if (node.type === "predictionsPanelBackground") return null;
            return node.id;
        };

        const nodes = buildLinkPredNodes(hoveredId, selectedId);
        const commonProps = {
            displayNodes: nodes,
            displayEdges: LINK_PRED_BASE_EDGES,
            isLegendOpen,
            legendItems: LINK_PRED_LEGEND_ITEMS,
            tierLabels: TIER_LABELS,
            onNodeMouseEnter: (_: unknown, node: Node) =>
                setHoveredId(resolveKey(node)),
            onNodeMouseLeave: () => setHoveredId(null),
            onNodeClick: (_: unknown, node: Node) => {
                const k = resolveKey(node);
                if (!k) return;
                setSelectedId((p) => (p === k ? null : k));
            },
            onPaneClick: () => setSelectedId(null),
            onLegendOpenChange: setIsLegendOpen,
            predictions
        };

        return (
            <>
                <div
                    className="w-full overflow-hidden rounded border"
                    style={{ height: 500 }}>
                    <XaiSubgraph
                        {...commonProps}
                        onOpenFullscreen={() => setIsFullscreen(true)}
                    />
                </div>
                <Sheet open={isFullscreen} onOpenChange={setIsFullscreen}>
                    <SheetContent
                        className="w-[99%] rounded"
                        style={{ height: "98vh" }}
                        side="top">
                        <SheetTitle className="sr-only">
                            Graph fullscreen view
                        </SheetTitle>
                        <div className="flex h-full w-full flex-col">
                            <XaiSubgraph
                                {...commonProps}
                                onOpenFullscreen={() => {}}
                                onCloseFullscreen={() => setIsFullscreen(false)}
                            />
                        </div>
                    </SheetContent>
                </Sheet>
            </>
        );
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        // Legend toggle exposes the "Predicted node types" section (TLP-only).
        const legendBtn = canvas.getByRole("button", {
            name: "Toggle legend"
        });
        await userEvent.click(legendBtn);
        await expect(canvas.getByText("Predicted node types")).toBeVisible();
        await userEvent.click(legendBtn);
    }
};
