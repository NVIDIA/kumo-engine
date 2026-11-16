// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useState } from "react";

import {
    Alert,
    Card,
    Description,
    HopGraph,
    HopGraphEdge,
    HopGraphNode,
    Sheet,
    SheetContent,
    Skeleton
} from "@kumo-ai/kumo-ui-catalog";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";

import { JSGraph } from "types/graphs";

import {
    NeighborhoodGraphResult,
    useNeighborhoodGraph
} from "./useNeighborhoodGraph";

interface SubgraphsSamplingProps {
    graph: JSGraph | null;
    predictiveQuery: string;
    modelPlanYaml: string;
    /** True while the model plan is being generated */
    loading: boolean;
}

const SubgraphsSampling = ({
    graph,
    predictiveQuery,
    modelPlanYaml,
    loading
}: SubgraphsSamplingProps) => {
    const [showModal, setShowModal] = useState(false);
    const {
        nodes,
        edges,
        isReady,
        notReadyReason,
        configuredHopCount,
        renderedHopCount
    } = useNeighborhoodGraph(graph, predictiveQuery, modelPlanYaml);
    const suppressedHopCount = configuredHopCount - renderedHopCount;

    // Cache last renderable graph to avoid flicker during revalidation.
    // useState (not useRef) so that clearing the cache triggers a re-render
    // and the not-ready message shows immediately.
    const [lastRenderable, setLastRenderable] = useState<{
        nodes: HopGraphNode[];
        edges: HopGraphEdge[];
    } | null>(null);

    useEffect(() => {
        setLastRenderable(
            isReady && nodes.length > 0 ? { nodes, edges } : null
        );
    }, [isReady, nodes, edges]);

    const displayNodes = isReady ? nodes : (lastRenderable?.nodes ?? []);
    const displayEdges = isReady ? edges : (lastRenderable?.edges ?? []);
    const hasRenderable = isReady || (lastRenderable?.nodes?.length ?? 0) > 0;

    return (
        <div className="flex h-full flex-col">
            {/* Description */}
            <Description
                size="sm"
                variant="secondary"
                className="!mb-2 px-1 font-medium leading-5">
                Preview of how tables from your graph will be used in generating
                subgraphs for Kumo to generate prediction for the entity.
            </Description>

            {/* Suppressed-hop warning */}
            {isReady && suppressedHopCount > 0 && (
                <Alert variant="warn" className="mb-2">
                    Showing {renderedHopCount} of {configuredHopCount}{" "}
                    configured hops. Hops retracing the same edge in reverse are
                    excluded, matching how the model samples during training.
                </Alert>
            )}

            {/* Graph area */}
            <Card className="relative min-h-0 flex-1 overflow-y-auto pb-0">
                {loading ? (
                    <div className="flex flex-col gap-2 px-1 pt-2">
                        <Skeleton className="h-5 w-40" />
                        <Skeleton className="h-40 w-full" />
                    </div>
                ) : hasRenderable ? (
                    <HopGraph
                        nodes={displayNodes}
                        edges={displayEdges}
                        className="w-full"
                        onOpenFullscreen={() => setShowModal(true)}
                    />
                ) : (
                    <NotReadyMessage reason={notReadyReason} />
                )}
            </Card>

            {/* Full-screen sheet — slides down from top, matching the Graphs tab */}
            <Sheet open={showModal} onOpenChange={setShowModal}>
                <SheetContent className="rounded w-[99%] h-[98vh]" side="top">
                    <div className="flex h-full w-full flex-col">
                        <HopGraph
                            nodes={displayNodes}
                            edges={displayEdges}
                            onCloseFullscreen={() => setShowModal(false)}
                        />
                    </div>
                </SheetContent>
            </Sheet>
        </div>
    );
};

// ─── NotReadyMessage ──────────────────────────────────────────────────────────
// Shown in the graph card when the hop graph cannot be rendered yet.
// Each notReadyReason maps to a user-facing explanation of what is missing.
function NotReadyMessage({
    reason
}: {
    reason: NeighborhoodGraphResult["notReadyReason"];
}) {
    const messages: Record<NonNullable<typeof reason>, string> = {
        // User hasn't provided both a predictive query and a model plan yet
        "missing-inputs":
            "Write a predictive query and generate a model plan to see the subgraph sampling preview.",
        // FOR EACH clause is absent from the predictive query
        "no-target-table":
            "Could not determine the target table from the FOR EACH clause.",
        // Graph schema has no foreign-key links to traverse
        "no-schema-edges":
            "No foreign-key relationships found in this graph's schema.",
        // Model plan YAML has no neighbor_sampling block
        "no-hops":
            "No neighbor sampling configuration found in the model plan.",
        // neighbor_sampling exists but produces no edges (e.g. all budgets are
        // zero, or hops don't start from hop1)
        "invalid-hops": "Neighbor sampling configuration looks incorrect."
    };

    const text = reason ? messages[reason] : null;

    return (
        <div className="flex h-full min-h-45 items-center justify-center px-4">
            <Description
                size="sm"
                variant="secondary"
                className="!mb-0 text-center">
                {text}
            </Description>
        </div>
    );
}

export default WithErrorBoundary(SubgraphsSampling);
