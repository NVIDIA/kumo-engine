// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useRef } from "react";

import {
    forceCenter,
    forceLink,
    forceManyBody,
    forceSimulation,
    forceX,
    forceY,
    Simulation,
    SimulationLinkDatum,
    SimulationNodeDatum
} from "d3-force";

import {
    Edge,
    Node,
    ReactFlowState,
    useReactFlow,
    useStore
} from "@xyflow/react";

type UseForceLayoutOptions = {
    strength: number;
    distance: number;
    setIsLayoutStable: (stable: boolean) => void;
};

type SimNodeType = SimulationNodeDatum & Node;

const elementCountSelector = (state: ReactFlowState) =>
    state.nodes.length + state.edges.length;

interface SimulationNode extends Node {
    x?: number;
    y?: number;
    vx?: number;
    vy?: number;
    fx?: number | null;
    fy?: number | null;
    index?: number;
}

interface SimulationLink extends SimulationLinkDatum<SimulationNode> {
    source: string | SimulationNode;
    target: string | SimulationNode;
}

function useForceLayout({
    strength = -1000,
    distance = 350,
    setIsLayoutStable
}: UseForceLayoutOptions) {
    const elementCount = useStore(elementCountSelector);
    const { setNodes, getNodes, getEdges } = useReactFlow();
    const simulationRef = useRef<Simulation<
        SimulationNode,
        SimulationLink
    > | null>(null);
    const requestRef = useRef<number | null>(null);
    const timeoutRef = useRef<NodeJS.Timeout | null>(null);
    const isMountedRef = useRef(true);
    const previousDraggingStateRef = useRef<Map<string, boolean>>(new Map());

    useEffect(() => {
        isMountedRef.current = true;
        return () => {
            isMountedRef.current = false;
        };
    }, []);

    const startSimulation = useCallback(() => {
        if (timeoutRef.current) {
            clearTimeout(timeoutRef.current);
            timeoutRef.current = null;
        }

        timeoutRef.current = setTimeout(() => {
            if (!isMountedRef.current) return;

            setIsLayoutStable(false);
            const nodes = getNodes();
            const edges = getEdges();

            const simulationNodes: SimNodeType[] = nodes?.map((node: Node) => ({
                ...node,
                x: node.position.x + (Math.random() - 0.5) * 200,
                y: node.position.y + (Math.random() - 0.5) * 200,
                fx: node.dragging ? node.position.x : null,
                fy: node.dragging ? node.position.y : null
            }));

            const simulationLinks: SimulationLink[] = edges?.map(
                (edge: Edge) => ({
                    source: edge.source,
                    target: edge.target
                })
            );

            const centerX = 0;
            const centerY = 0;

            const simulation = forceSimulation<SimulationNode, SimulationLink>()
                .nodes(simulationNodes as SimulationNode[])
                .force("charge", forceManyBody().strength(strength))
                .force(
                    "link",
                    forceLink<SimulationNode, SimulationLink>(simulationLinks)
                        .id((d) => d.id)
                        .strength(0.5)
                        .distance(distance)
                )
                .force(
                    "center",
                    forceCenter(window.innerWidth / 2, window.innerHeight / 2)
                ) // Very weak center force
                .force("x", forceX().x(centerX).strength(0.05))
                .force("y", forceY().y(centerY).strength(0.05))
                .alpha(1)
                .alphaDecay(0.05);

            simulationRef.current = simulation;

            let isUpdating = false;

            simulation.on("tick", () => {
                if (!isUpdating && isMountedRef.current) {
                    isUpdating = true;
                    requestAnimationFrame(() => {
                        if (!isMountedRef.current) {
                            isUpdating = false;
                            return;
                        }

                        const nodePositionMap = new Map<
                            string,
                            {
                                x: number;
                                y: number;
                                simNode: SimulationNode;
                            }
                        >();

                        simulationNodes.forEach((simNode) => {
                            nodePositionMap.set(simNode.id, {
                                x: simNode.x ?? 0,
                                y: simNode.y ?? 0,
                                simNode
                            });
                        });

                        setNodes((prevNodes: Node[]) =>
                            prevNodes.map((node) => {
                                const simData = nodePositionMap.get(node.id);
                                if (!simData) return node;

                                const { x, y, simNode } = simData;
                                const wasDragging =
                                    previousDraggingStateRef.current.get(
                                        node.id
                                    ) || false;

                                if (node.dragging) {
                                    simNode.fx = node.position.x;
                                    simNode.fy = node.position.y;
                                    previousDraggingStateRef.current.set(
                                        node.id,
                                        true
                                    );
                                    return node;
                                } else if (wasDragging && !node.dragging) {
                                    simNode.fx = null;
                                    simNode.fy = null;
                                    previousDraggingStateRef.current.set(
                                        node.id,
                                        false
                                    );
                                }

                                if (
                                    Math.abs(x - node.position.x) < 0.01 &&
                                    Math.abs(y - node.position.y) < 0.01
                                ) {
                                    return node;
                                }

                                return {
                                    ...node,
                                    position: { x, y }
                                };
                            })
                        );
                        isUpdating = false;
                    });
                }
            });

            const stopWhenStable = () => {
                if (!isMountedRef.current) {
                    simulation.stop();
                    return;
                }

                if (simulation.alpha() < 0.03) {
                    simulation.stop();
                    setIsLayoutStable(true);
                    requestRef.current = null;
                } else {
                    requestRef.current = requestAnimationFrame(stopWhenStable);
                }
            };

            requestRef.current = requestAnimationFrame(stopWhenStable);
        }, 200);
    }, [distance, getEdges, getNodes, setIsLayoutStable, setNodes, strength]);

    useEffect(() => {
        if (simulationRef.current) {
            simulationRef.current.stop();
            simulationRef.current = null;
        }

        if (requestRef.current !== null) {
            cancelAnimationFrame(requestRef.current);
            requestRef.current = null;
        }

        if (timeoutRef.current !== null) {
            clearTimeout(timeoutRef.current);
            timeoutRef.current = null;
        }

        previousDraggingStateRef.current.clear();

        startSimulation();

        return () => {
            if (simulationRef.current) {
                simulationRef.current.stop();
                simulationRef.current = null;
            }

            if (requestRef.current !== null) {
                cancelAnimationFrame(requestRef.current);
                requestRef.current = null;
            }

            if (timeoutRef.current !== null) {
                clearTimeout(timeoutRef.current);
                timeoutRef.current = null;
            }
        };
    }, [elementCount, startSimulation]);
}

export default useForceLayout;
