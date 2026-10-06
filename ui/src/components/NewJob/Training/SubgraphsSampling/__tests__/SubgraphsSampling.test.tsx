// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import { QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createQueryClient } from "tests/testUtil";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { graphDetail } from "pages/Graphs/__tests__/MockResponse/GraphDetail.response";

import { JSGraph } from "types/graphs";

import SubgraphsSampling from "../SubgraphsSampling";

// ── Mock HopGraph to avoid ReactFlow / ResizeObserver complexity in unit tests.
// We capture the nodes/edges passed to it so we can assert on them.
const mockHopGraph = vi.hoisted(() =>
    vi.fn(({ nodes, onOpenFullscreen, onCloseFullscreen }) => (
        <div data-testid="hop-graph">
            <span data-testid="node-count">{nodes?.length ?? 0}</span>
            {onOpenFullscreen && (
                <button
                    onClick={onOpenFullscreen}
                    data-testid="open-fullscreen">
                    Open fullscreen
                </button>
            )}
            {onCloseFullscreen && (
                <button
                    onClick={onCloseFullscreen}
                    data-testid="close-fullscreen">
                    Close fullscreen
                </button>
            )}
        </div>
    ))
);

vi.mock("@kumo-ai/kumo-ui-catalog", async (importOriginal) => {
    const actual =
        await importOriginal<typeof import("@kumo-ai/kumo-ui-catalog")>();
    return { ...actual, HopGraph: mockHopGraph };
});

// ── Graph fixture (H&M schema from existing mock data) ─────────────────────────
const graph: JSGraph = {
    id: graphDetail.id,
    name: graphDetail.name_alias,
    tables: graphDetail.graph.tables as JSGraph["tables"],
    groups: graphDetail.graph.col_groups.map((g) => ({ columns: g.columns })),
    totalLinks: graphDetail.num_links,
    totalTables: graphDetail.num_tables
};

const predictiveQuery = "PREDICT age FOR EACH customers";

const modelPlanYaml = `
neighbor_sampling:
  num_neighbors:
    - [12, 13]
`;

describe("SubgraphsSampling", () => {
    let queryClient = createQueryClient();

    const renderComponent = async (
        props: Partial<Parameters<typeof SubgraphsSampling>[0]> = {}
    ) => {
        await act(async () => {
            render(
                <MemoryRouter>
                    <QueryClientProvider client={queryClient}>
                        <SubgraphsSampling
                            graph={null}
                            predictiveQuery=""
                            modelPlanYaml=""
                            loading={false}
                            {...props}
                        />
                    </QueryClientProvider>
                </MemoryRouter>
            );
        });
    };

    beforeEach(() => {
        queryClient = createQueryClient();
        mockHopGraph.mockClear();
    });

    // ── loading state ──────────────────────────────────────────────────────────

    it("shows skeleton while model plan is loading", async () => {
        await renderComponent({ loading: true });
        expect(screen.queryByTestId("hop-graph")).not.toBeInTheDocument();
        // Skeleton renders when loading=true; absence of hop-graph confirms the
        // graph area is replaced by the loading state.
    });

    // ── placeholder states ─────────────────────────────────────────────────────

    it("shows placeholder with missing-inputs message when no data provided", async () => {
        await renderComponent();
        await waitFor(() => {
            expect(
                screen.getByText(
                    /Write a predictive query and generate a model plan/i
                )
            ).toBeInTheDocument();
        });
    });

    it("shows no-hops placeholder when model plan lacks neighbor_sampling", async () => {
        await renderComponent({
            graph,
            predictiveQuery,
            modelPlanYaml: "model_type: GNN"
        });
        await waitFor(() => {
            expect(
                screen.getByText(/No neighbor sampling configuration found/i)
            ).toBeInTheDocument();
        });
    });

    it("shows no-target-table placeholder when FOR EACH clause is absent", async () => {
        // resolveTargetTable returns null only when the FOR EACH clause
        // is missing entirely (not when it names an unknown table).
        await renderComponent({
            graph,
            predictiveQuery: "PREDICT x",
            modelPlanYaml
        });
        await waitFor(() => {
            expect(
                screen.getByText(/Could not determine the target table/i)
            ).toBeInTheDocument();
        });
    });

    // ── successful render ──────────────────────────────────────────────────────

    it("renders HopGraph when graph, query, and model plan are all provided", async () => {
        await renderComponent({ graph, predictiveQuery, modelPlanYaml });
        await waitFor(() => {
            expect(screen.getByTestId("hop-graph")).toBeInTheDocument();
        });
    });

    it("passes at least 2 nodes to HopGraph (entity + hop-1)", async () => {
        await renderComponent({ graph, predictiveQuery, modelPlanYaml });
        await waitFor(() => {
            const count = parseInt(
                screen.getByTestId("node-count").textContent ?? "0",
                10
            );
            expect(count).toBeGreaterThanOrEqual(2);
        });
    });

    // ── fullscreen sheet ───────────────────────────────────────────────────────

    it("opens fullscreen Sheet when 'Open fullscreen' is clicked", async () => {
        await renderComponent({ graph, predictiveQuery, modelPlanYaml });

        await waitFor(() =>
            expect(screen.getByTestId("open-fullscreen")).toBeInTheDocument()
        );

        await userEvent.click(screen.getByTestId("open-fullscreen"));

        await waitFor(() => {
            // Sheet opens: there should now be two HopGraph instances
            expect(screen.getAllByTestId("hop-graph")).toHaveLength(2);
        });
    });

    it("closes fullscreen Sheet when the close button is clicked", async () => {
        await renderComponent({ graph, predictiveQuery, modelPlanYaml });

        await waitFor(() =>
            expect(screen.getByTestId("open-fullscreen")).toBeInTheDocument()
        );

        await userEvent.click(screen.getByTestId("open-fullscreen"));

        // Sheet is open — click the close button rendered by the fullscreen HopGraph
        await waitFor(() =>
            expect(screen.getAllByTestId("hop-graph")).toHaveLength(2)
        );

        await userEvent.click(screen.getByTestId("close-fullscreen"));

        await waitFor(() => {
            expect(screen.getAllByTestId("hop-graph")).toHaveLength(1);
        });
    });
});
