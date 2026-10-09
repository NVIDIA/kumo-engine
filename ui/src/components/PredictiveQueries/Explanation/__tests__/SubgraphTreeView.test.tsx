// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { KumoapiExplainSubgraphSubgraph } from "gen/v0/types";
import { createQueryClient } from "tests/testUtil";

import { GraphXaiScores } from "types/explorer";

import { SubgraphTreeView } from "../SubGraph/SubgraphTreeView";

import mockGraphXaiScore from "./MockResponses/GraphXaiScores.json";
import mockSubgraphResponse from "./MockResponses/SubgraphResponse.json";

// Mock only the ReactFlow-based components that can't run in jsdom.
// All other catalog components use their real implementations.
const mockXaiSubgraph = vi.hoisted(() =>
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    vi.fn(({ displayNodes, onOpenFullscreen }: any) => (
        <div data-testid="xai-subgraph">
            <span data-testid="canvas-node-count">
                {displayNodes?.length ?? 0}
            </span>
            {onOpenFullscreen && (
                <button
                    onClick={onOpenFullscreen}
                    data-testid="open-fullscreen">
                    Open fullscreen
                </button>
            )}
        </div>
    ))
);

vi.mock("@kumo-ai/kumo-ui-catalog", async (importOriginal) => {
    const actual =
        await importOriginal<typeof import("@kumo-ai/kumo-ui-catalog")>();
    return {
        ...actual,
        xaiSubgraphApplyLayout: (nodes: unknown[]) => nodes,
        XaiSubgraph: mockXaiSubgraph
    };
});

const subgraph =
    mockSubgraphResponse as unknown as KumoapiExplainSubgraphSubgraph;
const xaiData = mockGraphXaiScore as unknown as GraphXaiScores;

describe("SubgraphTreeView", () => {
    let queryClient: QueryClient;

    beforeEach(() => {
        queryClient = createQueryClient();
    });

    afterEach(() => {
        queryClient.clear();
    });

    type Props = {
        subgraph: KumoapiExplainSubgraphSubgraph | undefined;
        entityId: string;
        xaiData?: GraphXaiScores;
        anchorTime: string;
        setActiveSubGraph: () => void;
    };

    const renderComponent = async (props: Partial<Props> = {}) => {
        const merged: Props = {
            subgraph,
            entityId: "10",
            anchorTime: "",
            setActiveSubGraph: vi.fn(),
            ...props
        };
        await act(async () => {
            render(
                <MemoryRouter>
                    <QueryClientProvider client={queryClient}>
                        <SubgraphTreeView {...merged} />
                    </QueryClientProvider>
                </MemoryRouter>
            );
        });
    };

    it("renders Subgraph heading", async () => {
        await renderComponent();
        expect(screen.getByText("Subgraph")).toBeInTheDocument();
    });

    it("renders total node count", async () => {
        await renderComponent();
        expect(screen.getByText(/\d+ total/)).toBeInTheDocument();
    });

    it("renders Column Contributions heading", async () => {
        await renderComponent();
        expect(screen.getByText("Column Contributions")).toBeInTheDocument();
    });

    it("shows click-a-node placeholder when no node is selected", async () => {
        await renderComponent();
        expect(
            screen.getByText(/Click a node to see each/)
        ).toBeInTheDocument();
    });

    it("renders Full toggle with label", async () => {
        await renderComponent();
        expect(screen.getByText("Full")).toBeInTheDocument();
        expect(screen.getByRole("switch")).toBeInTheDocument();
    });

    it("Full toggle starts unchecked", async () => {
        await renderComponent();
        expect(screen.getByRole("switch")).toHaveAttribute(
            "data-state",
            "unchecked"
        );
    });

    it("Full toggle turns on when clicked", async () => {
        await renderComponent();
        await userEvent.click(screen.getByRole("switch"));
        await waitFor(() => {
            expect(screen.getByRole("switch")).toHaveAttribute(
                "data-state",
                "checked"
            );
        });
    });

    it("Full toggle turns back off when clicked again", async () => {
        await renderComponent();
        const switchEl = screen.getByRole("switch");
        await userEvent.click(switchEl);
        await waitFor(() =>
            expect(switchEl).toHaveAttribute("data-state", "checked")
        );
        await userEvent.click(switchEl);
        await waitFor(() =>
            expect(switchEl).toHaveAttribute("data-state", "unchecked")
        );
    });

    it("renders Table and Size filter controls", async () => {
        await renderComponent();
        expect(screen.getAllByText("Table").length).toBeGreaterThan(0);
        expect(screen.getAllByText("Size").length).toBeGreaterThan(0);
    });

    it("handles undefined subgraph without crashing", async () => {
        await renderComponent({ subgraph: undefined });
        expect(screen.getByText("Subgraph")).toBeInTheDocument();
        expect(screen.getByText("0 total")).toBeInTheDocument();
    });

    it("renders with xaiData provided", async () => {
        await renderComponent({ xaiData });
        expect(screen.getByText("Subgraph")).toBeInTheDocument();
        expect(screen.getByText(/\d+ total/)).toBeInTheDocument();
    });

    it("mounts the XaiSubgraph canvas", async () => {
        await renderComponent();
        // Two instances: normal view + fullscreen Sheet (always mounted).
        expect(
            screen.getAllByTestId("xai-subgraph").length
        ).toBeGreaterThanOrEqual(1);
    });
});
