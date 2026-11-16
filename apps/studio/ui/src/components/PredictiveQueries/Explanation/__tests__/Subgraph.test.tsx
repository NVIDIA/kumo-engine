// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import { QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { KumoapiExplainSubgraphSubgraph } from "gen/v0/types";
import { mockAxiosInstance } from "tests/mockAxios";
import { createQueryClient } from "tests/testUtil";

import { GraphNodeData, GraphXaiScores } from "types/explorer";

import SubGraph, {
    GraphTableView,
    GraphTableViewProps,
    SubgraphProps,
    SubgraphSumaryTable,
    SubgraphSumaryTableProps
} from "../SubGraph/SubGraph";

import {
    MissingSubgraphSummary,
    SelectedNode,
    SubgraphSummary
} from "./MockResponses/CategoryResponses";
import mockGraphXaiScore from "./MockResponses/GraphXaiScores.json";
import mockSungraphResponse from "./MockResponses/SubgraphResponse.json";

const defaultProps = {
    subgraph: mockSungraphResponse as unknown as KumoapiExplainSubgraphSubgraph,
    isDataLoading: false,
    xaiData: mockGraphXaiScore as unknown as GraphXaiScores,
    anchorTime: "",
    show: true,
    setActiveSubGraph: vi.fn(),
    entityId: "10"
};

const getCellTexts = (row: HTMLElement) =>
    within(row)
        .getAllByRole("cell")
        .map((cell) => cell.textContent?.trim());

const expectTableHeaders = (
    tableHeader: HTMLElement,
    expectedHeaders: string[]
) => {
    const headers = within(tableHeader).getAllByRole("columnheader");
    const headerTexts = headers.map((th) =>
        th.textContent?.trim().toLowerCase()
    );

    expectedHeaders.forEach((expectedHeader) => {
        expect(headerTexts).toContain(expectedHeader.toLowerCase());
    });
};

describe("SubGraph", () => {
    const queryClient = createQueryClient();
    const renderSubGraph = async (props: SubgraphProps = defaultProps) => {
        await act(async () => {
            render(
                <MemoryRouter>
                    <QueryClientProvider client={queryClient}>
                        <SubGraph {...props} />
                    </QueryClientProvider>
                </MemoryRouter>
            );
        });
    };

    beforeEach(() => {
        queryClient.clear();
        vi.clearAllMocks();
    });

    it("Should render `Subgraph` when all the props are set", async () => {
        await renderSubGraph();

        expect(await screen.findByText("Subgraph")).toBeInTheDocument();

        const spinner = await screen.findByRole("status", { name: /Loading/i });
        expect(spinner).toBeInTheDocument();

        await waitFor(() => {
            expect(screen.getByText(/FINANCE_ACCOUNT/i)).toBeInTheDocument();
            expect(screen.getByText(/FINANCE_ORDERS/i)).toBeInTheDocument();
        });
    });

    it("Should show `Select a node` message when no node is selected", async () => {
        await renderSubGraph();

        await waitFor(() => {
            expect(
                screen.getByText(
                    "Select a node in the graph to view its details."
                )
            ).toBeInTheDocument();
        });
    });

    it("Should render loading state when `isDataLoading` is set as true", async () => {
        await renderSubGraph({
            ...defaultProps,
            isDataLoading: true
        });

        const skeletonElement = document.querySelector(
            '[data-slot="skeleton"]'
        );
        expect(skeletonElement).toBeInTheDocument();
    });

    it("Should render error message when no graph data is available", async () => {
        await renderSubGraph({
            ...defaultProps,
            subgraph: {
                subgraph_store: {
                    nodes: [],
                    edges: []
                }
            } as unknown as KumoapiExplainSubgraphSubgraph
        });

        expect(
            await screen.findByText(/Input graph data is not available/)
        ).toBeInTheDocument();
    });

    it("Should handle empty `xaiData` gracefully", async () => {
        await renderSubGraph({
            ...defaultProps,
            xaiData: {} as GraphXaiScores
        });

        expect(await screen.findByText("Subgraph")).toBeInTheDocument();
    });

    it("Should render with only one node (invalid case)", async () => {
        await renderSubGraph({
            ...defaultProps,
            subgraph: {
                subgraph_store: {
                    nodes: [
                        {
                            id: "node1",
                            data: {
                                cols: [],
                                tableName: "",
                                idName: "",
                                value: "",
                                totalScore: 0
                            }
                        }
                    ],
                    edges: []
                }
            } as unknown as KumoapiExplainSubgraphSubgraph
        });

        expect(
            await screen.findByText(/Input graph data is not available/)
        ).toBeInTheDocument();
    });

    it("Should render with exactly one valid node", async () => {
        await renderSubGraph({
            ...defaultProps,
            subgraph: {
                subgraph_store: {
                    nodes: [
                        {
                            id: "node1",
                            data: {
                                cols: [{ name: "col1", value: "val1" }],
                                tableName: "TABLE1",
                                idName: "id",
                                value: "123",
                                totalScore: 1.0
                            }
                        }
                    ],
                    edges: []
                }
            } as unknown as KumoapiExplainSubgraphSubgraph
        });

        expect(
            await screen.findByText(/Input graph data is not available/)
        ).toBeInTheDocument();
    });

    it("Should handle setActiveSubGraph callback", async () => {
        const setActiveSubGraphMock = vi.fn();
        await renderSubGraph({
            ...defaultProps,
            setActiveSubGraph: setActiveSubGraphMock
        });

        await waitFor(() => {
            expect(screen.getByText("Subgraph")).toBeInTheDocument();
        });
    });

    it("Should not make node_pred_rfm API calls", async () => {
        await renderSubGraph();

        await waitFor(() => {
            expect(mockAxiosInstance.get).not.toHaveBeenCalledWith(
                expect.stringContaining("node_pred_rfm")
            );
        });
    });
});

describe("GraphTableView", () => {
    const renderGraphTableView = async (props: GraphTableViewProps) => {
        await act(async () => {
            render(
                <MemoryRouter>
                    <GraphTableView {...props} />
                </MemoryRouter>
            );
        });
    };

    beforeEach(() => {
        vi.clearAllMocks();
    });

    it("Should render `GraphTableView`", async () => {
        await renderGraphTableView({
            selectedNode: SelectedNode as unknown as GraphNodeData
        });

        const tableHeader = await screen.findByTestId("table-header");
        expectTableHeaders(tableHeader, ["Column", "Value", "Score"]);

        const rows = screen.getAllByRole("row");

        //First row
        const [cell1, cell2, cell3] = getCellTexts(rows[1]);
        expect(cell1).toBe(SelectedNode.idName);
        expect(cell2).toBe(String(SelectedNode.value));
        expect(cell3).toBe("—");

        //Second row
        const [cell4, cell5, cell6] = getCellTexts(rows[2]);
        expect(cell4).toBe(SelectedNode.cols[0].name);
        expect(cell5).toBe(String(SelectedNode.cols[0].value));
        expect(cell6).toBe(SelectedNode.cols[0].score.toFixed(2));
    });

    it("Should display `No data found!` when the prop selectedNode is set as `null`", async () => {
        await renderGraphTableView({
            selectedNode: null
        });

        expect(await screen.findByText("No data found!")).toBeInTheDocument();
    });

    it("Should sort the table content", async () => {
        await renderGraphTableView({
            selectedNode: SelectedNode as unknown as GraphNodeData
        });

        const tableHeader = await screen.findByTestId("table-header");
        expectTableHeaders(tableHeader, ["Column", "Value", "Score"]);

        const getScoreColumnValues = () =>
            screen
                .getAllByRole("row")
                .slice(1)
                .map((r) =>
                    parseFloat(r.querySelectorAll("td")[2]?.textContent || "")
                )
                .filter(Number.isFinite);

        const beforeSort = getScoreColumnValues();

        const scoreHeader = screen.getByText("Score");
        await userEvent.click(scoreHeader);

        const afterSort = getScoreColumnValues();

        const expected = [...beforeSort].sort((a, b) => b - a);
        expect(afterSort).toEqual(expected);
    });

    it("Should display tooltip for long values", async () => {
        const nodeWithLongValue: GraphNodeData = {
            cols: [
                {
                    name: "LongColumn",
                    value: "This is a very long value that exceeds fifteen characters",
                    grad_score: 0.5
                }
            ],
            tableName: "TEST_TABLE",
            idName: "ID",
            value: "123",
            totalScore: 1.0
        };

        await renderGraphTableView({
            selectedNode: nodeWithLongValue
        });

        await waitFor(() => {
            expect(screen.getByText("LongColumn")).toBeInTheDocument();
        });

        const valueCell = screen.getByText(/This is a very .../);
        expect(valueCell).toBeInTheDocument();
    });

    it("Should handle null value correctly", async () => {
        const nodeWithNullValue: GraphNodeData = {
            cols: [
                {
                    name: "NullColumn",
                    value: null as unknown as string,
                    grad_score: 0.5
                }
            ],
            tableName: "TEST_TABLE",
            idName: "ID",
            value: "123",
            totalScore: 1.0
        };

        await renderGraphTableView({
            selectedNode: nodeWithNullValue
        });

        await waitFor(() => {
            expect(screen.getByText("NullColumn")).toBeInTheDocument();
        });
    });

    it("Should filter out invalid data rows", async () => {
        const nodeWithInvalidData: GraphNodeData = {
            cols: [
                {
                    name: "ValidColumn",
                    value: "ValidValue",
                    grad_score: 0.5
                },
                {
                    name: "AnotherValid",
                    value: "AnotherValue",
                    grad_score: 0.3
                }
            ],
            tableName: "TEST_TABLE",
            idName: "ID",
            value: "456",
            totalScore: 1.0
        };

        await renderGraphTableView({
            selectedNode: nodeWithInvalidData
        });

        const rows = screen.getAllByRole("row");
        expect(rows.length).toBe(4);
    });

    it("Should display emdash symbol for undefined scores", async () => {
        const nodeWithMixedScores: GraphNodeData = {
            cols: [
                {
                    name: "Column1",
                    value: "Value1",
                    grad_score: 0.5
                },
                {
                    name: "Column2",
                    value: "Value2",
                    grad_score: undefined
                }
            ],
            tableName: "TEST_TABLE",
            idName: "ID",
            value: "789",
            totalScore: 1.0
        };

        await renderGraphTableView({
            selectedNode: nodeWithMixedScores
        });

        await waitFor(() => {
            const emdashSymbols = screen.getAllByText("—");
            expect(emdashSymbols.length).toBeGreaterThan(0);
        });
    });

    it("Should handle empty cols array", async () => {
        const nodeWithEmptyCols: GraphNodeData = {
            cols: [],
            tableName: "TEST_TABLE",
            idName: "ID",
            value: "123",
            totalScore: 1.0
        };

        await renderGraphTableView({
            selectedNode: nodeWithEmptyCols
        });

        const rows = screen.getAllByRole("row");
        expect(rows.length).toBe(2);
    });
});

describe("SubgraphSumaryTable", () => {
    const renderSubgraphSumaryTable = async (
        props: SubgraphSumaryTableProps
    ) => {
        await act(async () => {
            render(
                <MemoryRouter>
                    <SubgraphSumaryTable {...props} />
                </MemoryRouter>
            );
        });
    };

    beforeEach(() => {
        vi.clearAllMocks();
    });

    it("Should render `SubgraphSumaryTable`", async () => {
        await renderSubgraphSumaryTable({
            subgraphSummary: SubgraphSummary
        });

        const tableHeader = await screen.findByTestId("table-header");
        expectTableHeaders(tableHeader, [
            "Property",
            "Current Entity",
            "Global"
        ]);

        const rows = screen.getAllByRole("row");

        //First row
        const [cell1, cell2, cell3] = getCellTexts(rows[1]);
        expect(cell1).toBe("FINANCE_TRANS Count");
        expect(cell2).toBe("123");
        expect(cell3).toBe("107");

        //Second row
        const [cell4, cell5, cell6] = getCellTexts(rows[2]);
        expect(cell4).toBe("Most Recent Fact");
        expect(cell5).toBe("1 days");
        expect(cell6).toBe("1.66 days");
    });

    it("Should display tooltip content when there are missing values", async () => {
        await renderSubgraphSumaryTable({
            subgraphSummary: MissingSubgraphSummary
        });

        const tableHeader = await screen.findByTestId("table-header");
        expectTableHeaders(tableHeader, [
            "Property",
            "Current Entity",
            "Global"
        ]);

        const rows = screen.getAllByRole("row");

        //First row
        const [cell1] = getCellTexts(rows[1]);
        expect(cell1).toBe("FINANCE_TRANS Count");

        //Second row
        const [cell4, cell5, cell6] = getCellTexts(rows[2]);
        expect(cell4).toBe("Most Recent Fact");
        expect(cell5).toBe("1 days");
        expect(cell6).toBe("1.66 days");

        const missingElements = screen.getAllByText("<MISSING>");
        expect(missingElements.length).toBe(2);

        await userEvent.hover(missingElements[0]);
        await waitFor(() =>
            expect(
                screen.getAllByText(
                    /Cannot extract corresponding value due to.*unsupported for.*task/i
                ).length
            ).toBeGreaterThan(0)
        );
    });

    it("Should display empty table message when subgraphSummary is empty", async () => {
        await renderSubgraphSumaryTable({
            subgraphSummary: []
        });

        expect(await screen.findByText("No data found!")).toBeInTheDocument();
    });
});
