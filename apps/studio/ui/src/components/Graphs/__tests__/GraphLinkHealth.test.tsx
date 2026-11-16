// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { RenderResult } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import {
    defaultGraphLinkHealthProps,
    setupGraphTests
} from "components/Graphs/__tests__/graphTestUtils";

import { EdgeStats } from "types/graphs";

import { GraphLinkHealthProps } from "../LinkHealth/GraphLinkHealth";

const mockEdgeStats: EdgeStats = {
    col_a: "users.id",
    col_a_table_name: "users",
    col_a_column_name: "id",
    col_a_table_id: "1",

    col_b: "orders.user_id",
    col_b_table_name: "orders",
    col_b_column_name: "user_id",
    col_b_table_id: "2",

    link_type: "FK",
    weak_edge: false,

    matches_a_in_b: 95,
    matches_a_in_b_original: 95,
    percent_a_in_b: "95%",

    matches_b_in_a: 90,
    matches_b_in_a_original: 90,
    percent_b_in_a: "90%"
};

describe("GraphLinkHealth", () => {
    const HEADER_TEXT = "Graph Link Health";
    const CALCULATING_MESSAGE =
        "Calculating edge statistics...this may take a few minutes. Please try again later!";

    const { graphLinkHealthMocks, renderGraphLinkHealth } = setupGraphTests();

    const renderComponent = async (
        overrides: Partial<GraphLinkHealthProps> = {}
    ) => {
        return await renderGraphLinkHealth({
            ...defaultGraphLinkHealthProps,
            ...overrides
        });
    };

    const verifyTableHeaders = async (
        findByText: RenderResult["findByText"]
    ) => {
        const expectedHeaders = [
            "Graph Link Health",
            "Primary Key (A)",
            "Foreign Key (B)",
            "Link Type",
            "Matches (A in B)",
            "Matches (B in A)"
        ];

        for (const header of expectedHeaders) {
            const headerElement = await findByText(header);
            expect(headerElement).toBeInTheDocument();
        }
    };

    const verifyEmptyState = (
        queryByText: RenderResult["queryByText"],
        headerText: string = HEADER_TEXT
    ) => {
        expect(queryByText(headerText)).not.toBeInTheDocument();
    };

    describe("Success State", () => {
        it("Should render graph link health table with data when message is 'Success'", async () => {
            const { findByText } = await renderComponent({
                summaryParams: {
                    message: "Success",
                    graphsEdges: [mockEdgeStats]
                }
            });

            await verifyTableHeaders(findByText);
        });

        it("Should display table column names from mock data", async () => {
            const { getByText } = await renderComponent({
                summaryParams: {
                    message: "Success",
                    graphsEdges: [mockEdgeStats]
                }
            });

            expect(getByText("users")).toBeInTheDocument();
            expect(getByText("orders")).toBeInTheDocument();
            expect(getByText(".id")).toBeInTheDocument();
            expect(getByText(".user_id")).toBeInTheDocument();
        });

        it("Should display match statistics correctly", async () => {
            const { getByText } = await renderComponent({
                summaryParams: {
                    message: "Success",
                    graphsEdges: [mockEdgeStats]
                }
            });

            expect(getByText("(95%)")).toBeInTheDocument();
            expect(getByText("(90%)")).toBeInTheDocument();
        });

        it("Should render column names with empty table name fallback when table names are missing", async () => {
            const mockEdgeStatsMissingNames: EdgeStats = {
                ...mockEdgeStats,
                col_a_table_name: undefined as unknown as string,
                col_b_table_name: undefined as unknown as string
            };

            const { getByText, queryByText } = await renderComponent({
                summaryParams: {
                    message: "Success",
                    graphsEdges: [mockEdgeStatsMissingNames]
                }
            });

            expect(queryByText("users")).not.toBeInTheDocument();
            expect(queryByText("orders")).not.toBeInTheDocument();

            expect(getByText(".id")).toBeInTheDocument();
            expect(getByText(".user_id")).toBeInTheDocument();
        });
    });

    describe("Empty State", () => {
        it("Should not render when message is 'Success' but graphsEdges is empty", async () => {
            const { queryByText } = await renderComponent({
                summaryParams: {
                    message: "Success",
                    graphsEdges: []
                }
            });

            verifyEmptyState(queryByText);
        });

        it("Should not render when message is 'Fetching' with data in graphsEdges", async () => {
            const { queryByText } = await renderComponent({
                summaryParams: {
                    message: "Fetching",
                    graphsEdges: [mockEdgeStats]
                }
            });

            verifyEmptyState(queryByText);
        });

        it("Should not render when message is 'Fetching' and graphsEdges is empty", async () => {
            const { queryByText } = await renderComponent({
                summaryParams: {
                    message: "Fetching",
                    graphsEdges: []
                }
            });

            verifyEmptyState(queryByText);
        });

        it("Should not render when message is 'Calculating' with data in graphsEdges", async () => {
            const { queryByText } = await renderComponent({
                summaryParams: {
                    message: CALCULATING_MESSAGE,
                    graphsEdges: [mockEdgeStats]
                }
            });

            verifyEmptyState(queryByText);
        });

        it("Should not render when message is 'Calculating' and graphsEdges is empty", async () => {
            const { queryByText } = await renderComponent({
                summaryParams: {
                    message: CALCULATING_MESSAGE,
                    graphsEdges: []
                }
            });

            verifyEmptyState(queryByText);
        });
    });

    describe("Error Handling", () => {
        it("Should display 'No data found.' message for invalid graphsEdges", async () => {
            const { getByText } = await renderComponent({
                summaryParams: {
                    message: "Success",
                    graphsEdges: { col_a: false } as unknown as EdgeStats[]
                }
            });

            expect(getByText("No data found.")).toBeInTheDocument();
        });

        it("Should call handleError when message is unrecognized", async () => {
            const { handleError } = graphLinkHealthMocks;
            await renderComponent({
                summaryParams: {
                    graphsEdges: [],
                    message: "Unexpected error occurred!"
                },
                handleError
            });

            expect(handleError).toHaveBeenCalledWith(
                expect.objectContaining({
                    response: { data: "Unexpected error occurred!" },
                    message: "Unexpected error occurred!"
                }),
                "Something went wrong!"
            );
        });
    });

    describe("Edge Cases", () => {
        it("Should handle zero match percentages", async () => {
            const mockZeroMatches: EdgeStats = {
                ...mockEdgeStats,
                matches_a_in_b: 0,
                percent_a_in_b: "0%",
                matches_b_in_a: 0,
                percent_b_in_a: "0%"
            };

            const { getAllByText } = await renderComponent({
                summaryParams: {
                    message: "Success",
                    graphsEdges: [mockZeroMatches]
                }
            });

            const zeroPercents = getAllByText("(0%)");
            expect(zeroPercents.length).toBeGreaterThan(0);
        });

        it("Should handle 100% match percentages", async () => {
            const mockPerfectMatches: EdgeStats = {
                ...mockEdgeStats,
                matches_a_in_b: 100,
                percent_a_in_b: "100%",
                matches_b_in_a: 100,
                percent_b_in_a: "100%"
            };

            const { getAllByText } = await renderComponent({
                summaryParams: {
                    message: "Success",
                    graphsEdges: [mockPerfectMatches]
                }
            });

            const hundredPercents = getAllByText("(100%)");
            expect(hundredPercents.length).toBeGreaterThan(0);
        });

        it("Should not render when summaryParams is undefined", async () => {
            const { container } = await renderComponent({
                summaryParams: undefined
            });

            expect(container).toBeEmptyDOMElement();
        });
    });

    it("Should render multiple edge statistics when multiple edges are provided", async () => {
        const mockEdgeStats2: EdgeStats = {
            ...mockEdgeStats,
            col_a: "products.id",
            col_a_table_name: "products",
            col_a_column_name: "id",
            col_a_table_id: "3",
            col_b: "orders.product_id",
            col_b_column_name: "product_id"
        };

        const { getByText, getAllByText } = await renderComponent({
            summaryParams: {
                message: "Success",
                graphsEdges: [mockEdgeStats, mockEdgeStats2]
            }
        });

        expect(getByText("users")).toBeInTheDocument();
        expect(getAllByText("orders").length).toBeGreaterThan(0);
        expect(getByText("products")).toBeInTheDocument();
    });
});
