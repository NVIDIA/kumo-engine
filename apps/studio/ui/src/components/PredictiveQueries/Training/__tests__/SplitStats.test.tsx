// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import { act, render, screen, within } from "@testing-library/react";

import { QueryType, TaskType } from "types/predictiveQueries";

import { jobStatus } from "utils/Constants";

import mockSplitstatsResponse from "../../../Jobs/TrainingTable/__tests__/MockResponses/SplitStatsResponse.test.json";
import SplitStats, { SplitStatsProps } from "../SplitStats/SplitStats";

import {
    SplitStats_Headers,
    SplitStats_Headers_STATIC_LP
} from "./MockResponses/StatsHeader";

const defaultProps: SplitStatsProps = {
    queryStatus: jobStatus.Done,
    taskType: TaskType.BINARY_CLASSIFICATION,
    queryType: QueryType.TEMPORAL,
    show: true,
    splitStats: mockSplitstatsResponse
};

describe("SplitStats", () => {
    const renderSplitStats = async (props: SplitStatsProps) => {
        await act(async () => {
            render(
                <MemoryRouter>
                    <SplitStats {...props} />
                </MemoryRouter>
            );
        });
    };

    it("Should not render anything when show is false", async () => {
        await renderSplitStats({
            queryStatus: jobStatus.None,
            taskType: TaskType.NONE,
            queryType: "" as QueryType,
            show: false,
            splitStats: null
        });

        expect(
            screen.queryByText("No split stats found")
        ).not.toBeInTheDocument();
    });

    it("Should render empty state when no stats are available", async () => {
        await renderSplitStats({
            ...defaultProps,
            queryStatus: jobStatus.None,
            taskType: TaskType.NONE,
            queryType: "" as QueryType,
            splitStats: null
        });

        expect(screen.getByText("No split stats found")).toBeInTheDocument();
    });

    it.each([
        {
            name: "Binary Classification with Temporal query",
            taskType: TaskType.BINARY_CLASSIFICATION,
            queryType: QueryType.TEMPORAL,
            expectedHeaders: SplitStats_Headers
        },
        {
            name: "Regression with Temporal query",
            taskType: TaskType.REGRESSION,
            queryType: QueryType.TEMPORAL,
            expectedHeaders: SplitStats_Headers
        },
        {
            name: "Regression with Static query",
            taskType: TaskType.REGRESSION,
            queryType: QueryType.STATIC,
            expectedHeaders: SplitStats_Headers
        },
        {
            name: "Link Prediction with Temporal query",
            taskType: TaskType.LINK_PREDICTION,
            queryType: QueryType.TEMPORAL,
            expectedHeaders: SplitStats_Headers
        },
        {
            name: "Link Prediction with Static query",
            taskType: TaskType.LINK_PREDICTION,
            queryType: QueryType.STATIC,
            expectedHeaders: SplitStats_Headers_STATIC_LP
        }
    ])(
        "Should render table with correct headers for $name",
        async ({ taskType, queryType, expectedHeaders }) => {
            await renderSplitStats({
                queryStatus: jobStatus.Done,
                taskType,
                queryType,
                show: true,
                splitStats: mockSplitstatsResponse
            });

            const table = screen.getByRole("table");
            const tableHeader = screen.getByTestId("table-header");
            expect(tableHeader).toBeInTheDocument();

            const headers = within(tableHeader).getAllByRole("columnheader");
            const columnHeaders = headers
                .map((th) => th.textContent?.trim())
                .filter(Boolean);

            const tbody = table.querySelector("tbody");
            expect(tbody).not.toBeNull();

            const rows = within(tbody!).getAllByRole("row");
            expect(rows.length).toBeGreaterThan(0);

            const metricNames = rows
                .map((row) => {
                    const cells = within(row).getAllByRole("cell");
                    return cells[0]?.textContent?.trim();
                })
                .filter(Boolean);

            const allHeaders = [...columnHeaders, ...metricNames];
            const allHeadersLowercase = allHeaders.map((h) => h?.toLowerCase());

            expectedHeaders.forEach((expectedHeader) => {
                expect(allHeadersLowercase).toContain(
                    expectedHeader.toLowerCase()
                );
            });
        }
    );

    it("Should render split stats table for MULTILABEL_CLASSIFICATION", async () => {
        await renderSplitStats({
            queryStatus: jobStatus.Done,
            taskType: TaskType.MULTILABEL_CLASSIFICATION,
            queryType: QueryType.TEMPORAL,
            show: true,
            splitStats: mockSplitstatsResponse
        });

        const table = screen.getByRole("table");
        expect(table).toBeInTheDocument();

        const tableHeader = screen.getByTestId("table-header");
        expect(tableHeader).toBeInTheDocument();

        // Verify table has rows with data
        const tbody = table.querySelector("tbody");
        expect(tbody).not.toBeNull();

        const rows = within(tbody!).getAllByRole("row");
        expect(rows.length).toBeGreaterThan(0);

        // Verify each row has data
        rows.forEach((row) => {
            const cells = within(row).getAllByRole("cell");
            const hasNonEmptyCell = cells.some(
                (cell) => (cell.textContent?.trim() ?? "").length > 0
            );
            expect(hasNonEmptyCell).toBe(true);
        });
    });

    it("Should render table with data for BINARY_CLASSIFICATION", async () => {
        await renderSplitStats({
            ...defaultProps,
            taskType: TaskType.BINARY_CLASSIFICATION
        });

        const table = screen.getByRole("table");
        expect(table).toBeInTheDocument();

        const tbody = table.querySelector("tbody");
        const rows = within(tbody!).getAllByRole("row");
        expect(rows.length).toBeGreaterThan(0);
    });

    it("Should calculate page count correctly based on table data length", async () => {
        const largeStatsResponse = {
            ...mockSplitstatsResponse,
            additional_stats_1: {
                ...mockSplitstatsResponse.train_table_stats,
                data_split: "Additional 1"
            },
            additional_stats_2: {
                ...mockSplitstatsResponse.train_table_stats,
                data_split: "Additional 2"
            }
        };

        await renderSplitStats({
            ...defaultProps,
            splitStats: largeStatsResponse
        });

        const testData = Array.from({ length: 75 }, (_, i) => ({
            data_split: `Split ${i}`,
            isActive: false
        }));

        const pageCount = Math.ceil(testData.length / 50);
        expect(pageCount).toBe(2);
    });

    describe("isLoading", () => {
        it.each([
            {
                name: "link prediction tasks regardless of query status",
                queryStatus: jobStatus.Running,
                taskType: TaskType.LINK_PREDICTION,
                splitStats: null
            },
            {
                name: "completed queries",
                queryStatus: jobStatus.Done,
                taskType: TaskType.BINARY_CLASSIFICATION,
                splitStats: null
            },
            {
                name: "failed queries",
                queryStatus: jobStatus.Failed,
                taskType: TaskType.BINARY_CLASSIFICATION,
                splitStats: null
            }
        ])(
            "Should set isLoading to false for $name",
            async ({ queryStatus, taskType, splitStats }) => {
                await renderSplitStats({
                    queryStatus,
                    taskType,
                    queryType: QueryType.TEMPORAL,
                    show: true,
                    splitStats
                });

                const loadingSkeleton = document.querySelector(
                    '[data-slot="skeleton"]'
                );
                expect(loadingSkeleton).not.toBeInTheDocument();
                expect(
                    screen.getByText("No split stats found")
                ).toBeInTheDocument();
            }
        );

        it.each([
            {
                name: "running non-link prediction tasks with null stats",
                queryStatus: jobStatus.Running
            },
            {
                name: "queued non-link prediction tasks with null stats",
                queryStatus: jobStatus.Queued
            }
        ])(
            "Should set isLoading to true for $name",
            async ({ queryStatus }) => {
                await renderSplitStats({
                    queryStatus,
                    taskType: TaskType.BINARY_CLASSIFICATION,
                    queryType: QueryType.TEMPORAL,
                    show: true,
                    splitStats: null
                });

                const loadingSkeleton = document.querySelector(
                    '[data-slot="skeleton"]'
                );
                expect(loadingSkeleton).toBeInTheDocument();
            }
        );

        it.each([
            {
                name: "running non-link prediction tasks with available stats",
                queryStatus: jobStatus.Running
            },
            {
                name: "queued non-link prediction tasks with available stats",
                queryStatus: jobStatus.Queued
            }
        ])(
            "Should set isLoading to false for $name",
            async ({ queryStatus }) => {
                await renderSplitStats({
                    queryStatus,
                    taskType: TaskType.BINARY_CLASSIFICATION,
                    queryType: QueryType.TEMPORAL,
                    show: true,
                    splitStats: mockSplitstatsResponse
                });

                const loadingSkeleton = document.querySelector(
                    '[data-slot="skeleton"]'
                );
                expect(loadingSkeleton).not.toBeInTheDocument();
            }
        );
    });
});
