// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import {
    TimeFrameStats,
    TrainingTimeFrameStatsData,
    TrainTableGenStats
} from "types/stats";

import { jobStatus } from "utils/Constants";

import mockGenerationStatsResponse from "../../../Jobs/TrainingTable/__tests__/MockResponses/GenerationStatsResponse.test.json";
import TimeframeStats, {
    TimeframeStatsProps
} from "../TimeframeStats/TimeframeStats";

import mockColumnDef from "./MockResponses/ColumnDefResponse.json";

const mockGenerationStats = {
    allStats: mockGenerationStatsResponse as unknown as TrainTableGenStats,
    tableData: {
        data: mockGenerationStatsResponse.iteration_data as TimeFrameStats[],
        columns: mockColumnDef
    },
    selectedStats: null,
    activeIndex: 71,
    finishedIterations: 72,
    expectedIterations: 72,
    receivedResponse: true,
    isTemporal: true,
    queryStatus: jobStatus.Done
} as TrainingTimeFrameStatsData;

describe("TimeframeStats", () => {
    const renderTimeframeStats = async (props: TimeframeStatsProps) => {
        await act(async () => {
            render(
                <MemoryRouter>
                    <TimeframeStats {...props} />
                </MemoryRouter>
            );
        });
    };

    it("Should conditionally render table based on show prop", async () => {
        await renderTimeframeStats({
            queryStatus: "",
            data: {} as TrainingTimeFrameStatsData,
            show: false,
            expandRow: false
        });

        expect(screen.queryByPlaceholderText("Search")).not.toBeInTheDocument();
        expect(screen.queryByTestId("table-header")).not.toBeInTheDocument();

        await renderTimeframeStats({
            queryStatus: jobStatus.None,
            data: {
                allStats: null,
                tableData: {
                    data: [] as TimeFrameStats[],
                    columns: []
                },
                selectedStats: null,
                activeIndex: 0,
                finishedIterations: 0,
                expectedIterations: 0,
                receivedResponse: false,
                isTemporal: false,
                queryStatus: jobStatus.None
            } as unknown as TrainingTimeFrameStatsData,
            show: true,
            expandRow: false
        });

        expect(screen.getByPlaceholderText("Search")).toBeInTheDocument();
        expect(screen.getByTestId("table-header")).toBeInTheDocument();
    });

    it("Should render `TimeframeStats` when all the props are set", async () => {
        await renderTimeframeStats({
            queryStatus: jobStatus.Done,
            data: mockGenerationStats,
            show: true,
            expandRow: false
        });

        expect(screen.getByPlaceholderText("Search")).toBeInTheDocument();
        const tableHeader = screen.getByTestId("table-header");
        mockColumnDef.forEach(({ header }) => {
            expect(tableHeader).toHaveTextContent(header);
        });

        expect(
            screen.getAllByText(
                mockGenerationStatsResponse.iteration_data[0].min_date
            )[0]
        ).toBeInTheDocument();
        expect(
            screen.getAllByText(
                mockGenerationStatsResponse.iteration_data[0].num_entities
            )
        ).toBeTruthy();
        expect(
            screen.getByText(
                mockGenerationStatsResponse.iteration_data[0]
                    .target_distribution.count.True!
            )
        ).toBeInTheDocument();
        expect(
            screen.getByText(
                mockGenerationStatsResponse.iteration_data[0]
                    .target_distribution.count.False!
            )
        ).toBeInTheDocument();
    });

    it.each([
        {
            name: "Running status with no response",
            queryStatus: jobStatus.Running,
            dataOverrides: { allStats: null, receivedResponse: false },
            expectSkeleton: true
        },
        {
            name: "Queued status with no response",
            queryStatus: jobStatus.Queued,
            dataOverrides: { allStats: null, receivedResponse: false },
            expectSkeleton: true
        },
        {
            name: "Done status with receivedResponse true",
            queryStatus: jobStatus.Done,
            dataOverrides: { allStats: null, receivedResponse: true },
            expectSkeleton: false
        },
        {
            name: "Done status with allStats and receivedResponse true",
            queryStatus: jobStatus.Done,
            dataOverrides: { receivedResponse: true },
            expectSkeleton: false
        }
    ])(
        "Should handle loading skeleton for $name",
        async ({ queryStatus, dataOverrides, expectSkeleton }) => {
            await renderTimeframeStats({
                queryStatus,
                data: {
                    ...mockGenerationStats,
                    ...dataOverrides
                },
                show: true,
                expandRow: false
            });

            const tableHeader = screen.getByTestId("table-header");
            mockColumnDef.forEach(({ header }) => {
                expect(tableHeader).toHaveTextContent(header);
            });

            expect(document.querySelectorAll('[data-slot="skeleton"]').length)[
                expectSkeleton ? "toBeGreaterThan" : "toBe"
            ](0);
        }
    );

    it("Should handle pagination navigation", async () => {
        await renderTimeframeStats({
            queryStatus: jobStatus.Done,
            data: mockGenerationStats,
            show: true,
            expandRow: false
        });

        const pageNext = screen.getByTestId("pagination-next");
        await userEvent.click(pageNext);
        expect(
            screen.getAllByText(
                mockGenerationStatsResponse.iteration_data[71].min_date
            )[0]
        ).toBeInTheDocument();

        const pagePrev = screen.getByTestId("pagination-prev");
        await userEvent.click(pagePrev);
        expect(
            screen.getAllByText(
                mockGenerationStatsResponse.iteration_data[0].min_date
            )[0]
        ).toBeInTheDocument();

        await userEvent.click(screen.getByTestId("pagination-2"));
    });

    it("Should filter table content by date, entities, and samples", async () => {
        await renderTimeframeStats({
            queryStatus: jobStatus.Done,
            data: mockGenerationStats,
            show: true,
            expandRow: false
        });

        const searchInput = screen.getByPlaceholderText("Search");
        const searchTests = [
            {
                input: mockGenerationStatsResponse.iteration_data[71].min_date,
                shouldFind:
                    mockGenerationStatsResponse.iteration_data[71].min_date,
                shouldNotFind:
                    mockGenerationStatsResponse.iteration_data[0].min_date
            },
            {
                input: mockGenerationStatsResponse.iteration_data[62].num_entities.toString(),
                shouldFind:
                    mockGenerationStatsResponse.iteration_data[62].num_entities,
                shouldNotFind:
                    mockGenerationStatsResponse.iteration_data[0]
                        .target_distribution.count.False!
            },
            {
                input: mockGenerationStatsResponse.iteration_data[19].target_distribution.count.True!.toString(),
                shouldFind:
                    mockGenerationStatsResponse.iteration_data[19]
                        .target_distribution.count.True!,
                shouldNotFind:
                    mockGenerationStatsResponse.iteration_data[9]
                        .target_distribution.count.True!
            },
            {
                input: mockGenerationStatsResponse.iteration_data[70].target_distribution.count.False!.toString(),
                shouldFind:
                    mockGenerationStatsResponse.iteration_data[70]
                        .target_distribution.count.False!,
                shouldNotFind:
                    mockGenerationStatsResponse.iteration_data[0]
                        .target_distribution.count.False!
            }
        ];

        for (const { input, shouldFind, shouldNotFind } of searchTests) {
            await userEvent.clear(searchInput);
            await userEvent.type(searchInput, input);

            const foundElements = screen.queryAllByText(shouldFind);
            expect(foundElements.length).toBeGreaterThan(0);
            expect(screen.queryByText(shouldNotFind)).not.toBeInTheDocument();
        }
    });

    it("Should handle sorting by Anchor Date column", async () => {
        await renderTimeframeStats({
            queryStatus: jobStatus.Done,
            data: mockGenerationStats,
            show: true,
            expandRow: false
        });

        const cells = screen.getAllByRole("cell");
        expect(cells[0]).toHaveTextContent(
            mockGenerationStatsResponse.iteration_data[0].min_date
        );

        const anchorTime = screen.getByText("Anchor Date");
        await userEvent.click(anchorTime);

        const sortedDateCells = await screen.findAllByRole("cell");
        await waitFor(() => {
            expect(sortedDateCells[0]).toHaveTextContent(
                mockGenerationStatsResponse.iteration_data[71].min_date
            );
        });

        await userEvent.click(anchorTime);
        await waitFor(() => {
            expect(cells[0]).toHaveTextContent(
                mockGenerationStatsResponse.iteration_data[0].min_date
            );
        });
    });

    it("Should display empty state when table data is empty", async () => {
        await renderTimeframeStats({
            queryStatus: jobStatus.Done,
            data: {
                ...mockGenerationStats,
                tableData: {
                    data: [],
                    columns: mockColumnDef
                }
            },
            show: true,
            expandRow: false
        });

        expect(screen.getByText("No data found.")).toBeInTheDocument();
    });

    it("Should display header even when the `tableData` are set empty", async () => {
        await renderTimeframeStats({
            queryStatus: jobStatus.Done,
            data: {
                ...mockGenerationStats,
                tableData: {
                    data: [{}] as unknown as TimeFrameStats[],
                    columns: mockColumnDef
                }
            },
            show: true,
            expandRow: false
        });

        const tableHeader = screen.getByTestId("table-header");
        mockColumnDef.forEach(({ header }) => {
            expect(tableHeader).toHaveTextContent(header);
        });

        expect(screen.getByPlaceholderText("Search")).toBeInTheDocument();

        const cells = screen.getAllByRole("cell");
        expect(cells.length).toBeGreaterThan(0);
        cells.forEach((cell) => {
            expect(cell).toBeEmptyDOMElement();
        });

        expect(screen.queryByTestId("card")).not.toBeInTheDocument();
    });
});
