// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import { QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen, waitFor, within } from "@testing-library/react";
import { mockAxiosInstance } from "tests/mockAxios";
import { testResponseType } from "tests/testResponseType";
import { createQueryClient } from "tests/testUtil";

import OutputStatistics, {
    BPOutputStatisticsProps
} from "../OutputStatistics/OutputStatistics";

import { setPredictionStatsResponseType } from "./MockFunctions/PredictionStats.mock";
import { MockUnlinkedBatchPredictionJobData } from "./MockResponses/UnlinkedBatchPredictionJobData";

describe("OutputStatistics", () => {
    const queryClient = createQueryClient();

    const renderBPOverview = async (props: BPOutputStatisticsProps) => {
        await act(async () => {
            render(
                <MemoryRouter>
                    <QueryClientProvider client={queryClient}>
                        <OutputStatistics {...props} />
                    </QueryClientProvider>
                </MemoryRouter>
            );
        });
    };

    const checkApiCall = async () => {
        const id = "bp-job-1937b6a194fd47cda79d490860314028";
        await waitFor(() => {
            expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                `v1/batch_prediction_jobs/${id}/prediction_stats`,
                undefined,
                { headers: undefined, params: { project_id: "project-tests" } }
            );
        });
    };

    const outputStatisticsHeaders = [
        "Column",
        "Data Type",
        "Semantic Type",
        "Missing",
        "Invalid",
        "Total Count"
    ];

    const verifyTableHeaders = () => {
        const tableHeader = screen.getByTestId("table-header");
        const headers = within(tableHeader).getAllByRole("columnheader");
        const headerTexts = headers.map((th) =>
            th.textContent?.trim().toLowerCase()
        );

        outputStatisticsHeaders.forEach((expectedHeader) => {
            expect(headerTexts).toContain(expectedHeader.toLowerCase());
        });
    };

    const verifyTableColumns = (expectedLabels: string[]) => {
        const rows = screen.getAllByRole("row");
        const firstColumnCells = rows
            .map(
                (row) =>
                    within(row).queryAllByTestId("table-cell")[0] ||
                    within(row).queryAllByRole("cell")[0]
            )
            .filter(Boolean);

        const actualLabels = firstColumnCells.map((cell) =>
            cell.textContent?.trim()
        );
        expectedLabels.forEach((label) => {
            expect(actualLabels).toContain(label);
        });
    };

    const verifyComponentDisplay = (displayValue: "block" | "none") => {
        const expectedClass = displayValue === "none" ? "hidden" : "block";
        const divs = screen.getAllByRole("generic", { hidden: true });
        const target = divs.find((el) => el.classList.contains(expectedClass));
        expect(target).toBeInTheDocument();
    };

    beforeEach(() => {
        vi.clearAllMocks();
        queryClient.clear();
    });

    it("Should render `BPOverview` when all the props are set", async () => {
        await renderBPOverview({
            jobInfo: MockUnlinkedBatchPredictionJobData,
            show: true,
            isJobRunning: true,
            handleError: vi.fn()
        });
        await checkApiCall();

        verifyTableHeaders();

        const expectedLabels = [
            "ENTITY",
            "TARGET_PRED",
            "False_PROB",
            "True_PROB"
        ];

        verifyTableColumns(expectedLabels);
    });

    it.each([
        {
            scenario: "jobInfo is null",
            props: {
                jobInfo: null,
                show: true,
                isJobRunning: true,
                handleError: vi.fn()
            },
            shouldCallApi: false,
            displayValue: "block" as const
        },
        {
            scenario: "isJobRunning is false",
            props: {
                jobInfo: MockUnlinkedBatchPredictionJobData,
                show: true,
                isJobRunning: false,
                handleError: vi.fn()
            },
            shouldCallApi: true,
            displayValue: "block" as const
        },
        {
            scenario: "show is false",
            props: {
                jobInfo: MockUnlinkedBatchPredictionJobData,
                show: false,
                isJobRunning: false,
                handleError: vi.fn()
            },
            shouldCallApi: true,
            displayValue: "none" as const
        }
    ])(
        "Should not display `StatisticsTable` when $scenario",
        async ({ props, shouldCallApi, displayValue }) => {
            await renderBPOverview(props);
            if (shouldCallApi) {
                await checkApiCall();
            }

            verifyComponentDisplay(displayValue);
        }
    );

    it.each([
        {
            responseType: testResponseType.ERROR,
            expectedContent: "No data found",
            description:
                "Should display `No data found` when API response is ERROR"
        },
        {
            responseType: testResponseType.INVALID,
            expectedContent: "skeleton",
            description:
                "Should display skeleton loader when API response is INVALID"
        }
    ])("$description", async ({ responseType, expectedContent }) => {
        setPredictionStatsResponseType(responseType);
        await renderBPOverview({
            jobInfo: MockUnlinkedBatchPredictionJobData,
            show: true,
            isJobRunning: true,
            handleError: vi.fn()
        });
        await checkApiCall();

        verifyTableHeaders();

        if (expectedContent === "No data found") {
            expect(screen.getByText(/No data found/i)).toBeInTheDocument();
        } else {
            const skeletonElement = document.querySelector(
                '[data-slot="skeleton"]'
            );
            expect(skeletonElement).toBeInTheDocument();
        }
    });

    it("Should call handleError when API fails", async () => {
        const handleErrorMock = vi.fn();
        setPredictionStatsResponseType(testResponseType.ERROR);

        await renderBPOverview({
            jobInfo: MockUnlinkedBatchPredictionJobData,
            show: true,
            isJobRunning: true,
            handleError: handleErrorMock
        });
        await checkApiCall();

        expect(handleErrorMock).toHaveBeenCalled();
    });

    it("Should not display search input when search prop is false", async () => {
        await renderBPOverview({
            jobInfo: MockUnlinkedBatchPredictionJobData,
            show: true,
            isJobRunning: true,
            handleError: vi.fn()
        });
        await checkApiCall();

        const searchInput = screen.queryByPlaceholderText(/search/i);
        expect(searchInput).not.toBeInTheDocument();
    });
});
