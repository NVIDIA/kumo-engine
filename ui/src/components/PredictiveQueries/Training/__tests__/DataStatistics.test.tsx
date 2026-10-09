// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import { QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { mockAxiosInstance } from "tests/mockAxios";
import { verifyTextExists } from "tests/testHelpers";
import { testResponseType } from "tests/testResponseType";
import { createQueryClient } from "tests/testUtil";

import { setGenStatsResponseType } from "components/Jobs/TrainingTable/__tests__/MockFunctions/GenerationStats.mock";
import { setSplitStatsResponseType } from "components/Jobs/TrainingTable/__tests__/MockFunctions/SplitStats.mock";

import { QueryType, TaskType } from "types/predictiveQueries";

import { jobStatus } from "utils/Constants";

import DataStatistics, {
    DataStatisticsProps
} from "../DataStatistics/DataStatistics";

import {
    SplitStats_Headers,
    TimeframeStats_Headers_BinaryClass,
    TimeframeStats_Headers_Reg
} from "./MockResponses/StatsHeader";

const genTableId = "gen-traintable-job-bed43f8f2fc54ea985439b026dbe0865";
const JobId = "trainingjob-bed43f8f2fc54ea985439b026dbe0865";

const defaultProps: DataStatisticsProps = {
    id: genTableId,
    queryStatus: jobStatus.Done,
    queryStage: {
        stage_name: "Generating Training Data",
        last_updated_at: "2025-04-10T09:32:53.299464+00:000",
        detail: "Successfully completed generating train table."
    },
    taskType: TaskType.REGRESSION,
    queryType: QueryType.TEMPORAL,
    isSourceSDK: false,
    jobId: JobId,
    isTrainingPage: true
};

describe("DataStatistics", () => {
    const queryClient = createQueryClient();
    const renderDataStatistics = async (props: DataStatisticsProps) => {
        await act(async () => {
            render(
                <MemoryRouter>
                    <QueryClientProvider client={queryClient}>
                        <DataStatistics {...props} />
                    </QueryClientProvider>
                </MemoryRouter>
            );
        });
    };

    const checkApiCall = async () => {
        await waitFor(() => {
            const APIS_TO_CALL = [
                `v0/gentraintable_jobs/${genTableId}/generation_stats`,
                `v0/training_jobs/${JobId}/get_train_table_split_stats`
            ];
            expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                APIS_TO_CALL[0],
                undefined,
                {
                    headers: undefined,
                    params: { project_id: "project-tests" }
                }
            );
            expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                APIS_TO_CALL[1],
                undefined,
                {
                    headers: undefined,
                    params: { project_id: "project-tests" }
                }
            );
        });
    };

    const verifyDescriptions = async (type: "timeframe" | "splitStats") => {
        if (type === "timeframe") {
            const description1 = await screen.findByText(
                "In order to create your pQuery, Kumo generated training examples from prior historical timeframes to help your pQuery learn to extrapolate historic patterns into the future. These training examples are of the form [Timeframe, Entity, Target value], where only Entities that meet your pQuery filter criteria are included, and the target value is the ground truth value obtained by running your pQuery at that historic point in time."
            );
            expect(description1).toBeInTheDocument();

            const description2 = await screen.findByText(
                "These charts and tables show the distribution of those target values over different timeframes, and can be useful to spot anomalous trends (potentially meaning you need to fix your input data tables or change your pQuery)."
            );
            expect(description2).toBeInTheDocument();
        } else if (type === "splitStats") {
            const splitDescription = await screen.findByText(
                "Statistics about the machine learning data split are useful if you want to analyze your model's performance based on the quantity and distribution of training examples the model was trained on (X% of the training examples had a target value of Y in data split Z)"
            );
            expect(splitDescription).toBeInTheDocument();
        }
    };

    const verifyTableHeaders = (
        tableHeader: HTMLElement,
        expectedHeaders: string[]
    ) => {
        const headerElements = within(tableHeader).getAllByRole("columnheader");
        const headerTexts = headerElements.map((th) =>
            th.textContent?.trim().toLowerCase()
        );
        expectedHeaders.forEach((expectedHeader) => {
            expect(headerTexts).toContain(expectedHeader.toLowerCase());
        });
    };

    const verifyTableRows = (tbody: HTMLElement) => {
        const rows = within(tbody).getAllByRole("row");
        expect(rows.length).toBeGreaterThan(0);
        rows.forEach((row) => {
            const cells = within(row).getAllByRole("cell");
            const hasNonEmptyCell = cells.some(
                (cell) => (cell.textContent?.trim() ?? "").length > 0
            );
            expect(hasNonEmptyCell).toBe(true);
        });
    };

    beforeEach(() => {
        vi.clearAllMocks();
        queryClient.clear();

        setGenStatsResponseType(testResponseType.ALL);
        setSplitStatsResponseType(testResponseType.ALL);
    });

    it("Should render empty state when no stats are available", async () => {
        await renderDataStatistics({
            id: "",
            queryStatus: jobStatus.None,
            queryStage: {
                stage_name: "",
                last_updated_at: new Date().toISOString()
            },
            taskType: TaskType.NONE,
            queryType: QueryType.STATIC,
            isSourceSDK: false,
            jobId: null
        });

        const noStats = document.getElementById("no-split-stats");
        expect(noStats).toHaveTextContent("No split stats found");
    });

    it("Should render tabs for 'REGRESSION' task with 'Timeframe Chart' active by default", async () => {
        await renderDataStatistics({ ...defaultProps });
        await checkApiCall();

        verifyTextExists("Training Data Statistics");

        const splitStatsTab = screen.getAllByRole("tab");

        expect(splitStatsTab[0]).toHaveTextContent("Split stats");
        expect(splitStatsTab[1]).toHaveTextContent("Timeframe chart");
        expect(splitStatsTab[2]).toHaveTextContent("Timeframe stats");
        expect(splitStatsTab[0]).toHaveAttribute("data-state", "active");
    });

    it("Should render correct header title based on isTrainingPage prop", async () => {
        setGenStatsResponseType(testResponseType.ALL);
        setSplitStatsResponseType(testResponseType.ALL);

        // When isTrainingPage is true, should show "Training Data Statistics"
        await renderDataStatistics({
            ...defaultProps,
            isTrainingPage: true
        });
        await checkApiCall();

        const normalHeader = await screen.findByText(
            "Training Data Statistics"
        );
        expect(normalHeader).toBeInTheDocument();

        // Clear and re-render for the second test
        await act(async () => {
            const container = document.body;
            while (container.firstChild) {
                container.removeChild(container.firstChild);
            }
        });

        // When isTrainingPage is false/undefined, should show "TRAINING DATA STATISTICS"
        await renderDataStatistics({
            ...defaultProps,
            isTrainingPage: false
        });
        await checkApiCall();

        const capsHeader = await screen.findByText("Training Data Statistics");
        expect(capsHeader).toBeInTheDocument();
    });

    it("Should display the 'Timeframe Chart' tab with correct chart content for 'REGRESSION' task type", async () => {
        await renderDataStatistics(defaultProps);

        await checkApiCall();

        const timeframeChartTab = screen.getAllByRole("tab");
        expect(timeframeChartTab[1]).toHaveTextContent("Timeframe chart");
        expect(timeframeChartTab[1]).toHaveAttribute("data-state", "inactive");
        await userEvent.click(timeframeChartTab[1]);

        await verifyDescriptions("timeframe");
        verifyTextExists("Generated 72 out of 72 timeframes");

        const trainTableChart = document.getElementById(
            "train-table-stat-chart"
        );
        expect(trainTableChart).toBeInTheDocument();

        expect(
            await screen.findByText("Distribution of Regression Values")
        ).toBeInTheDocument();

        const title = screen.queryAllByText("Time Frames");
        expect(title.length).toBeGreaterThan(0);

        // Test checkbox interactions for regression
        const checkboxes = screen.getAllByTestId("legend-checkbox-0");
        expect(checkboxes).toHaveLength(2);

        checkboxes.forEach((checkbox) => {
            expect(checkbox).toHaveAttribute("data-state", "checked");
        });

        const checkbox2 = screen.getByTestId("legend-checkbox-1");
        expect(checkbox2).toHaveAttribute("data-state", "checked");

        for (const checkbox of checkboxes) {
            await userEvent.click(checkbox);
            expect(checkbox).not.toHaveAttribute("data-state", "checked");
        }

        await userEvent.click(checkbox2);
        expect(checkbox2).not.toBeChecked();
    });

    it("Should display 'Timeframe Chart' tab for BINARY_CLASSIFICATION task type", async () => {
        await renderDataStatistics({
            ...defaultProps,
            taskType: TaskType.BINARY_CLASSIFICATION
        });
        await checkApiCall();

        verifyTextExists("Training Data Statistics");

        const timeframeChartTab = screen.getAllByRole("tab");
        expect(timeframeChartTab[1]).toHaveTextContent("Timeframe chart");
        expect(timeframeChartTab[1]).toHaveAttribute("data-state", "inactive");
        await userEvent.click(timeframeChartTab[1]);

        await verifyDescriptions("timeframe");
        verifyTextExists("Generated 72 out of 72 timeframes");

        const trainTableChart = document.getElementById(
            "train-table-stat-chart"
        );
        expect(trainTableChart).toBeInTheDocument();

        verifyTextExists("Time Frames");
        verifyTextExists("Count of Entities with a Given Value");
        verifyTextExists("Training Table Statistics over Time");

        // Test checkbox interactions for binary classification
        const checkbox1 = screen.getByTestId("legend-checkbox-0");
        expect(checkbox1).toHaveAttribute("data-state", "checked");
        await userEvent.click(checkbox1);
        expect(checkbox1).not.toBeChecked();

        const checkbox2 = screen.getByTestId("legend-checkbox-1");
        expect(checkbox2).toHaveAttribute("data-state", "checked");
        await userEvent.click(checkbox2);
        expect(checkbox2).not.toBeChecked();
    });

    it.each([
        {
            taskType: TaskType.REGRESSION,
            headers: TimeframeStats_Headers_Reg
        },
        {
            taskType: TaskType.BINARY_CLASSIFICATION,
            headers: TimeframeStats_Headers_BinaryClass
        }
    ])(
        "Should render 'Timeframe Stats' tab for $taskType task type",
        async ({ taskType, headers }) => {
            await renderDataStatistics({ ...defaultProps, taskType });
            await checkApiCall();

            const timeframeStatsTab = screen.getAllByRole("tab");
            expect(timeframeStatsTab[2]).toHaveTextContent(/Timeframe Stats/i);
            expect(timeframeStatsTab[2]).toHaveAttribute(
                "data-state",
                "inactive"
            );
            await userEvent.click(timeframeStatsTab[2]);

            await verifyDescriptions("timeframe");

            const tableHeader = screen.getByTestId("table-header");
            verifyTableHeaders(tableHeader, headers);

            const table = screen.getByRole("table");
            const tbody = table.querySelector("tbody")!;
            verifyTableRows(tbody);
        }
    );

    it.each([
        {
            taskType: TaskType.REGRESSION,
            headers: SplitStats_Headers
        },
        {
            taskType: TaskType.BINARY_CLASSIFICATION,
            headers: SplitStats_Headers
        },
        {
            taskType: TaskType.MULTILABEL_CLASSIFICATION,
            headers: SplitStats_Headers
        }
    ])(
        "Should display 'Split Stats' for $taskType task type",
        async ({ taskType, headers }) => {
            await renderDataStatistics({ ...defaultProps, taskType });
            await checkApiCall();

            const splitStatsTab = screen.getAllByRole("tab");
            expect(splitStatsTab[0]).toHaveTextContent(/Split Stats/i);
            expect(splitStatsTab[0]).toHaveAttribute("data-state", "active");

            await verifyDescriptions("splitStats");

            const tableHeader = screen.getByTestId("table-header");
            verifyTableHeaders(tableHeader, headers);

            const table = screen.getByRole("table");
            const tbody = table.querySelector("tbody")!;
            const rows = within(tbody).getAllByRole("row");
            expect(rows.length).toBeGreaterThan(0);

            const headerElements =
                within(tableHeader).getAllByRole("columnheader");
            const headerTexts = headerElements.map((th) =>
                th.textContent?.trim().toLowerCase()
            );
            expect(headerTexts[0]).toBe("data statistics");

            const validSplits = ["holdout", "validation", "training"];
            const splitHeaders = headerTexts.slice(1); // Skip first column

            splitHeaders.forEach((header) => {
                expect(validSplits).toContain(header);
            });

            expect(splitHeaders).toHaveLength(validSplits.length);
        }
    );

    it("Should renders only Split Stats tab for the 'STATIC' QueryType", async () => {
        await renderDataStatistics({
            ...defaultProps,
            queryType: QueryType.STATIC
        });
        await checkApiCall();

        const splitStatsTab = screen.getAllByRole("tab");
        expect(splitStatsTab[0]).toHaveTextContent(/Split Stats/i);
        expect(splitStatsTab[0]).toHaveAttribute("data-state", "active");

        await verifyDescriptions("splitStats");

        const tableHeader = screen.getByTestId("table-header");
        verifyTableHeaders(tableHeader, SplitStats_Headers);

        const table = screen.getByRole("table");
        const tbody = table.querySelector("tbody")!;
        const rows = within(tbody).getAllByRole("row");
        expect(rows.length).toBeGreaterThan(0);

        const headerElements = within(tableHeader).getAllByRole("columnheader");
        const headerTexts = headerElements.map((th) =>
            th.textContent?.trim().toLowerCase()
        );

        expect(headerTexts[0]).toBe("data statistics");

        const validSplits = ["holdout", "validation", "training"];
        const splitHeaders = headerTexts.slice(1); // Skip first column

        splitHeaders.forEach((header) => {
            expect(validSplits).toContain(header);
        });
        expect(splitHeaders).toHaveLength(validSplits.length);
    });

    it.each([
        {
            queryType: QueryType.STATIC,
            columnHeader: "Training",
            percent: /80\.43\s*%/,
            count: "147941"
        },
        {
            queryType: QueryType.STATIC,
            columnHeader: "Validation",
            percent: /9\.79\s*%/,
            count: "18000"
        },
        {
            queryType: QueryType.STATIC,
            columnHeader: "Holdout",
            percent: /9\.79\s*%/,
            count: "18000"
        },
        {
            queryType: QueryType.TEMPORAL,
            columnHeader: "Training",
            percent: /80\.43\s*%/,
            count: "147941",
            predictionWindow: /1993-01-31 00:00:00 - 1998-05-05 00:00:00/
        },
        {
            queryType: QueryType.TEMPORAL,
            columnHeader: "Validation",
            percent: /9\.79\s*%/,
            count: "18000",
            predictionWindow: /1998-05-05 00:00:00 - 1998-09-02 00:00:00/
        },
        {
            queryType: QueryType.TEMPORAL,
            columnHeader: "Holdout",
            percent: /9\.79\s*%/,
            count: "18000",
            predictionWindow: /1998-09-02 00:00:00 - 1998-12-31 00:00:00/
        }
    ])(
        "Should render $columnHeader card values for $queryType QueryType",
        async ({
            queryType,
            columnHeader,
            percent,
            count,
            predictionWindow
        }) => {
            await renderDataStatistics({ ...defaultProps, queryType });
            await checkApiCall();

            await waitFor(() => {
                const splitStatsTable = document.getElementById("split_stats");
                expect(splitStatsTable).toBeInTheDocument();

                const tableHeader = within(splitStatsTable!).getByTestId(
                    "table-header"
                );
                expect(tableHeader).toBeInTheDocument();
                expect(
                    within(tableHeader).getByText(columnHeader)
                ).toBeInTheDocument();

                const percentElements = within(splitStatsTable!).queryAllByText(
                    percent
                );
                expect(percentElements.length).toBeGreaterThan(0);

                const countElements = within(splitStatsTable!).queryAllByText(
                    count
                );
                expect(countElements.length).toBeGreaterThan(0);

                if (predictionWindow) {
                    const predictionWindowElements = within(
                        splitStatsTable!
                    ).queryAllByText(predictionWindow);
                    expect(predictionWindowElements.length).toBeGreaterThan(0);
                }
            });
        }
    );

    // Checks that "No split stats found" message displays for different job statuses and query types when no data is available.
    const scenarios = [
        [jobStatus.Cancelled, QueryType.STATIC],
        [jobStatus.Cancelled, QueryType.TEMPORAL],
        [jobStatus.Running, QueryType.STATIC],
        [jobStatus.Running, QueryType.TEMPORAL],
        [jobStatus.Untrained, QueryType.STATIC],
        [jobStatus.Untrained, QueryType.TEMPORAL],
        [jobStatus.Failed, QueryType.STATIC],
        [jobStatus.Failed, QueryType.TEMPORAL]
    ] as const;

    it.each(scenarios)(
        "Should render 'No split stats found' for %s queryStatus with %s queryType",
        async (status, type) => {
            setGenStatsResponseType(testResponseType.EMPTY);
            setSplitStatsResponseType(testResponseType.EMPTY);
            await renderDataStatistics({
                ...defaultProps,
                queryStatus: status,
                queryType: type,
                isTrainingPage: true
            });
            await checkApiCall();

            expect(
                await screen.findByRole("heading", {
                    name: "Training Data Statistics"
                })
            ).toBeInTheDocument();

            // When data is empty, verify table shows with empty values
            await waitFor(() => {
                const splitStatsTable = document.getElementById("split_stats");
                expect(splitStatsTable).toBeInTheDocument();

                // Extract all cell values
                const tbody = splitStatsTable!.querySelector("tbody");
                const rows = within(tbody!).getAllByRole("row");
                const allCells = rows.flatMap((row) =>
                    within(row).getAllByRole("cell")
                );
                const cellValues = allCells.map((cell) =>
                    cell.textContent?.trim()
                );

                // Verify cells contain empty values (- or 0)
                const emptyValues = cellValues.filter(
                    (val) => val === "-" || val === "0"
                );
                expect(emptyValues.length).toBeGreaterThan(0);
            });
        }
    );

    it("Should shows filter empty warning when some timeframes are skipped", async () => {
        setGenStatsResponseType(testResponseType.FILTERED_OUT);
        await renderDataStatistics({
            ...defaultProps,
            queryStage: {
                ...defaultProps.queryStage,
                stage_name: "Done",
                last_updated_at: defaultProps?.queryStage
                    ?.last_updated_at as string
            }
        });

        expect(
            await screen.findByText(
                /Generated 7 out of 10 timeframes\s+\(3 were empty after applying filters\)/i
            )
        ).toBeInTheDocument();
    });

    it("Should use parent training job endpoint when isSourceSDK is true and jobId is present", async () => {
        await renderDataStatistics({
            ...defaultProps,
            isSourceSDK: true
        });

        await waitFor(() => {
            expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                `v0/gentraintable_jobs/${genTableId}/generation_stats`,
                undefined,
                { headers: undefined, params: { project_id: "project-tests" } }
            );
            expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                `v0/training_jobs/${JobId}/get_train_table_split_stats`,
                undefined,
                {
                    headers: undefined,
                    params: { project_id: "project-tests" }
                }
            );
            expect(
                mockAxiosInstance.get.mock.calls.map(([url]) => url)
            ).not.toContain(`v0/gentraintable_jobs/${genTableId}/split_stats`);
        });

        const splitStatsTab = screen.getAllByRole("tab");
        expect(splitStatsTab[0]).toHaveTextContent(/Split Stats/i);
        expect(splitStatsTab[0]).toHaveAttribute("data-state", "active");
    });

    it("Should pass gen-train-table id directly to split_stats when isSourceSDK is true and no parent jobId", async () => {
        const standaloneGenTableId =
            "gen-traintable-job-aabbccdd11223344aabbccdd11223344";
        await renderDataStatistics({
            ...defaultProps,
            id: standaloneGenTableId,
            isSourceSDK: true,
            jobId: null
        });

        await waitFor(() => {
            expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                `v0/gentraintable_jobs/${standaloneGenTableId}/split_stats`,
                undefined,
                { headers: undefined, params: { project_id: "project-tests" } }
            );
            expect(
                mockAxiosInstance.get.mock.calls.map(([url]) => url)
            ).not.toEqual(
                expect.arrayContaining([
                    expect.stringContaining("training_jobs")
                ])
            );
        });

        const splitStatsTab = screen.getAllByRole("tab");
        expect(splitStatsTab[0]).toHaveTextContent(/Split Stats/i);
        expect(splitStatsTab[0]).toHaveAttribute("data-state", "active");
    });

    it("Should handle empty tabs array gracefully", async () => {
        await renderDataStatistics({
            ...defaultProps,
            queryType: QueryType.STATIC
        });
        await checkApiCall();

        verifyTextExists("Training Data Statistics");
        const splitStatsTab = screen.getAllByRole("tab");
        expect(splitStatsTab.length).toBeGreaterThan(0);
    });
});
