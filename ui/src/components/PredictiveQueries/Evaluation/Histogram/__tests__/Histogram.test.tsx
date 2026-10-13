// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { act } from "react";
import { MemoryRouter } from "react-router";

import { QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { mockAxiosInstance } from "tests/mockAxios";
import { testResponseType } from "tests/testResponseType";
import { createQueryClient } from "tests/testUtil";
import { describe, it, vi } from "vitest";

import { PredictiveQueryInfo, TaskType } from "types/predictiveQueries";

import Histogram, { HistogramProps } from "../Histogram";

import { setHistogramResponseType } from "./MockFunctions/Histogram.mock";
import { query_info as query_info_raw } from "./MockResponses/HistogramResponse.json";

const query_info = query_info_raw as unknown as PredictiveQueryInfo;

const DISTRIBUTION_TITLE = "Distribution of Predictions";
const SUMMARY_STATS_LABEL = "Summary stats for:";

const trainingJobId = "trainingjob-00055000000000000000000000000000";

const defaultProps: HistogramProps = {
    queryId: "",
    trainingJobId: trainingJobId,
    handleError: vi.fn(),
    queryInfo: query_info,
    samples: ""
};

describe("Histogram", () => {
    const queryClient = createQueryClient();
    const renderHistogram = async (props: HistogramProps) => {
        await act(async () => {
            render(
                <MemoryRouter>
                    <QueryClientProvider client={queryClient}>
                        <Histogram {...props} />
                    </QueryClientProvider>
                </MemoryRouter>
            );
        });
    };

    beforeEach(() => {
        vi.clearAllMocks();
        queryClient.clear();
    });

    const checkApiCall = () =>
        waitFor(() =>
            expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                `v0/training_jobs/${trainingJobId}/eval/regression_histogram`
            )
        );

    const openListboxAndSelectOption = async (
        optionTestId: string,
        expectedText: string
    ) => {
        const statsButton = await screen.findByTestId("testid-combobox");
        await userEvent.click(statsButton);

        const listbox = await screen.findByTestId("listbox");
        const listboxScope = within(listbox);
        const option = listboxScope.getByTestId(optionTestId);
        expect(option).toHaveTextContent(expectedText);

        await userEvent.click(option);

        await waitFor(() =>
            expect(statsButton).toHaveTextContent(expectedText)
        );
        return statsButton;
    };

    const verifyTableRowsAndValues = (
        table: HTMLElement,
        expectedHeaders: string[],
        expectedValues: string[]
    ) => {
        const { getAllByRole } = within(table);
        const rows = getAllByRole("row");
        const headerCells = within(rows[0]).getAllByRole("cell");
        expect(headerCells.map((cell) => cell.textContent)).toEqual(
            expectedHeaders
        );

        const valueCells = within(rows[1]).getAllByRole("cell");
        expect(valueCells.map((cell) => cell.textContent)).toEqual(
            expectedValues
        );
    };

    it("Should render loading animation when queryId and trainingJobId are empty", async () => {
        await renderHistogram({
            ...defaultProps,
            queryId: "",
            trainingJobId: ""
        });

        const skeletonElement = document.querySelector(
            '[data-slot="skeleton"]'
        );
        expect(skeletonElement).toBeInTheDocument();
    });

    it("Should render Histogram with prediction distribution when props are set", async () => {
        await renderHistogram(defaultProps);

        await checkApiCall();

        await screen.findByText(DISTRIBUTION_TITLE);
        const description = document.querySelector('[data-slot="description"]');
        expect(description).toHaveTextContent(
            "Histogram of the predicted value alongside the actual labels (when available) for all entities in your Predictive Query, in the holdout set of the training table."
        );
    });

    it.each([
        {
            selection: "Actuals",
            optionTestId: "combobox-option-0",
            statsHeaders: ["Min", "Mean", "Max"],
            statsValues: ["0", "0.17", "2"],
            percentileHeaders: ["25%", "50%", "75%"],
            percentileValues: ["0", "0", "0"]
        },
        {
            selection: "Predictions",
            optionTestId: "combobox-option-1",
            statsHeaders: ["Min", "Mean", "Max"],
            statsValues: ["-0.00", "0.002", "0.017"],
            percentileHeaders: ["25%", "50%", "75%"],
            percentileValues: ["-0.00", "-0.00", "0.0003"]
        }
    ])(
        "Should render histogram tables with Min/Mean/Max and Percentiles for '$selection' selection",
        async ({
            selection,
            optionTestId,
            statsHeaders,
            statsValues,
            percentileHeaders,
            percentileValues
        }) => {
            await renderHistogram(defaultProps);

            await checkApiCall();

            expect(
                await screen.findByText(SUMMARY_STATS_LABEL)
            ).toBeInTheDocument();

            if (selection !== "Actuals") {
                await openListboxAndSelectOption(optionTestId, selection);
            } else {
                const statsButton = screen.getByTestId("testid-combobox");
                expect(statsButton).toHaveTextContent("Actuals");
            }

            const histogramTables = await screen.findAllByRole("table");

            // Verify Min/Mean/Max table
            verifyTableRowsAndValues(
                histogramTables[0],
                statsHeaders,
                statsValues
            );

            // Verify Percentiles table
            expect(screen.getByText("Percentiles")).toBeInTheDocument();
            verifyTableRowsAndValues(
                histogramTables[1],
                percentileHeaders,
                percentileValues
            );
        }
    );

    it("Should display message `distribution histogram for this query is not available` when dataSplit is Training", async () => {
        await renderHistogram(defaultProps);
        await checkApiCall();

        const dataSplit = await screen.findByTestId("testid-select-split");
        await userEvent.click(dataSplit);

        const trainingSplit = await screen.findByText("Training");
        await userEvent.click(trainingSplit);

        expect(
            screen.getByText(
                "The distribution histogram for this query is not available for the selected data split."
            )
        ).toBeInTheDocument();
    });

    it("Should show tooltip with correct info text", async () => {
        await renderHistogram(defaultProps);

        await checkApiCall();

        const tooltipTrigger = await screen.findByTestId("tooltip-trigger");
        expect(tooltipTrigger).toBeInTheDocument();

        await userEvent.hover(tooltipTrigger);

        const tooltipElements = await screen.findAllByText(
            /This plot can be used to spot-check the quality/i
        );

        const visibleTooltip = tooltipElements.find(
            (el) => el instanceof HTMLElement && el.offsetParent !== null
        );

        expect(visibleTooltip).toBeInTheDocument();
    });

    it("Should call handleError when API fails", async () => {
        setHistogramResponseType(testResponseType.ERROR);

        await renderHistogram(defaultProps);

        await waitFor(() => {
            expect(defaultProps.handleError).toHaveBeenCalledWith(
                expect.objectContaining({
                    status: 500,
                    data: expect.any(Error)
                }),
                "Failed to get Regression Heatmap Stats!"
            );
        });

        setHistogramResponseType(testResponseType.ALL);
    });

    it("Should show alert when queryInfo indicates forecast problem", async () => {
        const forecastQueryInfo = {
            ...query_info,
            taskType: TaskType.FORECASTING
        };

        await renderHistogram({
            ...defaultProps,
            queryInfo: forecastQueryInfo
        });

        expect(await screen.findByText(DISTRIBUTION_TITLE)).toBeInTheDocument();

        const alert = await screen.findByText(
            "Distribution stats are not available for forecasting"
        );
        expect(alert).toBeInTheDocument();

        expect(screen.queryByText(SUMMARY_STATS_LABEL)).not.toBeInTheDocument();
        expect(screen.queryByRole("table")).not.toBeInTheDocument();
    });

    it.each([
        {
            samples_value: "N/A"
        },
        {
            samples_value: "100"
        }
    ])(
        "Should display message when Holdout split is selected and samples is $samples_value",
        async ({ samples_value }) => {
            setHistogramResponseType(testResponseType.EMPTY);

            await renderHistogram({
                ...defaultProps,
                samples: samples_value
            });

            await checkApiCall();

            await waitFor(() => {
                const dataSplit = screen.getByTestId("testid-select-split");
                expect(dataSplit).toHaveTextContent("Holdout");
            });

            expect(
                screen.getByText(
                    "The distribution histogram for this query is not available for the selected data split."
                )
            ).toBeInTheDocument();

            setHistogramResponseType(testResponseType.ALL);
        }
    );
});
