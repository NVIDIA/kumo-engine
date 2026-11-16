// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import { QueryClientProvider } from "@tanstack/react-query";
import {
    act,
    fireEvent,
    render,
    screen,
    waitFor
} from "@testing-library/react";
import { mockAxiosInstance } from "tests/mockAxios";
import { testResponseType } from "tests/testResponseType";
import { createQueryClient } from "tests/testUtil";
import { beforeEach, describe, it, vi } from "vitest";

import ExperimentMonitoringMetrics, {
    ExperimentMonitoringMetricsProps
} from "components/PredictiveQueries/Training/ExperimentMonitoringMetrics/ExperimentMonitoringMetrics";

import { jobStatus } from "utils/Constants";

import { setManyScalarsResponseType } from "./MockFunctions/ManyScalar.mock";
import { setTrialTagsResponseType } from "./MockFunctions/TrialTags.mock";
import { manyScalarPostPayload } from "./MockResponses/ManyScalar";

const trainingjobId = "trainingjob-00046000000000000000000000000000";
const defaultProps: ExperimentMonitoringMetricsProps = {
    jobId: trainingjobId,
    trainingStatus: jobStatus.Done
};

describe("ExperimentMonitoringMetrics", () => {
    const queryClient = createQueryClient();
    const renderExperimentMonitoringMetrics = async (
        props: ExperimentMonitoringMetricsProps
    ) => {
        await act(async () => {
            render(
                <MemoryRouter>
                    <QueryClientProvider client={queryClient}>
                        <ExperimentMonitoringMetrics {...props} />
                    </QueryClientProvider>
                </MemoryRouter>
            );
        });
    };

    beforeEach(() => {
        queryClient.clear();
        vi.clearAllMocks();

        setTrialTagsResponseType(testResponseType.ALL);
        setManyScalarsResponseType(testResponseType.ALL);
    });

    const waitForAPICALL = async () => {
        await waitFor(() => {
            expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                `v0/tensorboard/${trainingjobId}/plugin/scalars/tags`
            );

            expect(mockAxiosInstance.post).toHaveBeenCalledWith(
                `v0/tensorboard/plugin/scalars/many_scalars`,
                manyScalarPostPayload,
                {
                    "headers": {
                        "Content-Type": "application/json"
                    }
                }
            );
        });
    };

    const getMetricsDropdown = async () => {
        return await screen.findByTestId(
            "testid-Experiment Monitoring Metricsdropdown"
        );
    };

    it("Should not render Experiment Monitoring Metrics when training status is not provided", async () => {
        await renderExperimentMonitoringMetrics({
            jobId: "",
            trainingStatus: ""
        });

        const textHeading = screen.queryByText("Experiment Monitoring Metrics");

        expect(textHeading).not.toBeInTheDocument();
        expect(document.body.innerHTML).toBe("<div></div>");
    });

    it("Should render `Experiment Monitoring Metrics` when training status is provided", async () => {
        await renderExperimentMonitoringMetrics(defaultProps);

        await waitForAPICALL();

        expect(
            await screen.findByText(/Experiment Monitoring Metrics/i)
        ).toBeInTheDocument();

        const dropdownContainer = await getMetricsDropdown();
        expect(dropdownContainer).toHaveTextContent("AP");
    });

    it.each([
        { chartTitle: "Train AP", metricName: "AP" },
        { chartTitle: "Val AP", metricName: "AP" }
    ])(
        "Should render chart with correct axis titles for $chartTitle",
        async ({ chartTitle, metricName }) => {
            await renderExperimentMonitoringMetrics(defaultProps);

            await waitForAPICALL();

            expect(await screen.findByText(chartTitle)).toBeInTheDocument();

            const xAxisTitle = screen.getAllByText("Epoch");
            xAxisTitle.forEach((x) => {
                expect(x).toBeVisible();
            });

            const yAxisTitle = screen.getAllByText(metricName);
            yAxisTitle.forEach((y) => {
                expect(y).toBeVisible();
            });
        }
    );

    it("Should render X-axis and Y-axis ticks and labels correctly", async () => {
        await renderExperimentMonitoringMetrics(defaultProps);

        await waitForAPICALL();

        // X-axis ticks
        const xAxisTicks = ["2", "3", "4", "5", "6"];
        await waitFor(() => {
            expect(screen.getAllByText(xAxisTicks[0]).length).toBeGreaterThan(
                0
            );
        });
        xAxisTicks.forEach((num) => {
            screen.getAllByText(num).forEach((elem) => {
                expect(elem).toBeInTheDocument();
            });
        });

        // Y-axis ticks
        const yAxisTicks = ["0.38", "0.41", "0.44", "0.47", "0.50", "0.53"];
        for (const num of yAxisTicks) {
            const elements = screen.getAllByText(num);
            expect(elements.length).toBeGreaterThan(0);
            elements.forEach((el) => expect(el).toBeInTheDocument());
        }
    });

    it("Should render legend with label 'Experiment 1' and toggle checkbox on click", async () => {
        await renderExperimentMonitoringMetrics(defaultProps);

        await waitForAPICALL();

        expect(screen.getByText("Experiment 1")).toBeInTheDocument();

        const checkboxes = screen.getAllByRole("checkbox");
        checkboxes.forEach(async (checkbox) => {
            expect(checkbox).toBeTruthy();

            // Initially checked?
            expect(checkbox).toHaveAttribute("aria-checked", "true");
            expect(checkbox).toHaveAttribute("data-state", "checked");

            // Click to uncheck
            fireEvent.click(checkbox);
            expect(checkbox).toHaveAttribute("aria-checked", "false");
            expect(checkbox).toHaveAttribute("data-state", "unchecked");

            // Click again to re-check
            fireEvent.click(checkbox);
            expect(checkbox).toHaveAttribute("aria-checked", "true");
            expect(checkbox).toHaveAttribute("data-state", "checked");
        });
    });

    it("Should expand dropdown on click and display all metric options", async () => {
        await renderExperimentMonitoringMetrics(defaultProps);

        await waitForAPICALL();

        const dropdownInput = await getMetricsDropdown();

        // Open the dropdown
        await act(async () => {
            fireEvent.click(dropdownInput);
        });

        const expectedOptions = ["AUROC", "AUPRC", "AP"];
        for (const option of expectedOptions) {
            expect(
                screen.getByRole("option", { name: option })
            ).toBeInTheDocument();
        }
    });

    it("Should allow selecting a different metric from the dropdown", async () => {
        await renderExperimentMonitoringMetrics(defaultProps);

        await waitForAPICALL();

        const dropdownInput = await getMetricsDropdown();

        // Open the dropdown
        await act(async () => {
            fireEvent.click(dropdownInput);
        });

        const selectedMetric = "AUROC";
        const aurocOption = screen.getByRole("option", {
            name: selectedMetric
        });
        await act(async () => {
            fireEvent.click(aurocOption);
        });

        expect(dropdownInput).toHaveAttribute("aria-expanded", "false");

        // Chart title reflects the new metric
        const chartTitle = screen.getByText("Train AUROC");
        expect(chartTitle).toBeInTheDocument();

        const chartTitle2 = screen.getByText("Val AUROC");
        expect(chartTitle2).toBeInTheDocument();

        const xAxisTitle = screen.getAllByText("Epoch");
        xAxisTitle.forEach((x) => {
            expect(x).toBeVisible();
        });

        const yAxisTitle = screen.getAllByText(selectedMetric);
        yAxisTitle.forEach((y) => {
            expect(y).toBeVisible();
        });
    });

    const shouldNotRenderStatuses = [
        jobStatus.Cancelled,
        jobStatus.Failed,
        jobStatus.Queued,
        jobStatus.Untrained
    ];

    it.each(shouldNotRenderStatuses)(
        "Should not render `Experiment Monitoring Metrics` when status is %s",
        async (status) => {
            await renderExperimentMonitoringMetrics({
                ...defaultProps,
                trainingStatus: status
            });
            await waitFor(() => {
                const textHeading = screen.queryByText(
                    "Experiment Monitoring Metrics"
                );
                expect(textHeading).not.toBeInTheDocument();
            });
        }
    );

    it("Should render `Experiment Monitoring Metrics` when status is RUNNING", async () => {
        await renderExperimentMonitoringMetrics({
            ...defaultProps,
            trainingStatus: jobStatus.Running
        });

        await waitForAPICALL();

        await waitFor(() => {
            const textHeading = screen.queryByText(
                "Experiment Monitoring Metrics"
            );
            expect(textHeading).toBeInTheDocument();
        });
    });

    it.each([
        { responseType: testResponseType.ERROR, description: "Error" },
        { responseType: testResponseType.INVALID, description: "Invalid" },
        { responseType: testResponseType.EMPTY, description: "Empty" }
    ])(
        "Should display empty dom when `tensorboard` API returns $description",
        async ({ responseType }) => {
            setTrialTagsResponseType(responseType);
            await renderExperimentMonitoringMetrics(defaultProps);

            expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                `v0/tensorboard/${trainingjobId}/plugin/scalars/tags`
            );

            expect(
                screen.queryByText(/Experiment Monitoring Metrics/i)
            ).not.toBeInTheDocument();
        }
    );

    it.each([
        { responseType: testResponseType.ERROR, description: "Error" },
        { responseType: testResponseType.EMPTY, description: "Empty" }
    ])(
        "Should render chart with empty data when `many_scalars` API returns $description",
        async ({ responseType }) => {
            setManyScalarsResponseType(responseType);
            await renderExperimentMonitoringMetrics(defaultProps);

            await waitForAPICALL();

            const dropdownContainer = await getMetricsDropdown();
            expect(dropdownContainer).toHaveTextContent("AP");

            expect(screen.getAllByText("Epoch").length).toBeGreaterThan(0);
            expect(screen.getAllByText("AP").length).toBeGreaterThan(0);

            const emptyPaths = document.querySelectorAll('path[d=""]');
            expect(emptyPaths.length).toBeGreaterThan(0);

            const dataLinePaths = Array.from(
                document.querySelectorAll("g[clip-path] > path")
            );
            const nonEmptyDataPaths = dataLinePaths.filter(
                (path) => path.getAttribute("d")?.trim() !== ""
            );
            expect(nonEmptyDataPaths.length).toBe(0);
        }
    );

    it("Should display `Something went wrong` when `many_scalars` API returns Invalid response", async () => {
        setManyScalarsResponseType(testResponseType.INVALID);
        await renderExperimentMonitoringMetrics(defaultProps);

        await waitForAPICALL();

        const alert = await screen.findByRole("alert");
        expect(alert).toHaveTextContent("Something went wrong");
        expect(alert).toHaveTextContent("Please contact Kumo support");
        expect(
            screen.getByRole("button", { name: "Refresh Page" })
        ).toBeInTheDocument();
    });

    it("Should move tuneMetric to the beginning of dropdown options when tuneMetric is provided", async () => {
        await renderExperimentMonitoringMetrics({
            ...defaultProps,
            tuneMetric: "auroc"
        });

        await waitForAPICALL();

        const dropdownInput = await getMetricsDropdown();
        expect(dropdownInput).toHaveTextContent("AUROC");

        expect(await screen.findByText("Train AUROC")).toBeInTheDocument();
        expect(await screen.findByText("Val AUROC")).toBeInTheDocument();
    });
});
