// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import { QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { mockAxiosInstance } from "tests/mockAxios";
import { createQueryClient } from "tests/testUtil";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { TaskType } from "types/predictiveQueries";

import BinaryClassification, {
    BinaryClassificationProps
} from "../BinaryClassification";

const defaultProps: BinaryClassificationProps = {
    queryId: "pquery-00027000000000000000000000000000",
    trainingJobId: "trainingjob-00046000000000000000000000000000",
    taskType: TaskType.BINARY_CLASSIFICATION,
    width: 200,
    handleError: vi.fn()
};

describe("BinaryClassification", () => {
    const queryClient = createQueryClient();
    const renderBinaryClassification = async (
        props: BinaryClassificationProps
    ) => {
        await act(async () => {
            render(
                <MemoryRouter>
                    <QueryClientProvider client={queryClient}>
                        <BinaryClassification {...props} />
                    </QueryClientProvider>
                </MemoryRouter>
            );
        });
    };

    const getSlider = () => screen.getByRole("slider");

    const getSwitch = () => screen.getByRole("switch");

    const getSliderValue = (slider: HTMLElement) =>
        parseFloat(slider.getAttribute("aria-valuenow") || "0");

    const getPrecisionValue = () => {
        const precisionRow = screen.getAllByText("Precision")[0].closest("tr");
        return precisionRow?.querySelectorAll("td")[1]?.textContent?.trim();
    };

    const getMetricValue = (metricName: string) => {
        const labelCell = screen.getAllByText(metricName)[0];
        const valueCell = labelCell.closest("tr")?.querySelectorAll("td")[1];
        return valueCell?.textContent?.trim();
    };

    const waitForClassificationThreshold = async () => {
        await waitFor(() => {
            expect(
                screen.getByText("Classification threshold for")
            ).toBeInTheDocument();
        });
    };

    beforeEach(() => {
        vi.clearAllMocks();
    });

    it("Should render BinaryClassification when all props are set", async () => {
        await renderBinaryClassification(defaultProps);

        const trainingJobId = "trainingjob-00046000000000000000000000000000";
        expect(mockAxiosInstance.get).toHaveBeenCalledWith(
            `v0/training_jobs/${trainingJobId}/eval/binary_classification`
        );

        expect(
            screen.getByText("Classification Tradeoffs & Cutoff")
        ).toBeInTheDocument();
    });

    it("Should display classification threshold when loaded successfully", async () => {
        await renderBinaryClassification(defaultProps);

        await waitForClassificationThreshold();
        expect(screen.getAllByText("True").length).toBeGreaterThan(0);

        const slider = getSlider();
        expect(slider).toBeInTheDocument();
        expect(slider).toHaveAttribute("aria-valuenow", "0.5");
    });

    it("Should display error message when no data is available", async () => {
        const handleError = vi.fn();
        await renderBinaryClassification({
            ...defaultProps,
            trainingJobId: "",
            handleError
        });

        await waitFor(() => {
            expect(
                screen.getByText(
                    "No data found or failed to get Evaluation Stats!"
                )
            ).toBeInTheDocument();
            expect(handleError).toHaveBeenCalled();
        });
    });

    it("Should change option to False under `Classification threshold for`", async () => {
        await renderBinaryClassification(defaultProps);

        await waitForClassificationThreshold();
        expect(screen.getAllByText("True").length).toBeGreaterThan(0);

        const initialPrecision = getPrecisionValue();
        expect(initialPrecision).toBeTruthy();

        const switchButton = getSwitch();
        expect(switchButton).toHaveAttribute("aria-checked", "true");

        // Switch to FALSE
        await userEvent.click(switchButton);
        expect(switchButton).toHaveAttribute("aria-checked", "false");

        const newPrecision = getPrecisionValue();
        expect(newPrecision).toBeTruthy();
        expect(newPrecision).not.toBe(initialPrecision);

        // Switch back to TRUE
        await userEvent.click(switchButton);
        expect(switchButton).toHaveAttribute("aria-checked", "true");

        const revertedPrecision = getPrecisionValue();
        expect(revertedPrecision).toBe(initialPrecision);
    });

    it("Should display confusion matrix when loaded successfully", async () => {
        await renderBinaryClassification(defaultProps);

        await waitFor(() => {
            expect(screen.getByText("Confusion Matrix")).toBeInTheDocument();
        });

        expect(screen.getByText("Predicted")).toBeInTheDocument();
        expect(screen.getByText("Actual")).toBeInTheDocument();

        expect(screen.getByText("151 (14.1%)")).toBeInTheDocument();
        expect(screen.getByText("268 (25.0%)")).toBeInTheDocument();
        expect(screen.getByText("48 (4.5%)")).toBeInTheDocument();
        expect(screen.getByText("607 (56.5%)")).toBeInTheDocument();
    });

    it("Should update confusion matrix when slider changes", async () => {
        await renderBinaryClassification(defaultProps);

        const initialValue = "151 (14.1%)";
        const newValue = "155 (14.4%)";
        const slider = getSlider();
        slider.focus();

        const initialConfusionValue = screen.getByText(initialValue);
        expect(initialConfusionValue).toBeInTheDocument();

        await userEvent.keyboard("{arrowright}{arrowright}");

        await waitFor(() => {
            expect(getSliderValue(slider)).toBeGreaterThan(0.5);
        });

        await waitFor(() => {
            expect(screen.queryByText(initialValue)).not.toBeInTheDocument();
            expect(screen.getByText(newValue)).toBeInTheDocument();
        });
    });

    it("Should display threshold metrics when loaded successfully", async () => {
        await renderBinaryClassification(defaultProps);

        expect(screen.getByText("Threshold Metrics")).toBeInTheDocument();

        const metricChecks: [string, string][] = [
            ["Precision", "0.69"],
            ["Recall", "0.93"],
            ["Accuracy", "0.71"],
            ["Predicted Positive Rate", "0.81"],
            ["F1", "0.79"]
        ];

        metricChecks.forEach(([metric, value]) => {
            const metricValue = getMetricValue(metric);
            expect(metricValue).toBeTruthy();
            expect(metricValue).toBe(value);
        });
    });

    it.each([
        {
            chartTitle: "ROC Curve" as const,
            labels: [
                { text: "Predictive Query Performance", index: 0 },
                { text: "Random Guesses", index: 0 }
            ]
        },
        {
            chartTitle: "Precision VS Recall" as const,
            labels: [
                { text: "Predictive Query Performance", index: 1 },
                { text: "Random Guesses", index: 1 }
            ]
        },
        {
            chartTitle: "Cumulative Gain" as const,
            labels: [
                { text: "Ideal Case", index: 0 },
                { text: "Lift Curve", index: 0 },
                { text: "Baseline", index: 0 }
            ]
        }
    ])(
        "Should display $chartTitle chart with correct labels",
        async ({ chartTitle, labels }) => {
            await renderBinaryClassification(defaultProps);

            await waitFor(() => {
                const title = screen.getByText(chartTitle);
                expect(title).toBeInTheDocument();
            });

            await waitFor(() => {
                labels.forEach(({ text }) => {
                    const labelElements = screen.getAllByText(text);
                    expect(labelElements.length).toBeGreaterThan(0);
                });
            });
        }
    );

    it("Should handle edge cases with min and max values", async () => {
        await renderBinaryClassification(defaultProps);

        const slider = getSlider();
        slider.focus();

        // Move slider to min
        await userEvent.keyboard("{home}");
        await waitFor(() => {
            expect(getSliderValue(slider)).toBeLessThanOrEqual(0.01);
        });

        // Move slider to max
        await userEvent.keyboard("{end}");
        await waitFor(() => {
            expect(getSliderValue(slider)).toBeGreaterThanOrEqual(0.99);
        });
    });

    it("Should move slider using arrow keys in both directions", async () => {
        await renderBinaryClassification(defaultProps);

        const slider = getSlider();
        const initialValue = getSliderValue(slider);

        slider.focus();

        // Move right
        await userEvent.keyboard("{arrowright}");
        await waitFor(() => {
            expect(getSliderValue(slider)).toBeGreaterThan(initialValue);
        });

        // Move left
        await userEvent.keyboard("{arrowleft}{arrowleft}");
        await waitFor(() => {
            expect(getSliderValue(slider)).toBeLessThanOrEqual(initialValue);
        });
    });

    it("Should update all metrics when threshold changes", async () => {
        await renderBinaryClassification(defaultProps);

        const slider = getSlider();
        const initialMetrics = {
            precision: getMetricValue("Precision"),
            recall: getMetricValue("Recall"),
            accuracy: getMetricValue("Accuracy")
        };

        slider.focus();
        await userEvent.keyboard("{arrowright}{arrowright}{arrowright}");

        await waitFor(() => {
            const newPrecision = getMetricValue("Precision");
            const newRecall = getMetricValue("Recall");
            const newAccuracy = getMetricValue("Accuracy");

            expect(newPrecision).not.toBe(initialMetrics.precision);
            expect(newRecall).not.toBe(initialMetrics.recall);
            expect(newAccuracy).not.toBe(initialMetrics.accuracy);
        });
    });
});
