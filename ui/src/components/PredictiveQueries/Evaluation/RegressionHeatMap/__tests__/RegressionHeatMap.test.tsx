// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import { QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { mockAxiosInstance } from "tests/mockAxios";
import { testResponseType } from "tests/testResponseType";
import { createQueryClient } from "tests/testUtil";
import { beforeEach, describe, it, vi } from "vitest";

import RegressionHeatMap, {
    RegressionHeatMapProps
} from "../RegressionHeatMap";

import { setRegressionHeatMapResponseType } from "./MockFunctions/RegressionHeatMap.mock";
import { expectedXTicks } from "./MockResponses/RegressionHeatMap";

const trainingJobId = "trainingjob-b9a218e7fcaa40a48e823b54732c86f2";
const defaultProps: RegressionHeatMapProps = {
    queryId: trainingJobId,
    trainingJobId: trainingJobId,
    dataSplit: "test",
    handleError: vi.fn()
};

describe("RegressionHeatMap", () => {
    const queryClient = createQueryClient();
    const renderRegressionHeatMap = async (props: RegressionHeatMapProps) => {
        await act(async () => {
            render(
                <MemoryRouter>
                    <QueryClientProvider client={queryClient}>
                        <RegressionHeatMap {...props} />
                    </QueryClientProvider>
                </MemoryRouter>
            );
        });
    };

    const checkApiCall = (trainingJobId: string) =>
        waitFor(() =>
            expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                `v0/training_jobs/${trainingJobId}/eval/regression_heatmap`
            )
        );

    const getNumericTickTexts = (): string[] => {
        return [...document.querySelectorAll("svg text")]
            .map((el) => el.textContent?.trim())
            .filter((text): text is string => !!text && !isNaN(Number(text)));
    };

    const waitForTickValue = async (value: string) => {
        await screen.findByText(value);
    };

    beforeEach(() => {
        vi.clearAllMocks();
        queryClient.clear();
    });

    it("Should render loading animation when all props are empty", async () => {
        await renderRegressionHeatMap({
            queryId: "",
            trainingJobId: "",
            dataSplit: "",
            handleError: vi.fn()
        });

        const skeletonElement = document.querySelector(
            '[data-slot="skeleton"]'
        );
        expect(skeletonElement).toBeInTheDocument();
    });

    it("Should render chart with axis labels, heat scale, and toggle outliers button", async () => {
        await renderRegressionHeatMap({
            ...defaultProps
        });

        await checkApiCall(trainingJobId);

        await waitFor(() => {
            expect(
                screen.getByText("Predicted vs Actual Values")
            ).toBeInTheDocument();
        });

        expect(await screen.findByText("Actual Values")).toBeInTheDocument();
        expect(screen.getByText("Predicted Values")).toBeInTheDocument();

        await waitForTickValue("71");
        const allNumericTexts = getNumericTickTexts();
        const heatScaleValues = allNumericTexts.slice(-2);
        expect(heatScaleValues).toEqual(["0", "71"]);

        const button = await screen.findByRole("button", {
            name: /Hide Outliers/i
        });

        // Verify initial tick values with outliers
        await waitForTickValue("0.08");
        const allTickTexts1 = getNumericTickTexts();
        const expectedYTicks = ["-0.04", "0", "0.04", "0.08"];
        expect(allTickTexts1).toEqual(
            expect.arrayContaining([...expectedXTicks, ...expectedYTicks])
        );

        await userEvent.click(button);
        expect(button.textContent).toBe("Show Outliers");

        // Verify tick values changed after hiding outliers
        const tickElementsAfter = await screen.findAllByText("-0.04");
        expect(tickElementsAfter.length).toBeGreaterThan(0);

        const allTickTexts2 = getNumericTickTexts();
        const expectedXTicks2 = ["-0.04", "0", "0.04"];
        expect(allTickTexts2.slice(0, -2)).toEqual(
            expect.arrayContaining([...expectedXTicks2])
        );

        // Click to show outliers again
        await userEvent.click(button);
        expect(button.textContent).toBe("Hide Outliers");
    });

    it.each([
        { dataSplit: "test", label: "test" },
        { dataSplit: "val", label: "val" }
    ])(
        "Should display x-axis and y-axis tick labels for the '$label' dataSplit",
        async ({ dataSplit }) => {
            await renderRegressionHeatMap({
                ...defaultProps,
                dataSplit
            });
            await checkApiCall(trainingJobId);

            await waitForTickValue("0.08");

            const allTickTexts = getNumericTickTexts();
            const expectedYTicks = ["-0.04", "0", "0.04", "0.08"];

            const filteredTickTexts = allTickTexts.slice(0, -2); //excluding the heat scale reading
            expect(filteredTickTexts).toEqual(
                expect.arrayContaining([...expectedXTicks, ...expectedYTicks])
            );
        }
    );

    it.each([
        {
            split: "Validation",
            shouldShowOutliers: true
        },
        {
            split: "Training",
            shouldShowOutliers: false
        }
    ])(
        "Should display chart for $split split",
        async ({ split, shouldShowOutliers }) => {
            await renderRegressionHeatMap(defaultProps);
            await checkApiCall(trainingJobId);

            const dropdown = await screen.findByRole("button", {
                name: "Holdout"
            });
            await userEvent.click(dropdown);

            const option = await screen.findByRole("option", { name: split });
            await userEvent.click(option);

            expect(
                screen.getByText("Predicted vs Actual Values")
            ).toBeInTheDocument();

            if (shouldShowOutliers) {
                expect(
                    screen.getByRole("button", { name: "Hide Outliers" })
                ).toBeInTheDocument();
            } else {
                expect(
                    screen.queryByRole("button", { name: "Hide Outliers" })
                ).not.toBeInTheDocument();
                //Add another check
            }
        }
    );

    it.each([
        {
            id: "trainingjob-0c4623b61d0643d6a29b32c90c62f39c",
            scenario: "predicted values are lower than actual values"
        },
        {
            id: "trainingjob-6f8321a97ace4897b30eb4a0600e41bd",
            scenario: "actual values are lower than predicted values"
        }
    ])("Should render correct tick labels when $scenario", async ({ id }) => {
        await renderRegressionHeatMap({
            ...defaultProps,
            trainingJobId: id
        });
        await checkApiCall(id);

        await waitFor(() => {
            const ticks = getNumericTickTexts();
            expect(ticks).toContain("0.5");
        });
    });

    it("Should handle case when no matching points are found for line", async () => {
        const id = "trainingjob-afbc69bc7e60441ab732d6ad95f4c687";
        await renderRegressionHeatMap({
            ...defaultProps,
            trainingJobId: id
        });
        await checkApiCall(id);

        await waitFor(() => {
            const lineAnnotations = document.querySelectorAll(
                "g.annotation-xy-line"
            );
            expect(lineAnnotations.length).toBe(0);
        });
    });

    it.each([
        { responseType: testResponseType.ERROR, scenario: "API fails" },
        {
            responseType: testResponseType.INVALID,
            scenario: "data is invalid"
        }
    ])("Should call handleError when $scenario", async ({ responseType }) => {
        setRegressionHeatMapResponseType(responseType);

        await renderRegressionHeatMap({
            ...defaultProps
        });
        await checkApiCall(trainingJobId);

        await waitFor(() => {
            expect(defaultProps.handleError).toHaveBeenCalledWith(
                expect.any(Object),
                "Failed to get Regression Heatmap Stats!"
            );
        });

        setRegressionHeatMapResponseType(testResponseType.ALL);
    });
});
