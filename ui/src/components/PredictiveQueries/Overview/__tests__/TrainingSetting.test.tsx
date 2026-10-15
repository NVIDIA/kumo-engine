// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { act } from "react";
import { MemoryRouter } from "react-router";

import { QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { mockAxiosInstance } from "tests/mockAxios";
import { createQueryClient } from "tests/testUtil";

import { jobStatus } from "utils/Constants";

import TrainingSetting, { TrainingSettingProps } from "../TrainingSetting";

import { ModelPlan } from "./MockResponses/ModelPlan";

const id = "trainingjob-00053000000000000000000000000000";
const defaultProps: TrainingSettingProps = {
    queryId: id,
    trainingJobId: id,
    modelPlan: ModelPlan,
    advancedOptions: "",
    generationPlan: "",
    trainingPlan: undefined,
    trainingstatus: jobStatus.Done,
    ctaMsgs: {
        "errors": [],
        "warnings": [],
        "info_items": [],
        "showCTAErrors": true,
        "showCTAWarnings": true
    },
    loadingStatus: false
};

const verifyTrainingSettingsHeader = () => {
    expect(screen.getByText("Training Settings")).toBeInTheDocument();
};

const verifyTabsExist = (tabNames: string[]) => {
    const tabList = screen.getByRole("tablist");
    tabNames.forEach((tabName) => {
        expect(tabList).toHaveTextContent(tabName);
    });
    return tabList;
};

const verifyTableHeaders = (expectedHeaders: string[]) => {
    const tableHeader = screen.getByTestId("table-header");
    expect(tableHeader).toBeInTheDocument();

    const headers = within(tableHeader).getAllByRole("columnheader");
    const headerTexts = headers.map((th) =>
        th.textContent?.trim().toLowerCase()
    );

    expectedHeaders.forEach((expectedHeader) => {
        expect(headerTexts).toContain(expectedHeader.toLowerCase());
    });

    return { tableHeader, headers };
};

const clickTabByName = async (tabName: string) => {
    const tab = screen.getByRole("tab", { name: tabName });
    await userEvent.click(tab);
    return tab;
};

const verifyTabPanelContent = (expectedContent: string) => {
    const tabPanel = screen.getByRole("tabpanel");
    expect(tabPanel).toHaveTextContent(expectedContent);
    return tabPanel;
};

describe("TrainingSetting", () => {
    const queryClient = createQueryClient();
    const renderTrainingSetting = async (props: TrainingSettingProps) => {
        await act(async () => {
            render(
                <MemoryRouter>
                    <QueryClientProvider client={queryClient}>
                        <TrainingSetting {...props} />
                    </QueryClientProvider>
                </MemoryRouter>
            );
        });
    };

    const checkApiCall = async () => {
        await waitFor(() => {
            expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                `v0/training_jobs/${id}/encoders`
            );
        });
    };

    beforeEach(() => {
        vi.clearAllMocks();
        queryClient.clear();
    });

    it("Should render `LinkPredBaseLineResult` with empty props", async () => {
        await renderTrainingSetting(defaultProps);
        await checkApiCall();

        verifyTrainingSettingsHeader();
        verifyTabsExist(["Model plan", "Column processing"]);
    });

    it("Should display Model Plan content", async () => {
        await renderTrainingSetting(defaultProps);
        await checkApiCall();

        verifyTabsExist(["Model plan"]);

        // Changed: tabPanel.textContent includes scroll area CSS and lacks
        // newlines (CodeMirror renders text split across spans). Use the
        // textbox role instead and check for key YAML substrings.
        const editor = screen.getByRole("textbox");
        expect(editor.textContent).toMatch(/training_table_generation:/i);
        expect(editor.textContent).toMatch(/training_job:/i);
    });

    it("Should display Column Processing on clicking the tab", async () => {
        await renderTrainingSetting(defaultProps);
        await checkApiCall();

        const columnProcessing = screen.getAllByRole("tab")[1];
        await userEvent.click(columnProcessing);

        expect(
            screen.getByText(
                /This section shows the Encoders used to process your raw data/i
            )
        ).toBeInTheDocument();

        verifyTableHeaders(["Connector", "Table", "Column", "Encoder"]);
    });

    it("Should render Advanced Options tab when modelPlan is empty", async () => {
        await renderTrainingSetting({
            ...defaultProps,
            modelPlan: "",
            advancedOptions: "advanced options text"
        });

        verifyTrainingSettingsHeader();
        verifyTabsExist(["Advanced options"]);
        verifyTabPanelContent("advanced options text");
    });

    it("Should render Generation Plan tab and its content", async () => {
        await renderTrainingSetting({
            ...defaultProps,
            generationPlan: "generation plan here"
        });

        const tab = await clickTabByName("Training table generation plan");
        expect(tab).toBeInTheDocument();

        verifyTabPanelContent("generation plan here");
    });

    it("Should render Training Plan tab and its content", async () => {
        const trainingPlanText = `
            training_table_generation:
            split: >-
                TimeSplit(
                    timestamp_boundaries=None,
                    ...
                )`;

        await renderTrainingSetting({
            ...defaultProps,
            trainingPlan: trainingPlanText
        });

        const tab = await clickTabByName("Model plan");
        expect(tab).toBeInTheDocument();

        await waitFor(() => {
            verifyTabPanelContent("training_table_generation");
        });
    });

    it("Should display info tooltip when ctaMsgs contains split_setup_validation", async () => {
        await renderTrainingSetting({
            ...defaultProps,
            ctaMsgs: {
                ...defaultProps.ctaMsgs,
                info_items: [
                    {
                        id: "split_adjustment_for_empty_validation_set",
                        description:
                            "Important information about split validation",
                        is_error: false,
                        title: "Split Validation",
                        is_dismissible: false
                    }
                ]
            }
        });

        verifyTrainingSettingsHeader();

        const svgIcon = document.querySelector("svg");
        expect(svgIcon).toBeInTheDocument();

        await userEvent.hover(svgIcon as Element);

        await waitFor(() => {
            const tooltips = screen.getAllByText(
                "Important information about split validation"
            );
            expect(tooltips.length).toBeGreaterThan(0);
        });
    });
});
