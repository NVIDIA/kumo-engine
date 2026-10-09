// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import { QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { mockAxiosInstance } from "tests/mockAxios";
import { createQueryClient } from "tests/testUtil";

import { graphsList } from "pages/Graphs/__tests__/MockResponse/GraphsList.response";

import { GraphFlowInfo } from "types/graphs";

import TrainingSetting, {
    TrainingSettingsProps
} from "../Settings/TrainingSetting";

const setField = vi.fn();
const changeRunMode = vi.fn();
const setNewGraph = vi.fn();
const setGraphFlowInfo = vi.fn();

const defaultProps: TrainingSettingsProps = {
    createJobArgs: {
        name: "",
        graph_id: "",
        query_yaml: "",
        query_dialect: "",
        model_planner_yaml: "",
        description: "",
        run_mode: null,
        graph_name: null
    },
    setField,
    changeRunMode,
    modelPlanLoading: false,
    setNameError: vi.fn(),
    showOverwrittenWarning: false,
    closeWarning: vi.fn(),
    handleError: vi.fn(),
    setNewGraph,
    setGroups: vi.fn(),
    groups: [],
    graphFlowInfo: {
        tables: {},
        name: "",
        id: "",
        description: "",
        selected_tables: [],
        selected_table_ids: [],
        flow_type: ""
    },
    setGraphFlowInfo,
    setSaveGraph: vi.fn(),
    graphExist: "",
    validatingGraph: false,
    setParentLoading: vi.fn(),
    projectId: "project-tests"
};

describe("TrainingSettings", () => {
    const queryClient = createQueryClient();

    const renderTrainingSettings = async (props: TrainingSettingsProps) => {
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
                "v1/graphs/",
                undefined,
                { headers: undefined, params: { project_id: "project-tests" } }
            );
        });
    };

    const navigateToModelTab = async () => {
        const modelTab = await screen.findByRole("tab", {
            name: "Model settings"
        });
        await userEvent.click(modelTab);
        return modelTab;
    };

    beforeEach(() => {
        queryClient.clear();
        vi.clearAllMocks();
    });

    it("Should toggle between Graph and Model settings tabs and display appropriate content", async () => {
        await renderTrainingSettings(defaultProps);
        await checkApiCall();

        const graphTab = await screen.findByRole("tab", {
            name: "Graph settings"
        });
        const modelTab = await screen.findByRole("tab", {
            name: "Model settings"
        });

        expect(graphTab).toHaveAttribute("aria-selected", "true");
        expect(modelTab).toHaveAttribute("aria-selected", "false");

        const selectGraphButton = await screen.findByRole("button", {
            name: "Select graph"
        });
        expect(selectGraphButton).toBeInTheDocument();

        await userEvent.click(modelTab);
        expect(modelTab).toHaveAttribute("aria-selected", "true");
        expect(graphTab).toHaveAttribute("aria-selected", "false");
    });

    it("Should display model configuration options when Model settings tab is selected", async () => {
        await renderTrainingSettings(defaultProps);
        await checkApiCall();

        await navigateToModelTab();

        expect(
            screen.queryByText(/Baseline Comparison/i)
        ).not.toBeInTheDocument();

        expect(screen.getByText("Run Mode")).toBeInTheDocument();
        expect(
            screen.getByText(
                "The Run Mode field is used to determine the default values for your model's Model Plan based on common usage patterns."
            )
        ).toBeInTheDocument();

        // Model Planner is now a separate tab, not in Model settings
        expect(
            screen.getByRole("tab", { name: "Model planner" })
        ).toBeInTheDocument();
    });

    it("Should display graph list dropdown with all graph options and call setField when a graph is selected", async () => {
        const mockSetField = vi.fn();
        await renderTrainingSettings({
            ...defaultProps,
            setField: mockSetField
        });
        await checkApiCall();

        const selectGraphButton = await screen.findByRole("button", {
            name: "Select graph"
        });
        await userEvent.click(selectGraphButton);

        const graphDropdown = await screen.findByRole("listbox");
        expect(graphDropdown).toBeInTheDocument();

        for (const graph of graphsList) {
            expect(await screen.findByText(graph.name)).toBeInTheDocument();
        }

        const firstGraph = graphsList[0];
        const option = await screen.findByText(firstGraph.name);
        await userEvent.click(option);

        await waitFor(() => {
            expect(mockSetField).toHaveBeenCalledWith(
                "graph_id",
                firstGraph.id
            );
            expect(mockSetField).toHaveBeenCalledWith(
                "graph_name",
                firstGraph.name
            );
        });
    });

    it("Should call setNewGraph when 'Create new graph' is selected", async () => {
        await renderTrainingSettings(defaultProps);
        await checkApiCall();

        const selectGraphButton = await screen.findByRole("button", {
            name: "Select graph"
        });
        await userEvent.click(selectGraphButton);

        const createNewOption = await screen.findByText("Create new graph");
        await userEvent.click(createNewOption);

        expect(setNewGraph).toHaveBeenCalledWith(true);
    });

    it("Should call changeRunMode when a run mode option is selected", async () => {
        await renderTrainingSettings(defaultProps);
        await navigateToModelTab();

        const runModeSection = await screen.findByText("Run Mode");
        const comboboxButton = within(
            runModeSection.closest("form")!
        ).getByTestId("testid-run-mode-select");
        await userEvent.click(comboboxButton);

        const options = ["BEST", "FAST", "NORMAL"];
        for (const option of options) {
            expect(await screen.findByText(option)).toBeInTheDocument();
        }

        const fastOption = await screen.findByText("FAST");
        await userEvent.click(fastOption);
        expect(changeRunMode).toHaveBeenCalledWith({
            label: "FAST",
            value: "FAST"
        });

        await userEvent.click(comboboxButton);
        const bestOption = await screen.findByText("BEST");
        await userEvent.click(bestOption);
        expect(changeRunMode).toHaveBeenCalledWith({
            label: "BEST",
            value: "BEST"
        });

        await userEvent.click(comboboxButton);
        const normalOption = await screen.findByText("NORMAL");
        await userEvent.click(normalOption);
        expect(changeRunMode).toHaveBeenCalledWith({
            label: "NORMAL",
            value: "NORMAL"
        });
    });

    it("Should show Form and YAML tabs in the Model planner tab", async () => {
        await renderTrainingSettings(defaultProps);

        const modelPlannerTab = await screen.findByRole("tab", {
            name: "Model planner"
        });
        await userEvent.click(modelPlannerTab);

        expect(
            await screen.findByRole("tab", { name: "Form" })
        ).toBeInTheDocument();
        expect(screen.getByRole("tab", { name: "YAML" })).toBeInTheDocument();
    });

    it("Should display `Something went wrong` for `Invalid` table list", async () => {
        await renderTrainingSettings({
            ...defaultProps,
            graphFlowInfo: { name: false } as unknown as GraphFlowInfo
        });

        const alert = await screen.findByRole("alert");
        expect(alert).toHaveTextContent(/Something went wrong in/i);
        expect(alert).toHaveTextContent(/GraphLinkSetup/i);
        expect(alert).toHaveTextContent(/Please contact Kumo support/i);
    });
});
