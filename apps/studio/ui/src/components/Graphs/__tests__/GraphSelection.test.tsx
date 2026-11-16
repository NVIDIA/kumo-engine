// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { act } from "react";
import { MemoryRouter } from "react-router";

import { QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { mockNavigateCall } from "tests/mockRouter";
import { createQueryClient } from "tests/testUtil";

import { HandleErrorFunc } from "types/validation";

import {
    GraphSelection,
    GraphSelectionProps
} from "../GraphSelection/GraphSelection";

const setGraphName = vi.fn();
const setNameError = vi.fn();
const setSaveGraph = vi.fn();
const setSelectedGraph = vi.fn();
const onGraphSelect = vi.fn();

const defaultProps: GraphSelectionProps = {
    graphName: "",
    setGraphName,
    setNameError,
    editableName: false,
    graphExist: "",
    validatingGraph: false,
    setSaveGraph,
    selectedGraph: {
        value: "",
        label: ""
    },
    setSelectedGraph,
    previousSelected: "",
    handleError: {} as HandleErrorFunc,
    fromNewGraphPage: false,
    onGraphSelect,
    projectId: "project-tests"
};

describe("GraphSelection", () => {
    const queryClient = createQueryClient();
    const renderGraphSelection = async (props: GraphSelectionProps) => {
        await act(async () => {
            render(
                <MemoryRouter>
                    <QueryClientProvider client={queryClient}>
                        <GraphSelection {...props} />
                    </QueryClientProvider>
                </MemoryRouter>
            );
        });
    };

    const getGraphSelect = () => screen.getByTestId("testid-select-graph");

    const verifyGraphSelectExists = async () => {
        await waitFor(() => {
            expect(getGraphSelect()).toBeInTheDocument();
        });
    };

    const verifyGraphSelectDoesNotExist = () => {
        expect(
            screen.queryByTestId("testid-select-graph")
        ).not.toBeInTheDocument();
    };

    const clickGraphSelect = async () => {
        const select = getGraphSelect();
        await userEvent.click(select);
    };

    const selectCreateNewGraph = async () => {
        await verifyGraphSelectExists();
        await clickGraphSelect();

        await waitFor(() => {
            expect(screen.getByText("Create new graph")).toBeInTheDocument();
        });

        const createOption = screen.getByText("Create new graph");
        await userEvent.click(createOption);
    };

    const getBackButton = () => screen.getByRole("button", { name: /back/i });

    beforeEach(() => {
        vi.clearAllMocks();
        mockNavigateCall();
    });

    afterEach(() => {
        queryClient.clear();
        vi.clearAllMocks();
    });

    it("Should render graph selection section when fromNewGraphPage is false", async () => {
        await renderGraphSelection(defaultProps);

        await waitFor(() => {
            expect(screen.getByText("Graph")).toBeInTheDocument();
        });

        await verifyGraphSelectExists();
    });

    it("Should not render anything when fromNewGraphPage is true", async () => {
        await renderGraphSelection({
            ...defaultProps,
            fromNewGraphPage: true
        });

        expect(screen.queryByText("Graph")).not.toBeInTheDocument();
        verifyGraphSelectDoesNotExist();
    });

    it("Should render select dropdown with Create New Graph option", async () => {
        await renderGraphSelection(defaultProps);

        await verifyGraphSelectExists();
        await clickGraphSelect();

        await waitFor(() => {
            expect(screen.getByText("Create new graph")).toBeInTheDocument();
        });
    });

    it("Should call setSelectedGraph when a graph option is clicked", async () => {
        await renderGraphSelection(defaultProps);

        await verifyGraphSelectExists();
        await clickGraphSelect();

        await waitFor(() => {
            expect(screen.getByText("Create new graph")).toBeInTheDocument();
        });

        const createOption = screen.getByText("Create new graph");
        await userEvent.click(createOption);

        await waitFor(() => {
            expect(onGraphSelect).toHaveBeenCalled();
        });
    });

    it("Should not call setSelectedGraph when previousSelected matches selectedGraph value", async () => {
        await renderGraphSelection({
            ...defaultProps,
            selectedGraph: {
                value: "graph1",
                label: "Graph 1"
            },
            previousSelected: "graph1"
        });

        expect(setSelectedGraph).not.toHaveBeenCalled();
    });

    it("Should render skeleton when graphs are loading", async () => {
        render(
            <MemoryRouter>
                <QueryClientProvider client={queryClient}>
                    <GraphSelection {...defaultProps} />
                </QueryClientProvider>
            </MemoryRouter>
        );

        const skeletonElement = document.querySelector(
            '[data-slot="skeleton"]'
        );
        expect(skeletonElement).toBeInTheDocument();
    });

    it("Should render Back button when new graph is selected and saveGraphToReuse is false", async () => {
        await renderGraphSelection({
            ...defaultProps,
            selectedGraph: {
                value: "new",
                label: "Create New Graph"
            }
        });

        await selectCreateNewGraph();

        await waitFor(() => {
            expect(getBackButton()).toBeInTheDocument();
        });
    });

    it("Should call all callbacks when Back button is clicked", async () => {
        const setSelectedGraph = vi.fn();
        const setSaveGraph = vi.fn();
        const setGraphName = vi.fn();
        const onGraphSelect = vi.fn();

        await renderGraphSelection({
            ...defaultProps,
            setSelectedGraph,
            setSaveGraph,
            setGraphName,
            onGraphSelect,
            selectedGraph: {
                value: "new",
                label: "Create New Graph"
            }
        });

        await selectCreateNewGraph();

        await waitFor(() => {
            expect(getBackButton()).toBeInTheDocument();
        });

        await userEvent.click(getBackButton());

        await waitFor(() => {
            expect(setSelectedGraph).toHaveBeenCalledWith(null);
            expect(setSaveGraph).toHaveBeenCalledWith(false);
            expect(setGraphName).toHaveBeenCalledWith("");
            expect(onGraphSelect).toHaveBeenCalled();
        });
    });
});
