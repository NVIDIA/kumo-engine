// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { RenderResult, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { mockAxiosInstance } from "tests/mockAxios";

import { graphsList } from "pages/Graphs/__tests__/MockResponse/GraphsList.response";

import { GraphFlowInfo } from "types/graphs";

import { GraphLinkSetupProps } from "../LinkSetup/GraphLinkSetup";

import {
    baseGraphFlowInfo,
    baseGroups,
    mockTablesWithNoPKey
} from "./MockFunctions/GraphLinkGroups.mock";
import {
    defaultGraphLinkSetupProps,
    resetQueryClient,
    setupGraphTests
} from "./graphTestUtils";

let groups: typeof baseGroups;
let graphFlowInfo: typeof baseGraphFlowInfo;

describe("GraphLinkSetup", () => {
    const { graphLinkSetupMock, renderGraphLinkSetup } = setupGraphTests();

    beforeEach(() => {
        resetQueryClient();
        groups = structuredClone(baseGroups);
        graphFlowInfo = structuredClone(baseGraphFlowInfo);
    });

    afterEach(() => {
        vi.clearAllMocks();
        vi.resetAllMocks();
    });

    const renderComponent = async (
        overrides: Partial<GraphLinkSetupProps> = {}
    ) => {
        return await renderGraphLinkSetup({
            ...defaultGraphLinkSetupProps,
            ...overrides
        });
    };

    const checkApiCall = async () => {
        await waitFor(() => {
            const APIS_TO_CALL = ["v1/graphs/"];
            APIS_TO_CALL.forEach((API) => {
                expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                    `${API}`,
                    undefined,
                    {
                        headers: undefined,
                        params: { project_id: "project-tests" }
                    }
                );
            });
        });
    };

    const openGraphDropdown = async (
        getByTestId: RenderResult["getByTestId"]
    ) => {
        const dropdownButton = getByTestId("testid-select-graph");
        await userEvent.click(dropdownButton);
    };

    const selectCreateNewGraph = async (
        findByText: RenderResult["findByText"]
    ) => {
        const createNewOption = await findByText("Create new graph");
        await userEvent.click(createNewOption);
    };

    const clickSaveGraphCheckbox = async () => {
        const saveGraphCheckbox = document.querySelector(
            '[data-slot="checkbox"]'
        );
        if (!saveGraphCheckbox) {
            throw new Error("Reuse checkbox not found");
        }
        await userEvent.click(saveGraphCheckbox);
        return saveGraphCheckbox;
    };

    const openTableDropdown = async (
        getByTestId: RenderResult["getByTestId"]
    ) => {
        await waitFor(() => {
            expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                "v1/tables/",
                undefined,
                { headers: undefined, params: { project_id: "project-tests" } }
            );
        });
        const dropdownButton = getByTestId("testid-combobox");
        await userEvent.click(dropdownButton);
    };

    const selectTable = async (
        findByTestId: RenderResult["findByTestId"],
        optionIndex: number = 1
    ) => {
        const tableOption = await findByTestId(
            `combobox-option-${optionIndex}`
        );
        await userEvent.click(tableOption);
    };

    it("Should render the component with default props", async () => {
        const { getByText } = await renderComponent();
        await checkApiCall();

        expect(getByText("Graph")).toBeInTheDocument();
        expect(getByText("Tables & links")).toBeInTheDocument();
        expect(
            getByText(
                "You can change the tables and update links between them, the changes will be saved as a new graph without altering the original graph."
            )
        ).toBeInTheDocument();
    });

    it("Should render the empty state when 'show' is false", async () => {
        const { queryByText } = await renderComponent({ show: false });
        await checkApiCall();

        expect(queryByText("Graph")?.closest(".hidden")).not.toBeNull();
        expect(
            queryByText("Tables & links")?.closest(".hidden")
        ).not.toBeNull();
    });

    it("Should call setGroups when a group is marked as updated", async () => {
        const { setGroups } = graphLinkSetupMock;

        await renderComponent({
            groups,
            graphFlowInfo,
            setGroups
        });

        await checkApiCall();

        await waitFor(() => {
            expect(setGroups).toHaveBeenCalled();
        });
    });

    it("Should not call setGroups if graphFlowInfo is not provided", async () => {
        const { setGroups } = graphLinkSetupMock;

        await renderComponent({ setGroups });

        await checkApiCall();

        await waitFor(() => {
            expect(setGroups).not.toHaveBeenCalled();
        });
    });

    it("Should render graph list and call setSelectedGraph on graph selection", async () => {
        const { setSelectedGraph } = graphLinkSetupMock;

        const { getByText, getByTestId, findByText } = await renderComponent({
            setSelectedGraph
        });

        await checkApiCall();

        expect(getByText("Graph")).toBeInTheDocument();
        await openGraphDropdown(getByTestId);

        expect(await findByText("Create new graph")).toBeInTheDocument();

        for (const graph of graphsList) {
            expect(await findByText(graph.name)).toBeInTheDocument();
        }

        const firstGraph = graphsList[0];
        const firstOption = await findByText(firstGraph.name);
        await userEvent.click(firstOption);

        await waitFor(() => {
            expect(setSelectedGraph).toHaveBeenCalledWith({
                label: firstGraph.name,
                value: firstGraph.id
            });
        });
    });

    it("Should call setGraphName and setSaveGraph when 'Save graph to reuse later' is checked after selecting 'Create new graph'", async () => {
        const { setAllTables, setGraphName, setSaveGraph } = graphLinkSetupMock;

        const { getByText, getByTestId, findByText } = await renderComponent({
            setAllTables,
            setGraphName,
            setSaveGraph,
            newGraphName: "my graph"
        });

        await checkApiCall();

        expect(getByText("Graph")).toBeInTheDocument();
        await openGraphDropdown(getByTestId);
        await selectCreateNewGraph(findByText);
        await clickSaveGraphCheckbox();

        expect(setGraphName).toHaveBeenCalledWith("my graph");
        expect(setSaveGraph).toHaveBeenCalledWith(true);
    });

    it("Should not allow typing in the graph name input when editableName is false", async () => {
        const { setGraphName } = graphLinkSetupMock;

        const { getByPlaceholderText, getByTestId, findByText } =
            await renderComponent({
                newGraphName: "",
                setGraphName,
                editableName: false
            });

        await checkApiCall();
        await openGraphDropdown(getByTestId);
        await selectCreateNewGraph(findByText);
        await clickSaveGraphCheckbox();

        const input = getByPlaceholderText("Please enter graph name");
        expect(input).toBeInTheDocument();

        setGraphName.mockClear();

        await userEvent.type(input, "Test Graph");
        expect(setGraphName).not.toHaveBeenCalled();
        expect(input).toHaveAttribute("readOnly");
    });

    it("Should call setGraphName when graph name input is editable", async () => {
        const { setGraphName } = graphLinkSetupMock;

        const { getByPlaceholderText, getByTestId, findByText } =
            await renderComponent({
                newGraphName: "",
                setGraphName,
                editableName: true
            });

        await checkApiCall();
        await openGraphDropdown(getByTestId);
        await selectCreateNewGraph(findByText);
        await clickSaveGraphCheckbox();

        const input = getByPlaceholderText("Please enter graph name");
        expect(input).toBeInTheDocument();

        setGraphName.mockClear();

        await userEvent.type(input, "Test Graph");
        expect(setGraphName).toHaveBeenCalled();

        const typedValue = setGraphName.mock.calls
            .map((call) => call[0])
            .join("");
        expect(typedValue).toBe("Test Graph");
    });

    it("Should reset graph state when Remove graph button is clicked", async () => {
        const { setSelectedGraph, setGraphName, setSaveGraph } =
            graphLinkSetupMock;

        const { getByText, getByTestId, findByText } = await renderComponent({
            setSelectedGraph,
            setGraphName,
            setSaveGraph
        });

        await checkApiCall();
        await openGraphDropdown(getByTestId);
        await selectCreateNewGraph(findByText);

        expect(
            await findByText("Save graph to reuse later")
        ).toBeInTheDocument();
        await clickSaveGraphCheckbox();

        expect(
            getByText("Graph name is needed if saving for later use")
        ).toBeInTheDocument();

        const removeButton = getByText(/Remove graph/i);
        await userEvent.click(removeButton);

        expect(setSelectedGraph).toHaveBeenCalledWith(null);
        expect(setGraphName).toHaveBeenCalledWith("");
        expect(setSaveGraph).toHaveBeenCalledWith(false);
    });

    it("Should trigger error handler if graph name already exists", async () => {
        const { handleError } = graphLinkSetupMock;

        const { getByTestId, findByText } = await renderComponent({
            graphExist: "ltv_graph",
            handleError
        });

        await checkApiCall();
        await openGraphDropdown(getByTestId);
        await selectCreateNewGraph(findByText);

        expect(
            await findByText("Save graph to reuse later")
        ).toBeInTheDocument();
        await clickSaveGraphCheckbox();

        expect(handleError).toHaveBeenCalledWith(
            expect.objectContaining({
                message: "Graph already exists with name ltv_graph"
            }),
            "Graph already exists!"
        );
    });

    it("Should show error label when graph name is missing and save is checked", async () => {
        const { setGraphName, setNameError } = graphLinkSetupMock;

        const { getByTestId, findByText, getByText } = await renderComponent({
            setGraphName,
            setNameError
        });

        await checkApiCall();
        await openGraphDropdown(getByTestId);
        await selectCreateNewGraph(findByText);

        expect(
            await findByText("Save graph to reuse later")
        ).toBeInTheDocument();
        await clickSaveGraphCheckbox();

        expect(
            getByText("Graph name is needed if saving for later use")
        ).toBeInTheDocument();
    });

    it("Should call resetErrLabel when a table is selected", async () => {
        const { resetErrLabel } = graphLinkSetupMock;

        const { getByTestId, findByTestId } = await renderComponent({
            resetErrLabel,
            fromNewGraphPage: true
        });

        await openTableDropdown(getByTestId);
        await selectTable(findByTestId);

        expect(resetErrLabel).toHaveBeenCalled();
    });

    it("Should call setGraphFlowInfo when a table is selected", async () => {
        const { setGraphFlowInfo } = graphLinkSetupMock;

        const { getByTestId, findByTestId } = await renderComponent({
            setGraphFlowInfo,
            fromNewGraphPage: true
        });

        await openTableDropdown(getByTestId);
        await selectTable(findByTestId);

        expect(setGraphFlowInfo).toHaveBeenCalled();
    });

    it("Should call setAllTables when a table is selected", async () => {
        const { setAllTables } = graphLinkSetupMock;

        const { getByTestId, findByTestId } = await renderComponent({
            setAllTables,
            fromNewGraphPage: true
        });

        await openTableDropdown(getByTestId);
        await selectTable(findByTestId);

        await waitFor(() => {
            expect(setAllTables).toHaveBeenCalled();
        });
    });

    it("Should display 'No table found.' for invalid table search", async () => {
        const { setAllTables } = graphLinkSetupMock;

        const { getByTestId, getByPlaceholderText, findByText } =
            await renderComponent({
                setAllTables,
                fromNewGraphPage: true
            });

        await openTableDropdown(getByTestId);

        const searchInput = getByPlaceholderText("Search...");
        await userEvent.type(searchInput, "abc");

        expect(await findByText("No table found.")).toBeInTheDocument();
    });

    it("Should show warning when no primary key exists in selected tables", async () => {
        const { getByText } = await renderComponent({
            graphFlowInfo: {
                ...defaultGraphLinkSetupProps.graphFlowInfo,
                flow_type: "",
                selected_tables: ["table1", "table2"],
                tables: mockTablesWithNoPKey
            },
            isTableStage: false
        });

        await checkApiCall();

        expect(
            getByText(/At least one of the tables need to have a primary key/i)
        ).toBeInTheDocument();
    });

    it("Should display 'Something went wrong' for invalid table list", async () => {
        const { queryByRole } = await renderComponent({
            graphFlowInfo: { name: false } as unknown as GraphFlowInfo
        });

        const alert = queryByRole("alert");
        expect(alert).toBeInTheDocument();
        expect(alert).toHaveTextContent(/Something went wrong in/i);
        expect(alert).toHaveTextContent(/GraphLinkSetup/i);
        expect(alert).toHaveTextContent(/Please contact Kumo support/i);
    });

    it("Should call setTableList when tables are fetched", async () => {
        const { setTableList } = graphLinkSetupMock;

        await renderComponent({
            setTableList,
            fromNewGraphPage: true
        });

        await waitFor(() => {
            expect(setTableList).toHaveBeenCalled();
        });
    });

    it("Should not render GraphSelection when hideGraphSelection is true", async () => {
        const { queryByText, queryByPlaceholderText } = await renderComponent({
            hideGraphSelection: true
        });

        await waitFor(() => {
            expect(mockAxiosInstance.get).toHaveBeenCalled();
        });

        expect(queryByText("Graph")).not.toBeInTheDocument();

        expect(queryByPlaceholderText("Select graph")).not.toBeInTheDocument();

        expect(queryByText("Create new graph")).not.toBeInTheDocument();
    });

    it("Should not render GraphLinkGroups when no tables are in graphFlowInfo", async () => {
        const { queryByTestId } = await renderComponent({
            graphFlowInfo: {
                ...defaultGraphLinkSetupProps.graphFlowInfo,
                tables: {}
            },
            selectedGraph: { label: "Test Graph", value: "1" },
            isTableStage: true
        });

        await waitFor(() => {
            expect(mockAxiosInstance.get).toHaveBeenCalled();
        });

        expect(queryByTestId("fwdlinks-0")).not.toBeInTheDocument();
        expect(queryByTestId("bklinks-0")).not.toBeInTheDocument();
    });

    it("Should not render SelectTableDropdown when conditions are not met", async () => {
        const { queryByPlaceholderText, queryByText } = await renderComponent({
            selectedGraph: undefined,
            fromNewGraphPage: false,
            isTableStage: true
        });

        await checkApiCall();

        expect(queryByPlaceholderText("Select tables")).not.toBeInTheDocument();

        expect(queryByText("Tables & links")).not.toBeInTheDocument();
    });
});
