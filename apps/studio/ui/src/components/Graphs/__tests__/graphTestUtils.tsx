// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import { QueryClientProvider } from "@tanstack/react-query";
import { act, render } from "@testing-library/react";
import { createQueryClient } from "tests/testUtil";
import { vi } from "vitest";

import { OptionType } from "types/common";
import { FlowType } from "types/tables";
import { HandleErrorFunc } from "types/validation";

import {
    GraphLinkGroups,
    GraphLinkGroupsProps
} from "../LinkGroups/GraphLinkGroups";
import GraphLinkHealth, {
    GraphLinkHealthProps
} from "../LinkHealth/GraphLinkHealth";
import GraphLinkSetup, {
    GraphLinkSetupProps
} from "../LinkSetup/GraphLinkSetup";
import GraphNameInput, {
    GraphNameInputProps
} from "../NameInput/GraphNameInput";
import GraphOverview, { GraphOverviewProps } from "../Overview/GraphOverview";
import GraphSelectTables, {
    GraphSelectTablesProps
} from "../SelectTables/GraphSelectTables";

import {
    baseAllTables,
    baseBackwardLinks,
    baseGraphFlowInfo,
    baseGroups
} from "./MockFunctions/GraphLinkGroups.mock";
import { mockListResponse } from "./MockResponses/MockColumnStats";

// reset QueryClient
const queryClient = createQueryClient();

export const resetQueryClient = () => {
    queryClient.clear();
};

export const defaultGraphLinkGroupsProps: GraphLinkGroupsProps = {
    allTables: structuredClone(baseAllTables),
    graphFlowInfo: structuredClone(baseGraphFlowInfo),
    groups: structuredClone(baseGroups),
    setGroups: vi.fn(),
    backwardLinks: structuredClone(baseBackwardLinks),
    handleError: vi.fn(),
    show: true
};

export const defaultGraphLinkHealthProps: GraphLinkHealthProps = {
    summaryParams: {
        graphsEdges: [],
        message: ""
    },
    handleError: vi.fn() as HandleErrorFunc
};

export const defaultGraphNameInputProps: GraphNameInputProps = {
    name: "",
    setName: vi.fn(),
    readOnly: false,
    setError: vi.fn()
};

const mockGraphFlowInfo = {
    flow_type: "",
    selected_tables: [],
    tables: {},
    name: "",
    id: "",
    description: "",
    selected_table_ids: []
};

const mockOption: OptionType = { label: "", value: "" };

export const defaultGraphLinkSetupProps: GraphLinkSetupProps = {
    groups: [],
    setGroups: vi.fn(),
    selectedGraph: mockOption,
    setSelectedGraph: vi.fn(),
    graphFlowInfo: mockGraphFlowInfo,
    setGraphFlowInfo: vi.fn(),
    allTables: {},
    setAllTables: vi.fn(),
    show: true,
    setGraphName: vi.fn(),
    newGraphName: "",
    setNameError: vi.fn(),
    tableList: [],
    setTableList: vi.fn(),
    isTableStage: false,
    fromNewGraphPage: false,
    setTableIdMap: vi.fn(),
    handleError: vi.fn(),
    resetErrLabel: vi.fn(),
    editableName: true,
    setSaveGraph: vi.fn(),
    graphExist: "",
    validatingGraph: false,
    suggestLinks: vi.fn(),
    setLoading: vi.fn(),
    projectId: "project-tests"
};

export const defaultGraphOverviewProps: GraphOverviewProps = {
    graphId: "",
    onEditClick: vi.fn(),
    onDeleteClick: vi.fn(),
    onCopyClick: vi.fn(),
    hideMenuOptions: false
};

export const defaultGraphSelectTablesProps: GraphSelectTablesProps = {
    tableList: mockListResponse,
    graphFlowInfo: {
        name: "",
        id: "",
        description: "",
        tables: {},
        selected_tables: [],
        selected_table_ids: [],
        flow_type: FlowType.CREATE
    },
    setGraphFlowInfo: vi.fn(),
    allTables: {},
    setAllTables: vi.fn(),
    tableIdMap: {},
    updateGroups: vi.fn(),
    suggestLinks: vi.fn(),
    handleError: vi.fn(),
    setLoading: vi.fn(),
    projectId: "project-tests"
};

export const setupGraphTests = () => {
    const graphLinkHealthMocks = {
        handleError: vi.fn()
    };

    const graphLinkGroupsMocks = {
        setGroups: vi.fn(),
        handleError: vi.fn()
    };

    const graphNameInputMocks = {
        setName: vi.fn<(s: string) => void>(),
        setError: vi.fn(),
        mockSetValName: vi.fn(),
        validationStartSpy: vi.fn(),
        validationEndSpy: vi.fn(),
        onValidationEndSpy: vi.fn()
    };

    const graphLinkSetupMock = {
        setGroups: vi.fn(),
        setSelectedGraph: vi.fn(),
        setGraphFlowInfo: vi.fn(),
        setAllTables: vi.fn(),
        setGraphName: vi.fn(),
        setNameError: vi.fn(),
        setTableList: vi.fn(),
        setTableIdMap: vi.fn(),
        handleError: vi.fn(),
        resetErrLabel: vi.fn(),
        setSaveGraph: vi.fn(),
        suggestLinks: vi.fn()
    };

    const graphOverviewMock = {
        onEditClick: vi.fn(),
        onDeleteClick: vi.fn(),
        onCopyClick: vi.fn()
    };

    const graphSelectTablesMock = {
        setGfc: vi.fn(),
        setAllTables: vi.fn(),
        updateGroups: vi.fn(),
        suggestLinks: vi.fn(),
        handleError: vi.fn()
    };

    const renderGraphLinkGroups = async (
        props: Partial<GraphLinkGroupsProps> = {}
    ) => {
        return await act(async () => {
            return render(
                <MemoryRouter>
                    <GraphLinkGroups
                        {...defaultGraphLinkGroupsProps}
                        {...props}
                    />
                </MemoryRouter>
            );
        });
    };

    const renderGraphLinkHealth = async (
        props: Partial<GraphLinkHealthProps> = {}
    ) => {
        return await act(async () => {
            return render(
                <MemoryRouter>
                    <GraphLinkHealth
                        {...defaultGraphLinkHealthProps}
                        {...props}
                    />
                </MemoryRouter>
            );
        });
    };

    const renderGraphNameInput = async (
        props: Partial<GraphNameInputProps> = {}
    ) => {
        return await act(async () => {
            return render(
                <MemoryRouter>
                    <GraphNameInput
                        {...defaultGraphNameInputProps}
                        {...props}
                    />
                </MemoryRouter>
            );
        });
    };

    const renderGraphLinkSetup = async (
        props: Partial<GraphLinkSetupProps> = {}
    ) => {
        return await act(async () => {
            return render(
                <MemoryRouter>
                    <QueryClientProvider client={queryClient}>
                        <GraphLinkSetup
                            {...defaultGraphLinkSetupProps}
                            {...props}
                        />
                    </QueryClientProvider>
                </MemoryRouter>
            );
        });
    };

    const renderGraphOverview = async (
        props: Partial<GraphOverviewProps> = {}
    ) => {
        return await act(async () => {
            return render(
                <MemoryRouter>
                    <QueryClientProvider client={queryClient}>
                        <GraphOverview
                            {...defaultGraphOverviewProps}
                            {...props}
                        />
                    </QueryClientProvider>
                </MemoryRouter>
            );
        });
    };

    const renderGraphSelectTables = async (
        props: Partial<GraphSelectTablesProps> = {}
    ) => {
        return await act(async () => {
            return render(
                <MemoryRouter>
                    <QueryClientProvider client={queryClient}>
                        <GraphSelectTables
                            {...defaultGraphSelectTablesProps}
                            {...props}
                        />
                    </QueryClientProvider>
                </MemoryRouter>
            );
        });
    };

    return {
        graphLinkHealthMocks,
        renderGraphLinkHealth,
        graphLinkGroupsMocks,
        renderGraphLinkGroups,
        graphNameInputMocks,
        renderGraphNameInput,
        graphLinkSetupMock,
        renderGraphLinkSetup,
        graphOverviewMock,
        renderGraphOverview,
        graphSelectTablesMock,
        renderGraphSelectTables
    };
};
