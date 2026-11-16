// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { vi } from "vitest";

import { useConnectors } from "hooks/connector/connector";

import ConnectorsList from "../ConnectorsList";

const navigate = vi.fn();
const connector = {
    id: "connector-sales",
    projectId: "project-sales",
    identityId: "identity-sales",
    name: "Sales analytics",
    description: "",
    name_link: "/project/project-sales/connectors/connector-sales",
    created: "Jan 1, 2026",
    dateCreated: "Jan 1, 2026",
    sortCreated: "2026-01-01",
    updated: "Jan 1, 2026",
    dateUpdated: "Jan 1, 2026",
    sortUpdated: "2026-01-01",
    source: "databricks",
    config: {
        type: "databricks" as const,
        catalog: "sales",
        schema_name: "analytics",
        warehouse_id: "warehouse"
    },
    showDelete: true,
    timestamps: {
        created_at: "2026-01-01",
        updated_at: "2026-01-01"
    }
};

vi.mock("hooks/connector/connector", async () => {
    const { useEffect } =
        await vi.importActual<typeof import("react")>("react");
    return {
        useConnectors: vi.fn(),
        useConnectedTables: (
            _connectorId: string,
            _handleError: unknown,
            setLoading: (loading: boolean) => void
        ) => {
            useEffect(() => setLoading(false), [setLoading]);
            return { isLoading: false, tables: [], deleteTable: vi.fn() };
        },
        useCopyConnector: () => ({ copyConnector: vi.fn(), isCopying: false })
    };
});
vi.mock("hooks/project/projects", () => ({
    useProjectId: () => "project-sales",
    useProjectList: () => ({ projectListResponse: { data: [] } })
}));
vi.mock("hooks/validation/validation", () => ({
    useValidationState: () => [null, null, vi.fn(), vi.fn(), { current: null }]
}));
vi.mock("utils/navigator", () => ({
    useCustomNavigation: () => navigate
}));
vi.mock("components/Empty/EmptyEntities", () => ({
    default: () => <div>Project connector empty state</div>
}));

describe("ConnectorsList", () => {
    beforeEach(() => {
        vi.clearAllMocks();
        vi.mocked(useConnectors).mockReturnValue({
            isDataLoading: false,
            isConnectorsPresent: true,
            isDBConnectorsPresent: true,
            connectorList: [connector],
            deleteConnector: vi.fn()
        });
    });

    it("lists only the active project's connectors and creates in that project", () => {
        render(
            <MemoryRouter>
                <ConnectorsList />
            </MemoryRouter>
        );

        expect(screen.getByText("Sales analytics")).toBeInTheDocument();
        fireEvent.click(screen.getByRole("button", { name: "Add connector" }));
        expect(navigate).toHaveBeenCalledWith(
            "/project/project-sales/connectors/new-connector",
            { state: { connectorsPresent: true } }
        );
    });

    it("opens the copy dialog from the connector menu", async () => {
        render(
            <MemoryRouter
                initialEntries={[
                    "/project/project-sales/connectors/Sales analytics"
                ]}>
                <Routes>
                    <Route
                        path="/project/:projectId/connectors/:id"
                        element={<ConnectorsList />}
                    />
                </Routes>
            </MemoryRouter>
        );

        expect(
            screen.queryByTestId("copy-connector-dialog")
        ).not.toBeInTheDocument();
        await userEvent.click(
            await screen.findByTestId("more-options-dropdown")
        );
        await userEvent.click(
            await screen.findByRole("menuitem", { name: /Copy to project/ })
        );

        expect(
            await screen.findByTestId("copy-connector-dialog")
        ).toBeInTheDocument();
        expect(screen.getByLabelText("Connector name")).toHaveValue(
            "Sales analytics"
        );
    });

    it("uses the project empty state", () => {
        vi.mocked(useConnectors).mockReturnValue({
            isDataLoading: false,
            isConnectorsPresent: false,
            isDBConnectorsPresent: false,
            connectorList: [],
            deleteConnector: vi.fn()
        });

        render(
            <MemoryRouter>
                <ConnectorsList />
            </MemoryRouter>
        );
        expect(screen.getByText("Project connector empty state")).toBeVisible();
    });
});
import { MemoryRouter, Route, Routes } from "react-router";
