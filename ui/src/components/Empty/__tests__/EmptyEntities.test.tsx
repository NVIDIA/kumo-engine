// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import { fireEvent, render, screen } from "@testing-library/react";
import { mockNavigateCall, mockUseNavigate } from "tests/mockRouter";
import { vi } from "vitest";

import { PageNames } from "utils/Constants";

import EmptyEntities from "../EmptyEntities";

const emptyState = vi.hoisted(() => ({
    connectors: false,
    tables: false,
    graphs: false
}));

vi.mock("hooks/project/projects", () => ({
    useProjectId: () => "project-sales"
}));
vi.mock("hooks/connector/connector", () => ({
    useConnectors: () => ({
        isDataLoading: false,
        isConnectorsPresent: emptyState.connectors
    })
}));
vi.mock("hooks/graph/graphs", () => ({
    useGraphs: () => ({
        isDataLoading: false,
        isGraphsPresent: emptyState.graphs
    })
}));
vi.mock("hooks/tables/tables", () => ({
    useTables: () => ({
        isDataLoading: false,
        tablesPresent: emptyState.tables
    })
}));
vi.mock("hooks/jobs/list/jobList", () => ({
    useJobs: () => [[], 0, false, vi.fn(), false]
}));
describe("EmptyEntities", () => {
    beforeEach(() => {
        Object.assign(emptyState, {
            connectors: false,
            tables: false,
            graphs: false
        });
    });
    afterEach(() => vi.restoreAllMocks());

    it("creates the first connector inside the selected project", () => {
        mockNavigateCall();
        render(
            <MemoryRouter>
                <EmptyEntities currentPage={PageNames.Connector} />
            </MemoryRouter>
        );

        fireEvent.click(screen.getByRole("button", { name: "Add connector" }));
        expect(mockUseNavigate).toHaveBeenCalledWith(
            "/project/project-sales/connectors/new-connector"
        );
    });

    it.each([
        ["connector fallback", "", false, false],
        ["tables", PageNames.Table, false, false],
        ["graphs", PageNames.Jobs, true, false],
        ["training", PageNames.Jobs, true, true]
    ])(
        "opens the Studio installation guide from %s",
        (_, currentPage, tables, graphs) => {
            Object.assign(emptyState, { connectors: true, tables, graphs });
            const open = vi
                .spyOn(window, "open")
                .mockImplementation(() => null);
            render(
                <MemoryRouter>
                    <EmptyEntities currentPage={currentPage as string} />
                </MemoryRouter>
            );
            fireEvent.click(screen.getByRole("button", { name: "View docs" }));
            expect(open).toHaveBeenCalledWith(
                "https://docs.nvidia.com/sdgm/kumo-engine/1.0.0/kumo-studio-guide-coming-soon/preliminary-documentation/kumo-studio-code-guide/installation",
                "_blank"
            );
        }
    );
});
